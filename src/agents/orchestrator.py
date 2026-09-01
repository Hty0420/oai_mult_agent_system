"""
Multi-Agent Workflow Orchestrator using LangGraph

This module handles orchestration of different agents using LangGraph.
It dynamically routes user queries to appropriate agent based on content and context.
"""

import json
import asyncio
import time
import uuid
from typing import Dict, List, Optional, Any, Literal, TypedDict, Union, Annotated
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import MessagesState, StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .state import AgentState, RouterDecision
from .config import Config
from .router_agent import RouterAgent
from .segmentation_agent import SegmentationAgent
from .feature_extraction_agent import FeatureExtractionAgent
from .feature_combination_agent import FeatureCombinationAgent
from .classification_agent import ClassificationAgent
from .conversation_agent import ConversationAgent
from .orchestrator_agent import OrchestratorAgent
from .web_search_agent import WebSearchAgent
from db_utils.db_utils import (
    set_agent_context,
    set_request_context,
    DatabaseManager,
    get_request_context,
    get_trace_id,
)
from db_utils.trajectory import (
    assign_trajectory_partition,
    build_trajectory_event,
    current_policy_id,
    stable_json_hash,
)
from db_utils.error_categorizer import categorize_error
from db_utils.task_evaluator import evaluate_task_run, persist_task_evaluation
from db_utils.task_learning import (
    build_outcome_summary,
    build_task_identity,
    rule_feedback_label,
    terminal_task_status,
)
from .dataset_profiler import profile_kl_dataset
from .pipeline_planner import (
    build_trial_portfolio,
    generate_pipeline_candidates,
    load_capability_registry,
    load_history_priors,
    rank_pipeline_candidates,
    select_pipeline_candidate,
)
from .task_contracts import (
    build_checkpoint_config,
    infer_task_spec,
    required_agent_for_task,
)

# Load configuration
config = Config()

# Initialize memory
memory = MemorySaver()

class MultiAgentOrchestrator:
    """Orchestrates multiple agents for OA medical imaging analysis"""
    
    def __init__(self, config: Config):
        self.config = config
        
        # Initialize all agents
        self.router_agent = RouterAgent(config)
        self.segmentation_agent = SegmentationAgent(config)
        self.feature_extraction_agent = FeatureExtractionAgent(config)
        self.feature_combination_agent = FeatureCombinationAgent(config)
        self.classification_agent = ClassificationAgent(config)
        self.conversation_agent = ConversationAgent(config)
        self.orchestrator_agent = OrchestratorAgent(config)
        self.web_search_agent = WebSearchAgent(config)
        
        # Create the graph
        self.graph = self._create_graph()
    
    def _create_graph(self) -> StateGraph:
        """Create and configure the LangGraph for agent orchestration"""
        
        # Initialize the graph with AgentState
        workflow = StateGraph(AgentState)
        
        # Add nodes for each agent (using async wrappers)
        workflow.add_node("router", self._route_to_agent)
        workflow.add_node("segmentation_agent", self._async_wrapper(self.segmentation_agent.process))
        workflow.add_node("feature_extraction_agent", self._async_wrapper(self.feature_extraction_agent.process))
        workflow.add_node("feature_combination_agent", self._async_wrapper(self.feature_combination_agent.process))
        workflow.add_node("classification_agent", self._async_wrapper(self.classification_agent.process))
        workflow.add_node("conversation_agent", self._async_wrapper(self.conversation_agent.process))
        workflow.add_node("orchestrator_agent", self._async_wrapper(self.orchestrator_agent.process))
        workflow.add_node("web_search_agent", self._async_wrapper(self.web_search_agent.process))
        
        # Set the entry point
        workflow.set_entry_point("router")
        
        # Define routing logic
        workflow.add_conditional_edges(
            "router",
            self._should_continue,
            {
                "SEGMENTATION_AGENT": "segmentation_agent",
                "FEATURE_EXTRACTION_AGENT": "feature_extraction_agent",
                "FEATURE_COMBINATION_AGENT": "feature_combination_agent",
                "CLASSIFICATION_AGENT": "classification_agent",
                "CONVERSATION_AGENT": "conversation_agent",
                "ORCHESTRATOR_AGENT": "orchestrator_agent",
                "WEB_SEARCH_AGENT": "web_search_agent"
            }
        )
        
        # All agent nodes lead to END
        workflow.add_edge("segmentation_agent", END)
        workflow.add_edge("feature_extraction_agent", END)
        workflow.add_edge("feature_combination_agent", END)
        workflow.add_edge("classification_agent", END)
        workflow.add_edge("conversation_agent", END)
        workflow.add_edge("orchestrator_agent", END)
        workflow.add_edge("web_search_agent", END)
        
        # Compile the graph with memory
        return workflow.compile(checkpointer=memory)
    
    def _async_wrapper(self, func):
        """Wrapper to make both sync and async functions work with LangGraph"""
        async def async_wrapper(state: AgentState) -> AgentState:
            import inspect

            # Create a copy of the state and add stream_callback
            state_copy = state.copy()
            state_copy["stream_callback"] = getattr(self, "_current_stream_callback", None)

            # 设置 Agent 上下文，用于 tool_calls_log
            agent_name = state_copy.get("agent_name", "UNKNOWN")
            trace_id = state_copy.get("trace_id") or get_trace_id()
            set_agent_context(agent_name)

            start_time = time.time()
            success = True
            error_category = None
            error_message = None
            result = None

            try:
                # LangGraph executes this node on the task's existing event
                # loop.  Never create a short-lived loop here: MCP sessions
                # and AnyIO streams are loop-affine and must remain on the
                # same loop for the complete orchestration run.
                result = func(state_copy)
                if inspect.isawaitable(result):
                    result = await result
                if isinstance(result, dict) and result.get("success") is False:
                    success = False
                    error_category = result.get("error_category") or "step_failure"
                    error_message = result.get("error_message")
            except Exception as e:
                success = False
                error_message = str(e)
                error_category = categorize_error(error_message)
                # Re-raise so LangGraph can handle the error
                result = {
                    "output": f"Error during {agent_name} processing: {error_message}",
                    "agent_name": agent_name,
                }
                print(f"[Orchestrator] Agent {agent_name} 执行失败: {error_message}")
                import traceback
                traceback.print_exc()
            finally:
                total_latency_ms = int((time.time() - start_time) * 1000)
                tool_count = 0
                if isinstance(result, dict):
                    tool_calls = result.get("tool_calls", [])
                    tool_count = len(tool_calls) if tool_calls else 0

                # 记录 Agent 执行日志
                db = None
                try:
                    ctx = get_request_context()
                    user_id = ctx.get('user_id')
                    session_id = ctx.get('session_id')
                    if user_id and session_id:
                        agent_instance = getattr(func, "__self__", None)
                        llm = getattr(agent_instance, "llm", None)
                        model_name = (
                            getattr(llm, "model_name", None)
                            or getattr(llm, "model", None)
                        )
                        prompt_hash = stable_json_hash(
                            getattr(agent_instance, "system_prompt", "")
                        )
                        db = DatabaseManager()
                        db.save_agent_execution_log(
                            user_id=user_id,
                            session_id=session_id,
                            agent_name=agent_name,
                            trace_id=trace_id,
                            total_latency_ms=total_latency_ms,
                            tool_count=tool_count,
                            success=success,
                            error_category=error_category,
                            error_message=error_message,
                            model_name=str(model_name) if model_name else None,
                            prompt_hash=prompt_hash,
                        )
                        db.save_trajectory_event(
                            build_trajectory_event(
                                event_type="agent_completed",
                                trace_id=trace_id,
                                actor_type="agent",
                                actor_name=agent_name,
                                status="success" if success else "failed",
                                duration_ms=total_latency_ms,
                                payload={
                                    "tool_count": tool_count,
                                    "error_category": error_category,
                                    "model_name": str(model_name) if model_name else None,
                                    "prompt_hash": prompt_hash,
                                    "output_sha256": stable_json_hash(
                                        result.get("output", "")
                                        if isinstance(result, dict) else ""
                                    ),
                                },
                            )
                        )
                except Exception as log_error:
                    print(f"[Orchestrator] 记录 Agent 执行日志失败: {log_error}")
                finally:
                    if db is not None:
                        try:
                            db.close()
                        except Exception:
                            pass

            if result is None:
                result = {
                    "output": f"Agent {agent_name} returned no result",
                    "agent_name": agent_name,
                }

            # Remove stream_callback from result before returning to LangGraph
            if isinstance(result, dict) and "stream_callback" in result:
                del result["stream_callback"]

            return result
        return async_wrapper
    
    def _route_to_agent(self, state: AgentState) -> Dict[str, Any]:
        """Route query to appropriate agent"""
        print("\n[Orchestrator] Routing query to appropriate agent...")
        
        # Create a copy of the state and add stream_callback
        state_copy = state.copy()
        state_copy["stream_callback"] = getattr(self, "_current_stream_callback", None)
        
        # Use router agent to make decision
        decision = self.router_agent.route(state_copy)
        
        # Remove stream_callback from state before returning to LangGraph
        if "stream_callback" in decision["agent_state"]:
            del decision["agent_state"]["stream_callback"]
        
        return decision["agent_state"]
    
    def _should_continue(self, state: AgentState) -> str:
        """Determine if we should continue or end"""
        next_agent = state.get("next_agent", "CONVERSATION_AGENT")
        
        # Map agent names to graph nodes
        agent_mapping = {
            "SEGMENTATION_AGENT": "SEGMENTATION_AGENT",
            "FEATURE_EXTRACTION_AGENT": "FEATURE_EXTRACTION_AGENT",
            "FEATURE_COMBINATION_AGENT": "FEATURE_COMBINATION_AGENT",
            "CLASSIFICATION_AGENT": "CLASSIFICATION_AGENT",
            "CONVERSATION_AGENT": "CONVERSATION_AGENT",
            "ORCHESTRATOR_AGENT": "ORCHESTRATOR_AGENT",
            "WEB_SEARCH_AGENT": "WEB_SEARCH_AGENT"
        }
        
        return agent_mapping.get(next_agent, "CONVERSATION_AGENT")
    
    async def aprocess_query(self, query: Union[str, Dict], conversation_history: List[BaseMessage] = None, stream_callback=None) -> Dict[str, Any]:
        """
        Process a user query through the multi-agent system
        
        Args:
            query: User query (string or dict with text and image)
            conversation_history: List of previous messages
            stream_callback: Callback function for streaming output
            
        Returns:
            Dictionary with response and metadata
        """
        if conversation_history is None:
            conversation_history = []

        # A session owns conversational context; every request gets a distinct
        # run/trace and checkpoint namespace so execution state cannot leak.
        ctx = get_request_context()
        run_id = str(uuid.uuid4())
        set_request_context(
            user_id=ctx.get("user_id"),
            session_id=ctx.get("session_id"),
            agent_name=ctx.get("agent_name"),
            trace_id=run_id,
            workspace_id=ctx.get("workspace_id"),
            agent_version=ctx.get("agent_version"),
        )
        
        if isinstance(query, str):
            user_message = HumanMessage(content=query)
        elif isinstance(query, dict):
            text = query.get("text", "")
            user_message = HumanMessage(content=text)
        else:
            user_message = HumanMessage(content=str(query))
        
        query_text = query if isinstance(query, str) else str(query.get("text", "")) if isinstance(query, dict) else str(query)
        task_spec = infer_task_spec(query_text)
        goal_spec = dict(task_spec.get("goal_spec", {}))
        dataset_profile: dict[str, Any] = {}
        pipeline_candidates: list[dict[str, Any]] = []
        selected_pipeline: dict[str, Any] = {}
        planning_decision: dict[str, Any] = {}
        if (
            goal_spec.get("goal_type") == "KL_GRADING"
            and goal_spec.get("autonomy_mode") in {"auto", "partially_constrained"}
        ):
            try:
                registry = load_capability_registry()
                session_csv_paths: list[str] = []
                session_db = None
                try:
                    if ctx.get("user_id") is not None and ctx.get("session_id"):
                        session_db = DatabaseManager()
                        session_csv_paths = session_db.get_recent_session_csv_files(
                            ctx.get("user_id"), ctx.get("session_id")
                        )
                except Exception:
                    session_csv_paths = []
                finally:
                    if session_db is not None:
                        try:
                            session_db.close()
                        except Exception:
                            pass
                dataset_profile = profile_kl_dataset(
                    query_text,
                    preferred_paths=session_csv_paths,
                )
                planning_identity = build_task_identity(
                    task_spec=task_spec,
                    goal_spec=goal_spec,
                    dataset_profile=dataset_profile,
                    policy_id=current_policy_id(),
                )
                history = load_history_priors(
                    task_fingerprint=planning_identity["task_fingerprint"]
                )
                pipeline_candidates = rank_pipeline_candidates(
                    generate_pipeline_candidates(
                        goal_spec=goal_spec,
                        dataset_profile=dataset_profile,
                        registry=registry,
                    ),
                    history=history,
                    selection_policy=registry.get("selection_policy", {}),
                )
                selected_pipeline = select_pipeline_candidate(
                    pipeline_candidates
                ) or {}
                trial_portfolio = build_trial_portfolio(
                    pipeline_candidates,
                    search_policy=registry.get("search_policy", {}),
                )
                planning_decision = {
                    "registry_id": registry.get("registry_id"),
                    "registry_version": registry.get("version"),
                    "selection_policy_version": registry.get(
                        "selection_policy", {}
                    ).get("version"),
                    "llm_tie_tolerance": registry.get(
                        "selection_policy", {}
                    ).get("llm_tie_tolerance", 0.02),
                    "history_status": history.get("status"),
                    "initial_selected_candidate_id": selected_pipeline.get(
                        "candidate_id"
                    ),
                    "search_policy": registry.get("search_policy", {}),
                    "trial_portfolio": trial_portfolio,
                    "selection_basis": (
                        "training_only_nested_cv_then_prior_tie_break"
                    ),
                    "dataset_source": (
                        "current_session_registry"
                        if session_csv_paths
                        else "current_request_or_bounded_default"
                    ),
                }
            except Exception as planning_error:
                dataset_profile = {
                    "status": "blocked",
                    "issues": [
                        f"planning_initialization_failed:{type(planning_error).__name__}"
                    ],
                }
                planning_decision = {
                    "status": "blocked",
                    "reason": type(planning_error).__name__,
                }
        elif goal_spec.get("goal_type") == "BIOMARKER_DISCOVERY":
            try:
                dataset_profile = profile_kl_dataset(query_text)
                planning_decision = {
                    "status": (
                        "ready"
                        if dataset_profile.get("status") == "ready"
                        else "blocked"
                    ),
                    "plan_type": "fixed_biomarker_discovery_v1",
                    "selection_basis": "pre_registered_scientific_protocol",
                    "test_data_used_for_discovery": False,
                    "dataset_source": "current_request_or_bounded_default",
                }
            except Exception as planning_error:
                dataset_profile = {
                    "status": "blocked",
                    "issues": [
                        f"biomarker_profile_failed:{type(planning_error).__name__}"
                    ],
                }
                planning_decision = {
                    "status": "blocked",
                    "reason": type(planning_error).__name__,
                }
        task_identity = build_task_identity(
            task_spec=task_spec,
            goal_spec=goal_spec,
            dataset_profile=dataset_profile,
            policy_id=current_policy_id(),
            selected_pipeline=selected_pipeline,
        )
        data_partition = assign_trajectory_partition(
            task_identity["task_fingerprint"]
        )
        set_request_context(
            user_id=ctx.get("user_id"),
            session_id=ctx.get("session_id"),
            agent_name=ctx.get("agent_name"),
            trace_id=run_id,
            workspace_id=ctx.get("workspace_id"),
            agent_version=ctx.get("agent_version"),
            data_partition=data_partition,
        )
        initial_state = {
            "messages": conversation_history + [user_message],
            "current_input": query,
            "output": "",
            "agent_name": "",
            "tool_calls": [],
            "tool_results": [],
            "next_agent": "CONVERSATION_AGENT",
            "routing_decision_source": None,
            "trace_id": run_id,
            "task_run_id": run_id,
            "task_spec": task_spec,
            "goal_spec": goal_spec,
            "dataset_profile": dataset_profile,
            "pipeline_candidates": pipeline_candidates,
            "selected_pipeline": selected_pipeline,
            "planning_decision": planning_decision,
            "trial_results": [],
            "pipeline_trial_records": [],
            "trial_record_errors": [],
            "selected_feature_paths": {},
            "pipeline_search_manifest_path": None,
            "biomarker_result": {},
            "biomarker_audit": {},
            "biomarker_narrative": None,
            "biomarker_reporting": {},
            "current_artifacts": [],
            "success": None,
            "error_message": None,
            "error_category": None,
            "task_id": None,
            "task_fingerprint": task_identity["task_fingerprint"],
            "execution_fingerprint": task_identity["execution_fingerprint"],
            "outcome_summary": {},
            "execution_plan": {},
            "step_records": [],
            "task_evaluation": {},
            "policy_id": current_policy_id(),
        }

        # Create one authoritative task row for every routed request.  The
        # orchestrator agent enriches this row with its plan; it must not create
        # a second task or decide the terminal state before evaluation gates run.
        task_db = None
        try:
            if ctx.get("user_id") is not None and ctx.get("session_id"):
                task_db = DatabaseManager()
                task_id = task_db.create_task(
                    user_id=ctx.get("user_id"),
                    session_id=ctx.get("session_id"),
                    task_description=query_text,
                    trace_id=run_id,
                    policy_id=initial_state["policy_id"],
                    task_fingerprint=task_identity["task_fingerprint"],
                    execution_fingerprint=task_identity[
                        "execution_fingerprint"
                    ],
                )
                initial_state["task_id"] = task_id
                task_db.save_trajectory_event(
                    build_trajectory_event(
                        event_type="run_started",
                        trace_id=run_id,
                        task_id=task_id,
                        actor_type="control_plane",
                        actor_name="task_lifecycle",
                        status="running",
                        payload={
                            "query_sha256": stable_json_hash(query_text),
                            "task_fingerprint": task_identity[
                                "task_fingerprint"
                            ],
                            "execution_fingerprint": task_identity[
                                "execution_fingerprint"
                            ],
                            "dataset_hash": dataset_profile.get(
                                "dataset_hash"
                            ),
                            "objective_variant": goal_spec.get(
                                "objective_variant"
                            ),
                        },
                    )
                )
        except Exception as task_error:
            print(f"[Orchestrator] Failed to initialize task lifecycle: {task_error}")
        finally:
            if task_db is not None:
                try:
                    task_db.close()
                except Exception:
                    pass
        
        self._current_stream_callback = stream_callback

        if stream_callback:
            stream_callback("\n" + "=" * 60)
            stream_callback("📋 任务开始处理")
            stream_callback("=" * 60)

        start_time = time.time()

        def _evaluate_and_persist(run_result: dict) -> dict:
            observed_steps: list[dict[str, Any]] = []
            try:
                evaluated_agent = run_result.get("agent_name", "UNKNOWN")
                observed_steps = run_result.get("step_records", []) or [
                    {
                        "step": 1,
                        "agent": evaluated_agent,
                        "task": query_text,
                        "status": (
                            "failed" if run_result.get("success") is False else "success"
                        ),
                        "tool_calls": list(run_result.get("tool_calls", []) or []),
                    }
                ]
                evaluated_task_spec = run_result.get("task_spec") or task_spec
                report = evaluate_task_run(
                    trace_id=run_id,
                    run_id=run_result.get("task_run_id") or run_id,
                    task_id=run_result.get("task_id"),
                    policy_id=run_result.get("policy_id") or initial_state["policy_id"],
                    task_spec=evaluated_task_spec,
                    routed_agent=run_result.get("next_agent"),
                    step_records=observed_steps,
                    current_artifacts=run_result.get("current_artifacts", []) or [],
                    overall_success=run_result.get("success") is not False,
                    final_output=run_result.get("output", ""),
                    dataset_hash=(
                        run_result.get("dataset_profile", {}) or {}
                    ).get("dataset_hash"),
                )
                run_result["task_evaluation"] = report
            except Exception as evaluation_error:
                print(f"[Orchestrator] 生成任务级评价失败: {evaluation_error}")
                run_result["success"] = False
                run_result["outcome_summary"] = {
                    "schema_version": "1.0",
                    "outcome_status": "needs_review",
                    "task_success": False,
                    "completion_gate_status": "unknown",
                    "failed_gates": ["task_evaluation_unavailable"],
                    "requires_human_review": True,
                }
                fallback_db = None
                try:
                    if run_result.get("task_id"):
                        fallback_db = DatabaseManager()
                        fallback_db.update_task(
                            task_id=run_result["task_id"],
                            status="failed",
                            outcome_status="needs_review",
                            completion_gate_status="unknown",
                            outcome_summary=run_result["outcome_summary"],
                            error_message=(
                                "Deterministic task evaluation failed; "
                                "human review is required."
                            ),
                            error_category="evaluation_failure",
                        )
                except Exception as persist_error:
                    print(
                        "[Orchestrator] Failed to persist evaluation fallback: "
                        f"{persist_error}"
                    )
                finally:
                    if fallback_db is not None:
                        try:
                            fallback_db.close()
                        except Exception:
                            pass
                return {}

            final_identity = build_task_identity(
                task_spec=evaluated_task_spec,
                goal_spec=run_result.get("goal_spec") or goal_spec,
                dataset_profile=(
                    run_result.get("dataset_profile") or dataset_profile
                ),
                policy_id=(
                    run_result.get("policy_id") or initial_state["policy_id"]
                ),
                selected_pipeline=run_result.get("selected_pipeline") or {},
            )
            outcome = build_outcome_summary(
                evaluation=report,
                step_records=observed_steps,
                selected_pipeline=run_result.get("selected_pipeline") or {},
                dataset_hash=report.get("dataset_hash"),
                total_latency_ms=int((time.time() - start_time) * 1000),
            )
            run_result["task_fingerprint"] = final_identity["task_fingerprint"]
            run_result["execution_fingerprint"] = final_identity[
                "execution_fingerprint"
            ]
            run_result["outcome_summary"] = outcome
            run_result["success"] = bool(report.get("task_success"))
            if not run_result["success"]:
                failed_gates = ", ".join(outcome.get("failed_gates", []))
                verification_notice = (
                    "\n\n系统完成校验未通过，当前任务不能标记为完成。"
                    f"未通过项：{failed_gates or 'unknown'}。"
                )
                if verification_notice not in str(run_result.get("output", "")):
                    run_result["output"] = (
                        str(run_result.get("output", "")) + verification_notice
                    )

            evaluation_db = None
            try:
                evaluation_ctx = get_request_context()
                if evaluation_ctx.get("user_id") and evaluation_ctx.get("session_id"):
                    evaluation_db = DatabaseManager()
                    persist_task_evaluation(evaluation_db, report)
                    task_id = run_result.get("task_id")
                    if task_id:
                        evaluation_db.update_task(
                            task_id=task_id,
                            completed_steps=sum(
                                str(item.get("status", "")).lower() == "success"
                                for item in observed_steps
                            ),
                            status=terminal_task_status(outcome),
                            result_summary=json.dumps(
                                outcome, ensure_ascii=False
                            ),
                            error_message=(
                                None
                                if report.get("task_success")
                                else "Task completion gates did not all pass."
                            ),
                            error_category=(
                                None
                                if report.get("task_success")
                                else "completion_gate_failure"
                            ),
                            routed_agent=run_result.get("next_agent"),
                            task_fingerprint=final_identity[
                                "task_fingerprint"
                            ],
                            execution_fingerprint=final_identity[
                                "execution_fingerprint"
                            ],
                            outcome_status=outcome.get("outcome_status"),
                            completion_gate_status=outcome.get(
                                "completion_gate_status"
                            ),
                            outcome_summary=outcome,
                        )

                    expected_agent = required_agent_for_task(
                        evaluated_task_spec
                    )
                    observed_agent = run_result.get("next_agent")
                    if expected_agent:
                        evaluation_db.save_routing_feedback(
                            trace_id=run_id,
                            routing_log_id=run_result.get("routing_log_id"),
                            expected_agent=expected_agent,
                            observed_agent=observed_agent,
                            label_source="rule",
                            notes=(
                                "Independent deterministic label derived from "
                                "the current-request task specification."
                            ),
                        )

                    feedback_label = rule_feedback_label(outcome)
                    evaluation_db.save_task_feedback(
                        trace_id=run_id,
                        task_id=task_id,
                        feedback_source="rule",
                        outcome_label=feedback_label,
                        accepted=None,
                        correction_required=(feedback_label != "correct"),
                        confidence=1.0,
                        expected_agent=expected_agent,
                        expected_pipeline_id=(
                            (run_result.get("selected_pipeline") or {}).get(
                                "candidate_id"
                            )
                        ),
                        error_types=outcome.get("failed_gates", []),
                        notes="Deterministic task-completion gate result.",
                    )
                    evaluation_db.save_trajectory_event(
                        build_trajectory_event(
                            event_type="feedback_recorded",
                            trace_id=run_id,
                            task_id=task_id,
                            actor_type="evaluator",
                            actor_name="task-contract-evaluator",
                            status="success",
                            payload={
                                "feedback_source": "rule",
                                "outcome_label": feedback_label,
                                "expected_agent": expected_agent,
                            },
                        )
                    )
                    evaluation_db.save_trajectory_event(
                        build_trajectory_event(
                            event_type="run_completed",
                            trace_id=run_id,
                            task_id=task_id,
                            actor_type="control_plane",
                            actor_name="task_lifecycle",
                            status=(
                                "success"
                                if report.get("task_success")
                                else "failed"
                            ),
                            duration_ms=outcome.get("total_latency_ms"),
                            payload={
                                "task_fingerprint": final_identity[
                                    "task_fingerprint"
                                ],
                                "execution_fingerprint": final_identity[
                                    "execution_fingerprint"
                                ],
                                "outcome_status": outcome.get(
                                    "outcome_status"
                                ),
                                "failed_gates": outcome.get(
                                    "failed_gates", []
                                ),
                            },
                        )
                    )
            except Exception as evaluation_error:
                print(f"[Orchestrator] 记录任务级评价失败: {evaluation_error}")
            finally:
                if evaluation_db is not None:
                    try:
                        evaluation_db.close()
                    except Exception:
                        pass
            return report

        try:
            thread_config = build_checkpoint_config(
                user_id=ctx.get("user_id"),
                session_id=ctx.get("session_id"),
                run_id=run_id,
            )
            result = await self.graph.ainvoke(
                initial_state, config=thread_config
            )
            agent_name = result.get('agent_name', 'UNKNOWN')
            _evaluate_and_persist(result)

            if stream_callback:
                stream_callback("\n" + "=" * 60)
                if result.get("success") is False:
                    stream_callback("❌ 任务未完成")
                else:
                    stream_callback("✅ 任务处理完成")
                stream_callback("=" * 60)
        except Exception as e:
            import traceback
            traceback.print_exc()
            result = initial_state
            result["output"] = f"Error during processing: {str(e)}"
            result["agent_name"] = "ERROR"
            result["success"] = False
            _evaluate_and_persist(result)
            if stream_callback:
                stream_callback(f"\n❌ 处理过程中出错: {str(e)}")

        total_latency_ms = int((time.time() - start_time) * 1000)

        # 更新路由日志，补全 final_output_agent 和 total_latency_ms
        routing_log_id = result.get("routing_log_id")
        if routing_log_id:
            db = None
            try:
                db = DatabaseManager()
                db.update_routing_log(
                    routing_id=routing_log_id,
                    final_output_agent=result.get("agent_name", "UNKNOWN"),
                    total_latency_ms=total_latency_ms
                )
            except Exception as e:
                print(f"[Orchestrator] 更新路由日志失败: {e}")
            finally:
                if db is not None:
                    try:
                        db.close()
                    except Exception:
                        pass

        return {
            "messages": result["messages"],
            "response": result.get("output", ""),
            "agent_name": result.get("agent_name", "UNKNOWN"),
            "tool_calls": result.get("tool_calls", []),
            "tool_results": result.get("tool_results", []),
            "task_evaluation": result.get("task_evaluation", {}),
            "task_id": result.get("task_id"),
            "trace_id": run_id,
            "workspace_id": ctx.get("workspace_id"),
            "agent_version": ctx.get("agent_version"),
            "policy_id": current_policy_id(),
            "data_partition": data_partition,
            "task_fingerprint": result.get(
                "task_fingerprint", task_identity["task_fingerprint"]
            ),
            "execution_fingerprint": result.get(
                "execution_fingerprint",
                task_identity["execution_fingerprint"],
            ),
            "outcome_summary": result.get("outcome_summary", {}),
            "success": result.get("success") is True,
            "goal_spec": result.get("goal_spec", goal_spec),
            "dataset_profile": result.get("dataset_profile", dataset_profile),
            "pipeline_candidates": result.get(
                "pipeline_candidates", pipeline_candidates
            ),
            "selected_pipeline": result.get(
                "selected_pipeline", selected_pipeline
            ),
            "planning_decision": result.get(
                "planning_decision", planning_decision
            ),
            "trial_results": result.get("trial_results", []),
            "pipeline_trial_records": result.get(
                "pipeline_trial_records", []
            ),
            "trial_record_errors": result.get("trial_record_errors", []),
            "selected_feature_paths": result.get("selected_feature_paths", {}),
            "pipeline_search_manifest_path": result.get(
                "pipeline_search_manifest_path"
            ),
        }

    def process_query(
        self,
        query: Union[str, Dict],
        conversation_history: List[BaseMessage] = None,
        stream_callback=None,
    ) -> Dict[str, Any]:
        """Synchronous compatibility entry point.

        The workbench uses ``aprocess_query`` on its persistent execution
        loop. Standalone synchronous callers receive one task-scoped loop and
        explicitly close that loop's MCP connections before it exits.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError(
                "process_query() cannot run inside an active event loop; "
                "await aprocess_query() instead"
            )

        async def run_and_close_tools():
            from .tool_manager import reset_tool_manager

            try:
                return await self.aprocess_query(
                    query, conversation_history, stream_callback
                )
            finally:
                await reset_tool_manager()

        return asyncio.run(run_and_close_tools())


# Global orchestrator instance
_orchestrator = None

def get_orchestrator() -> MultiAgentOrchestrator:
    """Get or create the global orchestrator instance"""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = MultiAgentOrchestrator(Config())
    return _orchestrator


def process_query(query: Union[str, Dict], conversation_history: List[BaseMessage] = None, stream_callback=None) -> Dict[str, Any]:
    """
    Process a user query through the multi-agent system
    
    This is the main entry point for the multi-agent system.
    
    Args:
        query: User query (string or dict with text and image)
        conversation_history: List of previous messages
        stream_callback: Callback function for streaming output
        
    Returns:
        Dictionary with response and metadata
    """
    orchestrator = get_orchestrator()
    return orchestrator.process_query(query, conversation_history, stream_callback)


async def aprocess_query(
    query: Union[str, Dict],
    conversation_history: List[BaseMessage] = None,
    stream_callback=None,
) -> Dict[str, Any]:
    """Asynchronous entry point for long-lived application event loops."""
    orchestrator = get_orchestrator()
    return await orchestrator.aprocess_query(
        query, conversation_history, stream_callback
    )
