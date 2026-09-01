"""
Orchestrator Agent - Coordinates multiple agents for complex multi-step tasks

This agent analyzes complex user requests that require multiple agents to work together.
It identifies which agents to call, in what order, and coordinates their execution.
"""

from typing import Dict, List, Optional, Any, Mapping
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
import json
import uuid

from .state import AgentState
from .config import Config
from .segmentation_agent import SegmentationAgent
from .feature_extraction_agent import FeatureExtractionAgent
from .classification_agent import ClassificationAgent
from .pipeline_optimization_agent import PipelineOptimizationAgent
from .biomarker_discovery_agent import BiomarkerDiscoveryAgent
from .biomarker_verifier_agent import BiomarkerVerifierAgent
from .biomarker_reporter_agent import BiomarkerReporterAgent
from .conversation_agent import ConversationAgent
from .llm_json import parse_llm_json
from .path_utils import parse_tool_result_paths, infer_dataset_role
from .pipeline_planner import (
    materialize_pipeline_plan,
    materialize_pipeline_search_plan,
    select_pipeline_candidate,
)
from .task_contracts import (
    apply_selected_pipeline,
    infer_feature_type,
    materialize_biomarker_plan,
    normalize_orchestrator_plan,
)
from db_utils.db_utils import (
    DatabaseManager,
    get_request_context,
    get_trace_id,
    set_agent_context,
)
from db_utils.error_categorizer import categorize_error
from db_utils.trajectory import (
    build_artifact_record,
    build_trajectory_event,
    stable_json_hash,
)


class OrchestratorAgent:
    """Agent that coordinates multiple agents for complex multi-step tasks"""
    
    def __init__(self, config: Config):
        self.config = config
        self.llm = config.orchestrator_agent.llm if hasattr(config, 'orchestrator_agent') else config.router_agent.llm
        
        # Initialize sub-agents
        self.segmentation_agent = SegmentationAgent(config)
        self.feature_extraction_agent = FeatureExtractionAgent(config)
        self.classification_agent = ClassificationAgent(config)
        self.pipeline_optimization_agent = PipelineOptimizationAgent(config)
        self.biomarker_discovery_agent = BiomarkerDiscoveryAgent(config)
        self.biomarker_verifier_agent = BiomarkerVerifierAgent(config)
        self.biomarker_reporter_agent = BiomarkerReporterAgent(config)
        self.conversation_agent = ConversationAgent(config)
        
        # Map agent names to agent instances
        self.agents = {
            "SEGMENTATION_AGENT": self.segmentation_agent,
            "FEATURE_EXTRACTION_AGENT": self.feature_extraction_agent,
            "CLASSIFICATION_AGENT": self.classification_agent,
            "PIPELINE_OPTIMIZATION_AGENT": self.pipeline_optimization_agent,
            "BIOMARKER_DISCOVERY_AGENT": self.biomarker_discovery_agent,
            "BIOMARKER_VERIFIER_AGENT": self.biomarker_verifier_agent,
            "BIOMARKER_REPORTER_AGENT": self.biomarker_reporter_agent,
            "CONVERSATION_AGENT": self.conversation_agent
        }
        
        self.system_prompt = """你是医学影像多 agent 系统中的任务编排 agent。
你的职责是分析复杂用户请求，拆解任务步骤，并协调多个专门 agent 按正确顺序完成任务。

除非用户明确要求使用其他语言，否则所有 task_description、input_override、final_message、解释和总结都使用简体中文。
但 JSON key、agent 名称、tool 名称、参数名、文件路径、模型名必须保持英文原样，不要翻译。
编排协议必须严格保持英文标识，例如 "steps"、"agent"、"task_description"、"input_override"、"final_message"。

可用 Agents：
1. SEGMENTATION_AGENT - 使用 U-Mamba、LKM-UNet、LightM-UNet 等模型处理 MRI 图像分割
   - 当用户想分割 MRI 图像时使用
   - Tools: umamba_segment_mri, lkm_unet_segment_mri, lightm_unet_segment_mri

2. FEATURE_EXTRACTION_AGENT - 处理 MRI 图像特征提取
   - 当用户想提取 radiomics、MEAS 或 LDSE 特征时使用
   - CRITICAL: 数据增强工具（augment_dataset_offline、augment_online_extract_features）
     和质量评估工具（assess_feature_quality）只能在用户明确要求时加入，
     例如用户提到 "data augmentation"、"数据增强"、"augment"、"quality assessment"。
     不要仅因为用户提到分类或 LASSO 就主动加入这些步骤。
   - Tools: extract_radiomics_features, meas_features_extraction, ldse_features_extraction,
     ldse_train_test_features_extraction,
     augment_dataset_offline, augment_online_extract_features, assess_feature_quality
   - 训练集和测试集都有时，配对 LDSE 提取是 FEATURE_EXTRACTION_AGENT 内部的一步，
     使用一次 ldse_train_test_features_extraction，不要拆成多个 agent 步骤。

3. CLASSIFICATION_AGENT - 处理 KOA 分类和 KL 分级
   - 当用户想执行分类、KL 分级或读取分类结果时使用
   - Tools: koa_lasso_classification, koa_filter_classification, read_classification_results, etc.

4. CONVERSATION_AGENT - 处理普通问题和解释说明
   - 当用户提出一般问题、流程解释或结果解释时使用

5. BIOMARKER_DISCOVERY_AGENT - 执行预注册的 KL 0-4 横断面 MRI 候选标志物统计分析
   - 仅用训练集发现候选，冻结候选后才在测试集确认
   - Tool: run_biomarker_discovery

6. BIOMARKER_VERIFIER_AGENT - 独立复核标志物运行，不修改候选集合
   - Tool: audit_biomarker_discovery

7. BIOMARKER_REPORTER_AGENT - 仅根据独立审计通过的结构化事实自然讲解本次结果
   - 不调用工具，不改变候选或数字；生成文本必须通过事实一致性校验

任务分析规则：
1. 识别用户请求中的所有独立任务
2. 判断每个任务需要哪个或哪些 agent
3. 判断正确执行顺序
4. 有些任务可能需要同一个 agent 使用不同参数执行多次
5. 有些任务可能需要不同 agent 按顺序执行
6. IMPORTANT: 如果单个 agent 能在内部处理复杂请求，例如“分别对训练集和测试集提取特征”，应调用该 agent 一次并传入完整请求，而不是拆成多次调用。
7. PIPELINE RULE: 当某个操作的输出文件是另一个操作的输入时，必须拆成独立步骤，以便前一步生成的文件路径能明确传给下一步。具体规则：
   - "data augmentation -> feature extraction -> classification" = 3 个独立步骤，不要合并
   - "feature extraction -> classification" = 2 个独立步骤，不要合并
   - 只有在操作互相独立时，才可以合并到同一步，例如对不同数据集执行同一类独立操作

常见多 agent 任务模式：
1. 多个数据集的特征提取 -> FEATURE_EXTRACTION_AGENT，单次调用，由该 agent 内部处理多个数据集
2. 特征提取 + 分类 -> FEATURE_EXTRACTION_AGENT + CLASSIFICATION_AGENT，必须拆成独立步骤
3. 分割 + 特征提取 -> SEGMENTATION_AGENT + FEATURE_EXTRACTION_AGENT
4. 特征提取 + 分类 + 读取结果 -> FEATURE_EXTRACTION_AGENT + CLASSIFICATION_AGENT；若分类已有路径，可由 CLASSIFICATION_AGENT 单次处理分类和读取
5. 分类 + 读取结果 -> CLASSIFICATION_AGENT，单次调用，由该 agent 内部处理
6. 多种特征类型提取 -> FEATURE_EXTRACTION_AGENT，单次调用，由该 agent 内部处理
7. 数据增强 -> FEATURE_EXTRACTION_AGENT，使用 augment_dataset_offline，单次调用
   只有当用户明确要求 data augmentation / 数据增强时，才应用模式 7-9。
   CRITICAL: input_override 必须明确用户要求的特征类型，例如 LDSE、MEAS 或 radiomics。
8. 在线数据增强（train+test）+ LDSE + 分类：
   Step 1: FEATURE_EXTRACTION_AGENT - 仅对训练集调用 augment_online_extract_features，显式使用 dataset_role="train"；保留返回的 model_path 和训练增强特征 output_csv
   Step 2: FEATURE_EXTRACTION_AGENT - 对测试集调用 augment_online_extract_features，显式使用 dataset_role="test"，并使用 Step 1 返回的 ldse_model_path；保留测试增强特征 output_csv
   Step 3: CLASSIFICATION_AGENT - 直接使用 Step 1 和 Step 2 返回的两个增强特征 CSV 执行分类
   CRITICAL: augment_online_extract_features 已经提取特征；禁止再从不含 MaskPath 的增强特征 CSV 重复提取 LDSE。
9. 在线数据增强（train+test）+ LDSE（不分类）-> 使用模式 8 的前两个 FEATURE_EXTRACTION_AGENT 步骤后结束。
   MEAS/radiomics 在线增强不需要 LDSE model_path，但仍只能在用户明确要求数据增强时使用。

IMPORTANT: 必须严格输出 JSON，结构如下：
{{
  "steps": [
    {{
      "agent": "AGENT_NAME",
      "task_description": "用中文简要描述这一步做什么",
      "input_override": "可选：如需覆盖该步骤输入，在这里用中文说明"
    }},
    {{
      "agent": "ANOTHER_AGENT_NAME",
      "task_description": "用中文简要描述这一步做什么",
      "input_override": "可选：如需覆盖该步骤输入，在这里用中文说明"
    }}
  ],
  "selected_candidate_id": "可选：自动目标模式下从提供的候选 ID 中选择一个",
  "selection_reasoning": "可选：说明选择该候选的任务与数据依据",
  "final_message": "给用户看的中文最终总结"
}}

CRITICAL 规则：
        1. "steps" 数组必须包含完成用户请求所需的全部步骤
        2. 每个步骤必须指定要调用哪个 AGENT
        3. 步骤必须按正确逻辑顺序排列
        4. 如果用户要求为多个数据集提取特征，应调用 FEATURE_EXTRACTION_AGENT 一次并传入完整请求，由该 agent 内部处理多个数据集
        5. 如果用户同时要求特征提取和分类，必须同时包含两个操作对应的步骤
        6. 如果用户同时要求分类和读取结果，应调用 CLASSIFICATION_AGENT 一次，由该 agent 内部处理两个操作
        7. 每个步骤都应有清晰的 task_description
        8. "final_message" 应在所有步骤执行后用中文总结完成了什么
        9. IMPORTANT: 不必要时不要多次调用同一个 agent；只有当必须用不同参数覆盖输入且 agent 无法从原请求推断时，才多次调用同一 agent。
        10. PIPELINE RULE: 存在 output-to-input dependency 的步骤必须拆开。拆开后，Step 2 及后续步骤必须提供 input_override：
            - 明确区分输出类型：augment_dataset_offline 返回含 MaskPath 的标签 CSV，后续可提取特征；augment_online_extract_features 返回的已经是特征 CSV，应直接交给分类，不得再次提取
            - 引用前序步骤生成的真实文件路径；系统会以 "FILES GENERATED BY PREVIOUS STEPS" 附加
            - 聚焦一种操作类型，例如只做提取，不要写成“增强+提取”

示例：

        示例 1：用户要求分别为训练集和测试集提取 MEAS 特征，然后使用提取出的特征做 LASSO 分类，并读取分类报告
        steps 应包含：
        - Step 1: FEATURE_EXTRACTION_AGENT - 为训练集和测试集提取 MEAS 特征，单次调用
        - Step 2: CLASSIFICATION_AGENT - 使用 LASSO 分类并读取分类报告，单次调用

        示例 2：用户要求提取 MEAS 特征并使用 LASSO 分类
        steps 应包含：
        - Step 1: FEATURE_EXTRACTION_AGENT - 提取 MEAS 特征
        - Step 2: CLASSIFICATION_AGENT - 使用 LASSO 分类

        示例 3：用户要求同时使用 U-Mamba 和 LKM-UNet 分割 MRI 图像
        steps 应包含：
        - Step 1: SEGMENTATION_AGENT - 使用 U-Mamba 分割
        - Step 2: SEGMENTATION_AGENT - 使用 LKM-UNet 分割

        示例 4：用户要求对 train+test 数据增强，然后提取 LDSE 特征，再做 LASSO 分类
        steps 应包含 3 个独立步骤，因为测试在线增强依赖训练步骤返回的 model_path：
        - Step 1: FEATURE_EXTRACTION_AGENT - 对训练集进行在线数据增强并提取 LDSE 特征
          input_override: "对路径 <path> 的训练集 CSV 使用 dataset_role=train 进行在线数据增强并提取 LDSE 特征，返回 model_path 和 output_csv"
        - Step 2: FEATURE_EXTRACTION_AGENT - 使用训练模型对测试集进行在线数据增强并提取 LDSE 特征
          input_override: "对路径 <path> 的测试集 CSV 使用 dataset_role=test，并使用上一步返回的 ldse_model_path 进行在线数据增强，返回 output_csv"
        - Step 3: CLASSIFICATION_AGENT - 直接使用两个增强特征 CSV 进行 LASSO 分类
          input_override: "使用前两步返回的训练和测试 output_csv 进行 LASSO 分类，不再重复提取 LDSE"
        IMPORTANT: Step 1 必须明确用户请求的特征类型，例如 LDSE、MEAS 或 radiomics。

核心原则：始终根据用户意图识别全部必要步骤。参数缺失时，使用原始用户输入，让子 agent 处理参数提取。不要仅因为参数缺失就返回空 steps 数组。
"""
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "{input}")
        ])

    @staticmethod
    def _extract_output_paths(
        result: dict,
        task_description: str,
        accumulated_paths: list[tuple[str, str]],
    ) -> None:
        """Extract output file paths from a step's tool results for downstream steps.

        Preserve the canonical structured-result key so the next agent can parse
        paths containing whitespace without guessing token boundaries.
        """
        tool_results = result.get("tool_results", [])
        tool_calls = result.get("tool_calls", [])

        for tool_name, tool_result in zip(
            tool_calls + [""] * max(0, len(tool_results) - len(tool_calls)),
            tool_results,
        ):
            for key, output_path in parse_tool_result_paths(tool_result):
                item = (key, output_path)
                if item not in accumulated_paths:
                    accumulated_paths.append(item)

    @staticmethod
    def _format_output_paths(accumulated_paths: list[tuple[str, str]]) -> str:
        """Serialize canonical path fields as line-oriented structured handoff."""
        return "\n".join(
            f"  - {key}: {path}"
            for key, path in accumulated_paths
        )

    @staticmethod
    def _persist_pipeline_search(
        *,
        task_id: int | None,
        trace_id: str,
        policy_id: str | None,
        plan: dict[str, Any],
        result: Mapping[str, Any],
        candidates: list[dict[str, Any]],
        selected_pipeline: Mapping[str, Any],
    ) -> None:
        """Persist the authoritative search decision and matched CV evidence."""
        if not task_id:
            return
        db = None
        try:
            db = DatabaseManager()
            artifact_id = None
            manifest_path = result.get("pipeline_search_manifest_path")
            if manifest_path:
                artifact = build_artifact_record(
                    path=str(manifest_path),
                    artifact_type="pipeline_search_manifest",
                    created_by="PIPELINE_OPTIMIZATION_AGENT",
                    trace_id=trace_id,
                    task_id=task_id,
                    metadata={
                        "selected_candidate_id": selected_pipeline.get(
                            "candidate_id"
                        ),
                        "validation_scope": "training_only_nested_grouped_cv",
                        "test_data_used_for_selection": False,
                    },
                )
                db.register_artifact(artifact)
                artifact_id = artifact["artifact_id"]

            successful = [
                item for item in candidates if item.get("trial_status") == "success"
            ]
            persisted_trial_ids: list[str] = []
            for raw_record in result.get("pipeline_trial_records", []) or []:
                if not isinstance(raw_record, Mapping):
                    continue
                record = dict(raw_record)
                record["task_id"] = record.get("task_id") or task_id
                record["policy_id"] = record.get("policy_id") or policy_id
                persisted_trial_ids.append(db.save_pipeline_trial(record))
            for candidate in successful:
                for metric_name, metric_value in candidate.get(
                    "validation_metrics", {}
                ).items():
                    if metric_name not in {
                        "AUC", "PR_AUC", "Accuracy", "Precision",
                        "Sensitivity", "Specificity", "F1",
                    }:
                        continue
                    identity = stable_json_hash(
                        {
                            "trace_id": trace_id,
                            "candidate_id": candidate.get("candidate_id"),
                            "metric_name": metric_name,
                            "split_type": "cross_validation",
                            "evaluator_version": candidate.get(
                                "validation_evaluator_version"
                            ),
                        }
                    )
                    db.save_evaluation_result(
                        {
                            "evaluation_id": str(
                                uuid.uuid5(uuid.NAMESPACE_URL, identity)
                            ),
                            "trace_id": trace_id,
                            "task_id": task_id,
                            "artifact_id": artifact_id,
                            "policy_id": policy_id,
                            "task_type": "binary_kl_pipeline_search",
                            "method_name": candidate.get("candidate_id"),
                            "split_type": "cross_validation",
                            "dataset_hash": candidate.get(
                                "validation_dataset_hash"
                            ),
                            "evaluator_name": "pipeline_search",
                            "evaluator_version": candidate.get(
                                "validation_evaluator_version"
                            ) or "nested-grouped-cv-v1",
                            "record_type": "metric",
                            "metric_name": metric_name,
                            "metric_value": float(metric_value),
                            "gate_status": None,
                            "details": {
                                "validation_score": candidate.get(
                                    "validation_score"
                                ),
                                "patient_count": candidate.get(
                                    "validation_patient_count"
                                ),
                                "test_data_used": False,
                            },
                        }
                    )

            db.save_trajectory_event(
                build_trajectory_event(
                    event_type="pipeline_trial_completed",
                    trace_id=trace_id,
                    task_id=task_id,
                    actor_type="control_plane",
                    actor_name="pipeline_search",
                    status="success",
                    payload={
                        "trial_count": len(result.get("trial_results", [])),
                        "successful_trial_count": len(successful),
                        "persisted_trial_count": len(persisted_trial_ids),
                        "pipeline_trial_ids": persisted_trial_ids,
                        "validation_scope": "training_only_nested_grouped_cv",
                        "test_data_used_for_selection": False,
                        "candidate_results": [
                            {
                                "candidate_id": item.get("candidate_id"),
                                "trial_status": item.get("trial_status"),
                                "validation_score": item.get("validation_score"),
                                "auc_ci_lower": item.get(
                                    "validation_auc_lower_bound"
                                ),
                                "selection_stability": item.get(
                                    "validation_selection_stability"
                                ),
                            }
                            for item in candidates
                            if item.get("trial_status") != "not_run"
                        ],
                    },
                )
            )
            db.save_trajectory_event(
                build_trajectory_event(
                    event_type="pipeline_selected",
                    trace_id=trace_id,
                    task_id=task_id,
                    actor_type="control_plane",
                    actor_name="pipeline_search",
                    status="success",
                    payload={
                        "selected_candidate_id": selected_pipeline.get(
                            "candidate_id"
                        ),
                        "validation_score": selected_pipeline.get(
                            "validation_score"
                        ),
                        "validation_metrics": selected_pipeline.get(
                            "validation_metrics", {}
                        ),
                        "test_data_used_for_selection": False,
                    },
                )
            )
            db.update_task(
                task_id=task_id,
                plan_json=json.dumps(plan, ensure_ascii=False),
            )
        except Exception as exc:
            print(f"[OrchestratorAgent] Failed to persist pipeline search: {exc}")
        finally:
            if db is not None:
                try:
                    db.close()
                except Exception:
                    pass

    async def process(self, state: AgentState) -> AgentState:
        """Process complex multi-agent task"""
        stream_callback = state.get("stream_callback")

        messages = state["messages"]
        current_input = state["current_input"]

        task_id = state.get("task_id")

        input_text = ""
        if isinstance(current_input, str):
            input_text = current_input
        elif isinstance(current_input, dict):
            input_text = current_input.get("text", "")
        
        recent_context = ""
        for msg in messages[-6:]:
            if isinstance(msg, HumanMessage):
                recent_context += f"User: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                recent_context += f"Assistant: {msg.content}\n"

        goal_spec = state.get("goal_spec", {}) or {}
        autonomous_mode = bool(
            goal_spec.get("goal_type") == "KL_GRADING"
            and goal_spec.get("autonomy_mode") in {"auto", "partially_constrained"}
        )
        biomarker_mode = goal_spec.get("goal_type") == "BIOMARKER_DISCOVERY"
        autonomous_context = ""
        if autonomous_mode:
            candidate_summaries = [
                {
                    "candidate_id": item.get("candidate_id"),
                    "feature_type": item.get("feature_type"),
                    "classification_method": item.get("classification_method"),
                    "eligible": item.get("eligible"),
                    "rejection_reasons": item.get("rejection_reasons", []),
                    "score": item.get("score"),
                    "rank": item.get("rank"),
                    "agents": item.get("agents", []),
                    "tools": item.get("tools", []),
                }
                for item in state.get("pipeline_candidates", [])
            ]
            autonomous_context = f"""
        AUTONOMOUS GOAL CONTRACT:
        {json.dumps(goal_spec, ensure_ascii=False)}

        CURRENT DATASET PROFILE:
        {json.dumps(state.get('dataset_profile', {}), ensure_ascii=False)}

        CONTROL-PLANE VALIDATED PIPELINE CANDIDATES:
        {json.dumps(candidate_summaries, ensure_ascii=False)}

        Review the current goal, dataset profile, candidate prerequisites, tools,
        and ranking evidence. Return selected_candidate_id from the eligible list
        and explain the choice in selection_reasoning. Do not invent a new tool,
        path, metric, or completed result. The control plane will validate the
        proposed candidate before execution.
        """
        
        process_input = f"""
        User query: {input_text}

        Recent conversation context: {recent_context}

        {autonomous_context}

        You are Orchestrator Agent. Analyze the user's request and determine which agents to call and in what order.
        
        CRITICAL: Identify ALL steps needed to complete the user's request.
        - If user requests extracting features for multiple datasets, include appropriate steps
        - If user requests both feature extraction and classification, include steps for BOTH operations
        - If user requests both classification and reading results, include steps for BOTH operations
        
        IMPORTANT: ALWAYS identify and include ALL steps needed. Use the original user input for each step unless you need to override it.
        
        Respond in JSON format with steps array and final_message.
        """
        
        try:
            if biomarker_mode:
                plan = materialize_biomarker_plan(
                    dataset_profile=state.get("dataset_profile", {}),
                    current_input=input_text,
                )
            else:
                llm_response = self.llm.invoke(
                    self.prompt.format_messages(input=process_input)
                )
                print(f"[DEBUG] OrchestratorAgent - LLM 原始响应: {llm_response.content[:500] if len(llm_response.content) > 500 else llm_response.content}")
                try:
                    plan = parse_llm_json(llm_response.content)
                    print(f"[DEBUG] OrchestratorAgent - 解析后的 plan: {json.dumps(plan, ensure_ascii=False, indent=2)}")
                except (json.JSONDecodeError, ValueError) as e:
                    print(f"[DEBUG] OrchestratorAgent - JSON 解析失败: {e}")
                    plan = {
                        "steps": [],
                        "final_message": llm_response.content
                    }
            selected_pipeline = dict(state.get("selected_pipeline", {}) or {})
            effective_task_spec = dict(state.get("task_spec", {}) or {})
            planning_decision = dict(state.get("planning_decision", {}) or {})
            if biomarker_mode:
                planning_decision.update(
                    {
                        "selection_status": "pre_registered",
                        "selection_reasoning": "fixed_biomarker_discovery_v1",
                        "test_data_used_for_discovery": False,
                    }
                )
            elif autonomous_mode:
                llm_candidate_id = plan.get("selected_candidate_id")
                llm_selection_reasoning = plan.get("selection_reasoning")
                initial_leader = select_pipeline_candidate(
                    state.get("pipeline_candidates", []),
                    llm_candidate_id=(
                        str(llm_candidate_id) if llm_candidate_id else None
                    ),
                    llm_tie_tolerance=float(
                        planning_decision.get("llm_tie_tolerance", 0.02)
                    ),
                ) or {}
                plan = materialize_pipeline_search_plan(
                    trial_portfolio=planning_decision.get("trial_portfolio", {}),
                    dataset_profile=state.get("dataset_profile", {}),
                    user_query=input_text,
                )
                llm_proposal_accepted = bool(
                    llm_candidate_id
                    and str(llm_candidate_id) in {
                        str(item)
                        for item in planning_decision.get(
                            "trial_portfolio", {}
                        ).get("candidate_ids", [])
                    }
                )
                plan["llm_selection_reasoning"] = str(
                    llm_selection_reasoning or ""
                )
                plan["selection_reasoning"] = str(
                    planning_decision.get("selection_basis")
                    or "validated_capability_ranking"
                )
                # The registry/history leader seeds the bounded search but is
                # not authoritative until matched training-only validation.
                selected_pipeline = {}
                planning_decision.update(
                    {
                        "llm_selected_candidate_id": llm_candidate_id,
                        "llm_proposal_accepted": llm_proposal_accepted,
                        "llm_selection_reasoning": str(
                            llm_selection_reasoning or ""
                        ),
                        "initial_leader_candidate_id": initial_leader.get(
                            "candidate_id"
                        ),
                        "selected_candidate_id": None,
                        "selection_status": plan.get("selection_status"),
                        "selection_reasoning": str(
                            plan.get("selection_reasoning") or ""
                        ),
                    }
                )
            else:
                plan = normalize_orchestrator_plan(
                    plan,
                    task_spec=effective_task_spec,
                    current_input=input_text,
                )
            
            step_results = []
            all_messages = []
            successful_steps = 0
            had_step_failure = False
            pipeline_blocked = False
            step_records: list[dict[str, Any]] = []

            # 创建任务记录
            total_steps = len(plan.get("steps", []))
            routed_agent = state.get("next_agent", "ORCHESTRATOR_AGENT")
            trace_id = get_request_context().get('trace_id') or get_trace_id()
            plan_json_str = json.dumps(plan, ensure_ascii=False) if plan else None

            if total_steps > 0:
                db = None
                try:
                    ctx = get_request_context()
                    user_id = ctx.get('user_id')
                    session_id = ctx.get('session_id')
                    if user_id and session_id:
                        db = DatabaseManager()
                        if task_id:
                            db.update_task(
                                task_id=task_id,
                                total_steps=total_steps,
                                routed_agent=routed_agent,
                                plan_json=plan_json_str,
                            )
                        else:
                            # Compatibility fallback for direct Agent use.
                            task_id = db.create_task(
                                user_id=user_id,
                                session_id=session_id,
                                task_description=input_text,
                                total_steps=total_steps,
                                agent_name="ORCHESTRATOR_AGENT",
                                trace_id=trace_id,
                                routed_agent=routed_agent,
                                plan_json=plan_json_str,
                            )
                        db.save_trajectory_event(
                            build_trajectory_event(
                                event_type="plan_created",
                                trace_id=trace_id,
                                task_id=task_id,
                                actor_type="agent",
                                actor_name="ORCHESTRATOR_AGENT",
                                payload={
                                    "plan_sha256": stable_json_hash(plan),
                                    "total_steps": total_steps,
                                    "step_agents": [
                                        item.get("agent")
                                        for item in plan.get("steps", [])
                                    ],
                                    "autonomy_mode": goal_spec.get(
                                        "autonomy_mode"
                                    ),
                                    "selected_candidate_id": plan.get(
                                        "selected_candidate_id"
                                    ),
                                },
                            )
                        )
                        if autonomous_mode:
                            db.save_trajectory_event(
                                build_trajectory_event(
                                    event_type="pipeline_candidates_generated",
                                    trace_id=trace_id,
                                    task_id=task_id,
                                    actor_type="control_plane",
                                    actor_name="pipeline_planner",
                                    # Generation has finished here. Trial and
                                    # selection progress use their own events.
                                    status="success",
                                    payload={
                                        "dataset_hash": state.get(
                                            "dataset_profile", {}
                                        ).get("dataset_hash"),
                                        "candidates": [
                                            {
                                                "candidate_id": item.get(
                                                    "candidate_id"
                                                ),
                                                "eligible": item.get("eligible"),
                                                "score": item.get("score"),
                                                "rank": item.get("rank"),
                                                "rejection_reasons": item.get(
                                                    "rejection_reasons", []
                                                ),
                                            }
                                            for item in state.get(
                                                "pipeline_candidates", []
                                            )
                                        ],
                                        "selected_candidate_id": (
                                            selected_pipeline.get("candidate_id")
                                        ),
                                        "initial_leader_candidate_id": (
                                            planning_decision.get(
                                                "initial_leader_candidate_id"
                                            )
                                        ),
                                        "trial_portfolio": planning_decision.get(
                                            "trial_portfolio", {}
                                        ),
                                        "llm_selected_candidate_id": (
                                            planning_decision.get(
                                                "llm_selected_candidate_id"
                                            )
                                        ),
                                    },
                                )
                            )
                except Exception as e:
                    print(f"[OrchestratorAgent] 创建任务记录失败: {e}")
                finally:
                    if db is not None:
                        try:
                            db.close()
                        except Exception:
                            pass

            if stream_callback:
                stream_callback("\n📋 计划执行步骤:")
                for i, step in enumerate(plan.get("steps", []), 1):
                    stream_callback(f"   {i}. {step.get('agent', 'Unknown')} - {step.get('task_description', '')}")
            
            if "steps" in plan and plan["steps"]:
                # Accumulate output paths from previous steps for downstream context
                accumulated_paths: list[tuple[str, str]] = []
                current_artifacts: list[dict[str, Any]] = []
                selected_feature_paths: dict[str, str] = {}
                pipeline_trial_results: list[dict[str, Any]] = []
                pipeline_trial_records: list[dict[str, Any]] = []
                trial_record_errors: list[dict[str, str]] = []
                pipeline_search_manifest_path: str | None = None
                biomarker_result: dict[str, Any] = {}
                biomarker_audit: dict[str, Any] = {}
                biomarker_narrative: str = ""
                biomarker_reporting: dict[str, Any] = {}

                for i, step in enumerate(plan["steps"], 1):
                    agent_name = step.get("agent")
                    task_description = step.get("task_description", f"Step {i}")
                    input_override = step.get("input_override")

                    if pipeline_blocked:
                        blocked_message = "未执行：依赖的上游步骤失败。"
                        if stream_callback:
                            stream_callback(f"\n📌 步骤{i}: 跳过 {agent_name}")
                            stream_callback(f"   ⚠️ {blocked_message}")
                        all_messages.append(
                            f"步骤 {i}: {task_description}\n\n执行结果：\n{blocked_message}"
                        )
                        step_results.append(blocked_message)
                        step_records.append(
                            {
                                "step": i,
                                "agent": agent_name,
                                "task": task_description,
                                "status": "skipped",
                                "detail": blocked_message,
                                "tool_calls": [],
                            }
                        )
                        continue

                    if stream_callback:
                        stream_callback(f"\n📌 步骤{i}: 调用 {agent_name}")
                        stream_callback(f"   任务: {task_description}")

                    agent = self.agents.get(agent_name)
                    if not agent:
                        had_step_failure = True
                        pipeline_blocked = True
                        error_msg = f"智能体 {agent_name} 未找到"
                        if stream_callback:
                            stream_callback(f"   ❌ {error_msg}")
                        all_messages.append(f"步骤 {i}: {task_description}\n\n执行结果：\n{error_msg}")
                        step_results.append(error_msg)
                        step_records.append(
                            {
                                "step": i,
                                "agent": agent_name,
                                "task": task_description,
                                "status": "failed",
                                "detail": error_msg,
                                "tool_calls": [],
                            }
                        )
                        if task_id:
                            db_missing = None
                            try:
                                db_missing = DatabaseManager()
                                db_missing.save_task_step(
                                    task_id=task_id,
                                    step_index=i,
                                    agent_name=agent_name or "UNKNOWN",
                                    task_description=task_description,
                                    status="failed",
                                    trace_id=trace_id,
                                    error_category="invalid_params",
                                    error_message=error_msg,
                                )
                                db_missing.save_trajectory_event(
                                    build_trajectory_event(
                                        event_type="step_completed",
                                        trace_id=trace_id,
                                        task_id=task_id,
                                        step_id=str(i),
                                        actor_type="agent",
                                        actor_name=agent_name or "UNKNOWN",
                                        status="failed",
                                        payload={
                                            "task_description_sha256": stable_json_hash(task_description),
                                            "tool_count": 0,
                                            "error_category": "invalid_params",
                                        },
                                    )
                                )
                                db_missing.update_task(
                                    task_id=task_id,
                                    completed_steps=successful_steps,
                                    error_message=error_msg,
                                    error_category="invalid_params",
                                )
                            except Exception:
                                pass
                            finally:
                                if db_missing is not None:
                                    try:
                                        db_missing.close()
                                    except Exception:
                                        pass
                        continue

                    step_input = input_override if input_override else current_input

                    if (
                        agent_name == "CLASSIFICATION_AGENT"
                        and selected_pipeline
                        and selected_feature_paths
                    ):
                        step_input = (
                            "训练集内部候选试跑已完成。只执行以下唯一入选方案，"
                            "不得根据测试集结果切换方案。\n"
                            f"selected_candidate_id: {selected_pipeline.get('candidate_id')}\n"
                            f"feature_type: {selected_pipeline.get('feature_type')}\n"
                            f"classification_method: {selected_pipeline.get('classification_method')}\n"
                            f"model_recipe: {selected_pipeline.get('model_recipe')}\n"
                            "recommended_params: "
                            f"{json.dumps(selected_pipeline.get('validation_recommended_params', {}), ensure_ascii=False)}\n"
                            f"train_path: {selected_feature_paths.get('train', '')}\n"
                            f"test_path: {selected_feature_paths.get('test', '')}\n"
                            "完成最终分类后读取本次运行生成的分类报告。"
                        )

                    # Enrich subsequent steps with paths from previous steps
                    if accumulated_paths:
                        path_lines = self._format_output_paths(accumulated_paths)
                        step_input = (
                            f"{step_input}\n\n"
                            f"FILES GENERATED BY PREVIOUS STEPS (use these exact paths):\n"
                            f"{path_lines}"
                        )

                    step_state = {
                        "messages": messages,
                        "current_input": step_input,
                        "output": "",
                        "agent_name": "",
                        "tool_calls": [],
                        "tool_results": [],
                        "stream_callback": stream_callback,
                        "trace_id": trace_id,
                        "task_run_id": state.get("task_run_id") or trace_id,
                        "task_spec": effective_task_spec,
                        "goal_spec": goal_spec,
                        "dataset_profile": state.get("dataset_profile", {}),
                        "pipeline_candidates": state.get(
                            "pipeline_candidates", []
                        ),
                        "selected_pipeline": selected_pipeline,
                        "planning_decision": planning_decision,
                        "current_artifacts": list(current_artifacts),
                        "policy_id": state.get("policy_id"),
                        "task_id": task_id,
                        "biomarker_result": dict(biomarker_result),
                        "biomarker_audit": dict(biomarker_audit),
                    }

                    step_success = True
                    step_error_category = None
                    step_error_msg = None
                    step_output = ""
                    step_tool_calls = []
                    result: dict[str, Any] = {}

                    try:
                        set_agent_context(agent_name or "UNKNOWN")
                        result = await agent.process(step_state)
                        step_output = result.get("output", "")
                        step_tool_calls = result.get("tool_calls", [])
                        if result.get("success") is False:
                            raise RuntimeError(
                                result.get("error_message")
                                or result.get("output")
                                or f"{agent_name} reported failure"
                            )

                        # Extract output file paths from this step's results
                        previous_path_count = len(accumulated_paths)
                        self._extract_output_paths(
                            result, task_description, accumulated_paths
                        )
                        if agent_name == "PIPELINE_OPTIMIZATION_AGENT":
                            evaluated_candidates = result.get(
                                "pipeline_candidates", []
                            )
                            selected_pipeline = dict(
                                result.get("selected_pipeline", {}) or {}
                            )
                            selected_feature_paths = dict(
                                result.get("selected_feature_paths", {}) or {}
                            )
                            pipeline_trial_results = list(
                                result.get("trial_results", []) or []
                            )
                            pipeline_trial_records = list(
                                result.get("pipeline_trial_records", []) or []
                            )
                            trial_record_errors = list(
                                result.get("trial_record_errors", []) or []
                            )
                            pipeline_search_manifest_path = result.get(
                                "pipeline_search_manifest_path"
                            )
                            if not selected_pipeline or not all(
                                selected_feature_paths.get(role)
                                for role in ("train", "test")
                            ):
                                raise RuntimeError(
                                    "pipeline optimizer did not return a complete selected pipeline"
                                )
                            effective_task_spec = apply_selected_pipeline(
                                effective_task_spec, selected_pipeline
                            )
                            effective_task_spec["pipeline_search_performed"] = True
                            effective_task_spec["selection_split"] = "cross_validation"
                            current_artifacts = list(
                                result.get("current_artifacts", []) or []
                            )
                            accumulated_paths = [
                                ("train_features", selected_feature_paths["train"]),
                                ("test_features", selected_feature_paths["test"]),
                            ]
                            manifest_path = result.get(
                                "pipeline_search_manifest_path"
                            )
                            if manifest_path:
                                accumulated_paths.append(
                                    ("manifest_path", str(manifest_path))
                                )
                            state["pipeline_candidates"] = evaluated_candidates
                            plan["selected_candidate_id"] = selected_pipeline.get(
                                "candidate_id"
                            )
                            plan["selection_status"] = "selected_after_training_validation"
                            plan["selection_reason"] = {
                                "validation_score": selected_pipeline.get(
                                    "validation_score"
                                ),
                                "validation_metrics": selected_pipeline.get(
                                    "validation_metrics", {}
                                ),
                                "validation_scope": selected_pipeline.get(
                                    "validation_scope"
                                ),
                                "test_data_used_for_selection": False,
                            }
                            planning_decision.update(
                                {
                                    "selected_candidate_id": selected_pipeline.get(
                                        "candidate_id"
                                    ),
                                    "selection_status": plan.get(
                                        "selection_status"
                                    ),
                                    "selection_reasoning": (
                                        "matched_training_only_nested_cv"
                                    ),
                                    "test_data_used_for_selection": False,
                                    "successful_trial_count": sum(
                                        item.get("trial_status") == "success"
                                        for item in evaluated_candidates
                                    ),
                                }
                            )
                            self._persist_pipeline_search(
                                task_id=task_id,
                                trace_id=trace_id,
                                policy_id=state.get("policy_id"),
                                plan=plan,
                                result=result,
                                candidates=evaluated_candidates,
                                selected_pipeline=selected_pipeline,
                            )
                        if agent_name == "FEATURE_EXTRACTION_AGENT":
                            requested_feature_type = effective_task_spec.get(
                                "feature_type"
                            )
                            run_id = state.get("task_run_id") or trace_id
                            tool_results = result.get("tool_results", [])
                            tool_calls = result.get("tool_calls", [])
                            for tool_name, tool_result in zip(tool_calls, tool_results):
                                actual_feature_type = infer_feature_type(tool_name)
                                for key, path in parse_tool_result_paths(tool_result):
                                    if not path.lower().endswith(".csv"):
                                        continue
                                    actual_feature_type = actual_feature_type or infer_feature_type(path)
                                    if (
                                        requested_feature_type
                                        and actual_feature_type != requested_feature_type
                                    ):
                                        raise ValueError(
                                            "Feature artifact type mismatch: "
                                            f"requested {requested_feature_type}, got {actual_feature_type}"
                                        )
                                    current_artifacts.append(
                                        {
                                            "schema_version": "1.0",
                                            "run_id": run_id,
                                            "feature_type": actual_feature_type,
                                            "dataset_role": infer_dataset_role(path, key),
                                            "path": path,
                                            "producer_agent": agent_name,
                                            "producer_tool": tool_name,
                                        }
                                    )

                        if agent_name in {
                            "BIOMARKER_DISCOVERY_AGENT",
                            "BIOMARKER_VERIFIER_AGENT",
                        }:
                            run_id = state.get("task_run_id") or trace_id
                            requested_feature_type = effective_task_spec.get(
                                "feature_type"
                            )
                            for tool_name, tool_result in zip(
                                result.get("tool_calls", []),
                                result.get("tool_results", []),
                            ):
                                for key, path in parse_tool_result_paths(tool_result):
                                    current_artifacts.append(
                                        {
                                            "schema_version": "1.0",
                                            "run_id": run_id,
                                            "feature_type": requested_feature_type,
                                            "dataset_role": None,
                                            "artifact_type": key,
                                            "path": path,
                                            "producer_agent": agent_name,
                                            "producer_tool": tool_name,
                                        }
                                    )
                            if agent_name == "BIOMARKER_DISCOVERY_AGENT":
                                biomarker_result = dict(
                                    result.get("biomarker_result", {}) or {}
                                )
                            else:
                                biomarker_audit = dict(
                                    result.get("biomarker_audit", {}) or {}
                                )

                        if agent_name == "BIOMARKER_REPORTER_AGENT":
                            biomarker_narrative = str(
                                result.get("biomarker_narrative")
                                or result.get("output")
                                or ""
                            )
                            biomarker_reporting = dict(
                                result.get("biomarker_reporting", {}) or {}
                            )

                        if stream_callback:
                            stream_callback(f"   ✅ 完成 (工具: {', '.join(step_tool_calls) if step_tool_calls else '无'})")

                        all_messages.append(f"步骤 {i}: {task_description}\n\n执行结果：\n{step_output}")
                        step_results.append(step_output)
                        step_records.append(
                            {
                                "step": i,
                                "agent": agent_name,
                                "task": task_description,
                                "status": "success",
                                "detail": step_output,
                                "tool_calls": list(step_tool_calls),
                            }
                        )
                        successful_steps += 1

                        messages = messages + [AIMessage(content=step_output)]

                    except Exception as step_error:
                        step_success = False
                        had_step_failure = True
                        pipeline_blocked = True
                        step_error_msg = str(step_error)
                        step_error_category = categorize_error(step_error_msg)
                        if agent_name == "PIPELINE_OPTIMIZATION_AGENT" and task_id:
                            db_trial_failure = None
                            try:
                                failed_trials = list(
                                    result.get("trial_results", []) or []
                                )
                                db_trial_failure = DatabaseManager()
                                db_trial_failure.save_trajectory_event(
                                    build_trajectory_event(
                                        event_type="pipeline_trial_completed",
                                        trace_id=trace_id,
                                        task_id=task_id,
                                        actor_type="control_plane",
                                        actor_name="pipeline_search",
                                        status="failed",
                                        payload={
                                            "trial_count": len(failed_trials),
                                            "successful_trial_count": sum(
                                                str(item.get("status", "")).lower()
                                                == "success"
                                                for item in failed_trials
                                            ),
                                            "validation_scope": (
                                                "training_only_nested_grouped_cv"
                                            ),
                                            "test_data_used_for_selection": False,
                                            "error_category": step_error_category,
                                        },
                                    )
                                )
                            except Exception:
                                pass
                            finally:
                                if db_trial_failure is not None:
                                    try:
                                        db_trial_failure.close()
                                    except Exception:
                                        pass
                        if stream_callback:
                            stream_callback(f"   ❌ {step_error_msg}")
                        all_messages.append(f"步骤 {i}: {task_description}\n\n执行结果：\n{step_error_msg}")
                        step_results.append(step_error_msg)
                        step_records.append(
                            {
                                "step": i,
                                "agent": agent_name,
                                "task": task_description,
                                "status": "failed",
                                "detail": step_error_msg,
                                "tool_calls": list(step_tool_calls),
                            }
                        )
                    finally:
                        set_agent_context("ORCHESTRATOR_AGENT")
                        # 记录步骤明细
                        if task_id:
                            db_step = None
                            try:
                                db_step = DatabaseManager()
                                db_step.save_task_step(
                                    task_id=task_id,
                                    step_index=i,
                                    agent_name=agent_name,
                                    task_description=task_description,
                                    status='success' if step_success else 'failed',
                                    trace_id=trace_id,
                                    error_category=step_error_category,
                                    error_message=step_error_msg,
                                    output_summary=step_output[:500] if step_output else None,
                                )
                                db_step.save_trajectory_event(
                                    build_trajectory_event(
                                        event_type="step_completed",
                                        trace_id=trace_id,
                                        task_id=task_id,
                                        step_id=str(i),
                                        actor_type="agent",
                                        actor_name=agent_name or "UNKNOWN",
                                        status="success" if step_success else "failed",
                                        payload={
                                            "task_description_sha256": stable_json_hash(task_description),
                                            "tool_count": len(step_tool_calls),
                                            "output_sha256": stable_json_hash(step_output or ""),
                                            "error_category": step_error_category,
                                        },
                                    )
                                )
                            except Exception:
                                pass
                            finally:
                                if db_step is not None:
                                    try:
                                        db_step.close()
                                    except Exception:
                                        pass

                        # 更新任务进度
                        if task_id:
                            db_progress = None
                            try:
                                db_progress = DatabaseManager()
                                if not step_success:
                                    db_progress.update_task(
                                        task_id=task_id,
                                        error_message=step_error_msg,
                                        error_category=step_error_category,
                                    )
                                else:
                                    db_progress.update_task(
                                        task_id=task_id,
                                        completed_steps=successful_steps,
                                    )
                            except Exception:
                                pass
                            finally:
                                if db_progress is not None:
                                    try:
                                        db_progress.close()
                                    except Exception:
                                        pass
            
            # A planner's final_message is written before any tool runs and is
            # therefore not evidence of completion. Derive the final status only
            # from observed step outcomes.
            if step_records:
                status_label = "任务未完成" if had_step_failure else "任务已完成"
                status_lines = [f"{item['step']}. {item['status']}: {item['task']}" for item in step_records]
                final_message = f"{status_label}。\n" + "\n".join(status_lines)
            else:
                had_step_failure = True
                final_message = "任务未完成：没有生成可执行步骤。"

            planning_summary = ""
            if autonomous_mode:
                if selected_pipeline:
                    eligible_count = sum(
                        bool(item.get("eligible"))
                        for item in state.get("pipeline_candidates", [])
                    )
                    planning_summary = (
                        "自动规划结果："
                        f"从 {eligible_count} 个可执行候选中按预算试跑 "
                        f"{planning_decision.get('successful_trial_count', 0)} 个有效方案，"
                        f"最终选择 {selected_pipeline.get('candidate_id')}；"
                        f"训练集验证分 {selected_pipeline.get('validation_score')}。"
                        "候选选择未使用测试集标签或测试指标。\n"
                        f"当前分类口径：{goal_spec.get('label_policy')}。\n"
                    )
                else:
                    issues = state.get("dataset_profile", {}).get("issues", [])
                    rejection_reasons = sorted(
                        {
                            str(reason)
                            for item in state.get("pipeline_candidates", [])
                            for reason in item.get("rejection_reasons", [])
                        }
                    )
                    limitations = goal_spec.get("capability_limitations", [])
                    planning_summary = (
                        "自动规划结果：没有满足当前目标和数据前置条件的候选。"
                        f"数据问题：{', '.join(str(item) for item in issues) or 'none'}；"
                        f"候选拒绝原因：{', '.join(rejection_reasons) or 'unknown'}；"
                        f"能力边界：{' '.join(str(item) for item in limitations) or 'none'}。\n"
                    )

            biomarker_summary = str(
                locals().get("biomarker_narrative", "")
            ).strip()
            if biomarker_mode and not had_step_failure and biomarker_summary:
                # The front end displays ``output`` as the final assistant
                # message. For biomarker runs, show the independently audited,
                # reporter-authored narrative instead of internal step transcripts.
                full_response = biomarker_summary
            elif all_messages:
                full_response = planning_summary + "\n\n" + "=" * 70 + "\n\n".join(all_messages) + "\n\n" + "=" * 70 + "\n\n" + final_message
            else:
                full_response = planning_summary + final_message

            # Persist progress only. The root control plane commits the
            # terminal state after deterministic completion gates run.
            if task_id:
                db_final = None
                try:
                    db_final = DatabaseManager()
                    db_final.update_task(
                        task_id=task_id,
                        completed_steps=successful_steps,
                    )
                except Exception as e:
                    print(f"[OrchestratorAgent] 更新任务完成状态失败: {e}")
                finally:
                    if db_final is not None:
                        try:
                            db_final.close()
                        except Exception:
                            pass

            return {
                **state,
                "output": full_response,
                "agent_name": "ORCHESTRATOR_AGENT",
                "success": not had_step_failure,
                "tool_calls": [],
                "tool_results": step_results,
                "current_artifacts": locals().get("current_artifacts", []),
                "task_id": task_id,
                "execution_plan": plan,
                "step_records": step_records,
                "task_spec": effective_task_spec,
                "goal_spec": goal_spec,
                "dataset_profile": state.get("dataset_profile", {}),
                "pipeline_candidates": state.get("pipeline_candidates", []),
                "selected_pipeline": selected_pipeline,
                "planning_decision": planning_decision,
                "trial_results": locals().get("pipeline_trial_results", []),
                "pipeline_trial_records": locals().get(
                    "pipeline_trial_records", []
                ),
                "trial_record_errors": locals().get(
                    "trial_record_errors", []
                ),
                "selected_feature_paths": locals().get(
                    "selected_feature_paths", {}
                ),
                "pipeline_search_manifest_path": locals().get(
                    "pipeline_search_manifest_path"
                ),
                "biomarker_result": locals().get("biomarker_result", {}),
                "biomarker_audit": locals().get("biomarker_audit", {}),
                "biomarker_narrative": locals().get("biomarker_narrative", ""),
                "biomarker_reporting": locals().get("biomarker_reporting", {}),
            }

        except Exception as e:
            error_msg = f"任务编排过程中出错: {str(e)}"
            error_category = categorize_error(str(e))
            if stream_callback:
                stream_callback(f"\n❌ {error_msg}")
            import traceback
            traceback.print_exc()

            # Preserve error evidence; terminal status is committed by the
            # root control plane after deterministic evaluation.
            if task_id:
                db_err = None
                try:
                    db_err = DatabaseManager()
                    db_err.update_task(
                        task_id=task_id,
                        error_message=error_msg,
                        error_category=error_category,
                    )
                except Exception:
                    pass
                finally:
                    if db_err is not None:
                        try:
                            db_err.close()
                        except Exception:
                            pass

            return {
                **state,
                "output": error_msg,
                "agent_name": "ORCHESTRATOR_AGENT",
                "success": False,
                "error_category": error_category,
                "error_message": error_msg,
            }
