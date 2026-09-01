"""
State definitions for the multi-agent system
"""

from typing import Dict, List, Optional, Any, Literal, TypedDict, Union
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage

class AgentState(TypedDict):
    """State maintained across the workflow."""
    messages: List[BaseMessage]  # Conversation history
    agent_name: Optional[str]  # Current active agent
    current_input: Optional[Union[str, Dict]]  # Input to be processed
    has_image: bool  # Whether the current input contains an image
    image_type: Optional[str]  # Type of medical image if present
    output: Optional[str]  # Final output to user
    next_agent: Optional[str]  # Next agent to route to
    tool_calls: List[Dict]  # List of tool calls made
    tool_results: List[Dict]  # List of tool results
    routing_confidence: Optional[float]  # Router confidence score
    routing_reasoning: Optional[str]  # Router reasoning
    routing_decision_source: Optional[str]  # LLM proposal or LLM revision after validation
    routing_log_id: Optional[int]  # DB ID of routing log entry
    trace_id: Optional[str]  # Correlates all logs for one request
    policy_id: Optional[str]  # Versioned routing/orchestration policy
    task_run_id: Optional[str]  # Unique to one user request
    task_spec: Dict[str, Any]  # Deterministic current-request contract
    goal_spec: Dict[str, Any]  # Scientific goal and autonomy constraints
    dataset_profile: Dict[str, Any]  # Current-request train/test data profile
    pipeline_candidates: List[Dict[str, Any]]  # Validated and ranked candidate DAGs
    selected_pipeline: Dict[str, Any]  # Candidate selected by the control plane
    planning_decision: Dict[str, Any]  # LLM proposal and deterministic selection evidence
    trial_results: List[Dict[str, Any]]  # Matched training-only candidate validation records
    pipeline_trial_records: List[Dict[str, Any]]  # Immutable paired candidate records
    trial_record_errors: List[Dict[str, str]]  # Trial contract preparation errors
    selected_feature_paths: Dict[str, str]  # Current-run train/test features for the winner
    pipeline_search_manifest_path: Optional[str]  # Reproducible search decision artifact
    biomarker_result: Dict[str, Any]  # Candidate discovery result for this run
    biomarker_audit: Dict[str, Any]  # Independent verification result
    biomarker_narrative: Optional[str]  # Verified natural-language result explanation
    biomarker_reporting: Dict[str, Any]  # Writer validation and fallback metadata
    current_artifacts: List[Dict[str, Any]]  # Artifacts produced in this request only
    success: Optional[bool]  # Authoritative execution outcome from the active agent
    error_message: Optional[str]  # Failure detail when success is False
    error_category: Optional[str]  # Normalized failure category
    task_id: Optional[int]  # Persisted task row for orchestrated workflows
    task_fingerprint: Optional[str]  # Comparable decision-context fingerprint
    execution_fingerprint: Optional[str]  # Policy and selected-method fingerprint
    outcome_summary: Dict[str, Any]  # Deterministic terminal outcome summary
    execution_plan: Dict[str, Any]  # Normalized plan actually evaluated
    step_records: List[Dict[str, Any]]  # Observed step status and tool calls
    task_evaluation: Dict[str, Any]  # Deterministic task-level metrics and gates

class RouterDecision(TypedDict):
    """Output structure for the router agent."""
    agent: str
    reasoning: str
    confidence: float
