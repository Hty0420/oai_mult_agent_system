"""Deterministic task-level evaluation for multi-agent workflow runs.

The evaluator never asks an LLM whether a task succeeded. It derives metrics
and hard gates from the current TaskSpec, observed steps, tool calls, artifacts,
and authoritative terminal status. Records are portable and can be persisted in
the existing ``evaluation_results`` table.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import re
from typing import Any
import uuid

from .trajectory import (
    build_trajectory_event,
    current_policy_id,
    sanitize_for_audit,
    stable_json_hash,
)


TASK_EVALUATOR_NAME = "task-contract-evaluator"
TASK_EVALUATOR_VERSION = "1.5.0"

_CLASSIFIER_TOOLS = {
    "LASSO": "koa_lasso_classification",
    "Filter": "koa_filter_classification",
}
_FEATURE_TOOL_GROUPS = {
    "MEAS": frozenset({"meas_features_extraction"}),
    "LDSE": frozenset(
        {"ldse_features_extraction", "ldse_train_test_features_extraction"}
    ),
    "Radiomics": frozenset({"extract_radiomics_features"}),
}
_CARTILAGE_OPERATION_TOOLS = {
    "validation": frozenset({"validate_cartilage_segmentation"}),
    "analysis": frozenset({"analyze_cartilage_thickness"}),
    "batch_features": frozenset({"extract_cartilage_thickness_features"}),
    "mesh": frozenset({"generate_cartilage_3d_mesh"}),
}
_CARTILAGE_OUTPUT_TOOLS = frozenset(
    {
        "analyze_cartilage_thickness",
        "extract_cartilage_thickness_features",
        "generate_cartilage_3d_mesh",
    }
)
_POSITIVE_COMPLETION_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"任务已完成",
        r"任务处理完成",
        r"已完成全部",
        r"已成功完成",
        r"task (?:has been )?completed",
        r"successfully completed",
    )
)


def _status(value: Any) -> str:
    normalized = str(value or "unknown").strip().lower()
    return normalized if normalized in {"success", "failed", "skipped"} else "unknown"


def _classification_method(task_spec: Mapping[str, Any]) -> str | None:
    value = task_spec.get("classification_method")
    if value in _CLASSIFIER_TOOLS:
        return str(value)
    return None


def _task_type(task_spec: Mapping[str, Any]) -> str:
    if task_spec.get("goal_type") == "BIOMARKER_DISCOVERY":
        return "biomarker_discovery"
    parts: list[str] = []
    if task_spec.get("requires_feature_extraction"):
        parts.append("feature")
    if task_spec.get("requires_classification"):
        parts.append("classification")
    if task_spec.get("requires_report_read"):
        parts.append("report")
    return "_".join(parts) if parts else "general"


def _method_name(task_spec: Mapping[str, Any]) -> str:
    if task_spec.get("goal_type") == "BIOMARKER_DISCOVERY":
        return "CartilageThickness:ordinal_KL_candidate_discovery_v1"
    components = [
        str(value)
        for value in (
            task_spec.get("feature_type"),
            _classification_method(task_spec),
        )
        if value
    ]
    return ":".join(components) if components else "unknown"


def _required_agents(task_spec: Mapping[str, Any]) -> list[str]:
    if task_spec.get("goal_type") == "BIOMARKER_DISCOVERY":
        return [
            "FEATURE_EXTRACTION_AGENT",
            "BIOMARKER_DISCOVERY_AGENT",
            "BIOMARKER_VERIFIER_AGENT",
            "BIOMARKER_REPORTER_AGENT",
        ]
    agents: list[str] = []
    if task_spec.get("requires_feature_extraction"):
        agents.append(
            "PIPELINE_OPTIMIZATION_AGENT"
            if task_spec.get("pipeline_search_performed")
            else "FEATURE_EXTRACTION_AGENT"
        )
    if task_spec.get("requires_classification") or task_spec.get("requires_report_read"):
        agents.append("CLASSIFICATION_AGENT")
    return agents


def _required_tool_groups(task_spec: Mapping[str, Any]) -> list[tuple[str, frozenset[str]]]:
    groups: list[tuple[str, frozenset[str]]] = []
    if task_spec.get("goal_type") == "BIOMARKER_DISCOVERY":
        return [
            (
                "extract_CartilageThickness",
                frozenset({"extract_cartilage_thickness_features"}),
            ),
            ("discover_biomarkers", frozenset({"run_biomarker_discovery"})),
            ("audit_biomarkers", frozenset({"audit_biomarker_discovery"})),
        ]
    feature_type = task_spec.get("feature_type")
    if task_spec.get("requires_feature_extraction") and feature_type == "CartilageThickness":
        operation = str(task_spec.get("cartilage_operation") or "analysis")
        groups.append(
            (
                f"cartilage_{operation}",
                _CARTILAGE_OPERATION_TOOLS.get(
                    operation,
                    frozenset().union(*_CARTILAGE_OPERATION_TOOLS.values()),
                ),
            )
        )
    elif task_spec.get("requires_feature_extraction") and feature_type in _FEATURE_TOOL_GROUPS:
        groups.append((f"extract_{feature_type}", _FEATURE_TOOL_GROUPS[str(feature_type)]))
    method = _classification_method(task_spec)
    if task_spec.get("requires_classification"):
        groups.append(
            (
                f"classify_{method or 'requested'}",
                frozenset({_CLASSIFIER_TOOLS[method]})
                if method
                else frozenset(_CLASSIFIER_TOOLS.values()),
            )
        )
    if task_spec.get("requires_report_read"):
        groups.append(("read_report", frozenset({"read_classification_results"})))
    return groups


def _actual_tools(step_records: Sequence[Mapping[str, Any]]) -> list[str]:
    tools: list[str] = []
    for step in step_records:
        raw = step.get("tool_calls", [])
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
            tools.extend(str(name) for name in raw if name)
    return tools


def _requires_fresh_artifacts(
    task_spec: Mapping[str, Any], actual_tools: Sequence[str]
) -> bool:
    """Use the task/output contract, with compatibility for older checkpoints."""
    if task_spec.get("allow_history_reuse"):
        return False
    explicit_requirement = task_spec.get("requires_output_artifact")
    if explicit_requirement is not None:
        return bool(explicit_requirement)
    if (
        task_spec.get("feature_type") == "CartilageThickness"
        and actual_tools
        and not _CARTILAGE_OUTPUT_TOOLS.intersection(actual_tools)
    ):
        return False
    return bool(task_spec.get("requires_feature_extraction"))


def _contains_false_completion(output: str, overall_success: bool) -> bool:
    if overall_success:
        return False
    return any(pattern.search(str(output or "")) for pattern in _POSITIVE_COMPLETION_PATTERNS)


def _gate(status: str, *, applicable: bool = True, **details: Any) -> dict[str, Any]:
    if status not in {"pass", "fail", "unknown"}:
        raise ValueError("gate status must be pass, fail, or unknown")
    return {
        "status": status,
        "applicable": applicable,
        "details": sanitize_for_audit(details),
    }


def _metric(value: float, **details: Any) -> dict[str, Any]:
    return {"value": float(value), "details": sanitize_for_audit(details)}


def evaluate_task_run(
    *,
    trace_id: str,
    task_spec: Mapping[str, Any],
    routed_agent: str | None,
    step_records: Sequence[Mapping[str, Any]],
    current_artifacts: Sequence[Mapping[str, Any]],
    overall_success: bool,
    final_output: str,
    run_id: str | None = None,
    task_id: int | None = None,
    policy_id: str | None = None,
    dataset_hash: str | None = None,
) -> dict[str, Any]:
    """Evaluate one observed run and build idempotent evaluation records."""
    if not str(trace_id or "").strip():
        raise ValueError("trace_id is required")
    run_id = str(run_id or trace_id)
    policy_id = policy_id or current_policy_id()
    steps = [dict(step) for step in step_records if isinstance(step, Mapping)]
    artifacts = [dict(item) for item in current_artifacts if isinstance(item, Mapping)]

    required_agents = _required_agents(task_spec)
    actual_agents = [
        str(step.get("agent"))
        for step in steps
        if step.get("agent") and str(step.get("agent")) in set(required_agents)
    ]
    agent_sequence_ok = actual_agents == required_agents

    required_tool_groups = _required_tool_groups(task_spec)
    actual_tools = _actual_tools(steps)
    missing_tool_groups = [
        name for name, alternatives in required_tool_groups
        if not alternatives.intersection(actual_tools)
    ]
    tool_coverage = (
        1.0
        if not required_tool_groups
        else (len(required_tool_groups) - len(missing_tool_groups)) / len(required_tool_groups)
    )

    first_failure = next(
        (index for index, step in enumerate(steps) if _status(step.get("status")) == "failed"),
        None,
    )
    downstream_successes = []
    if first_failure is not None:
        downstream_successes = [
            int(step.get("step", index + 1))
            for index, step in enumerate(steps[first_failure + 1 :], start=first_failure + 1)
            if _status(step.get("status")) == "success"
        ]

    runtime_statuses = [_status(step.get("status")) for step in steps]
    observed_all_success = bool(steps) and all(status == "success" for status in runtime_statuses)
    runtime_consistent = bool(overall_success) == observed_all_success

    fresh_artifacts_required = _requires_fresh_artifacts(task_spec, actual_tools)
    requested_feature_type = task_spec.get("feature_type")
    artifact_scope_errors: list[str] = []
    for artifact in artifacts:
        if artifact.get("run_id") != run_id:
            artifact_scope_errors.append("run_id_mismatch")
        if requested_feature_type and artifact.get("feature_type") != requested_feature_type:
            artifact_scope_errors.append("feature_type_mismatch")
    # Missing output is an artifact-contract violation only after successful
    # execution.  When the tool itself fails, execution_succeeded is the root
    # failing gate and artifact scope must not obscure that cause.
    if fresh_artifacts_required and observed_all_success and not artifacts:
        artifact_scope_errors.append("missing_current_run_artifact")

    roles = {
        str(artifact.get("dataset_role"))
        for artifact in artifacts
        if artifact.get("dataset_role") in {"train", "test"}
    }
    paired_roles_required = bool(
        task_spec.get("requires_feature_extraction")
        and (
            task_spec.get("requires_classification")
            or task_spec.get("goal_type") == "BIOMARKER_DISCOVERY"
        )
    )
    artifact_roles_ok = not paired_roles_required or roles == {"train", "test"}

    report_order_ok = True
    if task_spec.get("requires_classification") and task_spec.get("requires_report_read"):
        method = _classification_method(task_spec)
        classifier = _CLASSIFIER_TOOLS.get(method or "")
        report_order_ok = False
        for step in steps:
            tools = [str(name) for name in step.get("tool_calls", [])]
            selected_classifier = classifier or next(
                (name for name in _CLASSIFIER_TOOLS.values() if name in tools), None
            )
            if selected_classifier in tools and "read_classification_results" in tools:
                report_order_ok = tools.index(selected_classifier) < tools.index("read_classification_results")
                if report_order_ok:
                    break

    structural_success = bool(
        overall_success
        and runtime_consistent
        and agent_sequence_ok
        and not missing_tool_groups
        and (not fresh_artifacts_required or not artifact_scope_errors)
        and artifact_roles_ok
        and not downstream_successes
        and report_order_ok
    )
    false_completion = _contains_false_completion(
        final_output, structural_success
    )

    gates = {
        "execution_succeeded": _gate(
            "pass" if observed_all_success else "fail",
            observed_step_statuses=runtime_statuses,
        ),
        "runtime_status_consistent": _gate(
            "pass" if runtime_consistent else "fail",
            overall_success=bool(overall_success),
            observed_step_statuses=runtime_statuses,
        ),
        "required_agent_sequence": _gate(
            "pass" if agent_sequence_ok else "fail",
            applicable=bool(required_agents),
            required_agents=required_agents,
            actual_agents=actual_agents,
            routed_agent=routed_agent,
        ),
        "required_tools_executed": _gate(
            "pass" if not missing_tool_groups else "fail",
            applicable=bool(required_tool_groups),
            required_groups=[name for name, _ in required_tool_groups],
            missing_groups=missing_tool_groups,
            actual_tools=actual_tools,
        ),
        "artifact_scope_valid": _gate(
            "pass" if (not fresh_artifacts_required or not artifact_scope_errors) else "fail",
            applicable=fresh_artifacts_required,
            artifact_count=len(artifacts),
            errors=sorted(set(artifact_scope_errors)),
            requested_feature_type=requested_feature_type,
        ),
        "artifact_roles_complete": _gate(
            "pass" if artifact_roles_ok else "fail",
            applicable=paired_roles_required,
            required_roles=["train", "test"] if paired_roles_required else [],
            observed_roles=sorted(roles),
        ),
        "downstream_blocked_after_failure": _gate(
            "pass" if not downstream_successes else "fail",
            applicable=first_failure is not None,
            first_failure_step=(first_failure + 1) if first_failure is not None else None,
            downstream_success_steps=downstream_successes,
        ),
        "no_false_completion": _gate(
            "pass" if not false_completion else "fail",
            false_completion_detected=false_completion,
            output_sha256=stable_json_hash(str(final_output or "")),
        ),
        "report_follows_current_classification": _gate(
            "pass" if report_order_ok else "fail",
            applicable=bool(
                task_spec.get("requires_classification")
                and task_spec.get("requires_report_read")
            ),
            actual_tools=actual_tools,
        ),
    }

    hard_gates_pass = all(gate["status"] == "pass" for gate in gates.values())
    task_success = bool(overall_success) and hard_gates_pass
    successful_steps = sum(status == "success" for status in runtime_statuses)
    step_success_ratio = successful_steps / len(steps) if steps else 0.0
    metrics = {
        "task_success": _metric(1.0 if task_success else 0.0),
        "step_success_ratio": _metric(
            step_success_ratio,
            successful_steps=successful_steps,
            total_steps=len(steps),
        ),
        "required_tool_coverage": _metric(
            tool_coverage,
            required_group_count=len(required_tool_groups),
            missing_group_count=len(missing_tool_groups),
        ),
        "false_completion": _metric(1.0 if false_completion else 0.0),
        "cross_run_artifact_violation": _metric(
            1.0 if "run_id_mismatch" in artifact_scope_errors else 0.0
        ),
    }

    task_type = _task_type(task_spec)
    method_name = _method_name(task_spec)
    records: list[dict[str, Any]] = []
    base = {
        "trace_id": str(trace_id),
        "task_id": task_id,
        "policy_id": policy_id,
        "task_type": task_type,
        "method_name": method_name,
        "split_type": "unknown",
        "dataset_hash": dataset_hash,
        "evaluator_name": TASK_EVALUATOR_NAME,
        "evaluator_version": TASK_EVALUATOR_VERSION,
    }
    for name, metric in metrics.items():
        evaluation_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"{trace_id}:{TASK_EVALUATOR_NAME}:{TASK_EVALUATOR_VERSION}:metric:{name}",
            )
        )
        records.append(
            {
                **base,
                "evaluation_id": evaluation_id,
                "record_type": "metric",
                "metric_name": name,
                "metric_value": metric["value"],
                "gate_status": None,
                "details": metric["details"],
            }
        )
    for name, gate in gates.items():
        evaluation_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"{trace_id}:{TASK_EVALUATOR_NAME}:{TASK_EVALUATOR_VERSION}:gate:{name}",
            )
        )
        records.append(
            {
                **base,
                "evaluation_id": evaluation_id,
                "record_type": "gate",
                "metric_name": name,
                "metric_value": None,
                "gate_status": gate["status"],
                "details": {
                    "applicable": gate["applicable"],
                    **gate["details"],
                },
            }
        )

    return {
        "schema_version": "1.0",
        "trace_id": str(trace_id),
        "task_id": task_id,
        "run_id": run_id,
        "policy_id": policy_id,
        "task_type": task_type,
        "method_name": method_name,
        "dataset_hash": dataset_hash,
        "evaluator_name": TASK_EVALUATOR_NAME,
        "evaluator_version": TASK_EVALUATOR_VERSION,
        "task_success": task_success,
        "metrics": metrics,
        "gates": gates,
        "records": records,
    }


def persist_task_evaluation(db: Any, report: Mapping[str, Any]) -> None:
    """Persist all records and one sanitized summary event using a DB-like object."""
    records = report.get("records", [])
    for record in records:
        db.save_evaluation_result(dict(record))
    failed_gates = sorted(
        name
        for name, gate in report.get("gates", {}).items()
        if gate.get("status") != "pass"
    )
    db.save_trajectory_event(
        build_trajectory_event(
            event_type="evaluation_recorded",
            trace_id=str(report["trace_id"]),
            task_id=report.get("task_id"),
            actor_type="evaluator",
            actor_name=TASK_EVALUATOR_NAME,
            policy_id=report.get("policy_id"),
            status="success" if report.get("task_success") else "failed",
            payload={
                "evaluator_version": report.get("evaluator_version"),
                "task_type": report.get("task_type"),
                "method_name": report.get("method_name"),
                "task_success": bool(report.get("task_success")),
                "record_count": len(records),
                "failed_gates": failed_gates,
            },
        )
    )
