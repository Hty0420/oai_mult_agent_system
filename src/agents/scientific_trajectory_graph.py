"""Create a privacy-safe visual summary of one completed Agent trajectory.

The PNG is intended for the right-hand result gallery, the Markdown report for
the adjacent evidence panel, and the JSON file is the machine-readable,
versioned source used by later replay and learning work.  Raw prompts, tool
arguments, patient identifiers and filesystem paths are deliberately excluded
from every artifact.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


SCIENTIFIC_TRAJECTORY_SCHEMA_VERSION = "2.0"
DEFAULT_GRAPH_DIRECTORY = "output/scientific_trajectory_graphs"
MAX_RENDERED_STEPS = 16

_SAFE_CODE = re.compile(r"[^A-Za-z0-9_.:-]+")

_AGENT_LABELS = {
    "ORCHESTRATOR_AGENT": "任务编排 Agent",
    "ROUTER_AGENT": "任务路由 Agent",
    "SEGMENTATION_AGENT": "影像分割 Agent",
    "FEATURE_EXTRACTION_AGENT": "特征提取 Agent",
    "FEATURE_COMBINATION_AGENT": "特征融合 Agent",
    "CLASSIFICATION_AGENT": "分类分析 Agent",
    "PIPELINE_OPTIMIZATION_AGENT": "流水线优化 Agent",
    "BIOMARKER_DISCOVERY_AGENT": "标志物发现 Agent",
    "BIOMARKER_VERIFIER_AGENT": "标志物验证 Agent",
    "BIOMARKER_REPORTER_AGENT": "标志物报告 Agent",
    "WEB_SEARCH_AGENT": "文献检索 Agent",
    "CONVERSATION_AGENT": "对话 Agent",
    "ERROR": "执行控制面",
    "UNKNOWN": "未识别 Agent",
}

_STATUS_LABELS = {
    "success": "通过",
    "completed": "完成",
    "failed": "失败",
    "skipped": "跳过",
    "running": "运行中",
    "pending": "等待",
    "unknown": "待复核",
}

_GATE_LABELS = {
    "execution_succeeded": "执行步骤全部成功",
    "runtime_status_consistent": "运行状态与步骤一致",
    "required_agent_sequence": "必需 Agent 顺序正确",
    "required_tools_executed": "必需工具均已执行",
    "artifact_scope_valid": "产物属于当前运行",
    "artifact_roles_complete": "训练/测试产物角色完整",
    "downstream_blocked_after_failure": "失败后下游已阻断",
    "no_false_completion": "未发生虚假完成",
    "report_follows_current_classification": "报告来自当前分类结果",
}

_REASON_LABELS = {
    "training_only_nested_cv_then_prior_tie_break": "训练内嵌套交叉验证，近似时按先验破同分",
    "matched_training_only_nested_cv": "基于匹配任务的训练内嵌套交叉验证",
    "pre_registered_scientific_protocol": "预注册科学流程",
    "training_only_candidate_search": "仅用训练数据进行候选搜索",
    "selected_after_training_validation": "训练内验证后选定",
    "current_session_registry": "当前会话已登记数据",
    "current_request_or_bounded_default": "当前请求或受限默认数据",
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _safe_token(value: Any, fallback: str) -> str:
    text = _SAFE_CODE.sub("-", str(value or "").strip()).strip(".-")
    return text[:80] or fallback


def _safe_code(value: Any, fallback: str = "unknown") -> str:
    """Return a bounded code-like label, never arbitrary user or path text."""
    text = str(value or "").strip()
    if not text:
        return fallback
    if len(text) > 100 or _SAFE_CODE.search(text):
        return fallback
    return text


def _agent_label(value: Any) -> str:
    code = _safe_code(value, "UNKNOWN").upper()
    return _AGENT_LABELS.get(code, code.replace("_", " ").title())


def _status(value: Any, *, default: str = "unknown") -> str:
    normalized = str(value or default).strip().lower()
    if normalized == "completed":
        return "success"
    return normalized if normalized in _STATUS_LABELS else default


def _tool_names(tool_calls: Sequence[Any] | None) -> list[str]:
    names: list[str] = []
    for call in tool_calls or []:
        if isinstance(call, Mapping):
            raw_name = call.get("name") or call.get("tool_name")
            if not raw_name and isinstance(call.get("function"), Mapping):
                raw_name = call["function"].get("name")
        else:
            raw_name = call
        name = _safe_code(raw_name, "")
        if name and name not in names:
            names.append(name)
    return names


def _artifact_types(artifacts: Sequence[Any] | None) -> list[str]:
    types: list[str] = []
    for artifact in artifacts or []:
        if not isinstance(artifact, Mapping):
            continue
        raw_type = (
            artifact.get("artifact_type")
            or artifact.get("feature_type")
            or artifact.get("dataset_role")
        )
        artifact_type = _safe_code(raw_type, "")
        if artifact_type and artifact_type not in types:
            types.append(artifact_type)
    return types


def _pipeline_label(selected: Mapping[str, Any]) -> str:
    parts = []
    for key in ("candidate_id", "method_name", "feature_type", "classifier"):
        value = _safe_code(selected.get(key), "")
        if value and value not in parts:
            parts.append(value)
    return " · ".join(parts[:3]) or "已选择候选方案"


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _first_finite_number(*values: Any) -> float | None:
    for value in values:
        number = _finite_number(value)
        if number is not None:
            return number
    return None


def _non_negative_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def _safe_codes(values: Sequence[Any] | None, *, limit: int = 12) -> list[str]:
    result: list[str] = []
    if isinstance(values, (str, bytes)):
        values = [values]
    for value in values or []:
        code = _safe_code(value, "")
        if code and code not in result:
            result.append(code)
        if len(result) >= limit:
            break
    return result


def _dataset_records(dataset_profile: Mapping[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    source_files = dataset_profile.get("source_files") or {}
    if not isinstance(source_files, Mapping):
        source_files = {}
    for role in ("train", "validation", "test", "external_test"):
        raw = source_files.get(role)
        if not isinstance(raw, Mapping):
            continue
        class_distribution: list[dict[str, Any]] = []
        raw_distribution = raw.get("class_distribution") or {}
        if isinstance(raw_distribution, Mapping):
            for label, count in sorted(
                raw_distribution.items(), key=lambda item: str(item[0])
            ):
                safe_label = _safe_code(label, "other")
                safe_count = _non_negative_int(count)
                if safe_count is not None:
                    class_distribution.append(
                        {"label": safe_label, "count": safe_count}
                    )
        sha256 = _safe_code(raw.get("sha256"), "")
        records.append(
            {
                "role": role,
                "row_count": _non_negative_int(raw.get("row_count")),
                "column_count": (
                    len(raw.get("columns") or [])
                    if isinstance(raw.get("columns"), Sequence)
                    and not isinstance(raw.get("columns"), (str, bytes))
                    else None
                ),
                "class_distribution": class_distribution,
                "sha256": sha256 or None,
            }
        )
    return records


def _candidate_records(run_result: Mapping[str, Any]) -> list[dict[str, Any]]:
    selected = dict(run_result.get("selected_pipeline") or {})
    selected_id = _safe_code(selected.get("candidate_id"), "")
    merged: dict[str, dict[str, Any]] = {}
    if selected_id:
        merged[selected_id] = dict(selected)

    for raw in run_result.get("pipeline_candidates") or []:
        if not isinstance(raw, Mapping):
            continue
        candidate_id = _safe_code(raw.get("candidate_id"), "")
        if candidate_id:
            merged.setdefault(candidate_id, {}).update(dict(raw))

    for raw in run_result.get("pipeline_trial_records") or []:
        if not isinstance(raw, Mapping):
            continue
        candidate_id = _safe_code(raw.get("candidate_id"), "")
        if candidate_id:
            merged.setdefault(candidate_id, {}).update(dict(raw))

    records: list[dict[str, Any]] = []
    for candidate_id, raw in merged.items():
        metrics = raw.get("validation_metrics") or raw.get("metrics") or {}
        if not isinstance(metrics, Mapping):
            metrics = {}
        details = raw.get("details") or {}
        if not isinstance(details, Mapping):
            details = {}
        failure_reason = (
            details.get("failure_reason")
            or raw.get("validation_failure_reason")
            or raw.get("failure_reason")
        )
        rejection_reasons = _safe_codes(raw.get("rejection_reasons") or [])
        safe_failure = _safe_code(failure_reason, "")
        if safe_failure and safe_failure not in rejection_reasons:
            rejection_reasons.append(safe_failure)

        records.append(
            {
                "candidate_id": candidate_id,
                "feature_type": _safe_code(raw.get("feature_type"), "unknown"),
                "classification_method": _safe_code(
                    raw.get("classification_method"), "unknown"
                ),
                "model_recipe": _safe_code(raw.get("model_recipe"), "unknown"),
                "status": _safe_code(
                    raw.get("trial_status") or raw.get("status"), "not_run"
                ),
                "validation_scope": _safe_code(
                    raw.get("validation_scope"), "unavailable"
                ),
                "validation_score": _first_finite_number(
                    raw.get("validation_score")
                ),
                "auc": _first_finite_number(raw.get("auc"), metrics.get("AUC")),
                "pr_auc": _first_finite_number(
                    raw.get("pr_auc"), metrics.get("PR_AUC")
                ),
                "f1": _first_finite_number(raw.get("f1"), metrics.get("F1")),
                "auc_ci_lower": _first_finite_number(
                    raw.get("auc_ci_lower"),
                    raw.get("validation_auc_lower_bound"),
                ),
                "auc_ci_upper": _first_finite_number(raw.get("auc_ci_upper")),
                "selection_stability": _first_finite_number(
                    raw.get("selection_stability"),
                    raw.get("validation_selection_stability"),
                ),
                "duration_ms": _non_negative_int(raw.get("duration_ms")),
                "patient_count": _non_negative_int(
                    raw.get("patient_count")
                    if raw.get("patient_count") is not None
                    else raw.get("validation_patient_count")
                ),
                "test_data_used_for_selection": bool(
                    raw.get("test_data_used")
                    or raw.get("test_data_used_for_selection")
                ),
                "rejection_reasons": rejection_reasons,
                "selected": candidate_id == selected_id,
            }
        )

    records.sort(
        key=lambda item: (
            not item["selected"],
            -(
                item["validation_score"]
                if item["validation_score"] is not None
                else -1.0
            ),
            item["candidate_id"],
        )
    )
    return records[:12]


def _gate_evidence(name: str, details: Mapping[str, Any]) -> str:
    if name == "execution_succeeded":
        statuses = _safe_codes(details.get("observed_step_statuses") or [], limit=8)
        return "步骤状态：" + " / ".join(statuses) if statuses else "按步骤状态检查"
    if name == "runtime_status_consistent":
        return f"最终状态：{bool(details.get('overall_success'))}"
    if name == "required_agent_sequence":
        required = _safe_codes(details.get("required_agents") or [], limit=8)
        actual = _safe_codes(details.get("actual_agents") or [], limit=8)
        return f"需要 {len(required)} 个，实际匹配 {len(actual)} 个"
    if name == "required_tools_executed":
        required = _safe_codes(details.get("required_groups") or [], limit=8)
        missing = _safe_codes(details.get("missing_groups") or [], limit=8)
        return (
            "缺少：" + "、".join(missing)
            if missing
            else f"{len(required)} 组必需工具已覆盖"
        )
    if name == "artifact_scope_valid":
        errors = _safe_codes(details.get("errors") or [], limit=8)
        return (
            "错误：" + "、".join(errors)
            if errors
            else f"核验 {int(details.get('artifact_count') or 0)} 个当前运行产物"
        )
    if name == "artifact_roles_complete":
        roles = _safe_codes(details.get("observed_roles") or [], limit=8)
        return "已观察角色：" + ("、".join(roles) if roles else "无需配对")
    if name == "downstream_blocked_after_failure":
        step = _non_negative_int(details.get("first_failure_step"))
        downstream = details.get("downstream_success_steps") or []
        return (
            f"首个失败步骤 {step}；异常下游成功 {len(downstream)} 个"
            if step is not None
            else "本次没有失败步骤"
        )
    if name == "no_false_completion":
        return "未检测到完成状态与执行证据矛盾"
    if name == "report_follows_current_classification":
        return "按当前分类工具调用顺序检查"
    return "确定性规则检查"


def _gate_records(task_evaluation: Mapping[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    gates = task_evaluation.get("gates") or {}
    if not isinstance(gates, Mapping):
        return records
    for raw_name, raw_gate in gates.items():
        if not isinstance(raw_gate, Mapping):
            continue
        name = _safe_code(raw_name, "unknown_gate")
        status = _safe_code(raw_gate.get("status"), "unknown")
        details = raw_gate.get("details") or {}
        if not isinstance(details, Mapping):
            details = {}
        records.append(
            {
                "gate_id": name,
                "label": _GATE_LABELS.get(name, name),
                "status": status if status in {"pass", "fail", "unknown"} else "unknown",
                "applicable": bool(raw_gate.get("applicable", True)),
                "evidence": _gate_evidence(name, details),
            }
        )
    return records


def _execution_records(run_result: Mapping[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, raw in enumerate(run_result.get("step_records") or [], start=1):
        if not isinstance(raw, Mapping):
            continue
        step = _non_negative_int(raw.get("step")) or index
        records.append(
            {
                "step": step,
                "agent": _safe_code(raw.get("agent"), "UNKNOWN"),
                "agent_label": _agent_label(raw.get("agent")),
                "status": _status(raw.get("status")),
                "tool_names": _tool_names(raw.get("tool_calls")),
                "duration_ms": _non_negative_int(raw.get("duration_ms")),
            }
        )
    return records[:32]


def _lineage_records(
    run_result: Mapping[str, Any], *, trace_id: str
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for raw in run_result.get("current_artifacts") or []:
        if not isinstance(raw, Mapping):
            continue
        artifact_type = _safe_code(
            raw.get("artifact_type") or raw.get("feature_type"), "unknown"
        )
        record = {
            "artifact_id": _safe_code(raw.get("artifact_id"), "unregistered"),
            "artifact_type": artifact_type,
            "dataset_role": _safe_code(raw.get("dataset_role"), "unspecified"),
            "feature_type": _safe_code(raw.get("feature_type"), "unspecified"),
            "producer_agent": _safe_code(
                raw.get("producer_agent") or raw.get("created_by"), "unavailable"
            ),
            "producer_tool": _safe_code(raw.get("producer_tool"), "unavailable"),
            "validation_status": _safe_code(
                raw.get("validation_status"), "unverified"
            ),
            "current_run": bool(raw.get("run_id"))
            and str(raw.get("run_id")) == str(trace_id),
            "source_artifact_ids": _safe_codes(
                raw.get("source_artifact_ids") or [], limit=12
            ),
        }
        records.append(record)
    return records[:32]


def _decision_record(run_result: Mapping[str, Any]) -> dict[str, Any]:
    planning = run_result.get("planning_decision") or {}
    if not isinstance(planning, Mapping):
        planning = {}
    portfolio = planning.get("trial_portfolio") or {}
    if not isinstance(portfolio, Mapping):
        portfolio = {}
    reason_codes = _safe_codes(
        [
            planning.get("selection_basis"),
            planning.get("selection_reasoning"),
            planning.get("selection_status"),
            planning.get("dataset_source"),
        ]
    )
    return {
        "selection_status": _safe_code(
            planning.get("selection_status") or planning.get("status"), "unavailable"
        ),
        "reason_codes": reason_codes,
        "reason_labels": [_REASON_LABELS.get(code, code) for code in reason_codes],
        "history_status": _safe_code(planning.get("history_status"), "unavailable"),
        "initial_candidate_id": _safe_code(
            planning.get("initial_selected_candidate_id"), "unavailable"
        ),
        "selected_candidate_id": _safe_code(
            planning.get("selected_candidate_id")
            or (run_result.get("selected_pipeline") or {}).get("candidate_id"),
            "unavailable",
        ),
        "search_mode": _safe_code(portfolio.get("mode"), "unavailable"),
        "search_budget": _non_negative_int(portfolio.get("budget")),
        "successful_trial_count": _non_negative_int(
            planning.get("successful_trial_count")
        ),
        "test_data_used_for_selection": bool(
            planning.get("test_data_used_for_selection")
        ),
    }


def _research_design_record(
    task_spec: Mapping[str, Any],
    goal_spec: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    constraints = goal_spec.get("constraints") or {}
    if not isinstance(constraints, Mapping):
        constraints = {}
    safe_constraints: dict[str, str] = {}
    for raw_key, raw_value in constraints.items():
        key = _safe_code(raw_key, "")
        value = _safe_code(raw_value, "")
        if key and value:
            safe_constraints[key] = value
    validation_scopes = _safe_codes(
        [item.get("validation_scope") for item in candidates], limit=8
    )
    return {
        "goal_type": _safe_code(goal_spec.get("goal_type"), "unavailable"),
        "objective_variant": _safe_code(
            goal_spec.get("objective_variant"), "unavailable"
        ),
        "target_column": _safe_code(goal_spec.get("target_column"), "unavailable"),
        "primary_metric": _safe_code(
            goal_spec.get("primary_metric"), "unavailable"
        ),
        "secondary_metrics": _safe_codes(
            goal_spec.get("secondary_metrics") or [], limit=8
        ),
        "autonomy_mode": _safe_code(
            goal_spec.get("autonomy_mode"), "unavailable"
        ),
        "capability_status": _safe_code(
            goal_spec.get("capability_status"), "unavailable"
        ),
        "constraints": safe_constraints,
        "validation_scopes": [
            item for item in validation_scopes if item != "unavailable"
        ],
        "requires_fresh_artifacts": bool(
            task_spec.get("requires_fresh_artifacts")
        ),
    }


def _recovery_record(
    execution: Sequence[Mapping[str, Any]], gates: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    first_failure = next(
        (item for item in execution if item.get("status") == "failed"), None
    )
    failed_gates = [
        str(item.get("gate_id"))
        for item in gates
        if item.get("status") == "fail"
    ]
    skipped = [
        int(item.get("step"))
        for item in execution
        if item.get("status") == "skipped" and item.get("step") is not None
    ]
    return {
        "failure_boundary_step": (
            int(first_failure["step"]) if first_failure is not None else None
        ),
        "failure_boundary_agent": (
            str(first_failure["agent"]) if first_failure is not None else None
        ),
        "failed_gates": failed_gates,
        "downstream_skipped_steps": skipped,
        "causal_root_verified": False,
        "recommended_resume_step": (
            int(first_failure["step"]) if first_failure is not None else None
        ),
    }


def _learning_record(
    *,
    success: bool,
    candidates: Sequence[Mapping[str, Any]],
    recovery: Mapping[str, Any],
    trace_id: str,
) -> dict[str, Any]:
    selected = next((item for item in candidates if item.get("selected")), None)
    if success and selected is not None:
        evidence_complete = bool(
            selected.get("validation_score") is not None
            and selected.get("auc") is not None
            and not selected.get("test_data_used_for_selection")
        )
        return {
            "type": "strategy_candidate",
            "status": "candidate_requires_cross_task_validation",
            "lesson_code": "prefer_selected_pipeline_in_matching_context",
            "candidate_id": selected.get("candidate_id"),
            "evidence_trace_ids": [trace_id],
            "evidence_complete": evidence_complete,
            "human_verified": False,
            "automatically_admitted_to_memory": False,
        }
    if not success:
        return {
            "type": "recovery_candidate",
            "status": "candidate_requires_causal_review",
            "lesson_code": "repair_from_observed_failure_boundary",
            "failure_boundary_step": recovery.get("failure_boundary_step"),
            "failed_gates": list(recovery.get("failed_gates") or []),
            "evidence_trace_ids": [trace_id],
            "evidence_complete": False,
            "human_verified": False,
            "automatically_admitted_to_memory": False,
        }
    return {
        "type": "none",
        "status": "no_generalizable_lesson_extracted",
        "lesson_code": None,
        "evidence_trace_ids": [trace_id],
        "evidence_complete": False,
        "human_verified": False,
        "automatically_admitted_to_memory": False,
    }


def _add_node(
    nodes: list[dict[str, Any]],
    *,
    node_id: str,
    kind: str,
    title: str,
    summary: str,
    status: str = "success",
    details: Mapping[str, Any] | None = None,
) -> None:
    nodes.append(
        {
            "id": node_id,
            "kind": kind,
            "title": title,
            "summary": summary,
            "status": _status(status),
            "details": dict(details or {}),
        }
    )


def build_scientific_trajectory_spec(
    run_result: Mapping[str, Any],
    *,
    trace_id: str,
    workspace_id: str | None = None,
    agent_version: str | None = None,
    policy_id: str | None = None,
    data_partition: str | None = None,
) -> dict[str, Any]:
    """Build a deterministic, privacy-safe graph specification."""
    result = dict(run_result or {})
    task_spec = dict(result.get("task_spec") or {})
    goal_spec = dict(result.get("goal_spec") or {})
    dataset_profile = dict(result.get("dataset_profile") or {})
    selected_pipeline = dict(result.get("selected_pipeline") or {})
    task_evaluation = dict(result.get("task_evaluation") or {})
    outcome = dict(result.get("outcome_summary") or {})
    step_records = [
        dict(item)
        for item in (result.get("step_records") or [])
        if isinstance(item, Mapping)
    ]

    task_type = _safe_code(
        task_spec.get("task_type") or goal_spec.get("goal_type"),
        "medical_imaging_research",
    )
    final_success = bool(result.get("success"))
    final_status = "success" if final_success else "failed"
    datasets = _dataset_records(dataset_profile)
    candidates = _candidate_records(result)
    research_design = _research_design_record(task_spec, goal_spec, candidates)
    gates = _gate_records(task_evaluation)
    execution = _execution_records(result)
    lineage = _lineage_records(result, trace_id=trace_id)
    decision = _decision_record(result)
    recovery = _recovery_record(execution, gates)
    learning = _learning_record(
        success=final_success,
        candidates=candidates,
        recovery=recovery,
        trace_id=str(trace_id),
    )
    passed_gates = sum(item.get("status") == "pass" for item in gates)
    failed_gate_count = sum(item.get("status") == "fail" for item in gates)
    total_tool_calls = sum(len(item.get("tool_names") or []) for item in execution)
    trial_duration_ms = sum(
        int(item["duration_ms"])
        for item in candidates
        if item.get("duration_ms") is not None
    )
    performance = {
        "total_latency_ms": _non_negative_int(outcome.get("total_latency_ms")),
        "step_count": len(execution),
        "successful_step_count": sum(
            item.get("status") == "success" for item in execution
        ),
        "tool_call_count": total_tool_calls,
        "candidate_trial_count": len(
            [item for item in candidates if item.get("status") != "not_run"]
        ),
        "candidate_trial_duration_ms": trial_duration_ms or None,
    }
    evaluator_versions = _safe_codes(
        [
            task_evaluation.get("evaluator_version"),
            *[
                item.get("validation_scope")
                for item in candidates
                if item.get("validation_scope") != "unavailable"
            ],
        ],
        limit=8,
    )
    reproducibility = {
        "task_fingerprint": _safe_code(
            result.get("task_fingerprint"), "unavailable"
        ),
        "execution_fingerprint": _safe_code(
            result.get("execution_fingerprint"), "unavailable"
        ),
        "dataset_hash": _safe_code(dataset_profile.get("dataset_hash"), "unavailable"),
        "agent_version": _safe_code(agent_version, "unavailable"),
        "policy_id": _safe_code(policy_id, "unavailable"),
        "registry_version": _safe_code(
            (result.get("planning_decision") or {}).get("registry_version"),
            "unavailable",
        ),
        "selected_model_recipe": _safe_code(
            selected_pipeline.get("model_recipe"), "unavailable"
        ),
        "evaluator_versions": evaluator_versions,
        "unavailable_fields": [
            "model_version",
            "prompt_hash",
            "tool_versions",
            "code_commit",
            "environment_lock_hash",
            "random_seed",
        ],
    }

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, str]] = []

    _add_node(
        nodes,
        node_id="task",
        kind="task",
        title="科研任务",
        summary=task_type,
        status="success",
        details={
            "task_id": result.get("task_id"),
            "task_fingerprint": _safe_code(
                result.get("task_fingerprint"), "unavailable"
            ),
        },
    )

    record_count = sum(
        int(item["row_count"])
        for item in datasets
        if item.get("row_count") is not None
    ) or None
    dataset_hash = _safe_code(dataset_profile.get("dataset_hash"), "")
    dataset_bits = [f"分区：{_safe_code(data_partition, 'unknown')}"]
    if research_design["objective_variant"] != "unavailable":
        dataset_bits.append(f"目标：{research_design['objective_variant']}")
    if record_count is not None:
        dataset_bits.append(f"记录：{record_count}")
    if dataset_hash:
        dataset_bits.append(f"数据哈希：{dataset_hash[:10]}…")
    _add_node(
        nodes,
        node_id="data_context",
        kind="data",
        title="数据与研究边界",
        summary=" · ".join(dataset_bits),
        status="success" if dataset_profile.get("status") != "blocked" else "failed",
        details={
            "data_partition": _safe_code(data_partition, "unknown"),
            "record_count": record_count,
            "dataset_hash": dataset_hash or None,
            "dataset_roles": [item["role"] for item in datasets],
            "issues": _safe_codes(dataset_profile.get("issues") or []),
        },
    )
    edges.append({"source": "task", "target": "data_context", "relation": "uses"})

    routed_agent = result.get("next_agent") or result.get("agent_name") or "UNKNOWN"
    _add_node(
        nodes,
        node_id="route",
        kind="decision",
        title="任务路由",
        summary=_agent_label(routed_agent),
        status="success" if routed_agent else "unknown",
        details={"routed_agent": _safe_code(routed_agent, "UNKNOWN")},
    )
    edges.append(
        {"source": "data_context", "target": "route", "relation": "constrains"}
    )

    raw_execution_plan = result.get("execution_plan") or {}
    execution_plan = (
        dict(raw_execution_plan) if isinstance(raw_execution_plan, Mapping) else {}
    )
    plan_steps = execution_plan.get("steps") or step_records
    plan_basis = "；".join(decision["reason_labels"][:2])
    _add_node(
        nodes,
        node_id="plan",
        kind="decision",
        title="执行计划",
        summary=(
            f"{len(plan_steps)} 个步骤 · {plan_basis}"
            if plan_basis
            else f"{len(plan_steps)} 个步骤 · 确定性门禁保护"
        ),
        status="success" if plan_steps else "unknown",
        details={"step_count": len(plan_steps)},
    )
    edges.append({"source": "route", "target": "plan", "relation": "creates"})
    previous_id = "plan"

    if selected_pipeline or result.get("pipeline_candidates"):
        selected_record = next(
            (item for item in candidates if item.get("selected")), None
        )
        selected_evidence = ""
        if selected_record:
            evidence_bits = []
            if selected_record.get("validation_score") is not None:
                evidence_bits.append(
                    f"综合分 {selected_record['validation_score']:.3f}"
                )
            if selected_record.get("auc") is not None:
                evidence_bits.append(f"AUC {selected_record['auc']:.3f}")
            if selected_record.get("auc_ci_lower") is not None:
                evidence_bits.append(
                    f"CI 下限 {selected_record['auc_ci_lower']:.3f}"
                )
            if selected_record.get("selection_stability") is not None:
                evidence_bits.append(
                    f"稳定性 {selected_record['selection_stability']:.3f}"
                )
            selected_evidence = " · " + " / ".join(evidence_bits[:3])
        _add_node(
            nodes,
            node_id="pipeline_selection",
            kind="decision",
            title="候选流水线选择",
            summary=(
                f"{len(candidates)} 个已记录候选 → "
                f"{_pipeline_label(selected_pipeline)}{selected_evidence}"
            ),
            status="success" if selected_pipeline else "unknown",
            details={
                "candidate_count": len(candidates),
                "selected_candidate_id": _safe_code(
                    selected_pipeline.get("candidate_id"), "unavailable"
                ),
            },
        )
        edges.append(
            {
                "source": previous_id,
                "target": "pipeline_selection",
                "relation": "selects",
            }
        )
        previous_id = "pipeline_selection"

    rendered_steps = step_records[:MAX_RENDERED_STEPS]
    if not rendered_steps:
        rendered_steps = [
            {
                "step": 1,
                "agent": result.get("agent_name") or routed_agent,
                "status": final_status,
                "tool_calls": result.get("tool_calls") or [],
            }
        ]

    for index, step in enumerate(rendered_steps, start=1):
        step_number = step.get("step")
        if not isinstance(step_number, int) or isinstance(step_number, bool):
            step_number = index
        step_id = f"step_{index}"
        tools = _tool_names(step.get("tool_calls"))
        tool_summary = "、".join(tools[:4]) if tools else "无外部工具"
        if len(tools) > 4:
            tool_summary += f" 等 {len(tools)} 项"
        _add_node(
            nodes,
            node_id=step_id,
            kind="execution",
            title=f"步骤 {step_number} · {_agent_label(step.get('agent'))}",
            summary=f"工具：{tool_summary}",
            status=_status(step.get("status"), default=final_status),
            details={
                "step": step_number,
                "agent": _safe_code(step.get("agent"), "UNKNOWN"),
                "tool_names": tools,
            },
        )
        edges.append({"source": previous_id, "target": step_id, "relation": "executes"})
        previous_id = step_id

    omitted_steps = max(0, len(step_records) - len(rendered_steps))
    if omitted_steps:
        _add_node(
            nodes,
            node_id="steps_omitted",
            kind="execution",
            title="其余执行步骤",
            summary=f"另有 {omitted_steps} 个步骤，完整记录见 JSON 清单",
            status=final_status,
            details={"omitted_step_count": omitted_steps},
        )
        edges.append(
            {"source": previous_id, "target": "steps_omitted", "relation": "continues"}
        )
        previous_id = "steps_omitted"

    artifacts = _artifact_types(result.get("current_artifacts"))
    if artifacts:
        artifact_summary = "、".join(artifacts[:5])
        if len(artifacts) > 5:
            artifact_summary += f" 等 {len(artifacts)} 类"
        _add_node(
            nodes,
            node_id="evidence",
            kind="evidence",
            title="科研产物与证据",
            summary=artifact_summary,
            status="success",
            details={"artifact_types": artifacts},
        )
        edges.append(
            {"source": previous_id, "target": "evidence", "relation": "produces"}
        )
        previous_id = "evidence"

    failed_gates = task_evaluation.get("failed_gates") or outcome.get("failed_gates") or []
    safe_failed_gates = [
        _safe_code(item, "unknown_gate") for item in failed_gates[:8]
    ]
    gate_status = _safe_code(
        outcome.get("completion_gate_status")
        or task_evaluation.get("completion_gate_status"),
        "pass" if final_success else "failed",
    )
    gate_summary = (
        f"{passed_gates}/{len(gates)} 项通过"
        if gates and not failed_gate_count and final_success
        else (
        "未通过：" + "、".join(safe_failed_gates)
        if safe_failed_gates
        else "需要人工复核"
        )
    )
    _add_node(
        nodes,
        node_id="evaluation",
        kind="verification",
        title="科学完成门禁",
        summary=gate_summary,
        status="success" if gate_status == "pass" and final_success else "failed",
        details={
            "completion_gate_status": gate_status,
            "failed_gates": safe_failed_gates,
        },
    )
    edges.append(
        {"source": previous_id, "target": "evaluation", "relation": "validated_by"}
    )

    outcome_status = _safe_code(
        outcome.get("outcome_status"), "completed" if final_success else "needs_review"
    )
    _add_node(
        nodes,
        node_id="outcome",
        kind="outcome",
        title="任务结果",
        summary=(
            "形成策略经验候选 · 尚需跨任务验证和人审"
            if learning.get("type") == "strategy_candidate"
            else (
                "形成恢复经验候选 · 尚需因果复核"
                if learning.get("type") == "recovery_candidate"
                else "未自动提取可泛化经验"
            )
        ),
        status=final_status,
        details={"outcome_status": outcome_status, "task_success": final_success},
    )
    edges.append(
        {"source": "evaluation", "target": "outcome", "relation": "determines"}
    )

    return {
        "schema_version": SCIENTIFIC_TRAJECTORY_SCHEMA_VERSION,
        "generated_at": _utc_now_iso(),
        "trace_id": _safe_code(trace_id, "unavailable"),
        "task_id": result.get("task_id"),
        "workspace_id": _safe_code(workspace_id, "unavailable"),
        "agent_version": _safe_code(agent_version, "unavailable"),
        "policy_id": _safe_code(policy_id, "unavailable"),
        "data_partition": _safe_code(data_partition, "unknown"),
        "query_sha256": _stable_hash(result.get("task_fingerprint") or trace_id),
        "privacy": {
            "contains_raw_prompt": False,
            "contains_tool_arguments": False,
            "contains_patient_identifiers": False,
            "contains_filesystem_paths": False,
        },
        "summary": {
            "task_type": task_type,
            "task_success": final_success,
            "outcome_status": outcome_status,
            "completion_gate_status": gate_status,
            "passed_gate_count": passed_gates,
            "failed_gate_count": failed_gate_count,
            "selected_candidate_id": decision["selected_candidate_id"],
            "total_latency_ms": performance["total_latency_ms"],
        },
        "sections": {
            "datasets": datasets,
            "research_design": research_design,
            "decision": decision,
            "candidate_comparison": candidates,
            "execution": execution,
            "artifact_lineage": lineage,
            "scientific_gates": gates,
            "performance": performance,
            "recovery": recovery,
            "learning_candidate": learning,
            "reproducibility": reproducibility,
        },
        "nodes": nodes,
        "edges": edges,
    }


def _md(value: Any) -> str:
    return str(value if value is not None else "—").replace("|", "\\|")


def _metric_text(value: Any) -> str:
    number = _finite_number(value)
    return "—" if number is None else f"{number:.3f}"


def _duration_text(value: Any) -> str:
    milliseconds = _non_negative_int(value)
    if milliseconds is None:
        return "未记录"
    if milliseconds < 1000:
        return f"{milliseconds} ms"
    return f"{milliseconds / 1000:.1f} s"


def _distribution_text(items: Sequence[Mapping[str, Any]]) -> str:
    if not items:
        return "未记录"
    return "；".join(
        f"{_md(item.get('label'))}: {_md(item.get('count'))}" for item in items
    )


def build_scientific_trajectory_markdown(spec: Mapping[str, Any]) -> str:
    """Render the privacy-safe spec as a detailed human-readable report."""
    summary = dict(spec.get("summary") or {})
    sections = dict(spec.get("sections") or {})
    datasets = list(sections.get("datasets") or [])
    research_design = dict(sections.get("research_design") or {})
    decision = dict(sections.get("decision") or {})
    candidates = list(sections.get("candidate_comparison") or [])
    execution = list(sections.get("execution") or [])
    lineage = list(sections.get("artifact_lineage") or [])
    gates = list(sections.get("scientific_gates") or [])
    performance = dict(sections.get("performance") or {})
    recovery = dict(sections.get("recovery") or {})
    learning = dict(sections.get("learning_candidate") or {})
    reproducibility = dict(sections.get("reproducibility") or {})

    success = bool(summary.get("task_success"))
    verdict = "✅ 完成且通过科学门禁" if success else "⚠️ 未完成或需要人工复核"
    lines = [
        "# 科学轨迹报告",
        "",
        f"> {verdict} · Trace `{_md(spec.get('trace_id'))}` · "
        f"数据分区 `{_md(spec.get('data_partition'))}`",
        "",
        "## 任务判定",
        "",
        "| 项目 | 结果 |",
        "|---|---|",
        f"| 任务类型 | `{_md(summary.get('task_type'))}` |",
        f"| 终态 | `{_md(summary.get('outcome_status'))}` |",
        f"| 科学门禁 | {_md(summary.get('passed_gate_count'))}/{len(gates)} 通过；"
        f"{_md(summary.get('failed_gate_count'))} 项失败 |",
        f"| 选中方案 | `{_md(summary.get('selected_candidate_id'))}` |",
        f"| 总耗时 | {_duration_text(summary.get('total_latency_ms'))} |",
        "",
        "## 数据与研究设计",
        "",
    ]
    if datasets:
        lines.extend(
            [
                "| 数据角色 | 记录数 | 字段数 | 类别分布 | 内容哈希 |",
                "|---|---:|---:|---|---|",
            ]
        )
        for dataset in datasets:
            digest = str(dataset.get("sha256") or "")
            lines.append(
                f"| `{_md(dataset.get('role'))}` | "
                f"{_md(dataset.get('row_count'))} | "
                f"{_md(dataset.get('column_count'))} | "
                f"{_md(_distribution_text(dataset.get('class_distribution') or []))} | "
                f"`{_md(digest[:12] + '…' if digest else '未记录')}` |"
            )
    else:
        lines.append("本任务没有可用的结构化数据画像。")

    constraints = research_design.get("constraints") or {}
    constraint_text = "、".join(
        f"{_md(key)}={_md(value)}" for key, value in constraints.items()
    ) or "未记录"
    lines.extend(
        [
            "",
            "| 研究设计项 | 记录 |",
            "|---|---|",
            f"| 科研目标 | `{_md(research_design.get('goal_type'))}` / "
            f"`{_md(research_design.get('objective_variant'))}` |",
            f"| 预测目标 | `{_md(research_design.get('target_column'))}` |",
            f"| 主指标 | `{_md(research_design.get('primary_metric'))}` |",
            f"| 次指标 | `{_md('、'.join(research_design.get('secondary_metrics') or []) or '未记录')}` |",
            f"| 验证设计 | `{_md('、'.join(research_design.get('validation_scopes') or []) or '未记录')}` |",
            f"| 自主程度 | `{_md(research_design.get('autonomy_mode'))}` |",
            f"| 能力边界 | `{_md(research_design.get('capability_status'))}` |",
            f"| 预设约束 | `{constraint_text}` |",
        ]
    )

    lines.extend(["", "## 自动规划与选择依据", ""])
    reason_labels = list(decision.get("reason_labels") or [])
    if reason_labels:
        lines.extend(f"- {reason}" for reason in reason_labels)
    else:
        lines.append("- 未记录结构化选择依据。")
    lines.extend(
        [
            f"- 历史证据状态：`{_md(decision.get('history_status'))}`",
            f"- 搜索模式与预算：`{_md(decision.get('search_mode'))}` / "
            f"{_md(decision.get('search_budget'))}",
            f"- 初始方案 → 最终方案：`{_md(decision.get('initial_candidate_id'))}` → "
            f"`{_md(decision.get('selected_candidate_id'))}`",
            "- 测试数据参与方案选择："
            + ("**是，存在泄漏风险**" if decision.get("test_data_used_for_selection") else "否"),
            "",
            "## 候选方案比较",
            "",
        ]
    )
    if candidates:
        lines.extend(
            [
                "| 方案 | 状态 | 综合分 | AUC（95% CI） | PR-AUC | F1 | 稳定性 | 耗时 |",
                "|---|---|---:|---|---:|---:|---:|---:|",
            ]
        )
        for candidate in candidates:
            marker = "✓ " if candidate.get("selected") else ""
            lower = _metric_text(candidate.get("auc_ci_lower"))
            upper = _metric_text(candidate.get("auc_ci_upper"))
            auc_text = _metric_text(candidate.get("auc"))
            if lower != "—" or upper != "—":
                auc_text += f"（{lower}–{upper}）"
            status = _md(candidate.get("status"))
            if candidate.get("rejection_reasons"):
                status += "：" + "、".join(candidate["rejection_reasons"][:3])
            lines.append(
                f"| {marker}`{_md(candidate.get('candidate_id'))}` | {status} | "
                f"{_metric_text(candidate.get('validation_score'))} | {auc_text} | "
                f"{_metric_text(candidate.get('pr_auc'))} | "
                f"{_metric_text(candidate.get('f1'))} | "
                f"{_metric_text(candidate.get('selection_stability'))} | "
                f"{_duration_text(candidate.get('duration_ms'))} |"
            )
        lines.append("")
        lines.append(
            "候选比较来自训练内验证记录；选中标记不代表临床有效性或外部泛化。"
        )
    else:
        lines.append("本任务没有进行可比较的候选流水线搜索。")

    lines.extend(["", "## 科学门禁矩阵", ""])
    if gates:
        lines.extend(
            [
                "| 门禁 | 状态 | 适用 | 可审计证据 |",
                "|---|---|---|---|",
            ]
        )
        status_labels = {"pass": "✅ 通过", "fail": "❌ 失败", "unknown": "⚠️ 未知"}
        for gate in gates:
            lines.append(
                f"| {_md(gate.get('label'))} | "
                f"{status_labels.get(str(gate.get('status')), '⚠️ 未知')} | "
                f"{'是' if gate.get('applicable') else '否'} | "
                f"{_md(gate.get('evidence'))} |"
            )
    else:
        lines.append("未获得结构化科学门禁记录，需要人工复核。")

    lines.extend(["", "## 执行步骤与工具", ""])
    if execution:
        lines.extend(
            [
                "| 步骤 | Agent | 状态 | 工具 | 步骤耗时 |",
                "|---:|---|---|---|---:|",
            ]
        )
        for step in execution:
            tools = "、".join(step.get("tool_names") or []) or "无外部工具"
            lines.append(
                f"| {_md(step.get('step'))} | {_md(step.get('agent_label'))} | "
                f"{_md(_STATUS_LABELS.get(str(step.get('status')), '待复核'))} | "
                f"{_md(tools)} | {_duration_text(step.get('duration_ms'))} |"
            )
    else:
        lines.append("未获得结构化步骤记录。")

    lines.extend(["", "## 产物血缘", ""])
    if lineage:
        lines.extend(
            [
                "| 产物 | 数据角色 | 特征类型 | 生产者 / 工具 | 上游产物 | 当前运行 | 验证状态 |",
                "|---|---|---|---|---|---|---|",
            ]
        )
        for artifact in lineage:
            artifact_id = str(artifact.get("artifact_id") or "unregistered")
            upstream = artifact.get("source_artifact_ids") or []
            upstream_text = "、".join(
                str(item)[:12] + ("…" if len(str(item)) > 12 else "")
                for item in upstream
            ) or "无 / 未登记"
            lines.append(
                f"| `{_md(artifact.get('artifact_type'))}`<br>"
                f"`{_md(artifact_id[:12] + ('…' if len(artifact_id) > 12 else ''))}` | "
                f"`{_md(artifact.get('dataset_role'))}` | "
                f"`{_md(artifact.get('feature_type'))}` | "
                f"`{_md(artifact.get('producer_agent'))}` / "
                f"`{_md(artifact.get('producer_tool'))}` | "
                f"`{_md(upstream_text)}` | "
                f"{'是' if artifact.get('current_run') else '否'} | "
                f"`{_md(artifact.get('validation_status'))}` |"
            )
    else:
        lines.append("本次运行没有登记可显示的科研产物血缘。")

    lines.extend(["", "## 性能与故障边界", ""])
    lines.extend(
        [
            f"- 步骤成功：{_md(performance.get('successful_step_count'))}/"
            f"{_md(performance.get('step_count'))}",
            f"- 工具调用：{_md(performance.get('tool_call_count'))} 次",
            f"- 候选试验：{_md(performance.get('candidate_trial_count'))} 个；累计 "
            f"{_duration_text(performance.get('candidate_trial_duration_ms'))}",
        ]
    )
    if recovery.get("failure_boundary_step") is not None:
        lines.extend(
            [
                f"- 首个观察到的失败边界：步骤 {recovery['failure_boundary_step']}，"
                f"Agent `{_md(recovery.get('failure_boundary_agent'))}`",
                f"- 建议恢复位置：步骤 {recovery.get('recommended_resume_step')}",
                "- 因果根因：**尚未验证**；失败边界不能自动等同于根因。",
            ]
        )
    else:
        lines.append("- 未观察到失败步骤；这不等同于已经证明科学结论正确。")

    lines.extend(["", "## 本次任务形成的经验候选", ""])
    if learning.get("type") == "strategy_candidate":
        lines.extend(
            [
                f"- 类型：策略经验候选 `{_md(learning.get('lesson_code'))}`",
                f"- 支持方案：`{_md(learning.get('candidate_id'))}`",
                "- 当前状态：需要在匹配任务上做跨任务验证，并由人类审核。",
            ]
        )
    elif learning.get("type") == "recovery_candidate":
        lines.extend(
            [
                f"- 类型：恢复经验候选 `{_md(learning.get('lesson_code'))}`",
                f"- 失败门禁：{_md('、'.join(learning.get('failed_gates') or []) or '未定位')}",
                "- 当前状态：需要反事实或人工因果复核。",
            ]
        )
    else:
        lines.append("本次任务没有自动提取可泛化经验。")
    lines.extend(
        [
            "- 自动写入长期记忆：**否**",
            "- 人工验证：**否**",
            "",
            "## 可复现性清单",
            "",
            "| 字段 | 记录 |",
            "|---|---|",
            f"| Agent 版本 | `{_md(reproducibility.get('agent_version'))}` |",
            f"| 策略版本 | `{_md(reproducibility.get('policy_id'))}` |",
            f"| 数据集哈希 | `{_md(reproducibility.get('dataset_hash'))}` |",
            f"| 任务指纹 | `{_md(reproducibility.get('task_fingerprint'))}` |",
            f"| 执行指纹 | `{_md(reproducibility.get('execution_fingerprint'))}` |",
            f"| 流水线注册表 | `{_md(reproducibility.get('registry_version'))}` |",
            f"| 选中模型配方 | `{_md(reproducibility.get('selected_model_recipe'))}` |",
            f"| 评估器/验证范围 | `{_md('、'.join(reproducibility.get('evaluator_versions') or []) or '未记录')}` |",
            f"| 尚未采集 | `{_md('、'.join(reproducibility.get('unavailable_fields') or []))}` |",
            "",
            "> 隐私说明：本报告不包含原始提示词、工具参数、患者标识、文件名或文件系统路径。"
            "它是科研过程审计记录，不构成临床诊断或模型临床有效性证明。",
            "",
        ]
    )
    return "\n".join(lines)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _output_directory(
    *,
    trace_id: str,
    workspace_id: str | None,
    output_root: str | os.PathLike[str] | None,
) -> Path:
    configured_root = output_root or os.getenv("SCIENTIFIC_TRAJECTORY_DIR")
    root = (
        Path(configured_root).expanduser()
        if configured_root
        else _project_root() / DEFAULT_GRAPH_DIRECTORY
    )
    workspace_segment = _safe_token(
        workspace_id, f"workspace-{_stable_hash(workspace_id)[:12]}"
    )
    trace_segment = _safe_token(trace_id, f"trace-{_stable_hash(trace_id)[:12]}")
    return (root / workspace_segment / trace_segment).resolve()


def _render_graph_png(spec: Mapping[str, Any], destination: Path) -> None:
    import matplotlib

    matplotlib.use("Agg", force=False)
    from matplotlib import pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    matplotlib.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    matplotlib.rcParams["axes.unicode_minus"] = False

    nodes = list(spec.get("nodes") or [])
    edge_lookup = {
        (str(edge.get("source")), str(edge.get("target"))): edge
        for edge in (spec.get("edges") or [])
    }
    node_count = max(1, len(nodes))
    figure_height = max(8.0, min(30.0, 1.38 * node_count + 2.3))
    figure, axis = plt.subplots(figsize=(9.2, figure_height), dpi=160)
    figure.patch.set_facecolor("#F6F8FB")
    axis.set_facecolor("#F6F8FB")
    axis.axis("off")

    fill_by_kind = {
        "task": "#E8F1FF",
        "data": "#EAF7F1",
        "decision": "#F2ECFF",
        "execution": "#E9F7F8",
        "evidence": "#FFF4E4",
        "verification": "#EDF6EC",
        "outcome": "#EAF2FF",
    }
    border_by_status = {
        "success": "#2F7D5C",
        "failed": "#B94A48",
        "skipped": "#A17120",
        "pending": "#64748B",
        "running": "#2563A6",
        "unknown": "#64748B",
    }

    positions: dict[str, tuple[float, float]] = {}
    top_y = float(node_count - 1)
    for index, node in enumerate(nodes):
        positions[str(node["id"])] = (0.0, top_y - index)

    for index in range(len(nodes) - 1):
        source = str(nodes[index]["id"])
        target = str(nodes[index + 1]["id"])
        if (source, target) not in edge_lookup:
            continue
        _, source_y = positions[source]
        _, target_y = positions[target]
        axis.add_patch(
            FancyArrowPatch(
                (0.0, source_y - 0.25),
                (0.0, target_y + 0.25),
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=1.4,
                color="#95A2B3",
                zorder=1,
            )
        )

    for node in nodes:
        node_id = str(node["id"])
        x, y = positions[node_id]
        status = _status(node.get("status"))
        kind = str(node.get("kind") or "execution")
        box = FancyBboxPatch(
            (x - 0.72, y - 0.25),
            1.44,
            0.5,
            boxstyle="round,pad=0.025,rounding_size=0.035",
            linewidth=1.9,
            edgecolor=border_by_status[status],
            facecolor=fill_by_kind.get(kind, "#F1F4F8"),
            zorder=2,
        )
        axis.add_patch(box)
        status_label = _STATUS_LABELS.get(status, "待复核")
        title = str(node.get("title") or "步骤")
        summary = str(node.get("summary") or "")
        axis.text(
            x - 0.66,
            y + 0.08,
            f"{title}  ·  {status_label}",
            ha="left",
            va="center",
            fontsize=11.5,
            fontweight="medium",
            color="#172033",
            zorder=3,
        )
        axis.text(
            x - 0.66,
            y - 0.105,
            summary[:115],
            ha="left",
            va="center",
            fontsize=9.5,
            color="#4B5B70",
            zorder=3,
        )

    trace_short = str(spec.get("trace_id") or "")[:12]
    axis.set_title(
        f"科学轨迹图  ·  {trace_short}  ·  {spec.get('data_partition', 'unknown')}",
        fontsize=15,
        fontweight="medium",
        color="#172033",
        pad=22,
        loc="left",
    )
    axis.text(
        -0.72,
        -1.02,
        "绿色边框表示通过；红色表示失败或需复核。图中不包含原始提示词、工具参数、患者标识或文件路径。",
        ha="left",
        va="center",
        fontsize=8.7,
        color="#66758A",
    )
    axis.set_xlim(-0.82, 0.82)
    axis.set_ylim(-1.25, top_y + 0.58)
    figure.tight_layout(pad=1.4)

    temporary = destination.with_name(f"{destination.stem}.tmp{destination.suffix}")
    figure.savefig(temporary, format="png", bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)
    temporary.replace(destination)


def generate_scientific_trajectory_graph(
    run_result: Mapping[str, Any],
    *,
    trace_id: str,
    workspace_id: str | None = None,
    agent_version: str | None = None,
    policy_id: str | None = None,
    data_partition: str | None = None,
    output_root: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Write the PNG, Markdown, and JSON artifacts and return absolute paths."""
    destination = _output_directory(
        trace_id=trace_id,
        workspace_id=workspace_id,
        output_root=output_root,
    )
    destination.mkdir(parents=True, exist_ok=True)

    image_path = destination / "scientific_trajectory.png"
    report_path = destination / "scientific_trajectory.md"
    manifest_path = destination / "scientific_trajectory.json"
    spec = build_scientific_trajectory_spec(
        run_result,
        trace_id=trace_id,
        workspace_id=workspace_id,
        agent_version=agent_version,
        policy_id=policy_id,
        data_partition=data_partition,
    )
    _render_graph_png(spec, image_path)
    report_text = build_scientific_trajectory_markdown(spec)
    temporary_report = report_path.with_name(
        f"{report_path.stem}.tmp{report_path.suffix}"
    )
    with temporary_report.open("w", encoding="utf-8") as stream:
        stream.write(report_text)
        if not report_text.endswith("\n"):
            stream.write("\n")
    temporary_report.replace(report_path)
    spec["rendered_artifact"] = {
        "image_file": image_path.name,
        "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
        "report_file": report_path.name,
        "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
    }

    temporary_manifest = manifest_path.with_name(
        f"{manifest_path.stem}.tmp{manifest_path.suffix}"
    )
    with temporary_manifest.open("w", encoding="utf-8") as stream:
        json.dump(spec, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    temporary_manifest.replace(manifest_path)

    return {
        "schema_version": SCIENTIFIC_TRAJECTORY_SCHEMA_VERSION,
        "image_path": str(image_path.resolve()),
        "report_path": str(report_path.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "node_count": len(spec["nodes"]),
        "edge_count": len(spec["edges"]),
    }


def scientific_trajectory_response_appendix(graph: Mapping[str, Any]) -> str:
    """Return parseable result text for the existing gallery/download pipeline."""
    image_path = Path(str(graph["image_path"])).as_posix()
    report_path = Path(str(graph["report_path"])).as_posix()
    manifest_path = Path(str(graph["manifest_path"])).as_posix()
    return (
        "\n\n### 科学轨迹图\n\n"
        "已根据本次任务的决策、执行、产物和科学门禁生成轨迹图。"
        "右侧结果区会自动展示概览图与结构化详报，PNG、Markdown 与可机器读取的 JSON 清单已保存。\n\n"
        f'file_path: "{image_path}"\n'
        f'report_path: "{report_path}"\n'
        f'manifest_path: "{manifest_path}"'
    )
