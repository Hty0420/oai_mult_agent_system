"""Deterministic task identity, closure, and feedback helpers.

The functions in this module deliberately have no LLM dependency.  They turn
one execution into a comparable, privacy-minimizing task record and keep the
authoritative terminal decision behind deterministic evaluation gates.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

from .trajectory import sanitize_for_audit, stable_json_hash


TASK_IDENTITY_SCHEMA_VERSION = "1.0"
TASK_OUTCOME_SCHEMA_VERSION = "1.0"


def _dataset_schema(dataset_profile: Mapping[str, Any]) -> dict[str, Any]:
    source_files = dataset_profile.get("source_files", {})
    result: dict[str, Any] = {}
    if not isinstance(source_files, Mapping):
        return result
    for role in ("train", "test"):
        profile = source_files.get(role)
        if not isinstance(profile, Mapping):
            continue
        result[role] = {
            "sha256": profile.get("sha256"),
            "row_count": profile.get("row_count"),
            "columns": sorted(str(item) for item in profile.get("columns", [])),
            "class_distribution": dict(
                sorted(
                    (str(key), int(value))
                    for key, value in dict(
                        profile.get("class_distribution", {})
                    ).items()
                )
            ),
        }
    return result


def build_task_identity(
    *,
    task_spec: Mapping[str, Any] | None,
    goal_spec: Mapping[str, Any] | None,
    dataset_profile: Mapping[str, Any] | None,
    policy_id: str | None = None,
    selected_pipeline: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build comparison and execution fingerprints for one task.

    ``task_fingerprint`` identifies the decision context and intentionally
    excludes the chosen pipeline and policy so alternative methods can be
    compared within the same context.  ``execution_fingerprint`` adds those
    choices and is used for reproducibility and duplicate detection.
    """
    task_spec = dict(task_spec or {})
    goal_spec = dict(goal_spec or task_spec.get("goal_spec") or {})
    dataset_profile = dict(dataset_profile or {})
    selected_pipeline = dict(selected_pipeline or {})

    context = {
        "schema_version": TASK_IDENTITY_SCHEMA_VERSION,
        "goal_type": goal_spec.get("goal_type") or task_spec.get("goal_type"),
        "objective_variant": (
            goal_spec.get("objective_variant")
            or task_spec.get("objective_variant")
        ),
        "target_column": goal_spec.get("target_column"),
        "label_policy": goal_spec.get("label_policy"),
        "autonomy_mode": (
            goal_spec.get("autonomy_mode") or task_spec.get("autonomy_mode")
        ),
        "constraints": sanitize_for_audit(
            goal_spec.get("constraints", {})
        ),
        "requires": {
            "feature_extraction": bool(
                task_spec.get("requires_feature_extraction")
            ),
            "classification": bool(task_spec.get("requires_classification")),
            "report_read": bool(task_spec.get("requires_report_read")),
            "biomarker_discovery": bool(
                task_spec.get("requires_biomarker_discovery")
            ),
            "biomarker_verification": bool(
                task_spec.get("requires_biomarker_verification")
            ),
            "biomarker_reporting": bool(
                task_spec.get("requires_biomarker_reporting")
            ),
        },
        "artifact_reuse_policy": task_spec.get("artifact_reuse_policy"),
        "dataset_hash": dataset_profile.get("dataset_hash"),
        "dataset_schema": _dataset_schema(dataset_profile),
        "profiler_version": dataset_profile.get("profiler_version"),
    }
    task_fingerprint = stable_json_hash(context)
    execution = {
        "task_fingerprint": task_fingerprint,
        "policy_id": policy_id,
        "selected_pipeline_id": (
            selected_pipeline.get("candidate_id")
            or task_spec.get("selected_pipeline_id")
        ),
        "feature_type": (
            selected_pipeline.get("feature_type")
            or task_spec.get("feature_type")
        ),
        "classification_method": (
            selected_pipeline.get("classification_method")
            or task_spec.get("classification_method")
        ),
    }
    return {
        "schema_version": TASK_IDENTITY_SCHEMA_VERSION,
        "task_fingerprint": task_fingerprint,
        "execution_fingerprint": stable_json_hash(execution),
        "context": context,
        "execution": execution,
    }


def build_outcome_summary(
    *,
    evaluation: Mapping[str, Any],
    step_records: Sequence[Mapping[str, Any]] = (),
    selected_pipeline: Mapping[str, Any] | None = None,
    dataset_hash: str | None = None,
    total_latency_ms: int | None = None,
) -> dict[str, Any]:
    """Summarize the observed outcome without trusting generated prose."""
    statuses = Counter(
        str(item.get("status") or "unknown").lower()
        for item in step_records
    )
    failed_gates = sorted(
        name
        for name, gate in dict(evaluation.get("gates", {})).items()
        if gate.get("status") != "pass"
    )
    task_success = bool(evaluation.get("task_success"))
    if task_success:
        outcome_status = "success"
    elif statuses.get("success", 0) > 0:
        outcome_status = "partial"
    elif not evaluation:
        outcome_status = "needs_review"
    else:
        outcome_status = "failure"

    false_completion = float(
        dict(evaluation.get("metrics", {}))
        .get("false_completion", {})
        .get("value", 0.0)
        or 0.0
    )
    selected_pipeline = dict(selected_pipeline or {})
    return sanitize_for_audit(
        {
            "schema_version": TASK_OUTCOME_SCHEMA_VERSION,
            "outcome_status": outcome_status,
            "task_success": task_success,
            "completion_gate_status": "pass" if task_success else "fail",
            "failed_gates": failed_gates,
            "false_completion": false_completion,
            "step_counts": {
                "success": statuses.get("success", 0),
                "failed": statuses.get("failed", 0),
                "skipped": statuses.get("skipped", 0),
                "unknown": statuses.get("unknown", 0),
                "total": len(step_records),
            },
            "selected_pipeline_id": selected_pipeline.get("candidate_id"),
            "method_name": evaluation.get("method_name"),
            "dataset_hash": dataset_hash or evaluation.get("dataset_hash"),
            "total_latency_ms": total_latency_ms,
            "requires_human_review": bool(
                outcome_status in {"partial", "needs_review"}
                or false_completion
            ),
        }
    )


def terminal_task_status(outcome_summary: Mapping[str, Any]) -> str:
    """Map a structured outcome to the existing authoritative task states."""
    return (
        "completed"
        if outcome_summary.get("completion_gate_status") == "pass"
        else "failed"
    )


def rule_feedback_label(outcome_summary: Mapping[str, Any]) -> str:
    status = str(outcome_summary.get("outcome_status") or "needs_review")
    return {
        "success": "correct",
        "partial": "partial",
        "failure": "incorrect",
        "needs_review": "unknown",
    }.get(status, "unknown")
