"""Typed, privacy-minimizing records for paired pipeline candidate trials."""

from __future__ import annotations

import math
import re
from typing import Any, Mapping
import uuid

from .trajectory import sanitize_for_audit, stable_json_hash


PIPELINE_TRIAL_SCHEMA_VERSION = "1.0"
PIPELINE_TRIAL_STATUSES = frozenset({"success", "failed"})
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SCORE_METRICS = ("AUC", "PR_AUC", "F1")


def _optional_sha256(value: Any, field: str) -> str | None:
    if value in (None, ""):
        return None
    normalized = str(value).strip().lower()
    if not _SHA256.fullmatch(normalized):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return normalized


def _unit_score(value: Any, field: str) -> float:
    score = float(value)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(f"{field} must be finite and within [0, 1]")
    return score


def atomic_candidate_config(candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Extract only fields that define the executable candidate semantics."""
    required = ("candidate_id", "feature_type", "classification_method")
    missing = [name for name in required if not candidate.get(name)]
    if missing:
        raise ValueError(f"candidate is missing atomic fields: {missing}")
    model_recipe = candidate.get("model_recipe") or {
        "LASSO": "L1_LOGISTIC",
        "FILTER": "ANOVA_TOPK_L2",
    }.get(str(candidate.get("classification_method") or "").strip().upper())
    if not model_recipe:
        raise ValueError("candidate model_recipe is required")
    recipe_config = candidate.get("recipe_config") or {}
    if not isinstance(recipe_config, Mapping):
        raise ValueError("candidate recipe_config must be an object")
    return sanitize_for_audit(
        {
            "candidate_id": str(candidate["candidate_id"]),
            "feature_type": str(candidate["feature_type"]),
            "classification_method": str(candidate["classification_method"]),
            "model_recipe": str(model_recipe),
            "recipe_config": dict(recipe_config),
            "registry_version": candidate.get("registry_version"),
        }
    )


def build_pipeline_trial_record(
    *,
    trace_id: str,
    candidate: Mapping[str, Any],
    result: Mapping[str, Any],
    task_id: int | None = None,
    task_fingerprint: str | None = None,
    dataset_hash: str | None = None,
    policy_id: str | None = None,
    duration_ms: int | None = None,
) -> dict[str, Any]:
    """Build one immutable record for a candidate attempted in a paired search."""
    if not str(trace_id or "").strip():
        raise ValueError("trace_id is required")
    config = atomic_candidate_config(candidate)
    config_hash = stable_json_hash(config)
    raw_status = str(result.get("status") or "failed").strip().lower()
    scope = str(
        result.get("validation_scope") or "training_only_nested_grouped_cv"
    )
    test_data_used = bool(result.get("test_data_used"))
    metrics = dict(result.get("metrics") or {})
    recipe_matches = str(result.get("model_recipe") or "") == config["model_recipe"]
    safe_success = (
        raw_status == "success"
        and scope == "training_only_nested_grouped_cv"
        and not test_data_used
        and recipe_matches
    )
    validated_metrics: dict[str, float] = {}
    if safe_success:
        try:
            validated_metrics = {
                name: _unit_score(metrics[name], f"metrics.{name}")
                for name in _SCORE_METRICS
            }
        except (KeyError, TypeError, ValueError):
            safe_success = False
            validated_metrics = {}
    status = "success" if safe_success else "failed"

    train_sha256 = _optional_sha256(result.get("train_sha256"), "train_sha256")
    normalized_dataset_hash = _optional_sha256(dataset_hash, "dataset_hash")
    normalized_fingerprint = _optional_sha256(
        task_fingerprint, "task_fingerprint"
    )
    intervals = dict(result.get("metric_intervals") or {})
    stability = dict(result.get("selection_stability") or {})
    auc_interval = intervals.get("AUC", {})
    auc_lower = None
    auc_upper = None
    stability_score = None
    if status == "success":
        try:
            auc_lower = _unit_score(auc_interval.get("lower"), "AUC CI lower")
            auc_upper = _unit_score(auc_interval.get("upper"), "AUC CI upper")
            if auc_lower > auc_upper:
                raise ValueError("AUC confidence interval is reversed")
        except (AttributeError, TypeError, ValueError):
            auc_lower = validated_metrics["AUC"]
            auc_upper = validated_metrics["AUC"]
        try:
            stability_score = _unit_score(
                stability.get("mean_pairwise_jaccard"),
                "selection stability",
            )
        except (TypeError, ValueError):
            stability_score = None

    identity = stable_json_hash(
        {
            "trace_id": str(trace_id),
            "candidate_config_sha256": config_hash,
            "train_sha256": train_sha256,
            "evaluator_version": result.get("evaluator_version"),
        }
    )
    folds = []
    for fold in result.get("folds", []) or []:
        if not isinstance(fold, Mapping):
            continue
        folds.append(
            {
                "fold": fold.get("fold"),
                "train_groups": fold.get("train_groups"),
                "validation_groups": fold.get("validation_groups"),
                "best_params": fold.get("best_params", {}),
                "inner_best_auc": fold.get("inner_best_auc"),
                "selected_feature_count": fold.get("selected_feature_count"),
                "validation_metrics": fold.get("validation_metrics", {}),
            }
        )
    failure_reason = result.get("error")
    if status == "failed" and not failure_reason:
        if test_data_used:
            failure_reason = "test_data_used_in_candidate_validation"
        elif not recipe_matches:
            failure_reason = "model_recipe_mismatch"
        else:
            failure_reason = "invalid_or_incomplete_candidate_evidence"

    record = {
        "trial_id": str(uuid.uuid5(uuid.NAMESPACE_URL, identity)),
        "schema_version": PIPELINE_TRIAL_SCHEMA_VERSION,
        "trace_id": str(trace_id),
        "task_id": task_id,
        "task_fingerprint": normalized_fingerprint,
        "dataset_hash": normalized_dataset_hash,
        "candidate_id": config["candidate_id"],
        "feature_type": config["feature_type"],
        "classification_method": config["classification_method"],
        "model_recipe": config["model_recipe"],
        "candidate_config_sha256": config_hash,
        "policy_id": policy_id,
        "status": status,
        "validation_scope": scope,
        "test_data_used": test_data_used,
        "evaluator_version": result.get("evaluator_version"),
        "train_sha256": train_sha256,
        "patient_count": result.get("patient_count"),
        "outer_folds": result.get("outer_folds"),
        "validation_score": (
            _unit_score(candidate.get("validation_score"), "validation_score")
            if status == "success" and candidate.get("validation_score") is not None
            else None
        ),
        "auc": validated_metrics.get("AUC"),
        "pr_auc": validated_metrics.get("PR_AUC"),
        "f1": validated_metrics.get("F1"),
        "auc_ci_lower": auc_lower,
        "auc_ci_upper": auc_upper,
        "selection_stability": stability_score,
        "duration_ms": (
            max(0, int(duration_ms)) if duration_ms is not None else None
        ),
        "recommended_params": dict(result.get("recommended_params") or {}),
        "metrics": metrics if status == "success" else {},
        "metric_intervals": intervals if status == "success" else {},
        "details": {
            "candidate_config": config,
            "folds": folds,
            "feature_count": result.get("feature_count"),
            "input_rows": result.get("input_rows"),
            "labeled_rows": result.get("labeled_rows"),
            "excluded_unlabeled_rows": result.get("excluded_unlabeled_rows"),
            "selection_stability": stability if status == "success" else {},
            "full_training_selection": (
                result.get("full_training_selection", {})
                if status == "success"
                else {}
            ),
            "failure_reason": failure_reason,
        },
    }
    return sanitize_for_audit(record)
