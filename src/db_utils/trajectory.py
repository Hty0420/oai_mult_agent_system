"""Deterministic trajectory and evaluation records for the multi-agent system.

This module deliberately has no database or LLM dependency.  It defines the
portable records written by the control/logging layer and keeps raw secrets,
patient identifiers, and filesystem paths out of generic event payloads.
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
import uuid


TRAJECTORY_SCHEMA_VERSION = "1.1"
DEFAULT_EVALUATOR_VERSION = "classification-manifest-v2"
TRAJECTORY_DATA_PARTITIONS = frozenset({"train", "validation", "test"})

EVENT_TYPES = frozenset(
    {
        "run_started",
        "route_decided",
        "plan_created",
        "agent_completed",
        "tool_completed",
        "step_completed",
        "artifact_registered",
        "evaluation_recorded",
        "evaluation_ingestion_completed",
        "run_completed",
        "feedback_recorded",
        "human_feedback_recorded",
        "policy_changed",
        "policy_candidate_generated",
        "pipeline_candidates_generated",
        "pipeline_trial_completed",
        "pipeline_selected",
    }
)

EVENT_STATUSES = frozenset({"pending", "running", "success", "failed", "skipped", "unknown"})
EVALUATION_SPLITS = frozenset(
    {"train", "validation", "cross_validation", "test", "external_test", "unknown"}
)

CLASSIFICATION_METRICS = (
    "AUC",
    "PR_AUC",
    "Accuracy",
    "Precision",
    "Sensitivity",
    "Specificity",
    "F1",
)

_SECRET_KEY = re.compile(
    r"(?:password|passwd|secret|token|api[_-]?key|authorization|cookie|credential)",
    re.IGNORECASE,
)
_PATIENT_KEY = re.compile(
    r"(?:patient|subject|case|mrn|accession|dicom).*?(?:id|uid|name)?$",
    re.IGNORECASE,
)
_PATH_KEY = re.compile(r"(?:path|file|directory|dir|uri)$", re.IGNORECASE)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def current_policy_id() -> str:
    """Return the explicitly configured policy identifier or a stable baseline."""
    value = os.getenv("MULTI_AGENT_POLICY_ID", "baseline-v1").strip()
    return value or "baseline-v1"


def current_agent_version() -> str:
    """Return the deploy/version label attached to new task trajectories."""
    value = os.getenv("MULTI_AGENT_VERSION", "multi-agent-v1").strip()
    return value or "multi-agent-v1"


def personal_workspace_id(user_id: Any) -> str:
    """Return the default logical workspace for a user without another account."""
    if user_id is None or not str(user_id).strip():
        raise ValueError("user_id is required to derive a personal workspace")
    return f"personal-{stable_json_hash(str(user_id))[:16]}"


def assign_trajectory_partition(
    identity: str,
    *,
    train_percent: int = 80,
    validation_percent: int = 10,
) -> str:
    """Assign a stable, leakage-resistant learning partition.

    Callers should pass a task fingerprint rather than a run ID. Repeated runs
    of the same task context will then remain in the same partition.
    """
    if not str(identity or "").strip():
        raise ValueError("identity is required for trajectory partitioning")
    if (
        isinstance(train_percent, bool)
        or isinstance(validation_percent, bool)
        or train_percent < 1
        or validation_percent < 1
        or train_percent + validation_percent >= 100
    ):
        raise ValueError("partition percentages must leave non-empty train/validation/test sets")
    bucket = int(stable_json_hash(str(identity))[:8], 16) % 100
    if bucket < train_percent:
        return "train"
    if bucket < train_percent + validation_percent:
        return "validation"
    return "test"


def stable_json_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint_text(value: str) -> str:
    return f"sha256:{hashlib.sha256(value.encode('utf-8')).hexdigest()}"


def sanitize_for_audit(
    value: Any,
    *,
    key: str | None = None,
    max_depth: int = 8,
    max_items: int = 100,
    max_string: int = 2000,
) -> Any:
    """Return a bounded JSON-compatible audit value.

    Secret-like and patient-identifier keys are redacted.  Path-like values are
    fingerprinted so trajectories remain comparable without duplicating local
    paths or identifiers embedded in filenames.
    """
    if max_depth < 0:
        return "[TRUNCATED_DEPTH]"

    if key and (_SECRET_KEY.search(key) or _PATIENT_KEY.search(key)):
        return "[REDACTED]"

    if value is None or isinstance(value, (bool, int)):
        return value

    if isinstance(value, float):
        return value if math.isfinite(value) else "[NON_FINITE]"

    if isinstance(value, (str, os.PathLike)):
        text = os.fspath(value)
        if key and _PATH_KEY.search(key):
            return _fingerprint_text(text)
        return text if len(text) <= max_string else text[:max_string] + "[TRUNCATED]"

    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for index, (raw_key, item) in enumerate(value.items()):
            if index >= max_items:
                result["__truncated_items__"] = len(value) - max_items
                break
            item_key = str(raw_key)
            result[item_key] = sanitize_for_audit(
                item,
                key=item_key,
                max_depth=max_depth - 1,
                max_items=max_items,
                max_string=max_string,
            )
        return result

    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        items = list(value)
        result = [
            sanitize_for_audit(
                item,
                max_depth=max_depth - 1,
                max_items=max_items,
                max_string=max_string,
            )
            for item in items[:max_items]
        ]
        if len(items) > max_items:
            result.append({"__truncated_items__": len(items) - max_items})
        return result

    return sanitize_for_audit(
        str(value),
        key=key,
        max_depth=max_depth - 1,
        max_items=max_items,
        max_string=max_string,
    )


def build_trajectory_event(
    *,
    event_type: str,
    trace_id: str,
    actor_type: str,
    actor_name: str,
    payload: Mapping[str, Any] | None = None,
    status: str = "success",
    task_id: int | None = None,
    step_id: str | None = None,
    parent_event_id: str | None = None,
    policy_id: str | None = None,
    duration_ms: int | None = None,
    event_id: str | None = None,
    user_id: int | None = None,
    workspace_id: str | None = None,
    agent_version: str | None = None,
    data_partition: str | None = None,
) -> dict[str, Any]:
    if event_type not in EVENT_TYPES:
        raise ValueError(f"Unsupported trajectory event_type: {event_type}")
    if status not in EVENT_STATUSES:
        raise ValueError(f"Unsupported trajectory status: {status}")
    if not str(trace_id or "").strip():
        raise ValueError("trace_id is required")
    if duration_ms is not None and (isinstance(duration_ms, bool) or duration_ms < 0):
        raise ValueError("duration_ms must be a non-negative integer")
    if data_partition is not None and data_partition not in TRAJECTORY_DATA_PARTITIONS:
        raise ValueError(f"Unsupported trajectory data_partition: {data_partition}")

    return {
        "event_id": event_id or str(uuid.uuid4()),
        "schema_version": TRAJECTORY_SCHEMA_VERSION,
        "trace_id": str(trace_id),
        "task_id": task_id,
        "step_id": step_id,
        "parent_event_id": parent_event_id,
        "event_type": event_type,
        "actor_type": str(actor_type),
        "actor_name": str(actor_name),
        "policy_id": policy_id or current_policy_id(),
        "user_id": user_id,
        "workspace_id": workspace_id,
        "agent_version": agent_version or current_agent_version(),
        "data_partition": data_partition,
        "status": status,
        "duration_ms": duration_ms,
        "payload": sanitize_for_audit(dict(payload or {})),
        "occurred_at": utc_now_iso(),
    }


def build_artifact_record(
    *,
    path: str | os.PathLike[str],
    artifact_type: str,
    created_by: str,
    trace_id: str | None = None,
    task_id: int | None = None,
    source_artifact_ids: Sequence[str] = (),
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise ValueError("artifact path must be an existing regular file")
    digest = file_sha256(resolved)
    artifact_id = f"ART-{digest[:32]}"
    return {
        "artifact_id": artifact_id,
        "trace_id": trace_id,
        "task_id": task_id,
        "artifact_type": str(artifact_type),
        "uri": str(resolved),
        "sha256": digest,
        "created_by": str(created_by),
        "source_artifact_ids": list(source_artifact_ids),
        "metadata": sanitize_for_audit(dict(metadata or {})),
        "validation_status": "unverified",
        "created_at": utc_now_iso(),
    }


def infer_classification_task_type(row: Mapping[str, Any]) -> str:
    task_type = str(row.get("task_type") or "").strip()
    if task_type:
        return task_type
    class_count = row.get("class_count")
    if isinstance(class_count, int):
        if class_count == 2:
            return "binary_kl"
        if class_count > 2:
            return "ordinal_kl"
    return "classification_unknown"


def build_experiment_evaluations(
    row: Mapping[str, Any],
    *,
    artifact_id: str | None = None,
    evaluator_version: str = DEFAULT_EVALUATOR_VERSION,
) -> list[dict[str, Any]]:
    """Convert one experiment row into versioned metric and gate records."""
    split_type = str(row.get("evaluation_split") or "external_test")
    if split_type not in EVALUATION_SPLITS:
        raise ValueError(f"Unsupported evaluation split: {split_type}")

    common = {
        "trace_id": row.get("trace_id"),
        "task_id": row.get("task_id"),
        "artifact_id": artifact_id,
        "policy_id": row.get("policy_id") or current_policy_id(),
        "task_type": infer_classification_task_type(row),
        "method_name": row.get("method_label") or row.get("classification_method") or "unknown",
        "split_type": split_type,
        "dataset_hash": row.get("test_sha256"),
        "evaluator_name": "classification_manifest",
        "evaluator_version": evaluator_version,
    }

    def evaluation_id(record_type: str, metric_name: str) -> str:
        identity = stable_json_hash(
            {
                "artifact_id": common["artifact_id"],
                "policy_id": common["policy_id"],
                "task_type": common["task_type"],
                "method_name": common["method_name"],
                "split_type": common["split_type"],
                "dataset_hash": common["dataset_hash"],
                "evaluator_name": common["evaluator_name"],
                "evaluator_version": common["evaluator_version"],
                "record_type": record_type,
                "metric_name": metric_name,
            }
        )
        return str(uuid.uuid5(uuid.NAMESPACE_URL, identity))

    records: list[dict[str, Any]] = []
    for metric_name in CLASSIFICATION_METRICS:
        metric_value = row.get(metric_name)
        if metric_value is None:
            continue
        if isinstance(metric_value, bool) or not isinstance(metric_value, (int, float)):
            raise TypeError(f"{metric_name} must be numeric")
        if not math.isfinite(float(metric_value)):
            raise ValueError(f"{metric_name} must be finite")
        records.append(
            {
                "evaluation_id": evaluation_id("metric", metric_name),
                **common,
                "record_type": "metric",
                "metric_name": metric_name,
                "metric_value": float(metric_value),
                "gate_status": None,
                "details": {},
            }
        )

    provenance_complete = bool(row.get("train_sha256") and row.get("test_sha256"))
    grouped_evaluation = bool(
        row.get("test_unique_patients")
        and row.get("NumPatients")
        and int(row["test_unique_patients"]) == int(row["NumPatients"])
    )
    for gate_name, passed, details in (
        (
            "provenance_complete",
            provenance_complete,
            {"requires": ["train_sha256", "test_sha256"]},
        ),
        (
            "patient_level_evaluation",
            grouped_evaluation,
            {
                "test_unique_patients": row.get("test_unique_patients"),
                "evaluated_patients": row.get("NumPatients"),
            },
        ),
        (
            "test_is_frozen",
            False,
            {"reason": "No frozen evaluation-set registry is linked to this historical run."},
        ),
    ):
        records.append(
            {
                "evaluation_id": evaluation_id("gate", gate_name),
                **common,
                "record_type": "gate",
                "metric_name": gate_name,
                "metric_value": 1.0 if passed else 0.0,
                "gate_status": "pass" if passed else "unknown" if gate_name == "test_is_frozen" else "fail",
                "details": sanitize_for_audit(details),
            }
        )
    scientific_quality = row.get("scientific_quality")
    if isinstance(scientific_quality, Mapping):
        for gate_name, gate in dict(
            scientific_quality.get("gates", {})
        ).items():
            if not isinstance(gate, Mapping):
                continue
            gate_status = str(gate.get("status") or "unknown")
            if gate_status not in {"pass", "fail", "unknown"}:
                gate_status = "unknown"
            records.append(
                {
                    "evaluation_id": evaluation_id("gate", str(gate_name)),
                    **common,
                    "record_type": "gate",
                    "metric_name": str(gate_name),
                    "metric_value": 1.0 if gate_status == "pass" else 0.0,
                    "gate_status": gate_status,
                    "details": sanitize_for_audit(gate.get("details", {})),
                }
            )
    return records


def build_policy_record(
    *,
    policy_id: str,
    policy_type: str,
    version: str,
    config: Mapping[str, Any],
    status: str = "draft",
    parent_policy_id: str | None = None,
    created_by: str = "system",
) -> dict[str, Any]:
    if status not in {"draft", "shadow", "active", "retired", "rolled_back"}:
        raise ValueError("invalid policy status")
    if not policy_id.strip() or not version.strip():
        raise ValueError("policy_id and version are required")
    sanitized = sanitize_for_audit(dict(config))
    return {
        "policy_id": policy_id,
        "policy_type": policy_type,
        "version": version,
        "status": status,
        "config": sanitized,
        "config_sha256": stable_json_hash(sanitized),
        "parent_policy_id": parent_policy_id,
        "created_by": created_by,
        "created_at": utc_now_iso(),
    }
