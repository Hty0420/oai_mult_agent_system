"""Deterministic scientific-quality checks for classification manifests."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping

from .trajectory import sanitize_for_audit


SCIENTIFIC_QUALITY_EVALUATOR = "classification-scientific-quality"
SCIENTIFIC_QUALITY_VERSION = "1.0.0"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_REQUIRED_ARTIFACTS = frozenset(
    {"schema", "nested_cv", "coefficients", "manifest"}
)
_SCORE_METRICS = (
    "AUC",
    "PR_AUC",
    "Accuracy",
    "Precision",
    "Sensitivity",
    "Specificity",
    "F1",
)
_COUNT_METRICS = ("NumPatients", "TN", "FP", "FN", "TP")
_QUALITY_GATE_NAMES = (
    "manifest_schema_complete",
    "metric_domain_valid",
    "confusion_metric_consistency",
    "patient_count_consistency",
    "artifact_files_complete",
    "input_hashes_verified",
    "train_test_inputs_distinct",
    "environment_provenance_complete",
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _gate(status: str, **details: Any) -> dict[str, Any]:
    if status not in {"pass", "fail", "unknown"}:
        raise ValueError("scientific quality gate status is invalid")
    return {"status": status, "details": sanitize_for_audit(details)}


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return numerator / denominator if denominator else 0.0


def _close(observed: Any, expected: float, tolerance: float = 1e-9) -> bool:
    return (
        isinstance(observed, (int, float))
        and not isinstance(observed, bool)
        and math.isfinite(float(observed))
        and math.isclose(
            float(observed), expected, rel_tol=tolerance, abs_tol=tolerance
        )
    )


def _result(
    *, manifest_sha256: str | None, gates: Mapping[str, Any]
) -> dict[str, Any]:
    supplied = dict(gates)
    gate_map = {
        name: supplied.get(
            name,
            _gate("unknown", reason="upstream_manifest_evidence_unavailable"),
        )
        for name in _QUALITY_GATE_NAMES
    }
    return {
        "schema_version": "1.0",
        "evaluator_name": SCIENTIFIC_QUALITY_EVALUATOR,
        "evaluator_version": SCIENTIFIC_QUALITY_VERSION,
        "manifest_sha256": manifest_sha256,
        "gates": gate_map,
        "hard_gates_pass": bool(gate_map)
        and all(gate["status"] == "pass" for gate in gate_map.values()),
    }


def evaluate_classification_manifest(
    manifest_path: str | Path,
) -> dict[str, Any]:
    """Verify a classification manifest and its locally available lineage.

    Missing historical inputs produce ``unknown`` hash verification rather than
    a fabricated pass. Digest mismatch, unsafe artifact path, inconsistent
    patient counts, or irreconcilable metrics are hard failures.
    """
    path = Path(manifest_path).resolve()
    gates: dict[str, dict[str, Any]] = {}
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        gates["manifest_schema_complete"] = _gate(
            "fail", error=type(exc).__name__
        )
        return _result(manifest_sha256=None, gates=gates)

    manifest_sha256 = _file_sha256(path)
    required_root = {
        "inputs",
        "versions",
        "random_seed",
        "counts",
        "metrics",
        "artifacts",
        "selected_feature_count",
        "final_c",
    }
    schema_complete = isinstance(manifest, Mapping) and required_root.issubset(
        manifest
    )
    inputs = manifest.get("inputs") if isinstance(manifest, Mapping) else None
    counts = manifest.get("counts") if isinstance(manifest, Mapping) else None
    metrics = manifest.get("metrics") if isinstance(manifest, Mapping) else None
    artifacts = manifest.get("artifacts") if isinstance(manifest, Mapping) else None
    versions = manifest.get("versions") if isinstance(manifest, Mapping) else None
    schema_complete = bool(
        schema_complete
        and isinstance(inputs, Mapping)
        and isinstance(counts, Mapping)
        and isinstance(metrics, Mapping)
        and isinstance(artifacts, Mapping)
        and isinstance(versions, Mapping)
    )
    gates["manifest_schema_complete"] = _gate(
        "pass" if schema_complete else "fail",
        required_root=sorted(required_root),
    )
    if not schema_complete:
        return _result(manifest_sha256=manifest_sha256, gates=gates)

    metric_domain_valid = set(_SCORE_METRICS + _COUNT_METRICS).issubset(metrics)
    if metric_domain_valid:
        metric_domain_valid = all(
            isinstance(metrics[name], (int, float))
            and not isinstance(metrics[name], bool)
            and math.isfinite(float(metrics[name]))
            and 0.0 <= float(metrics[name]) <= 1.0
            for name in _SCORE_METRICS
        ) and all(
            isinstance(metrics[name], int)
            and not isinstance(metrics[name], bool)
            and metrics[name] >= 0
            for name in _COUNT_METRICS
        )
    gates["metric_domain_valid"] = _gate(
        "pass" if metric_domain_valid else "fail",
        required_scores=list(_SCORE_METRICS),
        required_counts=list(_COUNT_METRICS),
    )

    confusion_consistent = False
    if metric_domain_valid:
        n = int(metrics["NumPatients"])
        tn, fp, fn, tp = (
            int(metrics["TN"]),
            int(metrics["FP"]),
            int(metrics["FN"]),
            int(metrics["TP"]),
        )
        precision = _safe_ratio(tp, tp + fp)
        sensitivity = _safe_ratio(tp, tp + fn)
        specificity = _safe_ratio(tn, tn + fp)
        f1 = _safe_ratio(
            2 * precision * sensitivity, precision + sensitivity
        )
        confusion_consistent = bool(
            n > 0
            and tn + fp + fn + tp == n
            and _close(metrics["Accuracy"], _safe_ratio(tp + tn, n))
            and _close(metrics["Precision"], precision)
            and _close(metrics["Sensitivity"], sensitivity)
            and _close(metrics["Specificity"], specificity)
            and _close(metrics["F1"], f1)
        )
    gates["confusion_metric_consistency"] = _gate(
        "pass" if confusion_consistent else "fail",
        evaluated_patients=metrics.get("NumPatients"),
        confusion_total=sum(
            int(metrics.get(name) or 0)
            for name in ("TN", "FP", "FN", "TP")
        ),
    )

    test_counts = counts.get("test") if isinstance(counts, Mapping) else None
    patient_count_consistent = False
    if isinstance(test_counts, Mapping) and metric_domain_valid:
        unique_patients = test_counts.get("unique_patients")
        class_counts = test_counts.get("class_counts")
        patient_count_consistent = bool(
            isinstance(unique_patients, int)
            and unique_patients == metrics["NumPatients"]
            and isinstance(class_counts, Mapping)
            and sum(int(value) for value in class_counts.values())
            == metrics["NumPatients"]
        )
    gates["patient_count_consistency"] = _gate(
        "pass" if patient_count_consistent else "fail",
        manifest_unique_patients=(
            test_counts.get("unique_patients")
            if isinstance(test_counts, Mapping)
            else None
        ),
        evaluated_patients=metrics.get("NumPatients"),
    )

    artifact_failures: list[str] = []
    if not _REQUIRED_ARTIFACTS.issubset(artifacts):
        artifact_failures.append("required_artifact_key_missing")
    for key in _REQUIRED_ARTIFACTS.intersection(artifacts):
        filename = artifacts[key]
        if (
            not isinstance(filename, str)
            or not filename
            or Path(filename).name != filename
        ):
            artifact_failures.append(f"{key}:unsafe_filename")
            continue
        artifact_path = (path.parent / filename).resolve()
        if artifact_path.parent != path.parent:
            artifact_failures.append(f"{key}:outside_run_directory")
        elif not artifact_path.is_file():
            artifact_failures.append(f"{key}:missing")
    gates["artifact_files_complete"] = _gate(
        "pass" if not artifact_failures else "fail",
        failures=artifact_failures,
    )

    input_failures: list[str] = []
    input_missing: list[str] = []
    input_hashes: dict[str, str] = {}
    for role in ("train", "test"):
        info = inputs.get(role)
        if not isinstance(info, Mapping):
            input_failures.append(f"{role}:metadata_missing")
            continue
        digest = info.get("sha256")
        source_path = info.get("path")
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            input_failures.append(f"{role}:invalid_sha256")
            continue
        input_hashes[role] = digest
        if not isinstance(source_path, str) or not source_path:
            input_failures.append(f"{role}:path_missing")
            continue
        resolved_source = Path(source_path)
        if not resolved_source.is_file():
            input_missing.append(role)
        elif _file_sha256(resolved_source) != digest:
            input_failures.append(f"{role}:sha256_mismatch")
    input_hash_status = (
        "fail" if input_failures else "unknown" if input_missing else "pass"
    )
    gates["input_hashes_verified"] = _gate(
        input_hash_status,
        failures=input_failures,
        unavailable_roles=input_missing,
    )
    distinct_inputs = bool(
        set(input_hashes) == {"train", "test"}
        and input_hashes["train"] != input_hashes["test"]
    )
    gates["train_test_inputs_distinct"] = _gate(
        "pass" if distinct_inputs else "fail"
    )

    version_keys = {"python", "numpy", "pandas", "scikit_learn"}
    environment_complete = version_keys.issubset(versions) and all(
        isinstance(versions[key], str) and versions[key].strip()
        for key in version_keys
    )
    gates["environment_provenance_complete"] = _gate(
        "pass" if environment_complete else "fail",
        required_versions=sorted(version_keys),
    )
    return _result(manifest_sha256=manifest_sha256, gates=gates)
