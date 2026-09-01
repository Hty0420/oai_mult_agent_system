"""Deterministic patient-level metrics and classification run artifacts."""

from __future__ import annotations

import hashlib
import json
import numbers
import os
import sys
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .leakage_safe_l1_logistic import LeakageSafeL1Logistic
from .patient_feature_table import PatientFeatureTable


METRIC_KEYS = (
    "NumPatients",
    "AUC",
    "PR_AUC",
    "Accuracy",
    "Precision",
    "Sensitivity",
    "Specificity",
    "F1",
    "TN",
    "FP",
    "FN",
    "TP",
)
COUNT_METRICS = frozenset(("NumPatients", "TN", "FP", "FN", "TP"))


def compute_binary_metrics(aggregated: pd.DataFrame) -> dict:
    """Compute fixed-threshold metrics from one row per source patient."""
    if not isinstance(aggregated, pd.DataFrame):
        raise TypeError("Aggregated patient predictions must be a pandas DataFrame.")
    required = ("group_id", "label", "probability")
    missing = [column for column in required if column not in aggregated]
    if missing:
        raise ValueError(f"Missing required patient metric column: {missing[0]}")
    if aggregated.empty:
        raise ValueError("Patient-level metrics require a non-empty frame.")

    groups = aggregated["group_id"]
    if groups.isna().any():
        raise ValueError("group_id values must be nonmissing.")
    if groups.duplicated().any():
        raise ValueError("Patient-level metrics reject duplicate group_id values.")

    labels = aggregated["label"]
    if not pd.api.types.is_integer_dtype(labels.dtype) or pd.api.types.is_bool_dtype(
        labels.dtype
    ):
        raise ValueError("Patient labels must use an integer dtype.")
    if labels.isna().any() or not labels.isin((0, 1)).all():
        raise ValueError("Patient labels must be binary integers 0 or 1.")
    if labels.nunique() != 2:
        raise ValueError("Patient-level AUC metrics require both classes.")

    try:
        probabilities = pd.to_numeric(aggregated["probability"], errors="raise").to_numpy(
            dtype=float, na_value=np.nan
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Patient probabilities must be numeric.") from exc
    if not np.isfinite(probabilities).all():
        raise ValueError("Patient probabilities must be finite.")
    if ((probabilities < 0.0) | (probabilities > 1.0)).any():
        raise ValueError("Patient probabilities must be between 0 and 1.")

    y_true = labels.to_numpy(dtype=int)
    y_pred = (probabilities >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp)
    return {
        "NumPatients": int(len(aggregated)),
        "AUC": float(roc_auc_score(y_true, probabilities)),
        "PR_AUC": float(average_precision_score(y_true, probabilities)),
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "Specificity": float(specificity),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("Artifact JSON values must be finite.")
        return value
    raise TypeError(f"Artifact value has unsupported JSON type: {type(value).__name__}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot_sha256(value: str | None, path: Path) -> str:
    if value is None:
        return _sha256(path)
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("Input snapshot SHA-256 must be a lowercase hexadecimal digest.")
    return value


def _validated_metrics(metrics: dict) -> dict[str, int | float]:
    if not isinstance(metrics, dict) or set(metrics) != set(METRIC_KEYS):
        raise ValueError("metrics must contain exactly the fixed patient-level metric keys.")
    validated: dict[str, int | float] = {}
    for key in METRIC_KEYS:
        value = metrics[key]
        if key in COUNT_METRICS:
            if isinstance(value, bool) or not isinstance(value, numbers.Integral):
                raise TypeError(f"Metric '{key}' must be a nonnegative integer.")
            plain_count = int(value)
            if plain_count < 0:
                raise ValueError(f"Metric '{key}' must be a nonnegative integer.")
            validated[key] = plain_count
            continue
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise TypeError(f"Metric '{key}' must be a finite number from 0 to 1.")
        plain_score = float(value)
        if not np.isfinite(plain_score) or not 0.0 <= plain_score <= 1.0:
            raise ValueError(f"Metric '{key}' must be a finite number from 0 to 1.")
        validated[key] = plain_score
    return validated


def _canonical_key(path: Path) -> str:
    return os.path.normcase(os.path.realpath(os.fspath(path)))


def _temporary_path(destination: Path) -> Path:
    return destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")


def _atomic_json(destination: Path, payload: Any) -> None:
    temporary = _temporary_path(destination)
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(
                _json_safe(payload),
                stream,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_csv(destination: Path, frame: pd.DataFrame) -> None:
    temporary = _temporary_path(destination)
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            frame.to_csv(stream, index=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _validate_input_file(value: str | Path, role: str) -> Path:
    path = Path(value)
    if not path.is_file():
        raise ValueError(f"The supplied {role} input path must be an existing regular file.")
    return path.resolve()


def _table_counts(table: PatientFeatureTable, role: str) -> dict[str, Any]:
    if not isinstance(table, PatientFeatureTable):
        raise TypeError(f"{role} table must be a PatientFeatureTable.")
    if (
        table.group_column not in table.frame
        or "CaseID" not in table.frame
        or "label" not in table.frame
    ):
        raise ValueError(f"{role} table is missing its patient, group, or label column.")
    counts = table.frame["label"].value_counts().sort_index()
    return {
        "rows": int(len(table.frame)),
        "unique_patients": int(table.frame["CaseID"].nunique()),
        "unique_groups": int(table.frame[table.group_column].nunique()),
        "class_counts": {str(key): int(value) for key, value in counts.items()},
    }


def _validate_suffix(suffix: str) -> None:
    if not isinstance(suffix, str):
        raise TypeError("Artifact suffix must be a string.")
    if (
        ".." in suffix
        or "/" in suffix
        or "\\" in suffix
        or ":" in suffix
        or "\x00" in suffix
    ):
        raise ValueError("Artifact suffix must not contain traversal or path separators.")


def write_classification_artifacts(
    *,
    output_dir: str | Path,
    suffix: str,
    train_path: str | Path,
    test_path: str | Path,
    train_table: PatientFeatureTable,
    test_table: PatientFeatureTable,
    model: LeakageSafeL1Logistic,
    metrics: dict,
    train_sha256: str | None = None,
    test_sha256: str | None = None,
) -> dict[str, str]:
    """Publish schema, nested-CV summary, coefficients, and manifest atomically."""
    _validate_suffix(suffix)
    train_input = _validate_input_file(train_path, "train")
    test_input = _validate_input_file(test_path, "test")
    if not isinstance(model, LeakageSafeL1Logistic) or model.final_pipeline is None:
        raise TypeError("model must be a fitted LeakageSafeL1Logistic.")
    if model.nested_cv_summary is None:
        raise ValueError("The fitted model is missing its nested CV summary.")
    validated_metrics = _validated_metrics(metrics)

    directory = Path(output_dir)
    if directory.exists() and not directory.is_dir():
        raise ValueError("output_dir exists but is not a directory.")
    directory = directory.resolve()

    basenames = {
        "schema": f"patient_feature_schema{suffix}.json",
        "nested_cv": f"nested_cv_summary{suffix}.json",
        "coefficients": f"model_coefficients{suffix}.csv",
        "manifest": f"classification_run_manifest{suffix}.json",
    }
    destinations = {
        key: (directory / name).resolve() for key, name in basenames.items()
    }
    destination_keys = {_canonical_key(path) for path in destinations.values()}
    if len(destination_keys) != len(destinations):
        raise ValueError("Artifact output filename collision detected.")
    input_keys = {_canonical_key(train_input), _canonical_key(test_input)}
    if destination_keys & input_keys:
        raise ValueError("Artifact output must not alias a supplied input file.")
    directory.mkdir(parents=True, exist_ok=True)

    train_counts = _table_counts(train_table, "train")
    test_counts = _table_counts(test_table, "test")
    coefficient_values = model.coefficients
    coefficient_frame = pd.DataFrame(
        {
            "feature": list(model.feature_columns),
            "coefficient": [coefficient_values[name] for name in model.feature_columns],
            "selected": [name in model.selected_features for name in model.feature_columns],
        }
    )
    if not np.isfinite(coefficient_frame["coefficient"].to_numpy(dtype=float)).all():
        raise ValueError("All model coefficients must be finite.")

    schema = {
        "feature_columns": list(model.feature_columns),
        "label_rule": "KLGrade >= 2",
        "region_pivot_rule": "Region rows are pivoted to deterministic region-major feature columns.",
        "group_column": train_table.group_column,
        "source_kind": train_table.source_kind,
        "ignored_extra_test_features": list(model.extra_test_features),
    }
    manifest = {
        "inputs": {
            "train": {
                "path": str(train_input),
                "sha256": _snapshot_sha256(train_sha256, train_input),
            },
            "test": {
                "path": str(test_input),
                "sha256": _snapshot_sha256(test_sha256, test_input),
            },
        },
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "random_seed": 42,
        "counts": {"train": train_counts, "test": test_counts},
        "final_c": model.nested_cv_summary["final_best_c"],
        "selected_feature_count": len(model.selected_features),
        "metrics": validated_metrics,
        "artifacts": basenames,
    }

    _atomic_json(destinations["schema"], schema)
    _atomic_json(destinations["nested_cv"], model.nested_cv_summary)
    _atomic_csv(destinations["coefficients"], coefficient_frame)
    _atomic_json(destinations["manifest"], manifest)
    return {key: str(path) for key, path in destinations.items()}
