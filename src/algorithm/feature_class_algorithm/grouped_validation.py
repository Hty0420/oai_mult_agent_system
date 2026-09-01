"""Patient-grouped validation helpers for binary classification."""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def _as_vector(values, name, *, non_empty=True):
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional input")
    if non_empty and len(array) == 0:
        raise ValueError(f"{name} must not be empty")
    return array


def _normalized_groups(groups, *, non_empty=True):
    values = _as_vector(groups, "groups", non_empty=non_empty)
    if pd.isna(values).any():
        raise ValueError("groups must not contain missing values")
    normalized = np.asarray([str(value) for value in values], dtype=object)
    if any(not group for group in normalized):
        raise ValueError("groups must not contain empty identifiers")
    return normalized


def _format_safe_group_ids(group_ids):
    """Return at most ten identifiers, redacting path-like values."""
    formatted = []
    for group_id in group_ids[:10]:
        lower_id = group_id.lower()
        is_path_like = (
            "/" in group_id
            or "\\" in group_id
            or ("image" in lower_id or "mask" in lower_id) and "." in group_id
        )
        formatted.append("<redacted>" if is_path_like else group_id)
    return ", ".join(formatted)


def _binary_labels(y):
    values = _as_vector(y, "labels")
    numeric = pd.to_numeric(pd.Series(values), errors="coerce")
    if numeric.isna().any() or not numeric.isin([0, 1]).all():
        raise ValueError("labels must contain only binary values 0 and 1")
    return numeric.astype(int).to_numpy()


def _validated_labels_and_groups(y, groups):
    labels = _binary_labels(y)
    group_values = _normalized_groups(groups)
    if len(labels) != len(group_values):
        raise ValueError("labels and groups must have matching lengths")
    return labels, group_values


def _reject_normalization_collisions(raw_groups, normalized_groups):
    """Reject different typed IDs that would collapse only after normalization."""
    representations = {}
    for raw_group, normalized_group in zip(raw_groups, normalized_groups):
        identity = (type(raw_group), repr(raw_group))
        representations.setdefault(normalized_group, set()).add(identity)
    collisions = [
        group_id for group_id, values in representations.items() if len(values) > 1
    ]
    if collisions:
        raise ValueError(
            "multiple raw identifiers for normalized source-patient IDs: "
            + _format_safe_group_ids(sorted(collisions))
        )


def assert_disjoint_groups(train_groups, test_groups) -> None:
    """Reject train/test source-patient overlap without exposing metadata."""
    train = _normalized_groups(train_groups, non_empty=False)
    test = _normalized_groups(test_groups, non_empty=False)
    overlap = sorted(set(train).intersection(test))
    if overlap:
        disclosed = _format_safe_group_ids(overlap)
        raise ValueError(f"train/test source-patient overlap: {disclosed}")


def group_counts_per_class(y, groups) -> dict[int, int]:
    """Count unique source patients in each binary class."""
    labels, group_values = _validated_labels_and_groups(y, groups)
    table = pd.DataFrame({"group_id": group_values, "label": labels})
    label_counts = table.groupby("group_id", sort=False)["label"].nunique()
    conflicts = label_counts[label_counts > 1].index.tolist()
    if conflicts:
        raise ValueError(
            "conflicting labels for source patients: "
            + _format_safe_group_ids(conflicts)
        )
    per_group = table.drop_duplicates("group_id")
    return {label: int((per_group["label"] == label).sum()) for label in (0, 1)}


def build_group_cv(y, groups, *, requested_splits: int, minimum_per_class: int):
    """Build deterministic, source-patient grouped stratified cross-validation."""
    if not isinstance(requested_splits, int) or requested_splits < 2:
        raise ValueError("requested_splits must be an integer of at least 2")
    if not isinstance(minimum_per_class, int) or minimum_per_class < 2:
        raise ValueError("minimum_per_class must be an integer of at least 2")
    labels, group_values = _validated_labels_and_groups(y, groups)
    _reject_normalization_collisions(_as_vector(groups, "groups"), group_values)
    counts = group_counts_per_class(labels, group_values)
    if counts[0] == 0 or counts[1] == 0:
        raise ValueError("both binary classes are required")
    n_splits = min(requested_splits, counts[0], counts[1])
    if n_splits < minimum_per_class:
        raise ValueError("insufficient source-patient groups per class for requested splits")
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)


def materialize_group_splits(y, groups, *, requested_splits: int):
    """Materialize deterministic patient-disjoint train/validation index splits."""
    cv = build_group_cv(
        y,
        groups,
        requested_splits=requested_splits,
        minimum_per_class=requested_splits,
    )
    return list(cv.split(np.zeros(len(y)), y, groups))


def aggregate_group_probabilities(groups, labels, probabilities) -> pd.DataFrame:
    """Average row probabilities to one sorted record per source patient."""
    group_values = _normalized_groups(groups)
    label_values = _binary_labels(labels)
    probability_values = _as_vector(probabilities, "probabilities")
    if len(group_values) != len(label_values) or len(groups) != len(probability_values):
        raise ValueError("groups, labels, and probabilities must have matching lengths")
    numeric_probabilities = pd.to_numeric(
        pd.Series(probability_values), errors="coerce"
    ).to_numpy(dtype=float)
    if not np.isfinite(numeric_probabilities).all():
        raise ValueError("probabilities must be finite")
    if ((numeric_probabilities < 0) | (numeric_probabilities > 1)).any():
        raise ValueError("probabilities must be within [0, 1]")
    table = pd.DataFrame({
        "group_id": group_values,
        "label": label_values,
        "probability": numeric_probabilities,
    })
    label_counts = table.groupby("group_id", sort=False)["label"].nunique()
    conflicts = label_counts[label_counts > 1].index.tolist()
    if conflicts:
        raise ValueError(
            "conflicting labels for source patients: "
            + _format_safe_group_ids(conflicts)
        )
    return (
        table.groupby("group_id", as_index=False, sort=True)
        .agg(label=("label", "first"), probability=("probability", "mean"))
        .loc[:, ["group_id", "label", "probability"]]
    )
