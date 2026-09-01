"""Leakage-safe, training-only validation for atomic KL pipeline recipes.

The public API deliberately accepts one training feature CSV and no held-out
test path. Candidate ranking therefore cannot observe test labels or metrics.
Every preprocessing and feature-selection step is fitted inside the inner
training folds of a patient-grouped nested cross-validation.
"""

from __future__ import annotations

from collections import Counter
import hashlib
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from algorithm.klgrade_policy import select_labeled_klgrade_rows

from .classification_artifacts import compute_binary_metrics
from .grouped_validation import aggregate_group_probabilities, build_group_cv
from .patient_feature_table import PatientFeatureTableBuilder


TRIAL_SCHEMA_VERSION = "1.1"
TRIAL_EVALUATOR_VERSION = "nested-grouped-cv-v2"
TRIAL_RANDOM_SEED = 42
TRIAL_C_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
TRIAL_BOOTSTRAP_SAMPLES = 500

MODEL_RECIPE_L1_LOGISTIC = "L1_LOGISTIC"
MODEL_RECIPE_ANOVA_TOPK_L2 = "ANOVA_TOPK_L2"
SUPPORTED_MODEL_RECIPES = frozenset(
    {MODEL_RECIPE_L1_LOGISTIC, MODEL_RECIPE_ANOVA_TOPK_L2}
)


def normalize_model_recipe(
    classification_method: str, model_recipe: str | None = None
) -> str:
    """Resolve a legacy method label to one explicit executable recipe."""
    method = str(classification_method or "").strip().upper()
    expected = {
        "LASSO": MODEL_RECIPE_L1_LOGISTIC,
        "FILTER": MODEL_RECIPE_ANOVA_TOPK_L2,
    }.get(method)
    if expected is None:
        raise ValueError("classification_method must be LASSO or Filter")
    recipe = str(model_recipe or expected).strip().upper()
    if recipe not in SUPPORTED_MODEL_RECIPES:
        raise ValueError(f"unsupported model_recipe: {recipe}")
    if recipe != expected:
        raise ValueError(
            f"classification_method {classification_method!r} is incompatible "
            f"with model_recipe {recipe!r}"
        )
    return recipe


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _training_table(train_path: str | Path):
    path = Path(train_path)
    if not path.is_file():
        raise ValueError("train_path must be an existing feature CSV")
    frame = pd.read_csv(path, dtype={"CaseID": str})
    selected = select_labeled_klgrade_rows(
        frame, dataset_role="pipeline-search training feature CSV"
    )
    table = PatientFeatureTableBuilder().build(selected.frame)
    columns = tuple(table.feature_columns)
    if not columns:
        raise ValueError("training feature schema must not be empty")
    X = table.frame.loc[:, columns].apply(pd.to_numeric, errors="raise")
    numeric = X.to_numpy(dtype=float, na_value=np.nan)
    for index, column in enumerate(columns):
        present = ~np.isnan(numeric[:, index])
        if not present.any():
            raise ValueError(f"feature '{column}' is entirely missing")
        if not np.isfinite(numeric[present, index]).all():
            raise ValueError(f"feature '{column}' contains non-finite values")
    y = table.frame["label"].astype(int).reset_index(drop=True)
    groups = table.frame[table.group_column].astype(str).reset_index(drop=True)
    X = X.reset_index(drop=True)
    return path.resolve(), selected, table, X, y, groups


def _candidate_pipeline(recipe: str) -> tuple[Pipeline, dict[str, list[Any]]]:
    if recipe == MODEL_RECIPE_L1_LOGISTIC:
        return (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            penalty="l1",
                            solver="liblinear",
                            max_iter=5000,
                            random_state=TRIAL_RANDOM_SEED,
                        ),
                    ),
                ]
            ),
            {"model__C": list(TRIAL_C_GRID)},
        )
    if recipe == MODEL_RECIPE_ANOVA_TOPK_L2:
        return (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("selector", SelectKBest(score_func=f_classif, k="all")),
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            penalty="l2",
                            solver="liblinear",
                            max_iter=5000,
                            random_state=TRIAL_RANDOM_SEED,
                        ),
                    ),
                ]
            ),
            {
                "selector__k": [5, 10, "all"],
                "model__C": [0.1, 1.0, 10.0],
            },
        )
    raise ValueError(f"unsupported model_recipe: {recipe}")


def _bounded_filter_grid(
    grid: dict[str, list[Any]], feature_count: int
) -> dict[str, list[Any]]:
    if "selector__k" not in grid:
        return grid
    values: list[Any] = []
    for value in grid["selector__k"]:
        bounded = min(int(value), feature_count) if value != "all" else "all"
        if bounded not in values:
            values.append(bounded)
    return {**grid, "selector__k": values}


def _plain_params(params: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value.item() if isinstance(value, np.generic) else value
        for key, value in params.items()
    }


def _selected_features(
    fitted: Pipeline, recipe: str, feature_names: tuple[str, ...]
) -> list[str]:
    if recipe == MODEL_RECIPE_ANOVA_TOPK_L2:
        support = fitted.named_steps["selector"].get_support()
    else:
        coefficients = np.asarray(fitted.named_steps["model"].coef_).reshape(-1)
        support = np.abs(coefficients) > 1e-12
    return [name for name, keep in zip(feature_names, support) if bool(keep)]


def _selection_summary(selected_by_fold: list[list[str]]) -> dict[str, Any]:
    sets = [set(items) for items in selected_by_fold]
    similarities: list[float] = []
    for left, right in combinations(sets, 2):
        union = left | right
        similarities.append(len(left & right) / len(union) if union else 0.0)
    counts = [len(items) for items in sets]
    frequency = Counter(name for items in sets for name in items)
    denominator = max(1, len(sets))
    return {
        "metric": "mean_pairwise_jaccard",
        "mean_pairwise_jaccard": (
            float(np.mean(similarities)) if similarities else 1.0
        ),
        "selected_count": {
            "mean": float(np.mean(counts)) if counts else 0.0,
            "min": min(counts) if counts else 0,
            "max": max(counts) if counts else 0,
        },
        "feature_frequency": {
            name: count / denominator for name, count in sorted(frequency.items())
        },
    }


def _bootstrap_intervals(
    patient_predictions: pd.DataFrame, *, samples: int
) -> dict[str, dict[str, Any]]:
    if not isinstance(samples, int) or samples < 100:
        raise ValueError("bootstrap_samples must be an integer of at least 100")
    labels = patient_predictions["label"].to_numpy(dtype=int)
    class_indices = [np.flatnonzero(labels == label) for label in (0, 1)]
    rng = np.random.default_rng(TRIAL_RANDOM_SEED)
    names = (
        "AUC", "PR_AUC", "Accuracy", "Precision",
        "Sensitivity", "Specificity", "F1",
    )
    distributions: dict[str, list[float]] = {name: [] for name in names}
    for _ in range(samples):
        indices = np.concatenate(
            [
                rng.choice(values, size=len(values), replace=True)
                for values in class_indices
            ]
        )
        bootstrap = patient_predictions.iloc[indices].reset_index(drop=True).copy()
        bootstrap["group_id"] = [
            f"bootstrap-{index}" for index in range(len(bootstrap))
        ]
        metrics = compute_binary_metrics(bootstrap)
        for name in names:
            distributions[name].append(float(metrics[name]))
    point_metrics = compute_binary_metrics(patient_predictions)
    return {
        name: {
            "point": float(point_metrics[name]),
            "lower": float(np.quantile(values, 0.025)),
            "upper": float(np.quantile(values, 0.975)),
            "confidence_level": 0.95,
            "method": "stratified_patient_bootstrap_percentile",
            "samples": samples,
        }
        for name, values in distributions.items()
    }


def validate_training_candidate(
    train_path: str | Path,
    classification_method: str,
    *,
    model_recipe: str | None = None,
    requested_outer_splits: int = 3,
    requested_inner_splits: int = 3,
    bootstrap_samples: int = TRIAL_BOOTSTRAP_SAMPLES,
) -> dict[str, Any]:
    """Return robust nested-CV evidence without reading any test artifact."""
    recipe = normalize_model_recipe(classification_method, model_recipe)
    path, selected, table, X, y, groups = _training_table(train_path)
    feature_names = tuple(str(item) for item in table.feature_columns)
    X_values = X.to_numpy(dtype=float, na_value=np.nan)
    y_values = y.to_numpy(dtype=int)
    group_values = groups.to_numpy(dtype=object)
    estimator, parameter_grid = _candidate_pipeline(recipe)
    parameter_grid = _bounded_filter_grid(parameter_grid, X.shape[1])
    outer_cv = build_group_cv(
        y_values,
        group_values,
        requested_splits=requested_outer_splits,
        minimum_per_class=2,
    )
    probabilities = np.full(len(y), np.nan, dtype=float)
    fold_records: list[dict[str, Any]] = []
    selected_by_fold: list[list[str]] = []

    for fold, (fit_index, validation_index) in enumerate(
        outer_cv.split(X_values, y_values, group_values), start=1
    ):
        fit_y = y_values[fit_index]
        fit_groups = group_values[fit_index]
        inner_cv = build_group_cv(
            fit_y,
            fit_groups,
            requested_splits=requested_inner_splits,
            minimum_per_class=2,
        )
        search = GridSearchCV(
            estimator=estimator,
            param_grid=parameter_grid,
            scoring="roc_auc",
            cv=inner_cv,
            refit=True,
            n_jobs=1,
            error_score="raise",
            return_train_score=False,
        )
        search.fit(X_values[fit_index], fit_y, groups=fit_groups)
        fold_probabilities = search.predict_proba(X_values[validation_index])[:, 1]
        probabilities[validation_index] = fold_probabilities
        fold_selected = _selected_features(
            search.best_estimator_, recipe, feature_names
        )
        selected_by_fold.append(fold_selected)
        fold_patient_predictions = aggregate_group_probabilities(
            group_values[validation_index],
            y_values[validation_index],
            fold_probabilities,
        )
        fold_records.append(
            {
                "fold": fold,
                "train_groups": int(len(set(fit_groups))),
                "validation_groups": int(
                    len(set(group_values[validation_index]))
                ),
                "best_params": _plain_params(search.best_params_),
                "inner_best_auc": float(search.best_score_),
                "selected_feature_count": len(fold_selected),
                "selected_features": fold_selected,
                "validation_metrics": compute_binary_metrics(
                    fold_patient_predictions
                ),
            }
        )

    if np.isnan(probabilities).any():
        raise RuntimeError("nested grouped CV did not produce all OOF probabilities")
    patient_predictions = aggregate_group_probabilities(
        group_values, y_values, probabilities
    )
    metrics = compute_binary_metrics(patient_predictions)
    full_training_cv = build_group_cv(
        y_values,
        group_values,
        requested_splits=requested_inner_splits,
        minimum_per_class=2,
    )
    full_training_search = GridSearchCV(
        estimator=estimator,
        param_grid=parameter_grid,
        scoring="roc_auc",
        cv=full_training_cv,
        refit=True,
        n_jobs=1,
        error_score="raise",
        return_train_score=False,
    )
    full_training_search.fit(X_values, y_values, groups=group_values)
    recommended_params = _plain_params(full_training_search.best_params_)
    normalized_method = str(classification_method).strip()
    return {
        "schema_version": TRIAL_SCHEMA_VERSION,
        "status": "success",
        "validation_scope": "training_only_nested_grouped_cv",
        "test_data_used": False,
        "classification_method": (
            "LASSO" if normalized_method.upper() == "LASSO" else "Filter"
        ),
        "model_recipe": recipe,
        "train_sha256": _sha256(path),
        "input_rows": selected.input_rows,
        "labeled_rows": selected.labeled_rows,
        "excluded_unlabeled_rows": selected.excluded_unlabeled_rows,
        "feature_count": int(len(feature_names)),
        "patient_count": int(patient_predictions["group_id"].nunique()),
        "outer_folds": len(fold_records),
        "inner_folds_requested": requested_inner_splits,
        "folds": fold_records,
        "metrics": metrics,
        "metric_intervals": _bootstrap_intervals(
            patient_predictions, samples=bootstrap_samples
        ),
        "selection_stability": _selection_summary(selected_by_fold),
        "recommended_params": recommended_params,
        "full_training_selection": {
            "best_params": recommended_params,
            "inner_best_auc": float(full_training_search.best_score_),
            "scope": "training_only_grouped_cv",
        },
        "random_seed": TRIAL_RANDOM_SEED,
        "evaluator_version": TRIAL_EVALUATOR_VERSION,
    }
