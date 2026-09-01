"""Leakage-safe nested grouped logistic classifiers with a locked schema."""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .grouped_validation import aggregate_group_probabilities, build_group_cv
from .patient_feature_table import PatientFeatureTable


C_GRID = np.logspace(-4, 4, 17)


def _oof_metrics(groups, labels, probabilities):
    aggregated = aggregate_group_probabilities(groups, labels, probabilities)
    y = aggregated["label"].to_numpy(dtype=int)
    probability = aggregated["probability"].to_numpy(dtype=float)
    prediction = (probability >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {
        "NumPatients": int(len(aggregated)), "AUC": float(roc_auc_score(y, probability)),
        "PR_AUC": float(average_precision_score(y, probability)),
        "Accuracy": float(accuracy_score(y, prediction)),
        "Precision": float(precision_score(y, prediction, zero_division=0)),
        "Sensitivity": float(recall_score(y, prediction, zero_division=0)),
        "Specificity": float(tn / (tn + fp)),
        "F1": float(f1_score(y, prediction, zero_division=0)),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


def _pipeline(*, penalty: str) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    penalty=penalty,
                    solver="liblinear",
                    max_iter=5000,
                    random_state=42,
                ),
            ),
        ]
    )


def _search(y: pd.Series, groups: pd.Series, *, penalty: str) -> GridSearchCV:
    inner_cv = build_group_cv(
        y,
        groups,
        requested_splits=5,
        minimum_per_class=2,
    )
    return GridSearchCV(
        estimator=_pipeline(penalty=penalty),
        param_grid={"model__C": C_GRID},
        scoring="roc_auc",
        cv=inner_cv,
        refit=True,
        n_jobs=1,
        error_score="raise",
        return_train_score=False,
    )


class _GroupedLogisticBase:
    penalty = ""

    def __init__(self) -> None:
        self.final_pipeline = None
        self.feature_columns = ()
        self.extra_test_features = ()
        self.nested_cv_summary = None

    @staticmethod
    def _require_table(table) -> PatientFeatureTable:
        if not isinstance(table, PatientFeatureTable):
            raise TypeError("Expected a PatientFeatureTable input.")
        if not isinstance(table.frame, pd.DataFrame):
            raise TypeError("PatientFeatureTable.frame must be a pandas DataFrame.")
        return table

    @classmethod
    def _training_data(cls, table):
        table = cls._require_table(table)
        columns = tuple(table.feature_columns)
        if not columns:
            raise ValueError("Training feature schema must not be empty.")
        if len(set(columns)) != len(columns):
            raise ValueError("Training feature schema contains duplicate feature names.")
        if any(not isinstance(column, str) or not column for column in columns):
            raise ValueError("Training feature names must be non-empty strings.")
        missing = [column for column in columns if column not in table.frame.columns]
        if missing:
            raise ValueError(f"Training table is missing locked feature: {missing[0]}")
        if table.frame.columns.duplicated().any():
            duplicated = table.frame.columns[table.frame.columns.duplicated()][0]
            raise ValueError(f"Training table contains duplicate column: {duplicated}")
        if "label" not in table.frame:
            raise ValueError("PatientFeatureTable is missing binary label column 'label'.")
        if table.group_column not in table.frame:
            raise ValueError(
                f"PatientFeatureTable is missing group column: {table.group_column}"
            )

        converted = {}
        for column in columns:
            try:
                values = pd.to_numeric(table.frame[column], errors="raise")
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Feature '{column}' must be numeric.") from exc
            numeric = values.to_numpy(dtype=float, na_value=np.nan)
            present = ~pd.isna(numeric)
            if not present.any():
                raise ValueError(f"Feature '{column}' has no usable observations; it is entirely missing.")
            if not np.isfinite(numeric[present]).all():
                raise ValueError(f"Feature '{column}' contains non-finite values.")
            converted[column] = numeric

        X = pd.DataFrame(converted, columns=columns)
        y = pd.Series(table.frame["label"].to_numpy()).reset_index(drop=True)
        groups = pd.Series(table.frame[table.group_column].to_numpy()).reset_index(drop=True)
        if len(X) == 0:
            raise ValueError("PatientFeatureTable must contain at least one row.")
        return columns, X, y, groups

    def fit(self, table: PatientFeatureTable):
        columns, X, y, groups = self._training_data(table)
        self.feature_columns = columns
        self.extra_test_features = ()

        outer_cv = build_group_cv(
            y,
            groups,
            requested_splits=5,
            minimum_per_class=3,
        )
        oof_probability = np.full(len(y), np.nan, dtype=float)
        folds = []
        for fold_index, (fit_idx, valid_idx) in enumerate(
            outer_cv.split(X, y, groups), start=1
        ):
            search = _search(
                y.iloc[fit_idx],
                groups.iloc[fit_idx],
                penalty=self.penalty,
            )
            search.fit(
                X.iloc[fit_idx],
                y.iloc[fit_idx],
                groups=groups.iloc[fit_idx],
            )
            oof_probability[valid_idx] = search.predict_proba(X.iloc[valid_idx])[:, 1]
            folds.append(
                {
                    "fold": int(fold_index),
                    "best_c": float(search.best_params_["model__C"]),
                    "train_groups": int(groups.iloc[fit_idx].nunique()),
                    "validation_groups": int(groups.iloc[valid_idx].nunique()),
                }
            )
        if np.isnan(oof_probability).any():
            raise RuntimeError(
                "Nested grouped CV did not produce one OOF probability per row."
            )

        final_search = _search(y, groups, penalty=self.penalty)
        final_search.fit(X, y, groups=groups)
        self.final_pipeline = final_search.best_estimator_
        self.nested_cv_summary = {
            "folds": tuple(folds),
            "oof_probabilities": tuple(float(value) for value in oof_probability),
            "oof_metrics": _oof_metrics(groups, y, oof_probability),
            "final_best_c": float(final_search.best_params_["model__C"]),
            "final_c_grid_auc": tuple(
                {
                    "c": float(parameters["model__C"]),
                    "mean_auc": float(mean_auc),
                }
                for parameters, mean_auc in zip(
                    final_search.cv_results_["params"],
                    final_search.cv_results_["mean_test_score"],
                )
            ),
        }
        return self

    def _prediction_data(self, table: PatientFeatureTable) -> pd.DataFrame:
        if self.final_pipeline is None:
            raise RuntimeError("The model must be fit before prediction.")
        table = self._require_table(table)
        missing = [
            column for column in self.feature_columns if column not in table.frame.columns
        ]
        if missing:
            raise ValueError(f"Prediction table is missing locked feature: {missing[0]}")
        declared = tuple(table.feature_columns)
        self.extra_test_features = tuple(
            column for column in declared if column not in self.feature_columns
        )
        converted = {}
        for column in self.feature_columns:
            try:
                values = pd.to_numeric(table.frame[column], errors="raise")
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Feature '{column}' must be numeric.") from exc
            numeric = values.to_numpy(dtype=float, na_value=np.nan)
            present = ~pd.isna(numeric)
            if present.any() and not np.isfinite(numeric[present]).all():
                raise ValueError(f"Feature '{column}' contains non-finite values.")
            converted[column] = numeric
        return pd.DataFrame(converted, columns=self.feature_columns)

    def predict_proba(self, table: PatientFeatureTable) -> np.ndarray:
        X = self._prediction_data(table)
        return np.asarray(self.final_pipeline.predict_proba(X)[:, 1], dtype=float)

    @property
    def coefficients(self) -> dict[str, float]:
        if self.final_pipeline is None:
            raise RuntimeError("The model must be fit before coefficients are available.")
        values = self.final_pipeline.named_steps["model"].coef_[0]
        return {
            column: float(value)
            for column, value in zip(self.feature_columns, values)
        }

    @property
    def selected_features(self) -> tuple[str, ...]:
        if self.final_pipeline is None:
            raise RuntimeError("The model must be fit before selected features are available.")
        if self.penalty == "l2":
            return self.feature_columns
        coefficients = self.final_pipeline.named_steps["model"].coef_[0]
        return tuple(
            column
            for column, coefficient in zip(self.feature_columns, coefficients)
            if coefficient != 0.0
        )


class LeakageSafeL1Logistic(_GroupedLogisticBase):
    penalty = "l1"


class LeakageSafeL2Baseline(_GroupedLogisticBase):
    penalty = "l2"
