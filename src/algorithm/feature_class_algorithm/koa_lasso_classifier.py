"""Patient-grouped, leakage-safe L1 Logistic classification integration."""

from __future__ import annotations

import hashlib
import io
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    precision_recall_curve,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from .classification_artifacts import (
    _atomic_csv,
    compute_binary_metrics,
    write_classification_artifacts,
)
from .grouped_validation import (
    aggregate_group_probabilities,
    assert_disjoint_groups,
)
from .leakage_safe_l1_logistic import (
    C_GRID,
    LeakageSafeL1Logistic,
    LeakageSafeL2Baseline,
)
from .patient_feature_table import PatientFeatureTable, PatientFeatureTableBuilder
from ..klgrade_policy import LabeledKLGradeRows, select_labeled_klgrade_rows


@dataclass(frozen=True)
class _FeatureCSVSnapshot:
    payload: bytes
    sha256: str
    frame: pd.DataFrame


class KOALassoClassifier:
    """Run primary grouped L1 Logistic and an exploratory grouped L2 baseline."""

    def __init__(self, train_path, test_path, output_dir, by_region=False):
        self.train_path = train_path
        self.test_path = test_path
        self.output_dir = output_dir
        self.by_region = by_region
        self.results = []
        self.diagnostics = []
        self.required_relative_outputs = ()
        self._train_input_sha256 = None
        self._test_input_sha256 = None

    @staticmethod
    def _required_output_contract(region_tokens=None):
        names = (
            "patient_feature_schema{suffix}.json",
            "nested_cv_summary{suffix}.json",
            "model_coefficients{suffix}.csv",
            "classification_run_manifest{suffix}.json",
            "LASSO_selected_features{suffix}.txt",
            "metrics_LASSO{suffix}.txt",
        )
        required = ["metrics_summary.csv"]
        if region_tokens is None:
            required.extend(name.format(suffix="") for name in names)
        else:
            for token in region_tokens.values():
                suffix = f"_region{token}"
                required.extend(
                    f"region_{token}/{name.format(suffix=suffix)}" for name in names
                )
        return tuple(required)

    @staticmethod
    def _read_feature_csv_snapshot(path: str, role: str) -> _FeatureCSVSnapshot:
        try:
            with open(path, "rb") as source:
                payload = source.read()
            header = pd.read_csv(io.BytesIO(payload), nrows=0)
            string_columns = {
                name: str
                for name in ("CaseID", "SourceCaseID")
                if name in header.columns
            }
            frame = pd.read_csv(io.BytesIO(payload), dtype=string_columns)
        except (OSError, pd.errors.ParserError, UnicodeError) as exc:
            raise ValueError(f"Unable to read the {role} feature CSV.") from exc
        return _FeatureCSVSnapshot(
            payload=payload,
            sha256=hashlib.sha256(payload).hexdigest(),
            frame=frame,
        )

    @staticmethod
    def _read_feature_csv(path: str, role: str) -> pd.DataFrame:
        return KOALassoClassifier._read_feature_csv_snapshot(path, role).frame

    @staticmethod
    def _safe_names(values) -> str:
        shown = []
        values = list(values)
        for value in values[:10]:
            rendered = str(value)
            lowered = rendered.lower()
            if (
                "/" in rendered
                or "\\" in rendered
                or (("image" in lowered or "mask" in lowered) and "." in rendered)
            ):
                rendered = "<redacted>"
            shown.append(rendered)
        suffix = " ..." if len(values) > 10 else ""
        return ", ".join(shown) + suffix

    @staticmethod
    def _region_token(region) -> str:
        token = str(region)
        invalid_characters = '<>:"/\\|?*'
        if (
            not token
            or ".." in token
            or token.endswith((".", " "))
            or any(character in invalid_characters for character in token)
            or any(ord(character) < 32 for character in token)
        ):
            raise ValueError("Region values must be non-empty safe names.")
        return token

    @classmethod
    def _region_tokens(cls, regions) -> dict:
        tokens = {region: cls._region_token(region) for region in regions}
        canonical = [token.casefold() for token in tokens.values()]
        if len(set(canonical)) != len(canonical):
            raise ValueError("Region values produce case-insensitive output collisions.")
        return tokens

    @staticmethod
    def _model_output_paths(directory: Path, suffix: str) -> list[Path]:
        names = [
            f"LASSO_selected_features{suffix}.txt",
            f"metrics_LASSO{suffix}.txt",
            f"patient_feature_schema{suffix}.json",
            f"nested_cv_summary{suffix}.json",
            f"model_coefficients{suffix}.csv",
            f"classification_run_manifest{suffix}.json",
            f"ROC曲线_LASSO{suffix}.png",
            f"ROC曲线_ALL{suffix}.png",
            f"PR曲线_LASSO{suffix}.png",
            f"PR曲线_ALL{suffix}.png",
            f"L1_Logistic_CV_AUC{suffix}.png",
            f"SHAP_Summary_LASSO{suffix}.png",
            f"SHAP_Summary_ALL{suffix}.png",
            f"SHAP_Waterfall_LASSO{suffix}.png",
            f"SHAP_Waterfall_ALL{suffix}.png",
        ]
        return [directory / name for name in names]

    def _preflight_output_paths(
        self, output_dir: Path, region_tokens: dict | None
    ) -> None:
        candidates = [output_dir / "metrics_summary.csv"]
        if region_tokens is None:
            candidates.extend(self._model_output_paths(output_dir, ""))
        else:
            for token in region_tokens.values():
                directory = output_dir / f"region_{token}"
                candidates.extend(
                    self._model_output_paths(directory, f"_region{token}")
                )
        canonical_outputs = [
            os.path.normcase(str(path.resolve())).casefold() for path in candidates
        ]
        if len(set(canonical_outputs)) != len(canonical_outputs):
            raise ValueError("Region values produce output filename collisions.")
        canonical_inputs = {
            os.path.normcase(str(Path(path).resolve())).casefold()
            for path in (self.train_path, self.test_path)
        }
        if canonical_inputs.intersection(canonical_outputs):
            raise ValueError("Classification output must not alias an input file.")

    def _preflight_summary_path(self, summary_path: Path) -> None:
        summary_key = os.path.normcase(str(summary_path.resolve())).casefold()
        input_keys = {
            os.path.normcase(str(Path(path).resolve())).casefold()
            for path in (self.train_path, self.test_path)
        }
        if summary_key in input_keys:
            raise ValueError("Classification summary must not alias an input file.")

    @staticmethod
    def _require_reusable_target(output_dir: Path) -> None:
        if not output_dir.exists():
            return
        if not output_dir.is_dir():
            raise ValueError("Classification output target is not a directory.")
        if any(output_dir.iterdir()):
            raise ValueError("Classification output directory must be empty; non-empty reuse is rejected.")

    @staticmethod
    def _create_staging_directory(output_dir: Path) -> Path:
        parent = output_dir.parent.resolve()
        parent.mkdir(parents=True, exist_ok=True)
        prefix = f".{output_dir.name}.stage-"
        stage = (parent / f"{prefix}{uuid.uuid4().hex}").resolve()
        if stage.parent != parent or not stage.name.startswith(prefix):
            raise ValueError("Unable to create a safe classification staging directory.")
        stage.mkdir(exist_ok=False)
        return stage

    @staticmethod
    def _cleanup_staging_directory(stage: Path, output_dir: Path) -> None:
        if not stage.exists():
            return
        parent = output_dir.parent.resolve()
        resolved = stage.resolve()
        prefix = f".{output_dir.name}.stage-"
        if resolved.parent != parent or not resolved.name.startswith(prefix):
            raise ValueError("Refusing to clean an unverified staging directory.")
        shutil.rmtree(resolved)

    @staticmethod
    def _publish_staged_directory(stage: Path, output_dir: Path) -> None:
        if output_dir.exists():
            if not output_dir.is_dir() or any(output_dir.iterdir()):
                raise ValueError("Classification output directory became non-empty before publication.")
            output_dir.rmdir()
        os.replace(stage, output_dir)

    @staticmethod
    def _assert_prediction_schema(
        train_table: PatientFeatureTable, test_table: PatientFeatureTable
    ) -> None:
        missing = [
            feature
            for feature in train_table.feature_columns
            if feature not in test_table.frame.columns
        ]
        if missing:
            shown = KOALassoClassifier._safe_names(missing)
            raise ValueError(f"Prediction table is missing locked feature: {shown}")

    @staticmethod
    def _source_groups(frame: pd.DataFrame) -> pd.Series:
        groups = frame["CaseID"].astype(str).copy()
        if "SourceCaseID" in frame:
            sources = frame["SourceCaseID"]
            usable = sources.notna() & sources.astype(str).str.strip().ne("")
            groups.loc[usable] = sources.loc[usable].astype(str)
        return groups

    @staticmethod
    def _validate_required_columns(frame: pd.DataFrame) -> None:
        missing = [column for column in ("CaseID", "KLGrade") if column not in frame]
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(missing)}.")

    @classmethod
    def _validate_global_group_grades(cls, frame: pd.DataFrame) -> None:
        groups = cls._source_groups(frame)
        grades = frame["KLGrade"]
        table = pd.DataFrame({"group": groups.to_numpy(), "grade": grades.to_numpy()})
        conflicting = table.groupby("group", sort=True)["grade"].nunique(dropna=False)
        identifiers = conflicting[conflicting > 1].index.tolist()
        if identifiers:
            raise ValueError(
                "KLGrade conflict for source-patient groups: "
                + cls._safe_names(identifiers)
            )

    def run(self):
        try:
            return self._run()
        except OSError as exc:
            raise ValueError(
                "Unable to publish classification outputs safely."
            ) from exc

    def _record_label_selection(
        self,
        role: str,
        selection: LabeledKLGradeRows,
    ) -> None:
        self.diagnostics.append(
            f"{role}: excluded {selection.excluded_unlabeled_rows} unlabeled "
            f"KLGrade=-1 row(s); {selection.labeled_rows} labeled row(s) remain"
        )

    def _run(self):
        output_dir = Path(self.output_dir).resolve()
        summary_path = output_dir / "metrics_summary.csv"

        self.results = []
        self.diagnostics = []
        self.required_relative_outputs = ()
        train_snapshot = self._read_feature_csv_snapshot(self.train_path, "training")
        test_snapshot = self._read_feature_csv_snapshot(self.test_path, "test")
        self._train_input_sha256 = train_snapshot.sha256
        self._test_input_sha256 = test_snapshot.sha256
        train = train_snapshot.frame
        test = test_snapshot.frame
        train_selection = select_labeled_klgrade_rows(
            train, dataset_role="training feature CSV"
        )
        test_selection = select_labeled_klgrade_rows(
            test, dataset_role="test feature CSV"
        )
        train = train_selection.frame
        test = test_selection.frame
        self._record_label_selection("train", train_selection)
        self._record_label_selection("test", test_selection)
        self._validate_required_columns(train)
        self._validate_required_columns(test)
        PatientFeatureTableBuilder._validate_case_ids(train)
        PatientFeatureTableBuilder._validate_case_ids(test)
        PatientFeatureTableBuilder._validate_regions(train)
        PatientFeatureTableBuilder._validate_regions(test)
        builder = PatientFeatureTableBuilder()
        self._preflight_summary_path(summary_path)
        stage = None
        published = False

        try:
            if not self.by_region:
                self._preflight_output_paths(output_dir, None)
                self._require_reusable_target(output_dir)
                train_table = builder.build(train)
                test_table = builder.build(test, allow_repeated_groups=True)
                self._validate_pair(train_table, test_table)
                stage = self._create_staging_directory(output_dir)
                self._execute_patient_model(
                    train_table,
                    test_table,
                    stage,
                    region_name=None,
                )
            else:
                if "Region" not in train or "Region" not in test:
                    raise ValueError("Region is required in both train and test CSVs.")
                train_regions = sorted(train["Region"].drop_duplicates().tolist(), key=str)
                region_tokens = self._region_tokens(train_regions)
                self._preflight_output_paths(output_dir, region_tokens)
                self._require_reusable_target(output_dir)
                test_regions = set(test["Region"].drop_duplicates().tolist())
                missing_regions = [region for region in train_regions if region not in test_regions]
                if missing_regions:
                    names = self._safe_names(missing_regions)
                    raise ValueError(f"Test CSV is missing training Region values: {names}")
                assert_disjoint_groups(
                    self._source_groups(train), self._source_groups(test)
                )
                analyzed_test = test.loc[test["Region"].isin(train_regions)]
                self._validate_global_group_grades(train)
                self._validate_global_group_grades(analyzed_test)
                extra_regions = sorted(test_regions.difference(train_regions), key=str)
                if extra_regions:
                    names = self._safe_names(extra_regions)
                    self.diagnostics.append(
                        f"Ignored {len(extra_regions)} extra test Region value(s): {names}"
                    )
                stage = self._create_staging_directory(output_dir)
                for region in train_regions:
                    token = region_tokens[region]
                    region_train = train.loc[train["Region"].eq(region)].copy()
                    region_test = test.loc[test["Region"].eq(region)].copy()
                    train_table = builder.build(region_train, by_region=True)
                    test_table = builder.build(
                        region_test,
                        by_region=True,
                        allow_repeated_groups=True,
                    )
                    self._validate_pair(train_table, test_table)
                    region_output = stage / f"region_{token}"
                    self._execute_patient_model(
                        train_table,
                        test_table,
                        region_output,
                        region_name=region,
                    )

            _atomic_csv(stage / "metrics_summary.csv", pd.DataFrame(self.results))
            self._publish_staged_directory(stage, output_dir)
            self.required_relative_outputs = self._required_output_contract(
                None if not self.by_region else region_tokens
            )
            published = True
        finally:
            if not published:
                self.results = []
                self.required_relative_outputs = ()
            if stage is not None and stage.exists():
                self._cleanup_staging_directory(stage, output_dir)

    def _validate_pair(
        self, train_table: PatientFeatureTable, test_table: PatientFeatureTable
    ) -> None:
        assert_disjoint_groups(
            train_table.frame[train_table.group_column],
            test_table.frame[test_table.group_column],
        )
        self._assert_prediction_schema(train_table, test_table)

    def _execute_patient_model(
        self,
        train_table: PatientFeatureTable,
        test_table: PatientFeatureTable,
        output_dir: str | Path,
        region_name=None,
    ) -> None:
        try:
            self._run_patient_model(
                train_table, test_table, output_dir, region_name=region_name
            )
        except ValueError as exc:
            message = str(exc)
            feature_names = (
                *train_table.feature_columns,
                *test_table.feature_columns,
            )
            for feature in feature_names:
                if self._safe_names([feature]) == "<redacted>":
                    message = message.replace(feature, "<redacted>")
            raise ValueError(message) from exc

    def _run_patient_model(
        self,
        train_table: PatientFeatureTable,
        test_table: PatientFeatureTable,
        output_dir: str | Path,
        region_name=None,
    ) -> None:
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        token = self._region_token(region_name) if region_name is not None else ""
        suffix = f"_region{token}" if token else ""

        primary = LeakageSafeL1Logistic().fit(train_table)
        if not primary.selected_features:
            raise ValueError(
                "L1 Logistic selected zero features from the locked training schema."
            )
        primary_aggregated = self._predict_patient_level(primary, test_table)
        primary_metrics = compute_binary_metrics(primary_aggregated)

        exploratory = LeakageSafeL2Baseline().fit(train_table)
        exploratory_aggregated = self._predict_patient_level(exploratory, test_table)
        exploratory_metrics = compute_binary_metrics(exploratory_aggregated)

        write_classification_artifacts(
            output_dir=directory,
            suffix=suffix,
            train_path=self.train_path,
            test_path=self.test_path,
            train_table=train_table,
            test_table=test_table,
            model=primary,
            metrics=primary_metrics,
            train_sha256=self._train_input_sha256,
            test_sha256=self._test_input_sha256,
        )
        self._write_selected_features(directory, suffix, primary.selected_features)
        self._write_metric_report(
            directory,
            suffix,
            primary_aggregated,
            primary_metrics,
        )

        self._safe_render(
            "LASSO", primary_aggregated, test_table, primary, directory, suffix
        )
        self._safe_render(
            "ALL_EXPLORATORY",
            exploratory_aggregated,
            test_table,
            exploratory,
            directory,
            suffix,
        )

        region_value = region_name if region_name is not None else "ALL"
        self.results.append(
            {"Region": region_value, "Model": "LASSO", **primary_metrics}
        )
        self.results.append(
            {
                "Region": region_value,
                "Model": "ALL_EXPLORATORY",
                **exploratory_metrics,
            }
        )

    @staticmethod
    def _predict_patient_level(model, test_table: PatientFeatureTable) -> pd.DataFrame:
        probabilities = model.predict_proba(test_table)
        return aggregate_group_probabilities(
            test_table.frame[test_table.group_column],
            test_table.frame["label"],
            probabilities,
        )

    @staticmethod
    def _write_selected_features(
        output_dir: Path, suffix: str, selected_features: tuple[str, ...]
    ) -> None:
        path = output_dir / f"LASSO_selected_features{suffix}.txt"
        path.write_text("\n".join(selected_features), encoding="utf-8")

    @staticmethod
    def _write_metric_report(
        output_dir: Path,
        suffix: str,
        aggregated: pd.DataFrame,
        metrics: dict,
    ) -> None:
        labels = aggregated["label"].to_numpy(dtype=int)
        predictions = (aggregated["probability"].to_numpy(dtype=float) >= 0.5).astype(int)
        lines = [
            "Model: LASSO patient-grouped L1 Logistic",
            *(f"{name}: {value}" for name, value in metrics.items()),
            "",
            "Patient-level classification report:",
            classification_report(labels, predictions, digits=4, zero_division=0),
        ]
        (output_dir / f"metrics_LASSO{suffix}.txt").write_text(
            "\n".join(lines), encoding="utf-8"
        )

    def _safe_render(
        self,
        model_name: str,
        aggregated: pd.DataFrame,
        row_test_table: PatientFeatureTable,
        model,
        output_dir: Path,
        suffix: str,
    ) -> list[str]:
        self._validate_render_state(aggregated, row_test_table, model)
        try:
            return self._render_plots_and_shap(
                aggregated, row_test_table, model, output_dir, suffix
            )
        except Exception as exc:  # plotting is optional; numeric outputs remain valid
            self.diagnostics.append(
                f"{model_name} rendering failed ({type(exc).__name__}); numeric artifacts remain available."
            )
            return []

    @staticmethod
    def _validate_render_state(
        aggregated: pd.DataFrame,
        row_test_table: PatientFeatureTable,
        model,
    ) -> None:
        if not isinstance(row_test_table, PatientFeatureTable):
            raise ValueError("Required render state has an invalid patient feature table.")
        if not isinstance(model, (LeakageSafeL1Logistic, LeakageSafeL2Baseline)):
            raise ValueError("Required render state has an invalid fitted model.")
        if model.final_pipeline is None or not model.feature_columns:
            raise ValueError("Required render state is missing the fitted pipeline or schema.")
        if any(
            feature not in row_test_table.frame.columns
            for feature in model.feature_columns
        ):
            raise ValueError("Required render state is missing a locked table feature.")
        pipeline = model.final_pipeline
        if not isinstance(pipeline, Pipeline):
            raise ValueError("Required render state has an invalid fitted pipeline.")
        if tuple(name for name, _ in pipeline.steps) != (
            "imputer",
            "scaler",
            "model",
        ):
            raise ValueError("Required render state has an invalid pipeline step order.")

        imputer = pipeline.named_steps["imputer"]
        scaler = pipeline.named_steps["scaler"]
        estimator = pipeline.named_steps["model"]
        if not isinstance(imputer, SimpleImputer):
            raise ValueError("Required render state has an invalid pipeline imputer.")
        if not isinstance(scaler, StandardScaler):
            raise ValueError("Required render state has an invalid pipeline scaler.")
        if not isinstance(estimator, LogisticRegression):
            raise ValueError("Required render state has an invalid pipeline model.")
        if estimator.penalty != model.penalty or estimator.solver != "liblinear":
            raise ValueError("Required render state has an incompatible pipeline model.")
        try:
            check_is_fitted(imputer, attributes=["statistics_"])
            check_is_fitted(scaler, attributes=["mean_", "scale_"])
            check_is_fitted(estimator, attributes=["coef_", "classes_"])
        except NotFittedError as exc:
            raise ValueError("Required render state has unfitted pipeline steps.") from exc

        feature_count = len(model.feature_columns)
        statistics = np.asarray(imputer.statistics_, dtype=float)
        scaler_mean = np.asarray(scaler.mean_, dtype=float)
        scaler_scale = np.asarray(scaler.scale_, dtype=float)
        coefficients = np.asarray(estimator.coef_, dtype=float)
        intercept = np.asarray(estimator.intercept_, dtype=float)
        classes = np.asarray(estimator.classes_)
        estimator_c = np.asarray(estimator.C)
        if (
            statistics.shape != (feature_count,)
            or scaler_mean.shape != (feature_count,)
            or scaler_scale.shape != (feature_count,)
            or coefficients.shape != (1, feature_count)
            or intercept.shape != (1,)
            or classes.shape != (2,)
            or estimator_c.ndim != 0
            or not np.isfinite(estimator_c)
            or estimator_c <= 0
            or not np.array_equal(classes, np.asarray([0, 1]))
            or not np.isfinite(statistics).all()
            or not np.isfinite(scaler_mean).all()
            or not np.isfinite(scaler_scale).all()
            or (scaler_scale <= 0).any()
            or not np.isfinite(coefficients).all()
            or not np.isfinite(intercept).all()
            or any(
                getattr(step, "n_features_in_", None) != feature_count
                for step in (imputer, scaler, estimator)
            )
        ):
            raise ValueError("Required render state has invalid fitted pipeline dimensions.")

        summary = model.nested_cv_summary
        if not isinstance(summary, dict):
            raise ValueError("Required render state is missing final C-grid data.")
        points = summary.get("final_c_grid_auc")
        if not isinstance(points, (tuple, list)) or len(points) != len(C_GRID):
            raise ValueError("Required render state is missing final C-grid data.")
        try:
            c_values = np.asarray([point["c"] for point in points], dtype=float)
            mean_auc = np.asarray(
                [point["mean_auc"] for point in points], dtype=float
            )
            best_c = float(summary["final_best_c"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Required render state has malformed C-grid data.") from exc
        expected_c_values = np.asarray(C_GRID, dtype=float)
        if (
            c_values.ndim != 1
            or mean_auc.ndim != 1
            or len(c_values) != len(mean_auc)
            or c_values.shape != expected_c_values.shape
            or not np.array_equal(c_values, expected_c_values)
            or not np.isfinite(c_values).all()
            or not np.isfinite(mean_auc).all()
            or not np.isfinite(best_c)
            or (c_values <= 0).any()
            or (mean_auc < 0).any()
            or (mean_auc > 1).any()
            or not np.equal(best_c, expected_c_values).any()
            or not np.equal(float(estimator_c), best_c)
        ):
            raise ValueError("Required render state has invalid C-grid or mean AUC data.")
        compute_binary_metrics(aggregated)

    def _render_plots_and_shap(
        self,
        aggregated,
        row_test_table,
        model,
        output_dir,
        suffix,
    ) -> list[str]:
        """Render patient-level curves and sample-level explanations."""
        from matplotlib import pyplot as plt

        directory = Path(output_dir)
        model_name = "LASSO" if model.penalty == "l1" else "ALL"
        created = []
        labels = aggregated["label"].to_numpy(dtype=int)
        probabilities = aggregated["probability"].to_numpy(dtype=float)

        fpr, tpr, _ = roc_curve(labels, probabilities)
        figure, axis = plt.subplots()
        axis.plot(fpr, tpr, label=model_name)
        axis.plot([0, 1], [0, 1], "k--")
        axis.set(xlabel="False positive rate", ylabel="True positive rate", title=f"Patient-level ROC{suffix}")
        axis.legend()
        roc_path = directory / f"ROC曲线_{model_name}{suffix}.png"
        figure.savefig(roc_path, bbox_inches="tight")
        plt.close(figure)
        created.append(str(roc_path))

        precision, recall, _ = precision_recall_curve(labels, probabilities)
        figure, axis = plt.subplots()
        axis.plot(recall, precision)
        axis.set(xlabel="Recall", ylabel="Precision", title=f"Patient-level PR{suffix}")
        pr_path = directory / f"PR曲线_{model_name}{suffix}.png"
        figure.savefig(pr_path, bbox_inches="tight")
        plt.close(figure)
        created.append(str(pr_path))

        if model.penalty == "l1":
            points = model.nested_cv_summary["final_c_grid_auc"]
            figure, axis = plt.subplots()
            axis.semilogx(
                [point["c"] for point in points],
                [point["mean_auc"] for point in points],
            )
            axis.axvline(
                model.nested_cv_summary["final_best_c"], color="red", linestyle="--"
            )
            axis.set(
                xlabel="Inverse regularization strength (C)",
                ylabel="Mean grouped CV AUC",
                title=f"L1 Logistic final grouped CV{suffix}",
            )
            c_auc_path = directory / f"L1_Logistic_CV_AUC{suffix}.png"
            figure.savefig(c_auc_path, bbox_inches="tight")
            plt.close(figure)
            created.append(str(c_auc_path))

        try:
            import shap

            raw = model._prediction_data(row_test_table)
            transformed = model.final_pipeline[:-1].transform(raw)
            estimator = model.final_pipeline.named_steps["model"]
            explainer = shap.LinearExplainer(
                estimator, transformed, feature_names=list(model.feature_columns)
            )
            shap_values = explainer(transformed)
            shap.summary_plot(
                shap_values,
                transformed,
                feature_names=list(model.feature_columns),
                show=False,
            )
            figure = plt.gcf()
            figure.suptitle(f"Sample-level SHAP Summary {model_name}{suffix}")
            summary_path = directory / f"SHAP_Summary_{model_name}{suffix}.png"
            figure.savefig(summary_path, bbox_inches="tight")
            plt.close(figure)
            created.append(str(summary_path))

            if len(transformed):
                shap.plots.waterfall(shap_values[0], show=False)
                figure = plt.gcf()
                figure.suptitle(f"Sample-level SHAP Waterfall {model_name}{suffix}")
                waterfall_path = directory / f"SHAP_Waterfall_{model_name}{suffix}.png"
                figure.savefig(waterfall_path, bbox_inches="tight")
                plt.close(figure)
                created.append(str(waterfall_path))
        except Exception as exc:
            self.diagnostics.append(
                f"{model_name} sample-level SHAP unavailable ({type(exc).__name__})."
            )
        return created


if __name__ == "__main__":
    classifier = KOALassoClassifier(
        train_path="../../datasets/ldse_extracted_features/train_ldse_features_summary.csv",
        test_path="../../datasets/ldse_extracted_features/test_ldse_features_summary.csv",
        output_dir="../results",
    )
    classifier.run()
