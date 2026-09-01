from dataclasses import dataclass

import pandas as pd


META_COLUMNS = {
    "CaseID", "KLGrade", "Region", "SourceCaseID", "ImagePath", "MaskPath",
    "image_path", "mask_path", "KneeSide", "Gender", "Age", "BMI",
    "Augmented", "image_id", "label", "_group_id",
}


@dataclass(frozen=True)
class PatientFeatureTable:
    frame: pd.DataFrame
    feature_columns: tuple[str, ...]
    group_column: str = "_group_id"
    source_kind: str = "single_row"


class PatientFeatureTableBuilder:
    def build(
        self,
        df: pd.DataFrame,
        *,
        by_region: bool = False,
        allow_repeated_groups: bool = False,
    ) -> PatientFeatureTable:
        work = df.copy()
        self._validate_required(work)
        self._validate_case_ids(work)
        self._validate_regions(work)
        work["CaseID"] = work["CaseID"].astype(str)
        raw_grades = work["KLGrade"]
        grades = pd.to_numeric(raw_grades, errors="coerce")
        invalid_grades = grades.isna() | ~grades.between(0, 4)
        if invalid_grades.any():
            raise ValueError(
                f"KLGrade contains {int(invalid_grades.sum())} invalid value(s); expected 0 through 4."
            )
        work["KLGrade"] = grades
        if not work["KLGrade"].between(0, 4).all():
            raise ValueError("KLGrade must contain only values from 0 through 4.")
        work["label"] = (work["KLGrade"] >= 2).astype(int)
        self._attach_group_id(work)
        self._validate_group_labels(work)
        self._validate_group_grades(work)
        if "Region" in work.columns and not by_region:
            if not allow_repeated_groups:
                self._validate_multiregion_groups(work)
            return self._pivot_regions(work)
        if "Region" in work:
            self._validate_case_region_uniqueness(work)
            if not allow_repeated_groups:
                self._validate_region_groups(work)
        elif not allow_repeated_groups:
            self._validate_unique_groups(work)
        else:
            self._validate_unique_cases(work)
        return self._build_single_row(
            work,
            source_kind="region" if "Region" in work else "single_row",
        )

    @staticmethod
    def _validate_required(work: pd.DataFrame) -> None:
        missing = [column for column in ("CaseID", "KLGrade") if column not in work]
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(missing)}.")

    @staticmethod
    def _validate_case_ids(work: pd.DataFrame) -> None:
        values = work["CaseID"]
        invalid = values.isna() | values.astype("string").str.strip().eq("").fillna(True)
        count = int(invalid.sum())
        if count:
            raise ValueError(
                f"CaseID contains {count} null or whitespace-only value(s)."
            )

    @staticmethod
    def _validate_regions(work: pd.DataFrame) -> None:
        if "Region" not in work:
            return
        values = work["Region"]
        invalid = values.isna() | values.astype("string").str.strip().eq("").fillna(True)
        if invalid.any():
            raise ValueError(
                f"Region contains {int(invalid.sum())} null or whitespace-only value(s)."
            )

    @staticmethod
    def _attach_group_id(work: pd.DataFrame) -> None:
        if "SourceCaseID" in work:
            source_ids = work["SourceCaseID"]
            use_source = source_ids.notna() & source_ids.astype(str).str.strip().ne("")
            work["_group_id"] = work["CaseID"]
            work.loc[use_source, "_group_id"] = source_ids.loc[use_source].astype(str)
        else:
            work["_group_id"] = work["CaseID"]

    def _validate_group_labels(self, work: pd.DataFrame) -> None:
        conflicting = work.groupby("_group_id", sort=True)["label"].nunique()
        conflicting = conflicting[conflicting > 1].index.tolist()
        if conflicting:
            self._raise_conflict("KLGrade", conflicting)

    def _validate_group_grades(self, work: pd.DataFrame) -> None:
        conflicting = work.groupby("_group_id", sort=True)["KLGrade"].nunique()
        conflicting = conflicting[conflicting > 1].index.tolist()
        if conflicting:
            self._raise_conflict("KLGrade", conflicting)

    def _validate_unique_groups(self, work: pd.DataFrame) -> None:
        duplicated = work.duplicated("_group_id", keep=False)
        if duplicated.any():
            identifiers = work.loc[duplicated, "_group_id"].drop_duplicates().tolist()
            self._raise_conflict("duplicate _group_id", identifiers)

    def _validate_multiregion_groups(self, work: pd.DataFrame) -> None:
        case_counts = work.groupby("_group_id", sort=True)["CaseID"].nunique()
        identifiers = case_counts[case_counts > 1].index.tolist()
        if identifiers:
            self._raise_conflict("duplicate _group_id", identifiers)

    def _validate_unique_cases(self, work: pd.DataFrame) -> None:
        duplicated = work.duplicated("CaseID", keep=False)
        if duplicated.any():
            identifiers = work.loc[duplicated, "CaseID"].drop_duplicates().tolist()
            self._raise_conflict("duplicate CaseID", identifiers)

    def _validate_case_region_uniqueness(self, work: pd.DataFrame) -> None:
        duplicated = work.duplicated(["CaseID", "Region"], keep=False)
        if duplicated.any():
            identifiers = work.loc[duplicated, "CaseID"].drop_duplicates().tolist()
            self._raise_conflict("duplicate CaseID and Region", identifiers)

    def _validate_region_groups(self, work: pd.DataFrame) -> None:
        duplicated = work.duplicated(["Region", "_group_id"], keep=False)
        if duplicated.any():
            identifiers = work.loc[duplicated, "_group_id"].drop_duplicates().tolist()
            self._raise_conflict("duplicate _group_id", identifiers)

    def _pivot_regions(self, work: pd.DataFrame) -> PatientFeatureTable:
        work = work.sort_values(["CaseID", "Region"])
        self._validate_case_region_uniqueness(work)

        self._validate_case_metadata(work)
        features = self._feature_columns(work)
        pivoted = work.pivot(index="CaseID", columns="Region", values=features)
        regions = list(pivoted.columns.levels[1])
        ordered_pivot_columns = pd.MultiIndex.from_tuples(
            [(feature, region) for region in regions for feature in features]
        )
        pivoted = pivoted.reindex(columns=ordered_pivot_columns)
        feature_columns = tuple(
            f"R{region}__{feature}" for region in regions for feature in features
        )
        pivoted.columns = feature_columns
        metadata_columns = ["KLGrade", "label", "_group_id"]
        if "SourceCaseID" in work:
            metadata_columns.insert(1, "SourceCaseID")
        metadata = work.drop_duplicates("CaseID").set_index("CaseID")[metadata_columns]
        frame = metadata.join(pivoted).reset_index()
        return PatientFeatureTable(frame, feature_columns, source_kind="multiregion")

    def _build_single_row(
        self, work: pd.DataFrame, *, source_kind: str
    ) -> PatientFeatureTable:
        features = self._feature_columns(work)
        frame = work.sort_values("CaseID").reset_index(drop=True)
        return PatientFeatureTable(frame, tuple(features), source_kind=source_kind)

    @staticmethod
    def _feature_columns(work: pd.DataFrame) -> list[str]:
        metadata = {column.casefold() for column in META_COLUMNS}
        features = [column for column in work.columns if column.casefold() not in metadata]
        for feature in features:
            try:
                work[feature] = pd.to_numeric(work[feature], errors="raise")
            except (TypeError, ValueError) as exc:
                safe_feature = PatientFeatureTableBuilder._safe_identifier(feature)
                raise ValueError(f"Feature '{safe_feature}' must be numeric.") from exc
        return features

    def _validate_case_metadata(self, work: pd.DataFrame) -> None:
        for column in ("KLGrade", "SourceCaseID", "_group_id", "label"):
            if column not in work:
                continue
            conflicting = work.groupby("CaseID", sort=True)[column].nunique(dropna=False)
            identifiers = conflicting[conflicting > 1].index.tolist()
            if identifiers:
                self._raise_conflict(column, identifiers)

    @staticmethod
    def _raise_conflict(kind: str, identifiers: list[object]) -> None:
        shown = ", ".join(
            PatientFeatureTableBuilder._safe_identifier(value)
            for value in identifiers[:10]
        )
        suffix = " ..." if len(identifiers) > 10 else ""
        raise ValueError(f"{kind} conflict for CaseID values: {shown}{suffix}")

    @staticmethod
    def _safe_identifier(value: object) -> str:
        rendered = str(value)
        lowered = rendered.lower()
        if (
            "/" in rendered
            or "\\" in rendered
            or (("image" in lowered or "mask" in lowered) and "." in rendered)
        ):
            return "<redacted>"
        return rendered
