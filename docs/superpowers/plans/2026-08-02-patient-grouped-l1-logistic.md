# Patient-Grouped L1 Logistic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace row-level/regression-style LASSO classification with a patient-level, group-isolated L1 Logistic workflow whose preprocessing, tuning, and selection are fit only inside training folds.

**Architecture:** Add a patient-table boundary, a grouped-validation boundary, and a leakage-safe L1 Logistic model, then reduce `KOALassoClassifier` to orchestration and reporting. Preserve the public constructor and MCP parameters while making the training schema authoritative and aggregating augmented test predictions by source patient.

**Tech Stack:** Python 3.12+, pandas 2.3+, NumPy, scikit-learn 1.7+, matplotlib, SHAP, pytest.

## Global Constraints

- Source specification: `docs/superpowers/specs/2026-08-02-patient-grouped-l1-logistic-design.md`.
- Preserve `KOALassoClassifier(train_path, test_path, output_dir, by_region=False).run()` and existing MCP parameter names.
- Analysis unit is patient/source patient; `SourceCaseID` takes precedence over `CaseID` for grouping.
- Training data alone determines feature names, preprocessing parameters, regularization, and selected features.
- Use `StratifiedGroupKFold(shuffle=True, random_state=42)` for inner and outer resampling.
- Use `C_GRID = np.logspace(-4, 4, 17)` and L1 Logistic with `solver="liblinear"`, `max_iter=5000`, `random_state=42`.
- Keep `KLGrade >= 2` as the positive-class definition.
- Do not add pickle/joblib model loading or externally supplied model artifacts.
- Do not expose image paths or full patient lists in errors; report at most ten conflicting identifiers.
- Do not implement calibration, DCA, ICC, IBSI preprocessing, ComBat, or external validation in this cycle.
- The workspace is not a Git repository. Replace commit steps with a verification checkpoint recorded in `.superpowers/sdd/2026-08-02-patient-grouped-l1-logistic/progress.md`.

---

## File Map

| File | Responsibility |
|---|---|
| `src/algorithm/feature_class_algorithm/patient_feature_table.py` | Validate and reshape MEAS/LDSE/Radiomics features into patient-level tables. |
| `src/algorithm/feature_class_algorithm/grouped_validation.py` | Group-overlap audit, stratified-group split creation, and source-patient probability aggregation. |
| `src/algorithm/feature_class_algorithm/leakage_safe_l1_logistic.py` | Nested grouped CV, final L1 Logistic fitting, schema lock, and prediction. |
| `src/algorithm/feature_class_algorithm/classification_artifacts.py` | Atomic JSON, schema, CV summary, coefficients, metrics, and run manifest. |
| `src/algorithm/feature_class_algorithm/koa_lasso_classifier.py` | Backward-compatible orchestration, charts, SHAP, and region dispatch. |
| `src/server/auto_experiment_server.py` | MCP validation and structured return of TXT/CSV/JSON artifacts. |
| `tests/test_patient_feature_table.py` | Patient/Region reshaping and metadata-conflict tests. |
| `tests/test_grouped_validation.py` | Group isolation and aggregation tests. |
| `tests/test_leakage_safe_l1_logistic.py` | Nested-CV, schema-lock, determinism, and L1-selection tests. |
| `tests/test_lasso_patient_pipeline.py` | End-to-end classifier/MCP-compatible output tests. |
| `tests/test_meas_lasso_metadata_compatibility.py` | Existing metadata regression updated for the new classifier internals. |
| `README.md` | Method and migration documentation. |

---

### Task 1: Build the patient-level feature-table boundary

**Files:**
- Create: `src/algorithm/feature_class_algorithm/patient_feature_table.py`
- Create: `tests/test_patient_feature_table.py`
- Create: `.superpowers/sdd/2026-08-02-patient-grouped-l1-logistic/progress.md`

**Interfaces:**
- Consumes: raw `pd.DataFrame` with `CaseID`, `KLGrade`, optional `Region`, optional `SourceCaseID`, and numeric features.
- Produces: `PatientFeatureTable(frame, feature_columns, group_column, source_kind)` and `PatientFeatureTableBuilder.build(df, by_region=False)`.

- [ ] **Step 1: Write the patient-wide Radiomics RED test**

```python
def test_multiregion_radiomics_becomes_one_row_per_case():
    raw = pd.DataFrame({
        "CaseID": ["p2", "p1", "p2", "p1"],
        "Region": [2, 2, 1, 1],
        "KLGrade": [3, 1, 3, 1],
        "shape_volume": [22.0, 12.0, 21.0, 11.0],
        "texture_entropy": [2.2, 1.2, 2.1, 1.1],
    })

    result = PatientFeatureTableBuilder().build(raw)

    assert result.frame["CaseID"].tolist() == ["p1", "p2"]
    assert result.feature_columns == (
        "R1__shape_volume", "R1__texture_entropy",
        "R2__shape_volume", "R2__texture_entropy",
    )
    assert result.frame.loc[0, "R1__shape_volume"] == 11.0
    assert result.frame.loc[1, "R2__texture_entropy"] == 2.2
    assert result.group_column == "_group_id"
    assert result.frame["_group_id"].tolist() == ["p1", "p2"]
```

- [ ] **Step 2: Run the RED test**

Run:

```powershell
python -m pytest tests/test_patient_feature_table.py::test_multiregion_radiomics_becomes_one_row_per_case -v
```

Expected: collection or import fails because `patient_feature_table.py` does not exist.

- [ ] **Step 3: Implement the immutable data contract and deterministic pivot**

```python
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
    def build(self, df: pd.DataFrame, *, by_region: bool = False) -> PatientFeatureTable:
        work = df.copy()
        self._validate_required(work)
        work["CaseID"] = work["CaseID"].astype(str)
        work["KLGrade"] = pd.to_numeric(work["KLGrade"], errors="raise")
        if not work["KLGrade"].between(0, 4).all():
            raise ValueError("KLGrade must contain only values from 0 through 4.")
        work["label"] = (work["KLGrade"] >= 2).astype(int)
        self._attach_group_id(work)
        self._validate_group_labels(work)
        if "Region" in work.columns and not by_region:
            return self._pivot_regions(work)
        return self._build_single_row(work, source_kind="region" if "Region" in work else "single_row")
```

Implement `_pivot_regions()` with `sort_values(["CaseID", "Region"])`, duplicate `(CaseID, Region)` rejection, per-case metadata consistency checks, `pivot(index="CaseID", columns="Region")`, literal column construction `R{region}__{feature}`, and `source_kind="multiregion"`. Use `pd.to_numeric(errors="raise")` for every feature selected by `META_COLUMNS` exclusion.

- [ ] **Step 4: Add conflict and single-row RED tests**

```python
@pytest.mark.parametrize("column,values", [
    ("KLGrade", [1, 2]),
    ("SourceCaseID", ["source-a", "source-b"]),
])
def test_multiregion_patient_metadata_conflict_is_rejected(column, values):
    raw = pd.DataFrame({
        "CaseID": ["p1", "p1"], "Region": [1, 2],
        "KLGrade": [1, 1], "SourceCaseID": ["source-a", "source-a"],
        "feature": [1.0, 2.0],
    })
    raw[column] = values
    with pytest.raises(ValueError, match=column):
        PatientFeatureTableBuilder().build(raw)

def test_duplicate_case_region_is_rejected():
    raw = pd.DataFrame({
        "CaseID": ["p1", "p1"], "Region": [1, 1],
        "KLGrade": [1, 1], "feature": [1.0, 2.0],
    })
    with pytest.raises(ValueError, match="duplicate.*CaseID.*Region"):
        PatientFeatureTableBuilder().build(raw)

def test_meas_or_ldse_table_keeps_one_row_per_case_and_prefers_source_group():
    raw = pd.DataFrame({
        "CaseID": ["p1_aug1", "p2"],
        "SourceCaseID": ["p1", None],
        "KLGrade": [1, 3], "feature": [1.0, 2.0],
    })
    result = PatientFeatureTableBuilder().build(raw)
    assert result.feature_columns == ("feature",)
    assert result.frame["_group_id"].tolist() == ["p1", "p2"]
```

- [ ] **Step 5: Implement conflict checks and run the whole Task 1 file**

Run:

```powershell
python -m pytest tests/test_patient_feature_table.py -v
```

Expected: all Task 1 tests pass.

- [ ] **Step 6: Record the Task 1 checkpoint**

Append the exact test command, pass count, and note that the workspace is non-Git to:

```text
.superpowers/sdd/2026-08-02-patient-grouped-l1-logistic/progress.md
```

---

### Task 2: Enforce group isolation and patient-level probability aggregation

**Files:**
- Create: `src/algorithm/feature_class_algorithm/grouped_validation.py`
- Create: `tests/test_grouped_validation.py`
- Modify: `.superpowers/sdd/2026-08-02-patient-grouped-l1-logistic/progress.md`

**Interfaces:**
- Consumes: label and group Series from `PatientFeatureTable`.
- Produces: `assert_disjoint_groups`, `build_group_cv`, `materialize_group_splits`, and `aggregate_group_probabilities`.

- [ ] **Step 1: Write RED tests for source-patient isolation**

```python
def test_train_test_source_overlap_is_rejected():
    with pytest.raises(ValueError, match="overlap.*source-2"):
        assert_disjoint_groups(
            pd.Series(["source-1", "source-2"]),
            pd.Series(["source-2", "source-3"]),
        )

def test_every_stratified_group_split_keeps_source_patients_isolated():
    y = pd.Series([0, 0, 0, 1, 1, 1, 0, 1])
    groups = pd.Series(["a", "b", "c", "d", "e", "f", "a", "d"])
    splits = materialize_group_splits(y, groups, requested_splits=3)
    for train_idx, valid_idx in splits:
        assert set(groups.iloc[train_idx]).isdisjoint(groups.iloc[valid_idx])
```

- [ ] **Step 2: Run RED and verify missing API failure**

Run:

```powershell
python -m pytest tests/test_grouped_validation.py -k "overlap or isolated" -v
```

Expected: import fails because the grouped-validation module is absent.

- [ ] **Step 3: Implement disjointness and dynamic split construction**

```python
from sklearn.model_selection import StratifiedGroupKFold

def assert_disjoint_groups(train_groups, test_groups) -> None:
    overlap = sorted(set(map(str, train_groups)) & set(map(str, test_groups)))
    if overlap:
        shown = ", ".join(overlap[:10])
        raise ValueError(f"Train/test source-patient overlap detected: {shown}")

def group_counts_per_class(y, groups) -> dict[int, int]:
    pairs = pd.DataFrame({"label": list(y), "group": list(map(str, groups))})
    conflicts = pairs.groupby("group")["label"].nunique()
    if (conflicts > 1).any():
        raise ValueError("A source patient has conflicting binary labels.")
    return pairs.drop_duplicates("group").groupby("label")["group"].nunique().to_dict()

def build_group_cv(y, groups, *, requested_splits: int, minimum_per_class: int):
    counts = group_counts_per_class(y, groups)
    if set(counts) != {0, 1}:
        raise ValueError("Both binary classes are required for grouped validation.")
    n_splits = min(requested_splits, counts[0], counts[1])
    if n_splits < minimum_per_class:
        raise ValueError(
            f"Grouped validation requires at least {minimum_per_class} source patients "
            f"per class; observed {counts}."
        )
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
```

`materialize_group_splits()` must call the following exact boundary and return integer index arrays for deterministic testing and manifest output:

```python
def materialize_group_splits(y, groups, *, requested_splits: int):
    cv = build_group_cv(
        y,
        groups,
        requested_splits=requested_splits,
        minimum_per_class=requested_splits,
    )
    return list(cv.split(np.zeros(len(y)), y, groups))
```

- [ ] **Step 4: Write RED tests for patient-level test aggregation**

```python
def test_augmented_test_probabilities_are_averaged_per_source_patient():
    aggregated = aggregate_group_probabilities(
        groups=pd.Series(["p1", "p1", "p2"]),
        labels=pd.Series([0, 0, 1]),
        probabilities=np.array([0.2, 0.4, 0.8]),
    )
    assert aggregated["group_id"].tolist() == ["p1", "p2"]
    assert aggregated["label"].tolist() == [0, 1]
    assert aggregated["probability"].tolist() == pytest.approx([0.3, 0.8])

def test_group_probability_aggregation_rejects_conflicting_labels():
    with pytest.raises(ValueError, match="conflicting"):
        aggregate_group_probabilities(
            pd.Series(["p1", "p1"]), pd.Series([0, 1]), np.array([0.2, 0.8])
        )
```

- [ ] **Step 5: Implement aggregation and run Task 2 tests**

```python
def aggregate_group_probabilities(groups, labels, probabilities) -> pd.DataFrame:
    frame = pd.DataFrame({
        "group_id": list(map(str, groups)),
        "label": np.asarray(labels, dtype=int),
        "probability": np.asarray(probabilities, dtype=float),
    })
    conflicts = frame.groupby("group_id")["label"].nunique()
    if (conflicts > 1).any():
        raise ValueError("A source patient has conflicting test labels.")
    return (
        frame.groupby("group_id", sort=True, as_index=False)
        .agg(label=("label", "first"), probability=("probability", "mean"))
    )
```

Run:

```powershell
python -m pytest tests/test_grouped_validation.py -v
```

Expected: all Task 2 tests pass.

- [ ] **Step 6: Record the Task 2 checkpoint**

Record command and pass count in the SDD progress ledger.

---

### Task 3: Implement nested grouped L1 Logistic fitting and schema lock

**Files:**
- Create: `src/algorithm/feature_class_algorithm/leakage_safe_l1_logistic.py`
- Create: `tests/test_leakage_safe_l1_logistic.py`
- Modify: `.superpowers/sdd/2026-08-02-patient-grouped-l1-logistic/progress.md`

**Interfaces:**
- Consumes: Task 1 `PatientFeatureTable` and Task 2 grouped-validation functions.
- Produces: `LeakageSafeL1Logistic.fit`, `LeakageSafeL2Baseline.fit`, `predict_proba`, `selected_features`, `coefficients`, `nested_cv_summary`, and `extra_test_features`.

- [ ] **Step 1: Write the RED contract and estimator-type test**

```python
def make_grouped_training_table():
    rows = []
    for index in range(12):
        label = index % 2
        rows.append({
            "CaseID": f"p{index}", "KLGrade": 3 if label else 1,
            "signal": float(label * 4 + index / 100),
            "noise": float((index * 7) % 5),
        })
    return PatientFeatureTableBuilder().build(pd.DataFrame(rows))

def test_final_estimator_is_l1_logistic_not_regression_lasso():
    model = LeakageSafeL1Logistic().fit(make_grouped_training_table())
    estimator = model.final_pipeline.named_steps["model"]
    assert isinstance(estimator, LogisticRegression)
    assert estimator.penalty == "l1"
    assert estimator.solver == "liblinear"
    assert model.selected_features
```

- [ ] **Step 2: Run RED and verify missing model failure**

Run:

```powershell
python -m pytest tests/test_leakage_safe_l1_logistic.py::test_final_estimator_is_l1_logistic_not_regression_lasso -v
```

Expected: import fails because the new model module is absent.

- [ ] **Step 3: Implement the real sklearn Pipeline and inner search factory**

```python
C_GRID = np.logspace(-4, 4, 17)

def _pipeline(*, penalty: str) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            penalty=penalty, solver="liblinear", max_iter=5000, random_state=42,
        )),
    ])

def _search(y, groups, *, penalty: str) -> GridSearchCV:
    inner_cv = build_group_cv(y, groups, requested_splits=5, minimum_per_class=2)
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
```

- [ ] **Step 4: Implement explicit outer nested CV and final refit**

The outer loop must pass outer-training groups with
`search.fit(X.iloc[fit_idx], y.iloc[fit_idx], groups=groups.iloc[fit_idx])`; do not use a helper that drops fit-time groups.

```python
outer_cv = build_group_cv(y, groups, requested_splits=5, minimum_per_class=3)
oof_probability = np.full(len(y), np.nan, dtype=float)
folds = []
for fold_index, (fit_idx, valid_idx) in enumerate(outer_cv.split(X, y, groups), start=1):
    search = _search(y.iloc[fit_idx], groups.iloc[fit_idx], penalty=self.penalty)
    search.fit(X.iloc[fit_idx], y.iloc[fit_idx], groups=groups.iloc[fit_idx])
    oof_probability[valid_idx] = search.predict_proba(X.iloc[valid_idx])[:, 1]
    folds.append({
        "fold": fold_index,
        "best_c": float(search.best_params_["model__C"]),
        "train_groups": int(groups.iloc[fit_idx].nunique()),
        "validation_groups": int(groups.iloc[valid_idx].nunique()),
    })
if np.isnan(oof_probability).any():
    raise RuntimeError("Nested grouped CV did not produce one OOF probability per row.")
final_search = _search(y, groups, penalty=self.penalty)
final_search.fit(X, y, groups=groups)
self.final_pipeline = final_search.best_estimator_
```

Store the ordered training schema before fitting. Reject any training feature that cannot be converted with `pd.to_numeric(errors="raise")` or is entirely missing.

Implement a shared private `_GroupedLogisticBase` containing the nested-CV algorithm. `LeakageSafeL1Logistic.penalty = "l1"` exposes nonzero-coefficient features as `selected_features`; `LeakageSafeL2Baseline.penalty = "l2"` exposes the entire locked training schema as its exploratory feature list. Both tune `C` independently with identical grouped folds.

- [ ] **Step 5: Write RED schema and determinism tests**

```python
def test_test_schema_is_train_locked_extra_ignored_missing_rejected():
    train = make_grouped_training_table()
    model = LeakageSafeL1Logistic().fit(train)
    test_frame = train.frame.copy()
    test_frame["extra_test_only"] = 99.0
    test = PatientFeatureTable(
        frame=test_frame,
        feature_columns=train.feature_columns + ("extra_test_only",),
        group_column=train.group_column,
        source_kind=train.source_kind,
    )
    probabilities = model.predict_proba(test)
    assert len(probabilities) == len(test.frame)
    assert model.extra_test_features == ("extra_test_only",)

    missing_frame = test_frame.drop(columns=[train.feature_columns[0]])
    missing = PatientFeatureTable(
        missing_frame,
        tuple(c for c in test.feature_columns if c != train.feature_columns[0]),
        train.group_column,
        train.source_kind,
    )
    with pytest.raises(ValueError, match=train.feature_columns[0]):
        model.predict_proba(missing)

def test_fixed_seed_repeats_folds_c_features_and_probabilities():
    table = make_grouped_training_table()
    first = LeakageSafeL1Logistic().fit(table)
    second = LeakageSafeL1Logistic().fit(table)
    assert first.nested_cv_summary == second.nested_cv_summary
    assert first.selected_features == second.selected_features
    assert first.predict_proba(table) == pytest.approx(second.predict_proba(table))

def test_exploratory_all_feature_baseline_is_grouped_l2_logistic():
    table = make_grouped_training_table()
    baseline = LeakageSafeL2Baseline().fit(table)
    estimator = baseline.final_pipeline.named_steps["model"]
    assert estimator.penalty == "l2"
    assert baseline.selected_features == table.feature_columns
```

- [ ] **Step 6: Write the test-only-distribution isolation regression**

Fit two models on the identical training table. Predict once on a normal test table and once on a copy containing extreme values. Assert the fitted imputer statistics, scaler means/scales, final `C`, and selected features are identical between the two independently fitted models. This protects against future code that concatenates train and test before preprocessing.

```python
def fitted_state(model):
    pipe = model.final_pipeline
    return (
        tuple(pipe.named_steps["imputer"].statistics_),
        tuple(pipe.named_steps["scaler"].mean_),
        tuple(pipe.named_steps["scaler"].scale_),
        float(pipe.named_steps["model"].C),
        model.selected_features,
    )
```

- [ ] **Step 7: Run Task 3 tests and record checkpoint**

Run:

```powershell
python -m pytest tests/test_leakage_safe_l1_logistic.py -v
```

Expected: all Task 3 tests pass with real sklearn estimators and no estimator mocks.

Record the exact output in the SDD ledger.

---

### Task 4: Create reproducible classification artifacts and patient-level metrics

**Files:**
- Create: `src/algorithm/feature_class_algorithm/classification_artifacts.py`
- Create: `tests/test_classification_artifacts.py`
- Modify: `.superpowers/sdd/2026-08-02-patient-grouped-l1-logistic/progress.md`

**Interfaces:**
- Consumes: train/test paths, patient tables, fitted `LeakageSafeL1Logistic`, aggregated patient probabilities, and output directory.
- Produces: atomic JSON/CSV artifacts and metric dictionaries used by `KOALassoClassifier`.

Use this exact writer boundary:

```python
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
) -> dict[str, str]:
    """Publish schema, nested-CV summary, coefficients, and manifest atomically."""
```

- [ ] **Step 1: Write RED tests for metrics at patient/source-patient level**

```python
def test_binary_metrics_use_aggregated_patients_not_augmented_rows():
    aggregated = pd.DataFrame({
        "group_id": ["p1", "p2", "p3", "p4"],
        "label": [0, 0, 1, 1],
        "probability": [0.1, 0.4, 0.6, 0.9],
    })
    metrics = compute_binary_metrics(aggregated)
    assert metrics["NumPatients"] == 4
    assert metrics["AUC"] == pytest.approx(1.0)
    assert metrics["Sensitivity"] == pytest.approx(1.0)
    assert metrics["Specificity"] == pytest.approx(1.0)
```

- [ ] **Step 2: Run RED, then implement metric calculation**

Use the fixed threshold `0.5` in this cycle. Return AUC, PR-AUC, accuracy, precision, sensitivity, specificity, F1, confusion-matrix counts, and patient count. Require both classes before calculating AUC.

```python
def compute_binary_metrics(aggregated: pd.DataFrame) -> dict:
    y = aggregated["label"].to_numpy(dtype=int)
    probability = aggregated["probability"].to_numpy(dtype=float)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Patient-level evaluation requires both binary classes.")
    prediction = (probability >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {
        "NumPatients": int(len(y)),
        "AUC": float(roc_auc_score(y, probability)),
        "PR_AUC": float(average_precision_score(y, probability)),
        "Accuracy": float(accuracy_score(y, prediction)),
        "Precision": float(precision_score(y, prediction, zero_division=0)),
        "Sensitivity": float(recall_score(y, prediction, zero_division=0)),
        "Specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "F1": float(f1_score(y, prediction, zero_division=0)),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }
```

- [ ] **Step 3: Write RED tests for schema, CV summary, coefficients, and manifest**

```python
def test_artifact_writer_emits_parseable_atomic_reproducibility_files(tmp_path):
    train_rows = [
        {
            "CaseID": f"train-{index}",
            "KLGrade": 3 if index % 2 else 1,
            "signal": float((index % 2) * 10 + index / 100),
            "noise": float(index % 3),
        }
        for index in range(12)
    ]
    test_rows = [
        {
            "CaseID": f"test-{index}",
            "KLGrade": 3 if index % 2 else 1,
            "signal": float((index % 2) * 10 + index / 100),
            "noise": float(index % 3),
        }
        for index in range(8)
    ]
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    pd.DataFrame(train_rows).to_csv(train_path, index=False)
    pd.DataFrame(test_rows).to_csv(test_path, index=False)
    builder = PatientFeatureTableBuilder()
    train_table = builder.build(pd.DataFrame(train_rows))
    test_table = builder.build(pd.DataFrame(test_rows))
    fitted_model = LeakageSafeL1Logistic().fit(train_table)

    paths = write_classification_artifacts(
        output_dir=tmp_path,
        suffix="",
        train_path=train_path,
        test_path=test_path,
        train_table=train_table,
        test_table=test_table,
        model=fitted_model,
        metrics={"AUC": 0.75, "NumPatients": 8},
    )
    schema = json.loads(Path(paths["schema"]).read_text(encoding="utf-8"))
    cv = json.loads(Path(paths["nested_cv"]).read_text(encoding="utf-8"))
    manifest = json.loads(Path(paths["manifest"]).read_text(encoding="utf-8"))
    coefficients = pd.read_csv(paths["coefficients"])
    assert schema["feature_columns"] == list(train_table.feature_columns)
    assert cv["final_best_c"] == fitted_model.nested_cv_summary["final_best_c"]
    assert set(coefficients.columns) == {"feature", "coefficient", "selected"}
    assert manifest["train_sha256"]
    assert manifest["test_sha256"]
    assert not list(tmp_path.glob("*.tmp"))
```

- [ ] **Step 4: Implement safe serialization**

Use a unique temporary sibling file, UTF-8 JSON with `ensure_ascii=False`, `sort_keys=True`, `indent=2`, flush/fsync, then `os.replace`. CSV writes use the same temporary-then-replace pattern. The manifest records absolute paths and SHA-256 but errors and user-visible diagnostics do not print image paths or unrestricted identifiers.

```python
def _write_json_atomic(path: Path, payload: dict) -> None:
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temp.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    except Exception:
        temp.unlink(missing_ok=True)
        raise
```

The schema contains feature order, label rule, region-pivot rule, group column, source kind, and ignored extra test features. The manifest contains input hashes, package versions, random seed, row/patient/group/class counts, final `C`, selected-feature count, and artifact names.

- [ ] **Step 5: Run Task 4 tests and checkpoint**

Run:

```powershell
python -m pytest tests/test_classification_artifacts.py -v
```

Expected: all Task 4 tests pass and no temporary files remain.

Record evidence in the progress ledger.

---

### Task 5: Integrate the new workflow behind `KOALassoClassifier` and MCP

**Files:**
- Modify: `src/algorithm/feature_class_algorithm/koa_lasso_classifier.py`
- Modify: `src/server/auto_experiment_server.py:360-461`
- Create: `tests/test_lasso_patient_pipeline.py`
- Modify: `tests/test_meas_lasso_metadata_compatibility.py`
- Modify: `.superpowers/sdd/2026-08-02-patient-grouped-l1-logistic/progress.md`

**Interfaces:**
- Consumes: Tasks 1–4 APIs.
- Produces: unchanged `KOALassoClassifier` constructor/run contract and unchanged MCP inputs with richer output artifacts.

- [ ] **Step 1: Write the end-to-end patient-wide RED test**

Create train/test CSVs with six source patients per class, two Regions per `CaseID`, and two informative features. Run the real classifier while replacing only the rendering boundary, not the estimator.

```python
def write_multiregion_csv(path, *, start: int, patients: int) -> Path:
    rows = []
    for offset in range(patients):
        label = offset % 2
        case_id = f"case-{start + offset}"
        for region in (1, 2):
            rows.append({
                "CaseID": case_id,
                "SourceCaseID": case_id,
                "Region": region,
                "KLGrade": 3 if label else 1,
                "signal": float(label * 20 + region + offset / 100),
                "texture": float(label * 10 + region * 0.5 + offset / 100),
            })
    pd.DataFrame(rows).to_csv(path, index=False)
    return path

def test_multiregion_classifier_reports_patient_count_not_region_rows(tmp_path):
    train_path = write_multiregion_csv(tmp_path / "train.csv", start=0, patients=12)
    test_path = write_multiregion_csv(tmp_path / "test.csv", start=100, patients=8)
    output = tmp_path / "out"

    classifier = KOALassoClassifier(
        str(train_path), str(test_path), str(output), by_region=False
    )
    classifier._render_plots_and_shap = lambda *args, **kwargs: []
    classifier.run()

    metrics = pd.read_csv(output / "metrics_summary.csv")
    assert metrics.loc[metrics["Model"] == "LASSO", "NumPatients"].item() == 8
    schema = json.loads((output / "patient_feature_schema.json").read_text())
    assert schema["source_kind"] == "multiregion"
    assert all(name.startswith(("R1__", "R2__")) for name in schema["feature_columns"])
```

- [ ] **Step 2: Run RED and confirm old row-level result or missing artifact failure**

Run:

```powershell
python -m pytest tests/test_lasso_patient_pipeline.py::test_multiregion_classifier_reports_patient_count_not_region_rows -v
```

Expected: failure because the current classifier treats Region rows as samples and does not write the schema JSON.

- [ ] **Step 3: Replace classifier internals while preserving its public API**

`run()` must:

1. Read CSVs with `dtype={"CaseID": str, "SourceCaseID": str}` when those columns exist.
2. For `by_region=False`, build train/test patient tables once and call `_run_patient_model(train_table, test_table, output_dir, region_name=None)`.
3. For `by_region=True`, lock the sorted training Region list; reject missing test Regions, ignore/record extra test Regions, and call `_run_patient_model(train_table, test_table, region_output, region_name=region)` once per training Region.
4. Call `assert_disjoint_groups` before model fitting.
5. Fit `LeakageSafeL1Logistic` only on the train patient table.
6. Predict row probabilities for test, aggregate by group, compute patient metrics, and write artifacts.
7. Populate `self.results` and `self.diagnostics` for the existing MCP response.

Remove `LassoCV`, direct `StandardScaler.fit_transform`, and train/test feature-intersection logic from `koa_lasso_classifier.py`.

- [ ] **Step 4: Preserve charts, report text, coefficients, and SHAP at the final-model boundary**

Replace `_evaluate_and_plot` with `_render_plots_and_shap(aggregated, row_test_table, model, output_dir, suffix) -> list[str]`. ROC/PR plots use `aggregated["label"]` and `aggregated["probability"]`. SHAP uses transformed test rows before group aggregation and labels the plot as sample-level explanation; global performance remains patient-level. Replace the regression MSE plot with `L1_Logistic_CV_AUC{suffix}.png` built from `cv_results_["mean_test_score"]` and `param_model__C` stored by the final search.

Keep the legacy filenames listed in the spec and add the new JSON/CSV files. The primary metrics row uses `Model="LASSO"`; the all-feature L2 baseline uses `Model="ALL_EXPLORATORY"` and the same grouped preprocessing/tuning boundary.

- [ ] **Step 5: Update MCP validation to allow train-fitted imputation**

In `auto_experiment_server.py`:

- Keep file existence, label presence, numeric-feature validation, and metadata exclusion.
- Remove the early NaN rejection at current lines 397–410.
- Add diagnostic counts such as `train: 3 missing feature values; imputation will be fit inside training folds` without exposing row identifiers.
- Catch `ValueError` from `clf.run()` and return `status="failed"`, the safe diagnostic message, `charts=[]`, `txt_files=[]`, and accumulated diagnostics.
- Include `.json` files in `txt_files` so Agent callers receive schema/CV/manifest paths.
- Replace the MCP docstring description of `LassoCV` with patient-grouped L1 Logistic and nested grouped CV.

- [ ] **Step 6: Rewrite the obsolete mocked LassoCV metadata test**

Replace `test_lasso_never_uses_numeric_snake_case_path_metadata` with a real builder/classifier-boundary test. Keep the numeric `image_path`/`mask_path` fixture and assert neither appears in `patient_feature_schema.json` nor `model_coefficients.csv`. Do not mock `LassoCV`, `StandardScaler`, or `LogisticRegression`.

- [ ] **Step 7: Add integration error tests**

```python
def run_without_plots(train_path, test_path, output):
    classifier = KOALassoClassifier(str(train_path), str(test_path), str(output))
    classifier._render_plots_and_shap = lambda *args, **kwargs: []
    classifier.run()
    return classifier

def test_classifier_rejects_train_test_source_patient_overlap(tmp_path):
    train = write_multiregion_csv(tmp_path / "train.csv", start=0, patients=12)
    test = write_multiregion_csv(tmp_path / "test.csv", start=100, patients=8)
    test_df = pd.read_csv(test, dtype={"CaseID": str, "SourceCaseID": str})
    test_df.loc[test_df["CaseID"] == "case-100", "SourceCaseID"] = "case-0"
    test_df.to_csv(test, index=False)
    with pytest.raises(ValueError, match="overlap.*case-0"):
        run_without_plots(train, test, tmp_path / "out")
    assert not (tmp_path / "out" / "metrics_summary.csv").exists()

def test_classifier_rejects_missing_test_feature_instead_of_intersection(tmp_path):
    train = write_multiregion_csv(tmp_path / "train.csv", start=0, patients=12)
    test = write_multiregion_csv(tmp_path / "test.csv", start=100, patients=8)
    pd.read_csv(test).drop(columns=["texture"]).to_csv(test, index=False)
    with pytest.raises(ValueError, match="R1__texture|R2__texture"):
        run_without_plots(train, test, tmp_path / "out")

def test_classifier_ignores_extra_test_feature_and_records_it(tmp_path):
    train = write_multiregion_csv(tmp_path / "train.csv", start=0, patients=12)
    test = write_multiregion_csv(tmp_path / "test.csv", start=100, patients=8)
    test_df = pd.read_csv(test)
    test_df["test_only"] = 99.0
    test_df.to_csv(test, index=False)
    run_without_plots(train, test, tmp_path / "out")
    schema = json.loads((tmp_path / "out" / "patient_feature_schema.json").read_text())
    assert schema["ignored_extra_test_features"] == ["R1__test_only", "R2__test_only"]

def test_classifier_rejects_region_label_conflict(tmp_path):
    train = write_multiregion_csv(tmp_path / "train.csv", start=0, patients=12)
    test = write_multiregion_csv(tmp_path / "test.csv", start=100, patients=8)
    train_df = pd.read_csv(train)
    train_df.loc[
        (train_df["CaseID"] == "case-0") & (train_df["Region"] == 2),
        "KLGrade",
    ] = 3
    train_df.to_csv(train, index=False)
    with pytest.raises(ValueError, match="KLGrade"):
        run_without_plots(train, test, tmp_path / "out")

def test_classifier_rejects_insufficient_groups_per_class(tmp_path):
    train = write_multiregion_csv(tmp_path / "train.csv", start=0, patients=4)
    test = write_multiregion_csv(tmp_path / "test.csv", start=100, patients=4)
    with pytest.raises(ValueError, match="at least 3 source patients per class"):
        run_without_plots(train, test, tmp_path / "out")

def test_classifier_aggregates_augmented_test_rows_by_source_case(tmp_path):
    train = write_multiregion_csv(tmp_path / "train.csv", start=0, patients=12)
    test = write_multiregion_csv(tmp_path / "test.csv", start=100, patients=8)
    test_df = pd.read_csv(test, dtype={"CaseID": str, "SourceCaseID": str})
    replacements = {
        "case-100": "source-negative-a", "case-102": "source-negative-a",
        "case-104": "source-negative-b", "case-106": "source-negative-b",
        "case-101": "source-positive-a", "case-103": "source-positive-a",
        "case-105": "source-positive-b", "case-107": "source-positive-b",
    }
    test_df["SourceCaseID"] = test_df["CaseID"].map(replacements)
    test_df.to_csv(test, index=False)
    run_without_plots(train, test, tmp_path / "out")
    metrics = pd.read_csv(tmp_path / "out" / "metrics_summary.csv")
    assert metrics.loc[metrics["Model"] == "LASSO", "NumPatients"].item() == 4
```

Each test must assert a user-actionable message and ensure no misleading metrics file was published after failure.

- [ ] **Step 8: Run Task 5 focused regression**

Run:

```powershell
python -m pytest tests/test_lasso_patient_pipeline.py tests/test_meas_lasso_metadata_compatibility.py tests/test_integration_feature_quality.py -v
```

Expected: all selected tests pass; no fake estimator remains in the MEAS compatibility suite.

- [ ] **Step 9: Record Task 5 checkpoint**

Record exact pass count, command, and any dependency warnings in the progress ledger.

---

### Task 6: Documentation, migration, and final verification

**Files:**
- Modify: `README.md:50-60,220-240`
- Modify: `docs/superpowers/specs/2026-08-02-patient-grouped-l1-logistic-design.md` only if implementation requires a user-approved clarification; otherwise leave it unchanged.
- Modify: `.superpowers/sdd/2026-08-02-patient-grouped-l1-logistic/progress.md`
- Create: `.superpowers/sdd/2026-08-02-patient-grouped-l1-logistic/final-report.md`

**Interfaces:**
- Consumes: all implementation and test evidence.
- Produces: accurate user documentation, verification record, and explicit environment limitations.

- [ ] **Step 1: Update README method and migration text**

Document these exact behavior changes:

- Radiomics `by_region=False` pivots Region features into one row per `CaseID`.
- `SourceCaseID` prevents augmented siblings from crossing CV folds.
- Median imputation, scaling, tuning, and L1 selection occur only inside training folds.
- Training schema is authoritative; missing test features fail and extra test features are ignored/recorded.
- `LassoCV` regression MSE plot is replaced by an L1 Logistic C–AUC plot.
- Test metrics are patient/source-patient level.
- New schema, nested-CV, coefficient, and manifest outputs are listed.

- [ ] **Step 2: Compile every changed production module**

Run:

```powershell
python -m py_compile src\algorithm\feature_class_algorithm\patient_feature_table.py src\algorithm\feature_class_algorithm\grouped_validation.py src\algorithm\feature_class_algorithm\leakage_safe_l1_logistic.py src\algorithm\feature_class_algorithm\classification_artifacts.py src\algorithm\feature_class_algorithm\koa_lasso_classifier.py src\server\auto_experiment_server.py
```

Expected: exit code 0 with no output.

- [ ] **Step 3: Run the complete new focused suite**

Run:

```powershell
python -m pytest tests/test_patient_feature_table.py tests/test_grouped_validation.py tests/test_leakage_safe_l1_logistic.py tests/test_classification_artifacts.py tests/test_lasso_patient_pipeline.py tests/test_meas_lasso_metadata_compatibility.py tests/test_integration_feature_quality.py -v
```

Expected: zero failures and zero errors.

- [ ] **Step 4: Run existing LDSE/Agent regression suites**

Run:

```powershell
python -m pytest tests/test_ldse_train_fitted_pipeline.py tests/test_feature_extraction_agent_plan_repair.py tests/test_agent_feature_paths.py tests/test_agent_prompt_language.py tests/test_tool_result_handoff.py -v
```

Expected: zero failures and zero errors.

- [ ] **Step 5: Run the dependency-collectable broad suite**

Run:

```powershell
python -m pytest -q --ignore=tests/test_classification_agent_path_scoping.py --ignore=tests/test_data_augmentation.py --ignore=tests/test_feature_extraction_agent_executes_plan.py --ignore=tests/test_integration_augmentation.py
```

Expected: zero failures. Report the exact pass count; do not reuse prior counts.

- [ ] **Step 6: Attempt full collection and record external blockers exactly**

Run:

```powershell
python -m pytest -q
```

If collection is blocked by missing `langchain_core`, `SimpleITK`, Python 3.12, or another environment dependency, copy the exact file/module blockers into `final-report.md`. Do not count blocked files as passing.

- [ ] **Step 7: Run static leakage-boundary scans**

Run:

```powershell
rg -n "LassoCV|_common_feats|_train_feats\s*&\s*_test_feats|fit_transform\(X_test|concat\(.*train.*test" src\algorithm\feature_class_algorithm src\server\auto_experiment_server.py
```

Expected: no production `LassoCV`, train/test feature intersection, test-set fitting, or train/test concatenation in the classification scope. Any documentation-only match must be adjudicated in the final report.

- [ ] **Step 8: Verify requirements line by line**

Read the design spec and record PASS/FAIL evidence for:

1. patient-wide Region pivot;
2. `SourceCaseID` grouping;
3. nested stratified group CV;
4. fold-local imputation/scaling/L1 selection;
5. train-locked schema;
6. train/test group overlap rejection;
7. patient-level test aggregation;
8. backward-compatible MCP inputs;
9. required legacy and new artifacts;
10. no unsafe model loader.

- [ ] **Step 9: Perform a scoped independent code review**

Review only the files in this plan against the design specification. Treat any Critical or Important leakage, patient-unit, schema, serialization, or evaluation finding as blocking. Fix findings with a new failing regression before changing production code, then rerun Steps 2–8.

- [ ] **Step 10: Close the progress ledger only with fresh evidence**

Write `final-report.md` with exact commands, exit codes, pass counts, environment blockers, changed files, review findings, and deferred non-goals. Mark the optimization complete only when no required work remains.
