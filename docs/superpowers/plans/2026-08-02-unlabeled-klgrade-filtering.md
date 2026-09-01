# Unlabeled KLGrade Filtering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Exclude only `KLGrade=-1` cases before MEAS/LDSE extraction and at the LASSO boundary, keep every sample count dynamic, and prevent the ClassificationAgent from reading a report after classification fails.

**Architecture:** Add one dependency-light KL-grade policy used by MEAS, LDSE, and LASSO. Apply it before any image/mask access or train-fitted LDSE preprocessing, preserve strict 0–4 validation downstream, and expose privacy-safe aggregate counts through diagnostics/manifests without changing public tool parameters.

**Tech Stack:** Python 3.11+, pandas, NumPy, scikit-learn, SimpleITK (production extraction), pytest, LangChain message types (agent integration).

## Global Constraints

- Source specification: `docs/superpowers/specs/2026-08-02-unlabeled-klgrade-filtering-design.md`.
- `KLGrade=-1` means unlabeled and must be excluded, never remapped to class 0 or class 1.
- Valid labeled grades remain the integers 0, 1, 2, 3, and 4; `KLGrade >= 2` remains the positive-class rule.
- Empty, nonnumeric, nonfinite, noninteger, `< -1`, and `> 4` grades must fail with aggregate-only messages.
- Filter before image/mask existence checks, MEAS extraction, LDSE scaler/PCA fitting, LDSE transform, patient-table construction, overlap checks, or model fitting.
- LDSE scaler/PCA fitting uses only labeled training cases; held-out test cases are transformed only with the frozen training artifact.
- Preserve current MEAS, LDSE, LASSO, and MCP parameter/return-path contracts.
- Keep original input-file SHA-256 provenance; do not rewrite user CSV files.
- Do not expose CaseID, ImagePath, MaskPath, raw invalid values, or patient-level exclusion lists in new diagnostics.
- Counts are dynamic. The current 383 train and 98 test cases are data-specific acceptance values, never production constants.
- Preserve transactional output behavior and reject non-empty output directory reuse.
- The workspace is not a Git repository. Replace commit steps with verification checkpoints in `.superpowers/sdd/2026-08-02-unlabeled-klgrade-filtering/progress.md`.

---

## File Map

| File | Responsibility |
|---|---|
| `src/algorithm/klgrade_policy.py` | Single privacy-safe contract for validating grades and selecting labeled rows. |
| `src/algorithm/KOA_feature_extraction/meas_feature_extraction.py` | Apply the policy before MEAS file checks/extraction and publish dynamic run counts. |
| `src/algorithm/KOA_feature_extraction/ldse_feature_extraction.py` | Apply the policy before train fitting and transform; distinguish unlabeled exclusions from extraction skips in artifacts. |
| `src/server/auto_experiment_server.py` | Print aggregate MEAS counts while preserving tool returns; retain structured LASSO failure results. |
| `src/algorithm/feature_class_algorithm/koa_lasso_classifier.py` | Defensively filter legacy feature CSVs before patient-table/model work and record diagnostics. |
| `src/agents/classification_agent.py` | Treat only explicit classification success as report-readable and stop the downstream report call on failure. |
| `tests/test_klgrade_policy.py` | Unit tests for accepted, excluded, rejected, immutable, and privacy-safe policy behavior. |
| `tests/test_meas_lasso_metadata_compatibility.py` | MEAS prefilter and no-path-access regressions. |
| `tests/test_ldse_train_fitted_pipeline.py` | LDSE train-only fitting, transform filtering, manifest counts, and rollback regressions. |
| `tests/test_lasso_patient_pipeline.py` | Dynamic filtering and patient-count/artifact regressions across feature-table styles. |
| `tests/test_classification_agent_path_scoping.py` | Classification result status parsing and report-call gating regressions. |
| `README.md` | User-facing unlabeled-label policy and new-data rerun instructions. |
| `.superpowers/sdd/2026-08-02-unlabeled-klgrade-filtering/progress.md` | Non-Git execution checkpoints and exact verification evidence. |

---

### Task 1: Create the shared KL-grade selection boundary

**Files:**
- Create: `src/algorithm/klgrade_policy.py`
- Create: `tests/test_klgrade_policy.py`
- Create: `.superpowers/sdd/2026-08-02-unlabeled-klgrade-filtering/progress.md`

**Interfaces:**
- Consumes: `select_labeled_klgrade_rows(frame: pd.DataFrame, *, dataset_role: str) -> LabeledKLGradeRows`.
- Produces: immutable `LabeledKLGradeRows(frame, input_rows, labeled_rows, excluded_unlabeled_rows)` used by Tasks 2–4.

- [ ] **Step 1: Write RED tests for selection, immutability, and dynamic counts**

```python
import numpy as np
import pandas as pd
import pytest

from src.algorithm.klgrade_policy import select_labeled_klgrade_rows


def test_selects_only_labeled_rows_without_mutating_input():
    source = pd.DataFrame({
        "CaseID": ["a", "b", "c", "d"],
        "KLGrade": ["-1", 0, 2.0, 4],
        "feature": [10.0, 11.0, 12.0, 13.0],
    })
    before = source.copy(deep=True)

    result = select_labeled_klgrade_rows(source, dataset_role="train")

    pd.testing.assert_frame_equal(source, before)
    assert result.frame["CaseID"].tolist() == ["b", "c", "d"]
    assert result.frame["KLGrade"].tolist() == [0, 2, 4]
    assert (result.input_rows, result.labeled_rows, result.excluded_unlabeled_rows) == (4, 3, 1)


def test_counts_change_with_input_instead_of_using_dataset_constants():
    frame = pd.DataFrame({
        "KLGrade": ([-1] * 30) + [index % 5 for index in range(420)]
    })
    result = select_labeled_klgrade_rows(frame, dataset_role="future train")
    assert result.input_rows == 450
    assert result.excluded_unlabeled_rows == 30
    assert result.labeled_rows == 420
```

- [ ] **Step 2: Run RED and verify the missing-module failure**

Run:

```powershell
python -m pytest tests/test_klgrade_policy.py -k "selects or constants" -v
```

Expected: collection fails because `src.algorithm.klgrade_policy` does not exist.

- [ ] **Step 3: Implement the immutable policy and exact validation order**

```python
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LabeledKLGradeRows:
    frame: pd.DataFrame
    input_rows: int
    labeled_rows: int
    excluded_unlabeled_rows: int


def select_labeled_klgrade_rows(
    frame: pd.DataFrame,
    *,
    dataset_role: str,
) -> LabeledKLGradeRows:
    if "KLGrade" not in frame.columns:
        raise ValueError(f"{dataset_role} is missing required column: KLGrade")

    numeric = pd.to_numeric(frame["KLGrade"], errors="coerce")
    finite = pd.Series(
        np.isfinite(numeric.to_numpy(dtype=float, na_value=np.nan)),
        index=frame.index,
    )
    integral = numeric.notna() & numeric.eq(np.floor(numeric))
    in_domain = numeric.between(-1, 4, inclusive="both")
    invalid = ~(finite & integral & in_domain)
    if invalid.any():
        raise ValueError(
            f"{dataset_role} KLGrade contains {int(invalid.sum())} invalid value(s); "
            "expected -1 for unlabeled or integers 0 through 4."
        )

    unlabeled = numeric.eq(-1)
    selected = frame.loc[~unlabeled].copy()
    selected["KLGrade"] = numeric.loc[~unlabeled].astype("int64")
    if selected.empty:
        raise ValueError(f"{dataset_role} has no labeled KLGrade rows after excluding -1.")

    return LabeledKLGradeRows(
        frame=selected,
        input_rows=int(len(frame)),
        labeled_rows=int(len(selected)),
        excluded_unlabeled_rows=int(unlabeled.sum()),
    )
```

- [ ] **Step 4: Add strict-invalid and privacy RED tests**

```python
@pytest.mark.parametrize("invalid", [None, "not-a-grade", -2, 5, 1.5, np.inf, -np.inf])
def test_rejects_every_invalid_grade_without_disclosing_value(invalid):
    frame = pd.DataFrame({"KLGrade": [0, invalid]})
    with pytest.raises(ValueError, match="1 invalid value") as error:
        select_labeled_klgrade_rows(frame, dataset_role="test")
    assert str(invalid) not in str(error.value)


def test_private_path_in_grade_is_not_echoed():
    private_value = "C:/private/patient-123/image.nii.gz"
    with pytest.raises(ValueError) as error:
        select_labeled_klgrade_rows(
            pd.DataFrame({"KLGrade": [0, private_value]}),
            dataset_role="train",
        )
    assert private_value not in str(error.value)
    assert "patient-123" not in str(error.value)


def test_all_unlabeled_rows_fail_explicitly():
    with pytest.raises(ValueError, match="no labeled"):
        select_labeled_klgrade_rows(
            pd.DataFrame({"KLGrade": [-1, "-1"]}),
            dataset_role="test",
        )
```

- [ ] **Step 5: Run the complete policy test file**

Run:

```powershell
python -m pytest tests/test_klgrade_policy.py -v
```

Expected: all policy tests pass.

- [ ] **Step 6: Record the Task 1 checkpoint**

Create `.superpowers/sdd/2026-08-02-unlabeled-klgrade-filtering/progress.md` with the exact command, pass count, and the statement that the workspace is non-Git.

---

### Task 2: Filter MEAS before any path or image work

**Files:**
- Modify: `src/algorithm/KOA_feature_extraction/meas_feature_extraction.py:1-249`
- Modify: `src/server/auto_experiment_server.py:292-315`
- Modify: `tests/test_meas_lasso_metadata_compatibility.py`
- Modify: `.superpowers/sdd/2026-08-02-unlabeled-klgrade-filtering/progress.md`

**Interfaces:**
- Consumes: `select_labeled_klgrade_rows()` from Task 1.
- Produces: unchanged `process_from_csv(csv_path, save=True) -> pd.DataFrame` plus `extractor.last_run_counts: dict[str, int]` with `input_rows`, `labeled_rows`, `excluded_unlabeled_rows`, `output_rows`, and `other_skipped_rows`.

- [ ] **Step 1: Write a RED test proving unlabeled paths are never inspected**

```python
import pytest


def test_meas_excludes_unlabeled_before_path_checks_or_extraction(tmp_path, monkeypatch):
    module = load_meas_module(monkeypatch)
    valid_image = tmp_path / "valid-image.nii.gz"
    valid_mask = tmp_path / "valid-mask.nii.gz"
    valid_image.write_bytes(b"image")
    valid_mask.write_bytes(b"mask")
    csv_path = tmp_path / "labels.csv"
    pd.DataFrame([
        {
            "CaseID": "unlabeled",
            "ImagePath": str(tmp_path / "must-not-be-checked-image.nii.gz"),
            "MaskPath": str(tmp_path / "must-not-be-checked-mask.nii.gz"),
            "KLGrade": -1,
        },
        {
            "CaseID": "labeled",
            "ImagePath": str(valid_image),
            "MaskPath": str(valid_mask),
            "KLGrade": 2,
        },
    ]).to_csv(csv_path, index=False)
    extractor = module.OAIZIB_MEAS_FeatureExtractor.__new__(
        module.OAIZIB_MEAS_FeatureExtractor
    )
    extractor.output_dir = str(tmp_path / "out")
    calls = []
    extractor.extract_features = lambda case_id, image, mask: (
        calls.append((case_id, image, mask)) or {"CaseID": case_id, "V_Femur": 1.0}
    )

    result = extractor.process_from_csv(str(csv_path), save=False)

    assert result["CaseID"].tolist() == ["labeled"]
    assert calls == [("labeled", str(valid_image), str(valid_mask))]
    assert extractor.last_run_counts == {
        "input_rows": 2,
        "labeled_rows": 1,
        "excluded_unlabeled_rows": 1,
        "output_rows": 1,
        "other_skipped_rows": 0,
    }
```

- [ ] **Step 2: Run the MEAS RED test**

Run:

```powershell
python -m pytest tests/test_meas_lasso_metadata_compatibility.py::test_meas_excludes_unlabeled_before_path_checks_or_extraction -v
```

Expected: FAIL because the unlabeled row is still processed or `last_run_counts` is absent.

- [ ] **Step 3: Integrate the policy before the MEAS loop**

Add the relative import and make `process_from_csv()` select before reading row paths:

```python
from ..klgrade_policy import select_labeled_klgrade_rows


selection = select_labeled_klgrade_rows(
    pd.read_csv(csv_path, dtype={"CaseID": str}),
    dataset_role="MEAS input",
)
df = selection.frame
self.last_run_counts = {
    "input_rows": selection.input_rows,
    "labeled_rows": selection.labeled_rows,
    "excluded_unlabeled_rows": selection.excluded_unlabeled_rows,
    "output_rows": 0,
    "other_skipped_rows": 0,
}
```

After building `features_df`, set:

```python
self.last_run_counts["output_rows"] = int(len(features_df))
self.last_run_counts["other_skipped_rows"] = int(
    selection.labeled_rows - len(features_df)
)
```

Initialize `self.last_run_counts = {}` in `__init__`. Keep the output columns limited to `CaseID`, MEAS features, and `KLGrade`; do not emit ImagePath/MaskPath variants.

- [ ] **Step 4: Add strict-label and dynamic-skip regressions**

```python
def test_meas_rejects_invalid_grade_before_extracting(tmp_path, monkeypatch):
    module = load_meas_module(monkeypatch)
    csv_path = tmp_path / "labels.csv"
    pd.DataFrame({
        "CaseID": ["bad"], "ImagePath": ["private-image"],
        "MaskPath": ["private-mask"], "KLGrade": [-2],
    }).to_csv(csv_path, index=False)
    extractor = module.OAIZIB_MEAS_FeatureExtractor.__new__(module.OAIZIB_MEAS_FeatureExtractor)
    extractor.output_dir = str(tmp_path / "out")
    extractor.extract_features = lambda *args: pytest.fail("extract_features must not run")
    with pytest.raises(ValueError, match="1 invalid value"):
        extractor.process_from_csv(str(csv_path), save=False)


def test_meas_distinguishes_unlabeled_from_missing_file_skips(tmp_path, monkeypatch):
    module = load_meas_module(monkeypatch)
    valid_image = tmp_path / "valid-image.nii.gz"
    valid_mask = tmp_path / "valid-mask.nii.gz"
    valid_image.write_bytes(b"image")
    valid_mask.write_bytes(b"mask")
    csv_path = tmp_path / "labels.csv"
    pd.DataFrame([
        {"CaseID": "u", "ImagePath": "unused", "MaskPath": "unused", "KLGrade": -1},
        {"CaseID": "missing", "ImagePath": "missing", "MaskPath": "missing", "KLGrade": 1},
        {
            "CaseID": "ok", "ImagePath": str(valid_image),
            "MaskPath": str(valid_mask), "KLGrade": 3,
        },
    ]).to_csv(csv_path, index=False)
    extractor = module.OAIZIB_MEAS_FeatureExtractor.__new__(module.OAIZIB_MEAS_FeatureExtractor)
    extractor.output_dir = str(tmp_path / "out")
    extractor.extract_features = lambda case_id, *_: {"CaseID": case_id, "V_Femur": 1.0}

    result = extractor.process_from_csv(str(csv_path), save=False)

    assert result["CaseID"].tolist() == ["ok"]
    assert extractor.last_run_counts == {
        "input_rows": 3,
        "labeled_rows": 2,
        "excluded_unlabeled_rows": 1,
        "output_rows": 1,
        "other_skipped_rows": 1,
    }
```

- [ ] **Step 5: Preserve the MCP path return and print aggregate counts**

In `meas_features_extraction()`, after `process_from_csv()`, print only `extractor.last_run_counts` values:

```python
counts = extractor.last_run_counts
print(
    "MEAS rows: "
    f"input={counts['input_rows']}, labeled={counts['labeled_rows']}, "
    f"excluded_unlabeled={counts['excluded_unlabeled_rows']}, "
    f"output={counts['output_rows']}, other_skipped={counts['other_skipped_rows']}"
)
```

Return the same `meas_features_summary.csv` path string as before.

- [ ] **Step 6: Run MEAS and metadata regressions**

Run:

```powershell
python -m pytest tests/test_meas_lasso_metadata_compatibility.py -v
```

Expected: all tests pass, including the existing `image_path`/`mask_path` metadata protections.

- [ ] **Step 7: Record the Task 2 checkpoint**

Append the exact command and pass count to the progress file.

---

### Task 3: Exclude unlabeled cases before LDSE fitting and transform

**Files:**
- Modify: `src/algorithm/KOA_feature_extraction/ldse_feature_extraction.py:1-714`
- Modify: `tests/test_ldse_train_fitted_pipeline.py`
- Modify: `.superpowers/sdd/2026-08-02-unlabeled-klgrade-filtering/progress.md`

**Interfaces:**
- Consumes: `select_labeled_klgrade_rows()` from Task 1.
- Produces: unchanged LDSE public methods and paths; model metadata and paired manifest gain dynamic labeled/excluded/other-skipped counts.

- [ ] **Step 1: Write RED tests for the train-fit and transform boundaries**

```python
def test_ldse_fit_never_loads_unlabeled_mask(tmp_path, monkeypatch, extractor_class):
    module = load_ldse_module(monkeypatch)
    csv_path, arrays = write_mask_csv(tmp_path)
    rows = pd.read_csv(csv_path, dtype={"CaseID": str}).to_dict("records")
    forbidden = tmp_path / "unlabeled-missing.nii.gz"
    rows.append({"CaseID": "unlabeled", "MaskPath": str(forbidden), "KLGrade": -1})
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    extractor = extractor_class(str(tmp_path / "out"), n_components=2)
    loaded = []
    patch_array_image_conversion(monkeypatch, module)
    extractor.load_image = lambda path: loaded.append(path) or arrays[os.path.normcase(os.path.abspath(path))]

    model_path = extractor.fit_shape_models(str(csv_path))

    assert os.path.isfile(model_path)
    assert str(forbidden) not in loaded
    metadata, _ = read_npz_model_artifact(model_path)
    assert metadata["train_input_rows"] == 7
    assert metadata["train_labeled_rows"] == 6
    assert metadata["train_excluded_unlabeled_rows"] == 1


def test_ldse_transform_excludes_unlabeled_before_mask_access(tmp_path, monkeypatch, extractor_class):
    module = load_ldse_module(monkeypatch)
    csv_path, arrays = write_mask_csv(tmp_path)
    extractor = extractor_class(str(tmp_path / "out"), n_components=2)
    attach_mask_arrays(monkeypatch, module, extractor, arrays)
    extractor.fit_shape_models(str(csv_path))
    forbidden = tmp_path / "unlabeled-missing.nii.gz"
    rows = pd.read_csv(csv_path, dtype={"CaseID": str}).to_dict("records")
    rows.append({"CaseID": "unlabeled", "MaskPath": str(forbidden), "KLGrade": -1})
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    loaded = []
    extractor.load_image = lambda path: (
        loaded.append(path) or arrays[os.path.normcase(os.path.abspath(path))]
    )

    result = extractor.transform_from_csv(str(csv_path), save=False)

    assert len(result) == 6
    assert "unlabeled" not in result["CaseID"].tolist()
    assert str(forbidden) not in loaded
    assert extractor.last_transform_counts == {
        "input_rows": 7,
        "labeled_rows": 6,
        "excluded_unlabeled_rows": 1,
        "output_rows": 6,
        "other_skipped_rows": 0,
    }
```

- [ ] **Step 2: Run LDSE RED tests**

Run:

```powershell
python -m pytest tests/test_ldse_train_fitted_pipeline.py -k "unlabeled_mask or excludes_unlabeled" -v
```

Expected: FAIL because `-1` remains in fit/transform input and metadata lacks the new counts.

- [ ] **Step 3: Filter inside both extractor entry points**

Import the policy and update `fit_shape_models()`:

```python
selection = select_labeled_klgrade_rows(
    self._read_csv_with_required_columns(train_csv_path),
    dataset_role="LDSE training input",
)
dataframe = selection.frame
```

Keep `train_csv_sha256` computed from the full original CSV. Add these artifact metadata keys:

```python
"train_input_rows": selection.input_rows,
"train_labeled_rows": selection.labeled_rows,
"train_excluded_unlabeled_rows": selection.excluded_unlabeled_rows,
```

Update `transform_from_csv()` before its row loop:

```python
selection = select_labeled_klgrade_rows(
    self._read_csv_with_required_columns(csv_path),
    dataset_role="LDSE transform input",
)
dataframe = selection.frame
self.last_transform_counts = {
    "input_rows": selection.input_rows,
    "labeled_rows": selection.labeled_rows,
    "excluded_unlabeled_rows": selection.excluded_unlabeled_rows,
}
```

After transformation, add `output_rows` and `other_skipped_rows = labeled_rows - output_rows`. Initialize `last_transform_counts = {}` in `__init__`.

- [ ] **Step 4: Make paired-run counts distinguish all three causes**

Change `_read_paired_ldse_input()` to return `LabeledKLGradeRows` after enforcing the KLGrade column. In `run_train_test_ldse_extraction()`, retain both selections for manifest construction while still fitting and transforming through the original CSV paths.

Preserve legacy total skip fields and add explicit fields:

```python
"train_input_rows": train_selection.input_rows,
"train_labeled_rows": train_selection.labeled_rows,
"train_excluded_unlabeled_rows": train_selection.excluded_unlabeled_rows,
"train_output_rows": int(len(train_features)),
"train_skipped_rows": int(train_selection.input_rows - len(train_features)),
"train_other_skipped_rows": int(train_selection.labeled_rows - len(train_features)),
"test_input_rows": test_selection.input_rows,
"test_labeled_rows": test_selection.labeled_rows,
"test_excluded_unlabeled_rows": test_selection.excluded_unlabeled_rows,
"test_output_rows": int(len(test_features)),
"test_skipped_rows": int(test_selection.input_rows - len(test_features)),
"test_other_skipped_rows": int(test_selection.labeled_rows - len(test_features)),
```

- [ ] **Step 5: Add paired-run dynamic count and rollback tests**

```python
def test_paired_ldse_manifest_counts_unlabeled_separately(tmp_path, monkeypatch, extractor_class):
    module = load_ldse_module(monkeypatch)
    train_csv, test_csv, arrays = write_paired_mask_csvs(tmp_path)
    train = pd.read_csv(train_csv, dtype={"CaseID": str})
    test = pd.read_csv(test_csv, dtype={"CaseID": str})
    train.loc[len(train)] = {
        "CaseID": "train-unlabeled",
        "MaskPath": str(tmp_path / "train-unlabeled.nii.gz"),
        "KLGrade": -1,
    }
    for index in range(2):
        test.loc[len(test)] = {
            "CaseID": f"test-unlabeled-{index}",
            "MaskPath": str(tmp_path / f"test-unlabeled-{index}.nii.gz"),
            "KLGrade": -1,
        }
    train.to_csv(train_csv, index=False)
    test.to_csv(test_csv, index=False)
    install_mask_io(monkeypatch, extractor_class, arrays, module=module)

    result = module.run_train_test_ldse_extraction(
        str(tmp_path / "paired-output"),
        str(train_csv),
        str(test_csv),
        n_components=2,
    )
    with open(result["manifest_path"], encoding="utf-8") as source:
        manifest = json.load(source)

    assert (
        manifest["train_input_rows"], manifest["train_labeled_rows"],
        manifest["train_excluded_unlabeled_rows"], manifest["train_output_rows"],
    ) == (7, 6, 1, 6)
    assert (
        manifest["test_input_rows"], manifest["test_labeled_rows"],
        manifest["test_excluded_unlabeled_rows"], manifest["test_output_rows"],
    ) == (5, 3, 2, 3)
    assert manifest["train_other_skipped_rows"] == 0
    assert manifest["test_other_skipped_rows"] == 0
    assert manifest["train_csv_sha256"] == hashlib.sha256(train_csv.read_bytes()).hexdigest()
    assert manifest["test_csv_sha256"] == hashlib.sha256(test_csv.read_bytes()).hexdigest()


def test_invalid_ldse_grade_rolls_back_before_model_fit(tmp_path, monkeypatch, extractor_class):
    module = load_ldse_module(monkeypatch)
    train_csv, test_csv, arrays = write_paired_mask_csvs(tmp_path)
    train = pd.read_csv(train_csv)
    train.loc[0, "KLGrade"] = -2
    train.to_csv(train_csv, index=False)
    install_mask_io(monkeypatch, extractor_class, arrays, module=module)
    output_root = tmp_path / "paired-output"

    with pytest.raises(ValueError, match="1 invalid value"):
        module.run_train_test_ldse_extraction(
            str(output_root), str(train_csv), str(test_csv), n_components=2
        )

    assert not (output_root / "ldse_run_manifest.json").exists()
    assert not (output_root / "train" / "ldse_features_summary.csv").exists()
    assert not (output_root / "test" / "ldse_features_summary.csv").exists()
    assert not (output_root / "model").exists()
```

- [ ] **Step 6: Run the complete LDSE lifecycle suite**

Run:

```powershell
python -m pytest tests/test_ldse_train_fitted_pipeline.py -v
```

Expected: all existing train-only fit, frozen-test transform, safe artifact, transaction, schema, and new filtering tests pass.

- [ ] **Step 7: Record the Task 3 checkpoint**

Append the exact command and pass count to the progress file.

---

### Task 4: Add the LASSO defensive filter with dynamic diagnostics

**Files:**
- Modify: `src/algorithm/feature_class_algorithm/koa_lasso_classifier.py:1-357`
- Modify: `tests/test_lasso_patient_pipeline.py`
- Modify: `.superpowers/sdd/2026-08-02-unlabeled-klgrade-filtering/progress.md`

**Interfaces:**
- Consumes: legacy or new MEAS/LDSE/Radiomics feature CSVs and `select_labeled_klgrade_rows()`.
- Produces: unchanged `KOALassoClassifier.run()` contract, privacy-safe diagnostics, and artifacts whose patient counts contain labeled cases only.

- [ ] **Step 1: Write a real-model RED test with nonconstant future-sized input**

```python
def test_lasso_dynamically_excludes_unlabeled_rows_and_counts_only_labeled(
    tmp_path, monkeypatch
):
    train_rows = _patient_rows("train-")
    test_rows = _patient_rows("test-")
    train_rows.extend([
        {"CaseID": "u-train-1", "KLGrade": -1, "signal": 999.0, "auxiliary": 999.0},
        {"CaseID": "u-train-2", "KLGrade": "-1", "signal": 998.0, "auxiliary": 998.0},
    ])
    test_rows.append(
        {"CaseID": "u-test", "KLGrade": -1, "signal": 997.0, "auxiliary": 997.0}
    )

    classifier, output_dir = _run_classifier(
        tmp_path, monkeypatch, train_rows, test_rows
    )

    assert any("train: excluded 2" in item and "12 labeled" in item for item in classifier.diagnostics)
    assert any("test: excluded 1" in item and "12 labeled" in item for item in classifier.diagnostics)
    summary = pd.read_csv(output_dir / "metrics_summary.csv")
    assert summary["NumPatients"].tolist() == [12, 12]
    manifest = json.loads(
        (output_dir / "classification_run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["counts"]["train"]["rows"] == 12
    assert manifest["counts"]["test"]["rows"] == 12
```

- [ ] **Step 2: Run the LASSO RED test**

Run:

```powershell
python -m pytest tests/test_lasso_patient_pipeline.py::test_lasso_dynamically_excludes_unlabeled_rows_and_counts_only_labeled -v
```

Expected: FAIL with the current `KLGrade contains ... invalid value(s)` error.

- [ ] **Step 3: Filter immediately after CSV reads and before patient validation**

Add a relative import and helper:

```python
from ..klgrade_policy import LabeledKLGradeRows, select_labeled_klgrade_rows


def _record_label_selection(self, role: str, selection: LabeledKLGradeRows) -> None:
    self.diagnostics.append(
        f"{role}: excluded {selection.excluded_unlabeled_rows} unlabeled "
        f"KLGrade=-1 row(s); {selection.labeled_rows} labeled row(s) remain"
    )
```

At the start of `_run()`, use:

```python
train_selection = select_labeled_klgrade_rows(
    self._read_feature_csv(self.train_path, "training"),
    dataset_role="training feature CSV",
)
test_selection = select_labeled_klgrade_rows(
    self._read_feature_csv(self.test_path, "test"),
    dataset_role="test feature CSV",
)
train = train_selection.frame
test = test_selection.frame
self._record_label_selection("train", train_selection)
self._record_label_selection("test", test_selection)
```

Then run the existing CaseID, Region, group, overlap, schema, finite-feature, class-count, and CV checks on the filtered frames. Do not weaken `PatientFeatureTableBuilder`'s 0–4 contract.

- [ ] **Step 4: Add strict-error, privacy, and feature-style regressions**

```python
@pytest.mark.parametrize("invalid", [-2, 5, 1.5, "not-a-grade"])
def test_lasso_still_rejects_invalid_non_unlabeled_grade(
    tmp_path, monkeypatch, invalid
):
    train_rows = _patient_rows("train-")
    train_rows[0]["KLGrade"] = invalid
    _disable_rendering(monkeypatch)
    train_path = _write_csv(tmp_path / "train.csv", train_rows)
    test_path = _write_csv(tmp_path / "test.csv", _patient_rows("test-"))
    output_dir = tmp_path / "output"
    with pytest.raises(ValueError, match="1 invalid value"):
        KOALassoClassifier(str(train_path), str(test_path), str(output_dir)).run()
    assert not output_dir.exists()


@pytest.mark.parametrize("metadata", [
    {"image_path": "private-image", "mask_path": "private-mask"},
    {"ImagePath": "private-image", "MaskPath": "private-mask"},
])
def test_unlabeled_filter_is_shared_by_meas_ldse_and_radiomics_style_tables(
    tmp_path, monkeypatch, metadata
):
    train_rows = _patient_rows("train-")
    test_rows = _patient_rows("test-")
    train_rows.append({
        "CaseID": "unlabeled", "KLGrade": -1,
        "signal": 999.0, "auxiliary": 999.0, **metadata,
    })

    _, output_dir = _run_classifier(
        tmp_path, monkeypatch, train_rows, test_rows
    )

    coefficients = pd.read_csv(output_dir / "model_coefficients.csv")
    schema = json.loads(
        (output_dir / "patient_feature_schema.json").read_text(encoding="utf-8")
    )
    assert not coefficients["feature"].str.contains(
        "image_path|mask_path|ImagePath|MaskPath", regex=True
    ).any()
    assert not set(schema["feature_columns"]).intersection(metadata)
```

Update existing assertions that assumed `classifier.diagnostics` contained only rendering messages so they filter by message content rather than asserting an exact total length.

- [ ] **Step 5: Run classifier, artifact, and metadata suites**

Run:

```powershell
python -m pytest tests/test_lasso_patient_pipeline.py tests/test_patient_feature_table.py tests/test_meas_lasso_metadata_compatibility.py tests/test_classification_artifacts.py tests/test_classification_output_contract.py -v
```

Expected: all tests pass; outputs remain transactional, input schemas remain train-locked, and only labeled patients appear in counts.

- [ ] **Step 6: Record the Task 4 checkpoint**

Append the exact command and pass count to the progress file.

---

### Task 5: Stop ClassificationAgent after failed classification

**Files:**
- Modify: `src/agents/classification_agent.py:1-606`
- Modify: `tests/test_classification_agent_path_scoping.py`
- Modify: `tests/test_tool_result_handoff.py`
- Modify: `.superpowers/sdd/2026-08-02-unlabeled-klgrade-filtering/progress.md`

**Interfaces:**
- Consumes: classification tool results as a Python dictionary or JSON string.
- Produces: `_structured_tool_status(result: object) -> str | None`; report calls are allowed only after explicit `status="success"`.

- [ ] **Step 1: Add RED tests for dictionary and JSON-string failures**

```python
import json
import pytest


@pytest.mark.parametrize("classification_result", [
    {
        "status": "failed",
        "message": "training feature CSV KLGrade contains 1 invalid value(s)",
        "output_dir": "LASSO_failed",
        "charts": [],
        "txt_files": [],
        "diagnostics": ["train: 14 rows"],
    },
    json.dumps({
        "status": "failed",
        "message": "training feature CSV KLGrade contains 1 invalid value(s)",
        "output_dir": "LASSO_failed",
        "charts": [],
        "txt_files": [],
        "diagnostics": ["train: 14 rows"],
    }),
])
def test_failed_classification_skips_report_reader(monkeypatch, classification_result):
    output_dir = "LASSO_failed"
    plan = json.dumps({
        "tools": [
            {
                "tool": "koa_lasso_classification",
                "arguments": {
                    "train_path": "train.csv",
                    "test_path": "test.csv",
                    "output_dir": output_dir,
                },
                "message": "Run classification.",
            },
            {
                "tool": "read_classification_results",
                "arguments": {"classification_result_dir": output_dir},
                "message": "Read report.",
            },
        ],
        "final_message": "done",
    })
    manager = FakeToolManager()
    manager.classification_result = classification_result

    async def fake_get_tool_manager():
        return manager

    monkeypatch.setattr(
        "agents.classification_agent.get_tool_manager", fake_get_tool_manager
    )
    config = SimpleNamespace(classification_agent=SimpleNamespace(llm=FakeLLM(plan)))
    result = asyncio.run(ClassificationAgent(config).process({
        "messages": [],
        "current_input": "classify train.csv and test.csv",
        "output": "",
        "agent_name": "",
        "tool_calls": [],
        "tool_results": [],
        "stream_callback": None,
    }))

    assert [name for name, _ in manager.calls] == ["koa_lasso_classification"]
    assert "KLGrade contains 1 invalid value" in result["output"]
    assert "目录不存在" not in result["output"]
```

Extend `FakeToolManager` in the test file with:

```python
class FakeToolManager:
    def __init__(self):
        self.calls = []
        self.classification_result = {
            "status": "success", "output_dir": "default", "charts": [], "txt_files": []
        }

    async def call_tool(self, tool_name, arguments):
        self.calls.append((tool_name, dict(arguments)))
        if tool_name in ("koa_lasso_classification", "koa_filter_classification"):
            return self.classification_result
        if tool_name == "read_classification_results":
            return "classification report"
        raise AssertionError(f"Unexpected tool: {tool_name}")
```

- [ ] **Step 2: Run Agent RED tests**

Run:

```powershell
python -m pytest tests/test_classification_agent_path_scoping.py -k "failed_classification" -v
```

Expected: FAIL because the agent still records the requested output directory and calls `read_classification_results`.

- [ ] **Step 3: Implement exact structured-status parsing**

Add a dependency-light module function:

```python
def _structured_tool_status(result: object) -> str | None:
    payload = result
    if isinstance(result, str):
        try:
            payload = json.loads(result)
        except (json.JSONDecodeError, TypeError):
            return None
    if not isinstance(payload, dict):
        return None
    status = payload.get("status")
    if not isinstance(status, str):
        return None
    normalized = status.strip().lower()
    return normalized or None
```

Do not infer success from prose such as `classification finished`.

Add a dependency-light AST regression to `tests/test_tool_result_handoff.py` so the
status contract remains testable even when LangChain is not installed:

```python
def load_dependency_light_status_helper():
    agent_path = os.path.join(
        os.path.dirname(__file__), "..", "src", "agents", "classification_agent.py"
    )
    with open(agent_path, encoding="utf-8") as source_file:
        tree = ast.parse(source_file.read(), filename=agent_path)
    status_function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_structured_tool_status"
    )
    namespace = {"json": json}
    module = ast.fix_missing_locations(
        ast.Module(body=[status_function], type_ignores=[])
    )
    exec(compile(module, agent_path, "exec"), namespace)
    return namespace["_structured_tool_status"]


def test_classification_status_helper_accepts_only_structured_status():
    status = load_dependency_light_status_helper()
    assert status({"status": "success"}) == "success"
    assert status(json.dumps({"status": "failed"})) == "failed"
    assert status("classification finished") is None
    assert status({"message": "success"}) is None
```

- [ ] **Step 4: Gate report execution on explicit success**

Track `classification_succeeded = False`. After a LASSO/Filter tool returns:

```python
classification_succeeded = _structured_tool_status(tool_result) == "success"
classification_output_dir = (
    arguments.get("output_dir") if classification_succeeded else None
)
```

Before invoking `read_classification_results`, skip it when `classification_succeeded` is false or `classification_output_dir` is absent. Preserve the original classification result in `tool_results` and `all_messages`; do not synthesize a report summary.

- [ ] **Step 5: Add a positive control for successful handoff**

```python
def test_successful_classification_reads_report_from_the_same_output_dir(monkeypatch):
    output_dir = "LASSO_ok"
    plan = json.dumps({
        "tools": [
            {
                "tool": "koa_lasso_classification",
                "arguments": {
                    "train_path": "train.csv", "test_path": "test.csv",
                    "output_dir": output_dir,
                },
                "message": "Run classification.",
            },
            {
                "tool": "read_classification_results",
                "arguments": {"classification_result_dir": "wrong-directory"},
                "message": "Read report.",
            },
        ],
        "final_message": "done",
    })
    manager = FakeToolManager()
    manager.classification_result = {
        "status": "success", "output_dir": output_dir,
        "charts": [], "txt_files": [], "diagnostics": [],
    }

    async def fake_get_tool_manager():
        return manager

    monkeypatch.setattr(
        "agents.classification_agent.get_tool_manager", fake_get_tool_manager
    )
    config = SimpleNamespace(classification_agent=SimpleNamespace(llm=FakeLLM(plan)))
    asyncio.run(ClassificationAgent(config).process({
        "messages": [], "current_input": "classify train.csv and test.csv",
        "output": "", "agent_name": "", "tool_calls": [],
        "tool_results": [], "stream_callback": None,
    }))

    assert manager.calls[1] == (
        "read_classification_results",
        {"classification_result_dir": output_dir},
    )
```

- [ ] **Step 6: Run Agent and tool-result handoff suites**

Run:

```powershell
python -m pytest tests/test_classification_agent_path_scoping.py tests/test_tool_result_handoff.py -v
```

Expected: all tests pass. If the local runtime lacks `langchain_core`, record that dependency collection blocker verbatim and run the status helper through a dependency-light AST test in `tests/test_tool_result_handoff.py`.

- [ ] **Step 7: Record the Task 5 checkpoint**

Append the exact command, pass count, and any optional dependency blocker to the progress file.

---

### Task 6: Document migration and verify the current and future-size contracts

**Files:**
- Modify: `README.md`
- Modify: `.superpowers/sdd/2026-08-02-unlabeled-klgrade-filtering/progress.md`

**Interfaces:**
- Consumes: completed Tasks 1–5.
- Produces: user workflow documentation and final evidence that counts are dynamic and current data yields 383/98 labeled cases.

- [ ] **Step 1: Add the README unlabeled-label policy**

Document these exact operational rules:

```markdown
### 未标注 KLGrade

- `KLGrade=-1` 表示未标注，MEAS、LDSE 和 LASSO 会动态排除；它不会映射为阴性或阳性。
- 0–4 以外的其他值仍会报错。样本数不固定，`有效数 = 输入数 - KLGrade=-1 数`。
- 新增训练病例后，重新运行 LDSE 训练/测试配对提取并在新的空目录中重新训练 LASSO。
- 仅新增测试病例时，使用训练集已拟合的 LDSE 工件 transform，不在测试集重新拟合。
- MEAS 可直接提取新增病例；训练集变化后仍需重新训练 LASSO。
- 分类失败时 Agent 直接返回原始失败信息，不再读取不存在的报告目录。
```

- [ ] **Step 2: Run focused verification**

Run:

```powershell
python -m pytest tests/test_klgrade_policy.py tests/test_meas_lasso_metadata_compatibility.py tests/test_ldse_train_fitted_pipeline.py tests/test_lasso_patient_pipeline.py tests/test_classification_agent_path_scoping.py tests/test_tool_result_handoff.py -q
```

Expected: all collectable focused tests pass; optional missing packages are reported separately rather than treated as product regressions.

- [ ] **Step 3: Run surrounding classification regressions**

Run:

```powershell
python -m pytest tests/test_patient_feature_table.py tests/test_grouped_validation.py tests/test_leakage_safe_l1_logistic.py tests/test_classification_artifacts.py tests/test_classification_output_contract.py tests/test_meas_lasso_metadata_compatibility.py tests/test_lasso_patient_pipeline.py -q
```

Expected: all tests pass with no new failures.

- [ ] **Step 4: Verify current data counts and the real defensive classifier path**

Run the shared selector against:

```text
src/LDSE_Features_TrainTest_20260802_044643_TrainTest/train/ldse_features_summary.csv
src/LDSE_Features_TrainTest_20260802_044643_TrainTest/test/ldse_features_summary.csv
```

Assert the aggregate-only results are:

```text
train: input=404, excluded_unlabeled=21, labeled=383
test: input=103, excluded_unlabeled=5, labeled=98
```

Then run the real classifier in a temporary output directory with rendering disabled at
the optional plotting boundary:

```powershell
@'
import json
import tempfile
from pathlib import Path

from src.algorithm.feature_class_algorithm.koa_lasso_classifier import KOALassoClassifier

train = "src/LDSE_Features_TrainTest_20260802_044643_TrainTest/train/ldse_features_summary.csv"
test = "src/LDSE_Features_TrainTest_20260802_044643_TrainTest/test/ldse_features_summary.csv"
KOALassoClassifier._render_plots_and_shap = lambda *args, **kwargs: []
with tempfile.TemporaryDirectory() as temporary_root:
    output = Path(temporary_root) / "classification"
    classifier = KOALassoClassifier(train, test, str(output))
    classifier.run()
    manifest = json.loads(
        (output / "classification_run_manifest.json").read_text(encoding="utf-8")
    )
    assert any("train: excluded 21" in item and "383 labeled" in item for item in classifier.diagnostics)
    assert any("test: excluded 5" in item and "98 labeled" in item for item in classifier.diagnostics)
    assert manifest["counts"]["train"]["rows"] == 383
    assert manifest["counts"]["test"]["rows"] == 98
'@ | python -
```

Expected: the command exits zero; the temporary directory is removed automatically.

- [ ] **Step 5: Verify a synthetic future-size dataset**

Run:

```powershell
python -m pytest tests/test_klgrade_policy.py::test_counts_change_with_input_instead_of_using_dataset_constants -v
```

Expected: PASS with the 450/30/420 dynamic input assertion; this test contains no reference to 383 or 98.

- [ ] **Step 6: Scan for forbidden implementations and unresolved markers**

Run:

```powershell
rg -n "383|98|KLGrade.*replace|KLGrade.*map|fit_transform.*test" src tests
rg -n "T[B]D|T[O]DO|implement la[t]er|fill in deta[i]ls" docs/superpowers/plans/2026-08-02-unlabeled-klgrade-filtering.md
```

Expected: no production-code hardcoding of 383/98, no `-1` label remapping, no test-set fitting, and no unresolved plan markers. Test/data-specific assertions and documentation references to 383/98 are allowed after manual inspection.

- [ ] **Step 7: Record final verification evidence**

Append every executed command, pass/fail count, dependency blocker, and the current 404→383 / 103→98 count evidence to the progress file. Do not claim the complete suite passed if optional dependency collection prevented it.
