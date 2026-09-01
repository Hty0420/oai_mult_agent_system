# LDSE Train-Fitted PCA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make LDSE train/test extraction fit scaler/PCA only on training masks, reuse the frozen model for both datasets, and leave an auditable, reproducible artifact trail.

**Architecture:** Split the extractor into explicit fit, load, and transform operations; build one paired train/test orchestration function on top; expose that function through MCP and normalize Agent plans to use it. Keep the single-CSV MCP tool as an explicit compatibility path whose mode and dataset role are validated.

**Tech Stack:** Python 3.12+, NumPy, pandas, SciPy, scikit-learn, SimpleITK, pickle, MCP/FastMCP, LangChain/LangGraph, pytest.

## Global Constraints

- `StandardScaler` and `PCA` fit only on the training CSV; test data may call only frozen `transform` operations.
- The paired MCP interface is `ldse_train_test_features_extraction(output_dir, train_csv_path, test_csv_path, n_components=50, random_seed=42)`.
- The single-CSV interface accepts only `mode="fit_transform"` or `mode="transform"`; `fit_transform` requires `dataset_role="train"`, and `transform` requires a verified model path.
- Sampling is deterministic from `random_seed`, normalized mask identity, and structure label; it must not depend on call order.
- Train and test output feature columns and order must be identical and determined by the training model.
- No test-set statistic may choose `n_components`, columns, imputation values, or any other fitted state.
- Existing historical CSV/model artifacts are not rewritten in place.
- The workspace has no `.git` directory, so task checkpoints record test evidence instead of creating commits.

---

## File Structure

- Modify `src/algorithm/KOA_feature_extraction/ldse_feature_extraction.py`: explicit model lifecycle, deterministic sampling, artifact validation, paired train/test orchestration, manifest generation.
- Modify `src/server/auto_experiment_server.py`: safe paired MCP tool and explicit-mode compatibility wrapper.
- Modify `src/agents/feature_extraction_agent.py`: tool documentation, examples, and deterministic normalization from unsafe dual LDSE calls to one paired call.
- Modify `src/agents/orchestrator_agent.py`: advertise the paired LDSE tool to the top-level routing prompt.
- Modify `tests/test_feature_extraction_agent_plan_repair.py`: replace the unsafe dual-call expectation and retain MEAS pair coverage.
- Modify `tests/test_agent_feature_paths.py`: prove nested `train/` and `test/` output paths are handed to classification correctly.
- Create `tests/test_ldse_train_fitted_pipeline.py`: extractor lifecycle, provenance, reproducibility, schema, and leakage regressions.
- Modify `README.md`: document the safe LDSE invocation and output artifacts.

### Task 1: Explicit LDSE Model Lifecycle and Deterministic Sampling

**Files:**
- Modify: `src/algorithm/KOA_feature_extraction/ldse_feature_extraction.py:12-196`
- Create: `tests/test_ldse_train_fitted_pipeline.py`

**Interfaces:**
- Consumes: existing `OAIZIB_Light_LDSE_FeatureExtractor(output_dir, n_components=50)` behavior.
- Produces: `OAIZIB_Light_LDSE_FeatureExtractor(output_dir, n_components=50, random_seed=42)`, `fit_shape_models(train_csv_path) -> str`, `load_shape_models(model_path) -> None`, `transform_from_csv(csv_path, output_dir=None, save=True) -> pd.DataFrame`, and explicit `process_from_csv(..., mode, dataset_role=None, model_path=None)`.

- [x] **Step 1: Add failing lifecycle and sampling tests**

Add a loader that stubs only unavailable SimpleITK during unit tests, then add these tests:

```python
def test_transform_without_model_fails_without_fitting(tmp_path, monkeypatch):
    module = load_ldse_module(monkeypatch)
    extractor = module.OAIZIB_Light_LDSE_FeatureExtractor(tmp_path / "out")
    fit_calls = []
    monkeypatch.setattr(
        extractor,
        "fit_shape_models",
        lambda csv_path: fit_calls.append(csv_path),
    )

    with pytest.raises(RuntimeError, match="fitted LDSE model"):
        extractor.transform_from_csv(tmp_path / "test.csv", save=False)

    assert fit_calls == []


def test_test_role_cannot_fit_transform(tmp_path, monkeypatch):
    module = load_ldse_module(monkeypatch)
    extractor = module.OAIZIB_Light_LDSE_FeatureExtractor(tmp_path / "out")

    with pytest.raises(ValueError, match="training data"):
        extractor.process_from_csv(
            tmp_path / "labels_test.csv",
            save=False,
            mode="fit_transform",
            dataset_role="test",
        )


def test_transform_rejects_csv_without_required_columns(tmp_path, monkeypatch):
    module = load_ldse_module(monkeypatch)
    csv_path = tmp_path / "invalid.csv"
    pd.DataFrame({"CaseID": ["001"]}).to_csv(csv_path, index=False)
    extractor = module.OAIZIB_Light_LDSE_FeatureExtractor(tmp_path / "out")
    extractor.pca_models = {1: object()}
    extractor.scalers = {1: object()}

    with pytest.raises(ValueError, match="MaskPath"):
        extractor.transform_from_csv(csv_path, save=False)


def test_boundary_sampling_is_stable_and_order_independent(tmp_path, monkeypatch):
    module = load_ldse_module(monkeypatch)
    mask = np.zeros((24, 24, 24), dtype=np.uint8)
    mask[4:20, 4:20, 4:20] = 1
    extractor = module.OAIZIB_Light_LDSE_FeatureExtractor(
        tmp_path / "out", random_seed=17
    )

    first = extractor.get_boundary_points(mask, 1, 100, sample_key="case-001")
    extractor.get_boundary_points(mask, 1, 100, sample_key="another-case")
    repeated = extractor.get_boundary_points(mask, 1, 100, sample_key="case-001")

    np.testing.assert_array_equal(first, repeated)
```

- [x] **Step 2: Run the new tests and verify the current implicit-fit behavior fails them**

Run:

```powershell
python -m pytest tests/test_ldse_train_fitted_pipeline.py -v
```

Expected: failures because `random_seed`, `transform_from_csv`, explicit `mode`, and the no-auto-fit guard do not exist yet.

- [x] **Step 3: Implement stable sampling and model readiness checks**

Add imports and these core helpers:

```python
import hashlib
import json
from pathlib import Path

MODEL_ARTIFACT_VERSION = 2


def _stable_seed(random_seed: int, sample_key: str, label: int) -> int:
    identity = os.path.normcase(os.path.abspath(str(sample_key)))
    payload = f"{random_seed}|{identity}|{label}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
```

Change `get_boundary_points` to accept `sample_key`, create
`np.random.default_rng(_stable_seed(...))`, and call `rng.choice(...)`. Pass each normalized mask path as
`sample_key` from fitting and transformation.

Add `_require_fitted_models()`:

```python
def _require_fitted_models(self):
    if not self.pca_models or not self.scalers:
        raise RuntimeError(
            "A fitted LDSE model is required for transform; "
            "fit on training data or load a verified model first."
        )
    if set(self.pca_models) != set(self.scalers):
        raise ValueError("LDSE artifact has mismatched PCA and scaler structures.")
```

- [x] **Step 4: Save and validate an auditable training artifact**

Make `fit_shape_models(train_csv_path)` reject a missing CSV or missing `MaskPath`, fit as it does now, fail if no
structure model can be fitted, and return the saved model path. Store:

```python
model_data = {
    "artifact_version": MODEL_ARTIFACT_VERSION,
    "fitted_dataset_role": "train",
    "train_csv_path": os.path.abspath(train_csv_path),
    "train_csv_sha256": _sha256_file(train_csv_path),
    "n_components_requested": self.n_components,
    "n_components_actual": {
        str(sid): int(model.n_components_) for sid, model in self.pca_models.items()
    },
    "random_seed": self.random_seed,
    "pca_solver": "full",
    "feature_columns": self._expected_feature_columns(),
    "structures": self.structures,
}
```

Store this JSON metadata as a non-object Unicode array in a compressed version-2 `.npz`, together with numeric
scaler mean/scale/variance arrays and PCA component/mean arrays for each fitted structure. Publish through a private
temporary file plus flush/fsync and atomic replace. Implement `load_shape_models(model_path)` with
`np.load(..., allow_pickle=False)` so it rejects a missing file, legacy pickle, unsupported version, a role other than
`train`, missing or malformed metadata, object/non-finite/oversized arrays, or incompatible scaler/PCA dimensions
before reconstructing only known sklearn classes and assigning model state.

- [x] **Step 5: Split transform from compatibility dispatch**

Move the row loop and CSV/PKL writing from current `process_from_csv()` into
`transform_from_csv(csv_path, output_dir=None, save=True)`. Call `_require_fitted_models()` before reading the CSV.
Validate `CaseID` and `MaskPath` before iterating; require `KLGrade` for the paired classification workflow and report
all missing columns in one `ValueError`.
Use training-derived `_expected_feature_columns()` to reindex output:

```python
columns = ["CaseID", *self._expected_feature_columns()]
if "KLGrade" in source_df.columns:
    columns.append("KLGrade")
features_df = pd.DataFrame(all_features).reindex(columns=columns)
```

Implement the compatibility dispatcher:

```python
def process_from_csv(
    self,
    csv_path,
    save=True,
    *,
    mode,
    dataset_role=None,
    model_path=None,
):
    if mode == "fit_transform":
        if dataset_role != "train":
            raise ValueError("fit_transform is allowed only for training data")
        self.fit_shape_models(csv_path)
    elif mode == "transform":
        if not model_path:
            raise ValueError("model_path is required in transform mode")
        self.load_shape_models(model_path)
    else:
        raise ValueError("mode must be 'fit_transform' or 'transform'")
    return self.transform_from_csv(csv_path, save=save)
```

- [x] **Step 6: Run Task 1 tests and record the checkpoint**

Run:

```powershell
python -m pytest tests/test_ldse_train_fitted_pipeline.py -v
```

Expected: all Task 1 tests pass; no test transformation calls `fit_shape_models`.

### Task 2: Paired Train/Test Pipeline, Schema Lock, and Manifest

**Files:**
- Modify: `src/algorithm/KOA_feature_extraction/ldse_feature_extraction.py:196-235`
- Modify: `tests/test_ldse_train_fitted_pipeline.py`

**Interfaces:**
- Consumes: Task 1 extractor lifecycle and verified model artifact.
- Produces: `run_train_test_ldse_extraction(output_dir, train_csv_path, test_csv_path, n_components=50, random_seed=42) -> dict[str, str]`.

- [x] **Step 1: Add a failing end-to-end paired pipeline test**

Create five small training masks and two test masks as NumPy arrays, write label CSVs whose image/mask paths exist,
and replace `extractor.load_image` plus `sitk.GetArrayFromImage` with array-backed fakes. Assert:

```python
result = module.run_train_test_ldse_extraction(
    tmp_path / "LDSE_Features_TrainTest_20260801_120000",
    train_csv,
    test_csv,
    n_components=2,
    random_seed=23,
)

assert set(result) == {
    "train_features_path", "test_features_path", "model_path", "manifest_path"
}
assert all(Path(path).exists() for path in result.values())

train_features = pd.read_csv(result["train_features_path"])
test_features = pd.read_csv(result["test_features_path"])
assert list(train_features.columns) == list(test_features.columns)

with open(result["manifest_path"], encoding="utf-8") as handle:
    manifest = json.load(handle)
assert manifest["fitted_dataset_role"] == "train"
assert manifest["train_csv_sha256"] == sha256_file(train_csv)
assert manifest["feature_columns_match"] is True
```

Wrap `fit_shape_models` with a counter and assert it receives only `train_csv` and is called once.

- [x] **Step 2: Run the paired test and verify it fails because the orchestration function is absent**

Run:

```powershell
python -m pytest tests/test_ldse_train_fitted_pipeline.py -k "paired" -v
```

Expected: FAIL because `run_train_test_ldse_extraction` is not defined.

- [x] **Step 3: Implement the paired orchestration function**

Use fixed output subdirectories and one extractor instance:

```python
def run_train_test_ldse_extraction(
    output_dir,
    train_csv_path,
    test_csv_path,
    n_components=50,
    random_seed=42,
):
    root = Path(output_dir)
    model_dir = root / "model"
    train_dir = root / "train"
    test_dir = root / "test"
    for directory in (model_dir, train_dir, test_dir):
        directory.mkdir(parents=True, exist_ok=True)

    extractor = OAIZIB_Light_LDSE_FeatureExtractor(
        str(model_dir), n_components=n_components, random_seed=random_seed
    )
    model_path = extractor.fit_shape_models(str(train_csv_path))
    train_df = extractor.transform_from_csv(str(train_csv_path), str(train_dir))
    test_df = extractor.transform_from_csv(str(test_csv_path), str(test_dir))
```

Compare `CaseID`- and `KLGrade`-excluded feature columns exactly. On mismatch, raise `ValueError` listing missing and
extra columns. Write `ldse_run_manifest.json` with absolute input/output paths, hashes, row counts, skipped counts,
requested/actual dimensions, `fitted_dataset_role="train"`, and `feature_columns_match=True`. Return absolute paths.

- [x] **Step 4: Run Task 2 tests and record the checkpoint**

Run:

```powershell
python -m pytest tests/test_ldse_train_fitted_pipeline.py -v
```

Expected: all extractor and paired pipeline tests pass.

### Task 3: Safe MCP Interfaces

**Files:**
- Modify: `src/server/auto_experiment_server.py:24-28,312-335`
- Modify: `tests/test_ldse_train_fitted_pipeline.py`

**Interfaces:**
- Consumes: `run_train_test_ldse_extraction(...)` and explicit extractor dispatcher from Tasks 1-2.
- Produces: MCP tools `ldse_train_test_features_extraction(...) -> dict[str, str]` and explicit-mode `ldse_features_extraction(...) -> str`.

- [x] **Step 1: Add a source-level MCP contract test that does not import optional server dependencies**

Parse `src/server/auto_experiment_server.py` with `ast` and assert both function names exist, the paired function has
`train_csv_path`, `test_csv_path`, `random_seed`, and calls `run_train_test_ldse_extraction`, while the legacy function
has `mode`, `dataset_role`, and `model_path`. This test avoids importing radiomics/MCP/SimpleITK in a minimal test
environment.

- [x] **Step 2: Run the MCP contract test and verify it fails**

Run:

```powershell
python -m pytest tests/test_ldse_train_fitted_pipeline.py -k "mcp_contract" -v
```

Expected: FAIL because the paired MCP function and explicit compatibility parameters are absent.

- [x] **Step 3: Implement the paired MCP wrapper**

Import the orchestration function and add:

```python
@mcp.tool()
def ldse_train_test_features_extraction(
    output_dir: Annotated[str, Field(description="LDSE 成对输出根目录。")],
    train_csv_path: Annotated[str, Field(description="训练 CSV 路径。")],
    test_csv_path: Annotated[str, Field(description="测试 CSV 路径。")],
    n_components: Annotated[int, Field(description="LDSE 编码维度。")]=50,
    random_seed: Annotated[int, Field(description="确定性采样种子。")]=42,
):
    return run_train_test_ldse_extraction(
        output_dir, train_csv_path, test_csv_path, n_components, random_seed
    )
```

- [x] **Step 4: Convert the single-CSV MCP wrapper to explicit safe modes**

Add required `mode`, optional `dataset_role` and `model_path`, and `random_seed`. Construct the extractor and call:

```python
feature_df = extractor.process_from_csv(
    csv_path,
    mode=mode,
    dataset_role=dataset_role,
    model_path=model_path,
)
```

Document in the docstring that no mode can fit test data and transform never falls back to training.

- [x] **Step 5: Run MCP contract plus pipeline tests and record the checkpoint**

Run:

```powershell
python -m pytest tests/test_ldse_train_fitted_pipeline.py -v
```

Expected: all tests pass and the server source remains valid Python.

### Task 4: Normalize FeatureExtractionAgent Plans to One Paired LDSE Call

**Files:**
- Modify: `src/agents/feature_extraction_agent.py:45-151,214-275,375-448`
- Modify: `src/agents/orchestrator_agent.py:55-65`
- Modify: `tests/test_feature_extraction_agent_plan_repair.py`

**Interfaces:**
- Consumes: paired MCP parameter names from Task 3 and `select_train_test_source_paths(...)`.
- Produces: `_ensure_paired_feature_extractions(...)` that preserves MEAS pair repair while replacing one or more unsafe LDSE calls with one paired call.

- [x] **Step 1: Replace the old LDSE plan repair expectation with a failing safe-plan test**

Change the existing test to assert:

```python
calls = decision["tools"]
assert len(calls) == 1
assert calls[0]["tool"] == "ldse_train_test_features_extraction"
assert calls[0]["arguments"]["train_csv_path"] == str(labels_train)
assert calls[0]["arguments"]["test_csv_path"] == str(labels_test)
assert "TrainTest" in calls[0]["arguments"]["output_dir"]
```

Add a second test with two pre-existing LDSE single calls and assert both are collapsed into one paired call. Add a
MEAS test asserting a single MEAS call is still cloned into train/test calls. Add a training-only LDSE test asserting
that the remaining single call receives `mode="fit_transform"` and `dataset_role="train"`. Add a test-only LDSE
case without `model_path` and assert the unsafe call is removed and `final_message` asks for a verified training model.

- [x] **Step 2: Run the plan repair tests and verify the LDSE assertions fail**

Run:

```powershell
python -m pytest tests/test_feature_extraction_agent_plan_repair.py -v
```

Expected: LDSE tests fail with two `ldse_features_extraction` calls; MEAS compatibility test passes or exposes a
pre-existing regression that must be preserved during the implementation.

- [x] **Step 3: Implement LDSE call normalization before the MEAS repair loop**

When both source paths exist and any LDSE single call is present:

```python
paired_call = {
    "tool": "ldse_train_test_features_extraction",
    "arguments": {
        "output_dir": paired_output_dir,
        "train_csv_path": train_source,
        "test_csv_path": test_source,
        "n_components": first_args.get("n_components", 50),
        "random_seed": first_args.get("random_seed", 42),
    },
    "message": "使用仅在训练集拟合的共享 PCA 模型提取训练集和测试集 LDSE 特征。",
}
```

Replace all existing `ldse_features_extraction` calls at the first LDSE call index. Convert a directory containing
`_Train_` or `_Test_` to `_TrainTest_`; otherwise append `_TrainTest`. Restrict the existing clone loop to
`meas_features_extraction`.

When only one source role exists, normalize a training call to explicit `fit_transform/train`. Permit a test-only
single call only when it uses `mode="transform"` and has `model_path`; otherwise remove it and append a concise model
requirement to `final_message`. When the LLM already emits the paired tool, overwrite its train/test CSV arguments with
the exact role-matched source paths collected from context so guessed paths cannot bypass the existing safety net.

- [x] **Step 4: Update the Agent prompt and examples**

Describe the paired tool and make it the only allowed route when both train/test sources exist. Describe explicit
single-CSV parameters and replace the old example that showed a standalone test LDSE fit. Keep the existing rule that
only user-requested feature families are called. Add `ldse_train_test_features_extraction` to the
FeatureExtractionAgent and OrchestratorAgent tool inventories.

- [x] **Step 5: Run Agent plan tests and record the checkpoint**

Run:

```powershell
python -m pytest tests/test_feature_extraction_agent_plan_repair.py tests/test_agent_prompt_language.py -v
```

Expected: paired LDSE normalization and MEAS compatibility tests pass.

### Task 5: Downstream Path Handoff and Documentation

**Files:**
- Modify: `tests/test_agent_feature_paths.py`
- Modify only if the new test fails: `src/agents/path_utils.py:53-112`
- Modify: `README.md`

**Interfaces:**
- Consumes: paired output layout `<root>/train/ldse_features_summary.csv` and `<root>/test/ldse_features_summary.csv`.
- Produces: reliable train/test path classification and user-facing safe invocation documentation.

- [x] **Step 1: Add the nested-layout path regression test**

```python
def test_selects_paired_ldse_nested_train_and_test_paths():
    paths = [
        "LDSE_Features_TrainTest_20260801_120000/train/ldse_features_summary.csv",
        "LDSE_Features_TrainTest_20260801_120000/test/ldse_features_summary.csv",
    ]

    train_path, test_path = select_train_test_feature_paths(paths)

    assert train_path == paths[0]
    assert test_path == paths[1]
```

- [x] **Step 2: Run the path test before changing production code**

Run:

```powershell
python -m pytest tests/test_agent_feature_paths.py -k "paired_ldse_nested" -v
```

Expected: PASS with the current component-based role inference. If it fails, update `infer_dataset_role()` so the
nearest parent directory named exactly `train` or `test` takes precedence over the shared `TrainTest` root, then rerun.

- [x] **Step 3: Document the safe workflow**

Add a README section showing the paired MCP arguments, output tree, and these guarantees:

```text
- scaler/PCA are fitted on labels_train.csv only
- labels_test.csv is transformed by the frozen training artifact
- model provenance and random seed are saved in ldse_run_manifest.json
- standalone test extraction requires a verified training model and mode="transform"
```

- [x] **Step 4: Run focused regression tests**

Run:

```powershell
python -m pytest tests/test_ldse_train_fitted_pipeline.py tests/test_feature_extraction_agent_plan_repair.py tests/test_agent_feature_paths.py tests/test_classification_agent_path_scoping.py -v
```

Expected: all collectable focused tests pass.

### Task 6: Final Verification and Evidence

**Files:**
- Modify if required by test findings: only files already listed in Tasks 1-5.

**Interfaces:**
- Consumes: complete safe LDSE implementation.
- Produces: verified behavior and a precise report of any environment-only collection blockers.

- [x] **Step 1: Compile modified Python modules**

Run:

```powershell
python -m py_compile src/algorithm/KOA_feature_extraction/ldse_feature_extraction.py src/server/auto_experiment_server.py src/agents/feature_extraction_agent.py src/agents/path_utils.py
```

Expected: exit code 0 with no syntax errors.

- [ ] **Step 2: Run all currently collectable tests**

Environment note (2026-08-01): left unchecked because the bundled Python 3.12 runtime lacks scipy,
scikit-learn, pytest, and SimpleITK, and the required final verification intentionally excludes four known
environment/integration files. Focused and supported-runtime results are recorded in the final fix report.

Run:

```powershell
python -m pytest -q
```

Expected: all tests pass when declared dependencies are installed. If collection stops on missing `SimpleITK`,
`langchain_core`, or another declared dependency, record the exact missing module and separately report the focused
tests that ran successfully; do not describe an uncollected full suite as passing.

- [x] **Step 3: Verify leakage invariants from artifacts**

Run the paired pipeline test once more and inspect its temporary manifest assertions. Confirm the model artifact has
`fitted_dataset_role="train"`, the manifest train hash matches the training CSV, exactly one fit occurred, and both
feature files share the same ordered columns.

- [x] **Step 4: Update plan checkboxes and hand off the first optimization**

Mark completed steps in this plan, summarize modified files and test counts, list environment blockers separately,
and name the next proposed optimization—LDSE physical-space normalization and fixed-length contour representation—
without starting it until this first optimization is accepted.
