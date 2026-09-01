# MEAS-LASSO Metadata Compatibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让新生成的 MEAS 特征表不再包含路径列，同时保证带 `image_path`、`mask_path` 的历史文件可通过 LASSO 校验和列对齐。

**Architecture:** 在数据源头移除 MEAS 路径元数据，并在读取侧的特征校验器和 LASSO 分类器中显式兼容历史字段。修改保持在现有提取、校验、分类三个边界内，不改变特征算法、Agent 路由或分类指标。

**Tech Stack:** Python 3.11/3.12、pandas、NumPy、scikit-learn、pytest、SimpleITK（生产环境；测试使用最小模块替身）

## Global Constraints

- 不改变 MEAS 数值特征公式、名称和数量。
- 不改变 `KLGrade < 2` 与 `KLGrade >= 2` 的二分类规则。
- 不修改已有 CSV；读取端兼容历史 `image_path`、`mask_path`。
- 除两个已知路径字段外，其他非数值未知字段仍必须被拒绝。
- 当前工作目录不是 Git 仓库；计划中的提交命令仅供未来 Git 工作区使用，本次执行跳过提交。

---

## File Structure

- Create: `tests/test_meas_lasso_metadata_compatibility.py` — 集中覆盖路径字段校验、分类列对齐、MEAS 输出与缺失掩码行为。
- Modify: `src/algorithm/feature_quality/quality_assessor.py` — 将 snake_case 路径字段识别为非特征元数据。
- Modify: `src/algorithm/feature_class_algorithm/koa_lasso_classifier.py` — 在训练/测试列对齐和最终特征丢弃阶段显式排除历史路径字段。
- Modify: `src/algorithm/KOA_feature_extraction/meas_feature_extraction.py` — 停止输出路径字段，并正确检查掩码文件存在性。

### Task 1: 历史 MEAS 文件通过特征校验

**Files:**
- Create: `tests/test_meas_lasso_metadata_compatibility.py`
- Modify: `src/algorithm/feature_quality/quality_assessor.py:12-29`

**Interfaces:**
- Consumes: `validate_feature_dataframe(df: pd.DataFrame, label_col: str = "KLGrade") -> tuple[bool, str, list[str]]`
- Produces: 路径元数据不会出现在返回的 `feature_cols` 中；未知字符串列仍触发失败。

- [ ] **Step 1: 写入历史字段回归测试**

```python
import pandas as pd

from algorithm.feature_quality.quality_assessor import validate_feature_dataframe


def test_validator_treats_snake_case_paths_as_metadata():
    df = pd.DataFrame({
        "CaseID": ["001", "002"],
        "image_path": ["a.nii.gz", "b.nii.gz"],
        "mask_path": ["a_mask.nii.gz", "b_mask.nii.gz"],
        "V_Femur": [1.0, 2.0],
        "KLGrade": [1, 3],
    })

    valid, message, feature_cols = validate_feature_dataframe(df)

    assert valid, message
    assert feature_cols == ["V_Femur"]


def test_validator_still_rejects_unknown_string_features():
    df = pd.DataFrame({
        "CaseID": ["001", "002"],
        "scanner_note": ["first", "second"],
        "V_Femur": [1.0, 2.0],
        "KLGrade": [1, 3],
    })

    valid, message, _ = validate_feature_dataframe(df)

    assert not valid
    assert "scanner_note" in message
```

- [ ] **Step 2: 运行测试并确认第一个测试因路径列被误判而失败**

Run: `python -m pytest -q tests/test_meas_lasso_metadata_compatibility.py::test_validator_treats_snake_case_paths_as_metadata`

Expected: FAIL，消息包含 `Non-numeric columns: ['image_path', 'mask_path']`。

- [ ] **Step 3: 最小修改非特征字段集合**

在 `NON_FEATURE_PATTERNS` 中显式加入：

```python
"image_path", "mask_path",
```

- [ ] **Step 4: 运行两个校验测试**

Run: `python -m pytest -q tests/test_meas_lasso_metadata_compatibility.py -k validator`

Expected: 2 passed。

- [ ] **Step 5: 在 Git 工作区提交**

```bash
git add tests/test_meas_lasso_metadata_compatibility.py src/algorithm/feature_quality/quality_assessor.py
git commit -m "fix: accept legacy MEAS path metadata"
```

当前非 Git 工作区跳过此步骤。

### Task 2: LASSO 列对齐排除历史路径字段

**Files:**
- Modify: `tests/test_meas_lasso_metadata_compatibility.py`
- Modify: `src/algorithm/feature_class_algorithm/koa_lasso_classifier.py:47-74,128-139`

**Interfaces:**
- Consumes: `KOALassoClassifier.run() -> None`
- Produces: 传给 `_run_lasso_for_dataset()` 的训练和测试 DataFrame 均不包含 `image_path`、`mask_path`。

- [ ] **Step 1: 写入列对齐回归测试**

```python
import pandas as pd

from algorithm.feature_class_algorithm.koa_lasso_classifier import KOALassoClassifier


def test_lasso_alignment_removes_snake_case_path_metadata(tmp_path):
    train = pd.DataFrame({
        "CaseID": ["1", "2"],
        "image_path": ["train1.nii.gz", "train2.nii.gz"],
        "mask_path": ["train1_mask.nii.gz", "train2_mask.nii.gz"],
        "V_Femur": [1.0, 2.0],
        "KLGrade": [1, 3],
    })
    test = pd.DataFrame({
        "CaseID": ["3", "4"],
        "image_path": ["test1.nii.gz", "test2.nii.gz"],
        "mask_path": ["test1_mask.nii.gz", "test2_mask.nii.gz"],
        "V_Femur": [1.5, 2.5],
        "KLGrade": [0, 4],
    })
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)

    classifier = KOALassoClassifier(
        str(train_path), str(test_path), str(tmp_path / "out")
    )
    captured = {}

    def capture_columns(df_train, df_test, output_dir, region_name=None):
        captured["train"] = list(df_train.columns)
        captured["test"] = list(df_test.columns)

    classifier._run_lasso_for_dataset = capture_columns
    classifier.run()

    assert "image_path" not in captured["train"]
    assert "mask_path" not in captured["train"]
    assert "image_path" not in captured["test"]
    assert "mask_path" not in captured["test"]
    assert "V_Femur" in captured["train"]
```

- [ ] **Step 2: 运行测试并确认列仍被传给内部分类方法**

Run: `python -m pytest -q tests/test_meas_lasso_metadata_compatibility.py::test_lasso_alignment_removes_snake_case_path_metadata`

Expected: FAIL，至少一个断言显示 `image_path` 或 `mask_path` 仍在列集合中。

- [ ] **Step 3: 最小修改两个元数据集合**

在 `KOALassoClassifier.run()` 的 `_meta_cols` 和 `_run_lasso_for_dataset()` 的 `drop_cols` 中分别加入：

```python
'image_path', 'mask_path',
```

- [ ] **Step 4: 运行列对齐测试**

Run: `python -m pytest -q tests/test_meas_lasso_metadata_compatibility.py::test_lasso_alignment_removes_snake_case_path_metadata`

Expected: PASS。

- [ ] **Step 5: 在 Git 工作区提交**

```bash
git add tests/test_meas_lasso_metadata_compatibility.py src/algorithm/feature_class_algorithm/koa_lasso_classifier.py
git commit -m "fix: exclude legacy paths from LASSO alignment"
```

当前非 Git 工作区跳过此步骤。

### Task 3: 新 MEAS 输出移除路径并修复掩码检查

**Files:**
- Modify: `tests/test_meas_lasso_metadata_compatibility.py`
- Modify: `src/algorithm/KOA_feature_extraction/meas_feature_extraction.py:166-250`

**Interfaces:**
- Consumes: `OAIZIB_MEAS_FeatureExtractor.extract_features(CaseID, image_path, mask_path) -> dict | None`
- Produces: 返回字典只包含 `CaseID` 和数值 MEAS 特征，不包含路径。
- Consumes: `OAIZIB_MEAS_FeatureExtractor.process_from_csv(csv_path, save=True) -> pd.DataFrame`
- Produces: 任一输入影像或掩码缺失时跳过病例。

- [ ] **Step 1: 为缺少 SimpleITK 的轻量测试环境增加模块加载帮助函数**

```python
import importlib
import sys
import types


def load_meas_module(monkeypatch):
    try:
        return importlib.import_module(
            "algorithm.KOA_feature_extraction.meas_feature_extraction"
        )
    except ModuleNotFoundError as exc:
        if exc.name != "SimpleITK":
            raise
        fake_sitk = types.ModuleType("SimpleITK")
        fake_sitk.BinaryThreshold = lambda image, **kwargs: image
        monkeypatch.setitem(sys.modules, "SimpleITK", fake_sitk)
        return importlib.import_module(
            "algorithm.KOA_feature_extraction.meas_feature_extraction"
        )
```

- [ ] **Step 2: 写入新输出不含路径的失败测试**

```python
def test_meas_extract_features_does_not_emit_paths(monkeypatch):
    module = load_meas_module(monkeypatch)
    extractor = module.OAIZIB_MEAS_FeatureExtractor.__new__(
        module.OAIZIB_MEAS_FeatureExtractor
    )
    extractor.labels_map = {1: "Femur"}
    extractor.structure_pairs = {}
    extractor.feature_name_map = {
        "volume": "V",
        "surface_area": "A",
        "surface_to_volume_ratio": "SVR",
    }
    extractor.load_image = lambda path: object()
    extractor.calculate_volume = lambda mask, label: 10.0
    extractor.calculate_surface_area = lambda mask, label: 5.0
    extractor.calculate_surface_to_volume_ratio = lambda volume, area: area / volume

    result = extractor.extract_features("001", "image.nii.gz", "mask.nii.gz")

    assert result["CaseID"] == "001"
    assert result["V_Femur"] == 10.0
    assert "image_path" not in result
    assert "mask_path" not in result
```

- [ ] **Step 3: 写入掩码不存在时跳过病例的失败测试**

```python
def test_meas_process_skips_case_when_mask_is_missing(tmp_path, monkeypatch):
    module = load_meas_module(monkeypatch)
    image_path = tmp_path / "image.nii.gz"
    image_path.write_bytes(b"image")
    missing_mask = tmp_path / "missing_mask.nii.gz"
    csv_path = tmp_path / "labels.csv"
    pd.DataFrame({
        "CaseID": ["001"],
        "ImagePath": [str(image_path)],
        "MaskPath": [str(missing_mask)],
        "KLGrade": [2],
    }).to_csv(csv_path, index=False)

    extractor = module.OAIZIB_MEAS_FeatureExtractor.__new__(
        module.OAIZIB_MEAS_FeatureExtractor
    )
    extractor.output_dir = str(tmp_path / "out")
    called = []
    extractor.extract_features = lambda *args: called.append(args) or {"CaseID": "001"}

    result = extractor.process_from_csv(str(csv_path), save=False)

    assert result.empty
    assert called == []
```

- [ ] **Step 4: 运行两个测试并确认分别因旧输出和重复路径检查而失败**

Run: `python -m pytest -q tests/test_meas_lasso_metadata_compatibility.py -k "meas_extract or mask_is_missing"`

Expected: 2 failed；第一个结果仍含路径字段，第二个仍调用了 `extract_features()`。

- [ ] **Step 5: 实施最小生产代码修改**

将 MEAS 基础字典改为：

```python
features = {"CaseID": CaseID}
```

并将存在性判断改为：

```python
if not os.path.exists(ImagePath) or not os.path.exists(MaskPath):
```

- [ ] **Step 6: 运行两个 MEAS 测试**

Run: `python -m pytest -q tests/test_meas_lasso_metadata_compatibility.py -k "meas_extract or mask_is_missing"`

Expected: 2 passed。

- [ ] **Step 7: 在 Git 工作区提交**

```bash
git add tests/test_meas_lasso_metadata_compatibility.py src/algorithm/KOA_feature_extraction/meas_feature_extraction.py
git commit -m "fix: keep MEAS summaries feature-only"
```

当前非 Git 工作区跳过此步骤。

### Task 4: 相关回归验证

**Files:**
- Test: `tests/test_meas_lasso_metadata_compatibility.py`
- Test: `tests/test_feature_quality.py`
- Test: `tests/test_feature_quality_tool.py`
- Test: `tests/test_integration_feature_quality.py`
- Test: `tests/test_agent_feature_paths.py`
- Test: `tests/test_classification_agent_path_scoping.py`

**Interfaces:**
- Consumes: Tasks 1-3 的最终行为。
- Produces: 可复现的测试结果与环境限制记录。

- [ ] **Step 1: 运行新增测试文件**

Run: `python -m pytest -q tests/test_meas_lasso_metadata_compatibility.py`

Expected: 5 passed。

- [ ] **Step 2: 运行不依赖 LangChain/SimpleITK 安装的相关测试**

Run: `python -m pytest -q tests/test_feature_quality.py tests/test_feature_quality_tool.py tests/test_integration_feature_quality.py tests/test_agent_feature_paths.py`

Expected: 全部通过。

- [ ] **Step 3: 尝试运行完整测试集并记录真实结果**

Run: `python -m pytest -q`

Expected: 在完整依赖环境中全部通过；当前系统环境若仍缺少 `langchain_core` 或 `SimpleITK`，应如实记录收集错误，不得宣称全量通过。

- [ ] **Step 4: 检查变更范围**

Run: `git diff -- src/algorithm/feature_quality/quality_assessor.py src/algorithm/feature_class_algorithm/koa_lasso_classifier.py src/algorithm/KOA_feature_extraction/meas_feature_extraction.py tests/test_meas_lasso_metadata_compatibility.py`

Expected: Git 工作区只包含设计范围内的三处生产代码和一个测试文件；当前非 Git 工作区改用逐文件内容检查。

- [ ] **Step 5: 在 Git 工作区提交最终验证状态**

```bash
git add src/algorithm/feature_quality/quality_assessor.py src/algorithm/feature_class_algorithm/koa_lasso_classifier.py src/algorithm/KOA_feature_extraction/meas_feature_extraction.py tests/test_meas_lasso_metadata_compatibility.py
git commit -m "test: cover MEAS to LASSO compatibility"
```

当前非 Git 工作区跳过此步骤。
