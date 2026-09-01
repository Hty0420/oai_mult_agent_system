# MEAS 与 LASSO 元数据兼容修复设计

**日期：** 2026-08-01  
**状态：** 已确认，待实施  
**范围：** MEAS 特征输出、特征文件校验、LASSO 输入兼容和相关回归测试

## 1. 背景与根因

当前 `OAIZIB_MEAS_FeatureExtractor.extract_features()` 会把输入影像路径以 `image_path`、`mask_path` 两个字符串字段写入每条 MEAS 特征记录，`process_from_csv()` 随后将其原样保存到 `meas_features_summary.csv`。

LASSO MCP 工具在创建 `KOALassoClassifier` 前，会调用 `validate_feature_dataframe()` 校验特征文件。校验器的非特征字段列表包含 `ImagePath`、`MaskPath`，但不包含带下划线的 `image_path`、`mask_path`。两列因此被误判为模型特征，并因字符串类型触发“Feature columns must be numeric”错误。

底层分类器本身通过 `select_dtypes(include=[np.number])` 只保留数值列，所以实际阻断点是 LASSO 前置校验，而不是 scikit-learn 模型训练。

## 2. 目标

1. 新生成的 MEAS 特征文件只包含样本标识、数值特征和标签，不再保存影像/掩码路径。
2. 已经生成且包含 `image_path`、`mask_path` 的历史 MEAS 文件仍可直接用于 LASSO 分类。
3. 校验器不得把常见路径字段命名变体误认为数值特征。
4. 修正 MEAS 输入文件存在性检查中重复检查 `ImagePath`、遗漏 `MaskPath` 的问题。
5. 使用自动化测试覆盖上述行为，防止回归。

## 3. 非目标

- 不改变 MEAS 数值特征的计算公式、名称或数量。
- 不改变 KL 二分类规则、LASSO 超参数或分类报告格式。
- 不重构 Agent 路由、MCP 进程或文件路径交接机制。
- 不修改现有历史 CSV 文件；兼容性由读取和校验逻辑提供。

## 4. 设计方案

采用“源头清理 + 读取兼容”的双层方案。

### 4.1 新 MEAS 输出

`extract_features()` 的返回字典保留 `CaseID` 和所有数值 MEAS 特征，不再加入 `image_path`、`mask_path`。`process_from_csv()` 继续追加 `KLGrade` 并保存 CSV/PKL。

新文件的预期结构为：

```text
CaseID,<MEAS 数值特征...>,KLGrade
```

该结构与仓库中现有的 MEAS 汇总样例以及 LDSE/Radiomics 特征表风格保持一致。

### 4.2 历史文件兼容

特征质量校验的非特征元数据集合新增 `image_path` 和 `mask_path`。因此，历史 CSV 中的这两列：

- 不参与数值类型校验；
- 不计入特征数量；
- 不参与缺失值检查、质量评估或模型训练。

同时在 `KOALassoClassifier` 的列对齐集合与最终丢弃列集合中加入这两个命名变体。虽然当前数值类型筛选已经能够排除它们，但显式声明可避免它们参与训练/测试列对齐，并使元数据规则在各层保持一致。

### 4.3 输入路径检查

将 MEAS 批处理中的：

```python
if not os.path.exists(ImagePath) or not os.path.exists(ImagePath):
```

更正为分别检查 `ImagePath` 和 `MaskPath`。任一文件缺失时跳过该病例，保持现有错误处理语义。

## 5. 数据流

```text
labels_train/test.csv
  -> meas_features_extraction
  -> meas_features_summary.csv（不含路径列）
  -> validate_feature_dataframe
  -> KOALassoClassifier
  -> 分类结果与报告
```

历史文件走同一链路；校验阶段会将 `image_path`、`mask_path` 识别为元数据后继续执行。

## 6. 错误处理

- 路径列以外的非数值特征列仍应被校验器拒绝，避免放宽规则掩盖真正的数据错误。
- `KLGrade` 缺失时继续返回明确的标签列错误。
- 数值特征含 NaN 时继续由 LASSO MCP 工具拒绝。
- 影像或掩码文件不存在时，MEAS 提取器继续跳过病例并打印提示。

## 7. 测试设计

### 7.1 校验器回归测试

构造包含 `CaseID`、`image_path`、`mask_path`、数值 MEAS 特征和 `KLGrade` 的 DataFrame，断言：

- `validate_feature_dataframe()` 返回有效；
- 路径列不出现在 `feature_cols`；
- 真正的数值特征仍在 `feature_cols`。

该测试在修复前应稳定失败，错误原因必须是两列被识别为非数值特征。

### 7.2 MEAS 输出测试

使用最小可控输入验证 `extract_features()` 返回值不包含 `image_path`、`mask_path`，同时保留 `CaseID` 和预期数值特征。为避免测试依赖真实大体积 MRI，可使用小型 SimpleITK 图像或针对图像加载与特征计算做最小替身。

### 7.3 路径存在性测试

构造图像存在但掩码不存在的标签记录，断言 MEAS 批处理跳过该病例且不会调用特征提取。这能捕获重复检查 `ImagePath` 的旧问题。

### 7.4 回归范围

运行：

- 特征质量与工具校验相关测试；
- MEAS 新增测试；
- Agent 路径选择和分类作用域测试（环境依赖可用时）；
- 最终可执行的完整测试集。

## 8. 验收标准

1. 新生成的 `meas_features_summary.csv` 不含 `image_path`、`mask_path`。
2. 带这两列的历史 MEAS CSV 能通过特征文件校验。
3. 路径列不会进入 LASSO 的训练/测试公共特征集合。
4. 掩码缺失时该病例被正确跳过。
5. 新增回归测试经历先失败、修复后通过；相关既有测试无回归。

## 9. 风险与兼容性

- 移除新文件中的路径列可能影响依赖这两列的外部脚本。本项目内没有发现后续组件依赖 MEAS 输出中的路径列；原始路径仍保存在输入标签 CSV 中。
- 历史文件不做原地修改，避免破坏已有实验记录。
- 兼容逻辑只扩展已知元数据字段，不会忽略任意字符串列。
