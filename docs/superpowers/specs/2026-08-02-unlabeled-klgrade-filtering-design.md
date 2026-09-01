# KLGrade=-1 未标注病例统一过滤设计

## 1. 背景与根因

当前训练标签文件有 404 行，其中 21 行 `KLGrade=-1`；测试标签文件有
103 行，其中 5 行 `KLGrade=-1`。CaseID 唯一且训练/测试标签与现有 LDSE
特征逐行对齐，250 个 LDSE 特征也没有缺失值。失败的根因不是 LDSE 特征
提取或标签合并，而是 LASSO 分类边界只接受 0–4 的有效 KL 分级。

用户已确认：`KLGrade=-1` 表示未标注，不需要参与特征拟合、模型训练或性能
评价。它不能映射为阴性或阳性。

## 2. 目标与非目标

### 目标

1. 在 MEAS、LDSE 监督特征提取前排除 `KLGrade=-1`。
2. 在 LASSO 公共分类入口再次过滤 `-1`，兼容历史特征 CSV。
3. 保留底层患者表的严格标签合同，只允许 0–4。
4. 以汇总计数记录排除情况，不披露患者 ID、图像或掩膜路径。
5. 分类失败后 ClassificationAgent 不再调用报告读取工具。
6. 对当前已上传数据，过滤后的验收基线为训练 383 例、测试 98 例；该数字仅用于
   数据特定的回归验证，不进入生产代码或通用测试的硬编码条件。

### 非目标

- 不将 `-1` 重新编码为任何 KL 等级或二分类标签。
- 不放宽对空值、非数值、`KLGrade < -1` 或 `KLGrade > 4` 的拒绝。
- 不改变 `KLGrade >= 2` 为阳性的临床标签规则。
- 不改变 L1 Logistic、嵌套分组交叉验证或患者级评价方法。
- 不修改原始用户 CSV；过滤只作用于本次运行的内存数据。

## 3. 统一标签策略

新增一个不依赖特征类型的标签策略模块，提供单一过滤边界。建议接口：

```python
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
    ...
```

处理顺序固定为：

1. 要求存在 `KLGrade` 列。
2. 使用隐私安全的数值转换；错误信息只给出数量，不回显原始值。
3. 仅将数值上恰好等于 `-1` 的行识别为未标注并排除。
4. 空值、非数值、`< -1`、非整数或 `> 4` 均明确失败。
5. 剩余值必须为整数 0、1、2、3、4。
6. 若过滤后没有已标注行，明确失败。
7. 返回全新 DataFrame，不修改调用者传入的数据。

字符串 `"-1"` 和数值 `-1` 语义相同。错误和 diagnostics 不列出 CaseID、
ImagePath、MaskPath 或无界值列表。

## 4. 各组件数据流

### 4.1 MEAS

`OAIZIB_MEAS_FeatureExtractor.process_from_csv()` 在检查影像/掩膜文件和调用
`extract_features()` 之前应用统一标签策略。未标注行完全不加载影像、不计算
MEAS 特征，也不进入输出 CSV/PKL。

提取器记录本次 `input_rows`、`labeled_rows` 和
`excluded_unlabeled_rows`，供 MCP/日志输出汇总诊断；现有 MEAS 特征 CSV 路径
返回合同保持不变。

### 4.2 LDSE

LDSE 的训练拟合和转换入口在读取 CSV 后立即应用统一标签策略：

- `fit_shape_models()` 只使用已标注训练行拟合 scaler/PCA；
- `transform_from_csv()` 只转换已标注行；
- 配对训练/测试流程继续保证 PCA 仅由过滤后的训练集拟合，测试集只 transform；
- 原始输入文件 SHA-256 仍记录完整用户文件，保持输入可追溯；
- manifest 新增训练/测试 `excluded_unlabeled_rows`，并区分输入行数、已标注
  行数、输出行数和其他跳过行数。

过滤在内存中完成，不生成或暴露患者级清洗清单。模型工件继续记录原始训练
输入哈希；PCA 实际拟合样本仅来自已标注训练行。

### 4.3 LASSO 公共入口

`KOALassoClassifier.run()` 在患者表构建、训练/测试重叠检查、模型拟合和输出
暂存之前，对训练和测试特征 CSV 应用同一策略。

这层属于防御性兼容：即使旧 MEAS/LDSE/Radiomics 特征文件仍带有 `-1`，分类
也会安全排除。分类器 diagnostics 分别记录训练和测试排除数量。过滤后仍执行
现有严格检查：

- 训练集必须同时包含两类；
- 每类来源患者数必须满足嵌套分组 CV；
- 测试集必须同时包含两类以计算完整患者级指标；
- CaseID/SourceCaseID、Region、schema、非有限特征和组重叠规则保持不变。

所有结果中的 `NumPatients`、类别计数、OOF 指标、测试指标和 manifest 均只统计
已标注患者。

### 4.4 MCP 与 ClassificationAgent

`koa_lasso_classification` 保持参数和返回结构不变。成功 diagnostics 应包含类似：

```text
train: excluded 21 unlabeled KLGrade=-1 row(s); 383 labeled row(s) remain
test: excluded 5 unlabeled KLGrade=-1 row(s); 98 labeled row(s) remain
```

ClassificationAgent 只在分类工具明确返回 `status="success"` 且输出目录通过必需
产物完整性验证后，才保存 `classification_output_dir` 并执行
`read_classification_results`。

若分类返回 `failed`：

- 清空/不设置报告目录；
- 跳过后续报告读取调用；
- 直接向用户返回原分类失败信息和 diagnostics；
- 不再生成“目录不存在”的二次错误或虚假报告总结。

工具结果可能是字典或 JSON 字符串，状态解析必须同时覆盖这两种形式；无法可靠
解析为成功时按失败处理。

## 5. 错误处理与安全

- 只有 `-1` 可自动排除。
- 空 `KLGrade`、非数值、`-2`、`5`、小数等级等继续失败。
- 错误只报告无效/排除数量，不回显敏感原始值。
- 过滤后空数据、单类数据或分组不足时，不创建分类结果目录。
- LDSE 配对运行失败时继续使用现有事务清理，不留下部分模型、特征或 manifest。
- 不修改原始 CSV，也不自动覆盖已有输出目录。

## 6. 测试设计

### 6.1 标签策略单元测试

- 数值 `-1` 和字符串 `"-1"` 被排除；0–4 保留且顺序不变。
- 输入 DataFrame 不被修改。
- 空值、非数值、`-2`、`5`、1.5 被隐私安全地拒绝。
- 全部为 `-1` 时明确失败。

### 6.2 MEAS

- `-1` 行不检查影像/掩膜路径且不调用 `extract_features()`。
- 有效行正常输出，排除计数正确。
- 输出不含未标注病例。

### 6.3 LDSE

- 训练 PCA/scaler 的输入不含 `-1` 行。
- 测试 transform 不含 `-1` 行。
- 配对 manifest 的原始输入、已标注、排除和输出计数一致。
- 测试数据仍不能参与 scaler/PCA 拟合。
- 原有输入哈希、schema 和事务清理回归继续通过。

### 6.4 LASSO

- 真实分类流程可读取含 `-1` 的旧特征 CSV，并只用有效患者训练/评价。
- 通用测试按输入动态断言：
  `labeled_rows = input_rows - excluded_unlabeled_rows`，且结果表、manifest 与
  diagnostics 中的患者计数均等于 `labeled_rows`。
- 当前数据的训练 383、测试 98 只作为数据特定回归用例，不作为通用计数合同。
- MEAS、LDSE 和 Radiomics 风格特征表共用同一过滤行为。
- `-2`、`5`、非数值仍失败且无输出。
- 过滤后单类或分组不足仍失败。

### 6.5 Agent

- 分类成功后才调用报告读取工具。
- 分类字典返回 `failed` 时不调用报告读取。
- 分类 JSON 字符串返回 `failed` 时不调用报告读取。
- 失败消息保留，且不出现“目录不存在”的二次总结。

## 7. 验证与成功标准

1. 新增测试先产生符合预期的 RED，再实施最小修复。
2. MEAS、LDSE、LASSO 和 ClassificationAgent 聚焦测试全部通过。
3. 现有患者级 L1 Logistic、LDSE 训练拟合、MEAS 元数据兼容和 Agent 路径回归
   全部通过。
4. 可收集广泛测试无新增失败；外部缺失依赖继续单独记录。
5. 静态检查确认没有把 `-1` 映射为二分类标签，也没有测试集拟合或特征交集。
6. 使用当前特征文件的数据特定防御性验证应显示训练排除 21 行、测试排除 5 行，
   保留 383/98 行；这组数字不用于限制未来数据量。完整 LDSE 重提取只在运行环境
   依赖可用时执行。

## 8. 动态数据量与新增数据

所有行数均在每次运行时从输入文件动态计算。训练集或测试集增减病例时，不会仅因
总行数不再是 404/103 或过滤后不再是 383/98 而失败：0–4 的新增病例自动纳入，
`-1` 的新增病例自动排除，输出计数、diagnostics 和 manifest 随输入更新。

通用一致性合同为：

```text
labeled_rows = input_rows - excluded_unlabeled_rows
classification_rows = labeled_rows
extraction_output_rows = labeled_rows - other_skipped_rows
```

当前的 404/103、21/5 和 383/98 只保留为当前数据快照的回归证据。未来新增数据后，
只要上述动态合同成立，就不应因样本数量变化报错；若特征提取还因文件缺失或提取
失败跳过病例，必须在 manifest 中另行计入 `other_skipped_rows`，不能误算为未标注。

新增数据的运行规则：

- 新增训练病例后，必须重新运行训练/测试 LDSE 配对提取，使 scaler/PCA 只在更新后
  的已标注训练集上重新拟合，再在新的空输出目录中重新训练 LASSO。
- 仅新增测试病例时，使用既有训练集拟合并冻结的 LDSE scaler/PCA 对新增测试病例做
  transform，不得用新增测试数据重新拟合；评价结果写入新的空输出目录。
- MEAS 不包含需要从训练集学习的 scaler/PCA；新增已标注病例可直接提取。若训练集
  发生变化，仍需重新训练 LASSO；若仅测试集变化，则用既有训练模型重新评价。

数据量变化本身不是错误，但以下真实约束仍会触发失败：存在除 `-1` 之外的非法
KLGrade、CaseID 缺失或空白、训练/测试来源患者重叠、特征 schema 不一致、非有限
特征值、过滤后单类、来源患者分组不足，或复用非空输出目录。

## 9. 兼容性与迁移

- MEAS、LDSE 和 LASSO 的现有工具参数保持不变。
- 新行为是安全迁移：过去遇到 `-1` 会整体失败，现在只排除明确的未标注行。
- 其他非法标签仍严格失败，避免数据问题被静默掩盖。
- 旧特征 CSV 无需手工编辑即可分类；若要求 `-1` 完全不参与 LDSE PCA，必须
  使用修复后的代码重新提取 LDSE 特征到新的空输出目录。
- README 增加未标注标签策略、排除计数和重新提取说明。
