# 患者级分组 L1 Logistic 分类流程设计

**日期：** 2026-08-02  
**状态：** 已完成架构与行为确认，等待书面规格审阅  
**范围：** `KOALassoClassifier`、患者级 Radiomics 表构建、训练集内部嵌套分组交叉验证、测试集最终评估及兼容性测试

## 1. 背景

当前分类流程存在三类研究方法风险：

1. Radiomics 汇总表中同一 `CaseID` 可对应多个 `Region` 行；将每一行直接视为独立样本会造成患者级伪重复。
2. 训练集与测试集通过特征列交集决定模型输入，意味着测试集参与了特征模式确定；训练阶段应锁定完整特征模式，测试集只能按该模式转换。
3. 当前使用回归型 `LassoCV` 对二分类标签筛选特征。新的主分析应使用以二项分布为目标的 L1 Logistic，并把缺失值处理、标准化、参数选择和特征选择放在训练折内部。

本设计遵循 `radiology-radiomics` 与 `radiology-stats` 的患者级分析、训练内选择、分组验证和独立测试集原则。

## 2. 目标

- Radiomics 多区域数据在建模前转换为一名患者一行的宽表。
- MEAS 和 LDSE 的单行患者特征保持原有表形态。
- 在线或离线增强样本使用 `SourceCaseID` 分组，禁止同源样本跨交叉验证折。
- 所有会学习数据分布或标签关系的步骤仅在相应训练折拟合。
- 训练集锁定模型特征模式；测试集不得通过列交集改变该模式。
- 使用嵌套 `StratifiedGroupKFold` 估计训练集内部性能并选择 L1 Logistic 正则化参数。
- 保留现有 `KOALassoClassifier` 构造方式、MCP 调用入口和主要报告产物。

## 3. 非目标

本轮不实现以下内容：

- MRI 物理空间重采样、强度标准化或 PyRadiomics IBSI 参数文件；
- ICC 特征稳定性分析；
- 校准曲线、校准斜率/截距、Brier score 或决策曲线分析；
- 多中心 ComBat、时间验证或外部验证；
- 临床变量模型、影像组学与临床联合模型；
- 新增模型反序列化接口或跨运行预测服务。

上述内容保留为后续独立优化周期。

## 4. 方案选择

采用模块化方案，而不是继续把数据整理、交叉验证、评估和绘图全部堆叠在 `koa_lasso_classifier.py` 中。

### 4.1 `PatientFeatureTableBuilder`

职责：将输入特征表转换为患者级建模表，并生成稳定的分组键。

建议位置：

```text
src/algorithm/feature_class_algorithm/patient_feature_table.py
```

核心接口：

```python
class PatientFeatureTableBuilder:
    def build(self, df: pd.DataFrame, *, by_region: bool = False) -> PatientFeatureTable:
        ...
```

`PatientFeatureTable` 至少包含：

```python
@dataclass(frozen=True)
class PatientFeatureTable:
    frame: pd.DataFrame
    feature_columns: tuple[str, ...]
    group_column: str
    source_kind: str
```

行为：

- 输入存在 `Region` 且 `by_region=False` 时，以 `CaseID` 为行索引，以 `Region` 为列前缀展开数值特征。
- 区域前缀规范为 `R{region}__{original_feature_name}`，例如 `R2__original_shape_Volume`。
- 输入存在 `Region` 且 `by_region=True` 时，不跨区域合并；由分类器逐个 Region 调用构建器，每个区域仍是一名患者一行。
- 输入没有 `Region` 时，保留 MEAS/LDSE 的一名 `CaseID` 一行结构。
- `SourceCaseID` 非空时，分组键取其字符串值；否则取 `CaseID`。
- 同一 `CaseID`、同一 `Region` 出现重复行时拒绝处理，不进行静默平均。
- 同一 `CaseID` 的不同 Region 必须具有一致的 `KLGrade` 和 `SourceCaseID`；冲突时拒绝处理。
- 同一分组键内的所有样本必须具有一致的二分类标签。
- `KLGrade` 仅接受 0–4；二分类标签仍定义为 `KLGrade >= 2`。
- 元数据字段不进入模型特征。

### 4.2 `LeakageSafeL1Logistic`

职责：执行训练集内部嵌套分组交叉验证、拟合最终模型、锁定特征模式并转换测试集。

建议位置：

```text
src/algorithm/feature_class_algorithm/leakage_safe_l1_logistic.py
```

核心接口：

```python
class LeakageSafeL1Logistic:
    def fit(self, train: PatientFeatureTable) -> "LeakageSafeL1Logistic":
        ...

    def predict_proba(self, test: PatientFeatureTable) -> np.ndarray:
        ...

    @property
    def selected_features(self) -> tuple[str, ...]:
        ...
```

训练 Pipeline 固定为：

```text
SimpleImputer(strategy="median")
→ StandardScaler()
→ LogisticRegression(penalty="l1", solver="liblinear", max_iter=5000, random_state=42)
```

参数选择：

- 候选 `C` 为 `np.logspace(-4, 4, 17)`。
- 内层评分使用 ROC AUC。
- 内层和外层均使用 `StratifiedGroupKFold(shuffle=True, random_state=42)`。
- 外层折数为 `min(5, 训练集中每类的独立分组数)`；每类至少需要 3 个独立分组。
- 每个外层训练子集重新计算可用的内层折数；每类至少需要 2 个独立分组，否则以清晰诊断终止。
- 嵌套外层产生训练集内部的 out-of-fold 概率，仅用于内部性能摘要，不参与最终测试评价。
- 完成内部验证后，在完整训练集上重新执行一次分组内层参数选择并拟合最终 Pipeline。

特征模式：

- 训练表决定有序 `feature_columns`。
- 全部缺失或无法转换为数值的训练特征在拟合前被拒绝，并在错误中列出字段；不从测试集推断如何处理。
- 测试表缺少任一训练特征时拒绝预测，并列出缺失字段。
- 测试表的额外特征被忽略并记录，不改变训练特征顺序。
- 训练集与测试集的分组键集合必须完全不相交；发现重叠时拒绝运行并列出有限数量的冲突标识。

### 4.3 `KOALassoClassifier` 兼容层

现有接口保持不变：

```python
KOALassoClassifier(train_path, test_path, output_dir, by_region=False).run()
```

兼容行为：

- `by_region=False`：Radiomics 多区域先转换为患者级宽表；MEAS/LDSE 直接进入患者级分类。
- `by_region=True`：训练集锁定 Region 列表，每个训练 Region 独立建立患者级模型，输出继续写入对应 `region_{region}` 目录；测试集缺少训练 Region 时失败，额外测试 Region 被忽略并记录。
- MCP 工具参数和 Agent 调用方式不改变。
- 保留 `metrics_summary.csv`、文本分类报告、ROC 图、PR 图和选中特征文件。
- 原 `LassoCV` 均方误差图不再代表正确的二分类目标，替换为 L1 Logistic 的交叉验证 `C`–AUC 摘要图；迁移说明写入 README。
- `ALL` 全特征模型不再作为主分析。为兼容旧报告，保留为明确标注的探索性 L2 Logistic 基线，并使用相同的训练内预处理和分组调参边界。

## 5. 数据流

```text
训练 CSV / 测试 CSV
        ↓
输入字段和 KLGrade 校验
        ↓
PatientFeatureTableBuilder
  ├─ Radiomics Region → 患者级宽表
  └─ MEAS/LDSE → 保持患者级单行
        ↓
构造 group_id（SourceCaseID 优先）
        ↓
训练/测试 group_id 重叠审计
        ↓
训练集内部嵌套 StratifiedGroupKFold
  └─ 每折重新拟合 imputer/scaler/L1 Logistic
        ↓
完整训练集重新选择 C 并拟合最终模型
        ↓
按训练特征模式转换测试集
        ↓
若测试集有同源增强样本，按 group_id 平均概率
        ↓
患者级最终指标与报告
```

测试集概率聚合规则：

- 每个 `group_id` 只有一行时直接使用该概率。
- 同一 `group_id` 有多个增强样本时，对阳性概率取算术平均。
- 同一 `group_id` 的真实标签不一致时拒绝评价。
- 聚合后再计算混淆矩阵、ROC、PR 和分类报告，避免增强样本放大患者权重。

## 6. 错误处理

以下情况必须以 `ValueError` 或领域专用异常终止，不得降级为静默修补：

- 缺少 `CaseID` 或 `KLGrade`；
- Radiomics 数据缺少有效 `Region` 值；
- 同一患者/区域重复；
- 同一患者不同 Region 的标签或来源患者不一致；
- 同一来源患者存在冲突标签；
- 训练集和测试集存在相同 `group_id`；
- 训练集无数值特征；
- 训练特征含无法处理的非数值值或整列缺失；
- 测试集缺少训练所需特征；
- 任一数据集只有一个类别；
- 独立分组数不足以完成规定的嵌套分组交叉验证；
- L1 Logistic 最终未选择任何特征。

错误消息必须包含数据集角色、问题类型和有限数量的字段或患者标识，且不得输出影像路径或其他敏感元数据。

## 7. 输出产物

保留：

- `metrics_summary.csv`
- `metrics_LASSO*.txt`
- `ROC曲线_LASSO*.png`
- `PR曲线_LASSO*.png`
- `LASSO_selected_features*.txt`
- 现有 SHAP 图；SHAP 使用最终 Logistic 模型和训练模式后的数据

新增：

- `patient_feature_schema*.json`：数据类型、训练特征顺序、区域展开规则、分组字段和标签规则；
- `nested_cv_summary*.json`：外层折数、每折最佳 `C`、训练集 out-of-fold 指标和最终最佳 `C`；
- `model_coefficients*.csv`：特征、标准化尺度下的 Logistic 系数和是否被 L1 选中；
- `classification_run_manifest*.json`：输入文件绝对路径、SHA-256、随机种子、软件版本、样本/分组/类别计数和输出文件清单。

所有 JSON 采用 UTF-8、稳定键顺序和原子写入。现阶段不新增可由外部路径加载的 pickle/joblib 模型文件。

## 8. 测试设计

### 8.1 患者级表

- 两名患者、多个 Region 正确展开为两行，列名带稳定区域前缀。
- Region 输入顺序变化不改变输出特征顺序。
- 同一患者标签冲突、来源患者冲突或重复 Region 行时失败。
- MEAS/LDSE 无 Region 输入保持一名患者一行。

### 8.2 分组与泄漏

- `SourceCaseID` 存在时优先作为分组键。
- 同源增强样本从不跨内层或外层折。
- 训练集与测试集共享来源患者时在拟合前失败。
- 测试集额外列被忽略；缺少训练特征时失败。
- 测试集异常值不会改变训练期 imputer、scaler、最佳 `C` 或选中特征。

### 8.3 模型

- 真实 Pipeline 的最终估计器为 L1 Logistic，不再是 `LassoCV`。
- 每个交叉验证拆分同时满足类别分层和分组隔离。
- 固定随机种子重复运行得到相同折、最佳 `C`、特征列表和概率。
- 训练或测试单类、分组数不足、零选中特征均返回明确错误。

### 8.4 评价与输出

- 同一 `SourceCaseID` 的多个测试增强概率先平均，再计算患者级指标。
- 主要既有报告文件仍生成。
- 新增 schema、nested CV、系数和 manifest 文件字段完整、可 JSON/CSV 读取。
- README 的 LASSO 方法说明与实际实现一致。

### 8.5 回归

- 现有 MEAS–LASSO 元数据兼容测试继续通过。
- 现有 LDSE 训练拟合/测试转换测试继续通过。
- 现有 MCP LASSO 参数和结果读取工具测试继续通过。
- 当前环境无法收集的可选依赖测试必须单独列为环境限制，不得计为通过。

## 9. 成功标准

- Radiomics 非区域模型不再把不同 Region 行当作独立患者。
- 所有训练型预处理和 L1 选择均发生在训练折内部。
- 同源患者不会跨训练/验证折或训练/测试边界。
- 测试集不再参与特征模式确定或超参数选择。
- 最终测试指标按患者/来源患者计算。
- 对外分类工具调用保持兼容，新增产物可追溯本次分析。
- 聚焦测试、可收集广泛测试和独立代码复核均无 Critical/Important 残留后，才关闭本优化周期。

## 10. 后续优化顺序

本设计完成后，后续独立周期依次为：

1. MRI/Mask 物理空间及 PyRadiomics IBSI 参数标准化；
2. 基于重复分割或 mask 扰动的 ICC 稳定性筛选；
3. 95% CI、校准、Brier score 和 DCA；
4. 多中心、时间或外部验证与中心效应分析；
5. CLEAR/IBSI/TRIPOD+AI 报告和完整可复现性包。
