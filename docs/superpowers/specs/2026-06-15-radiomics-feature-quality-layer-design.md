# Phase 1: 影像组学特征质量层设计

日期: 2026-06-15
状态: 待实施
关联: [[2026-06-15-radiomics-multi-agent-optimization]] (整体优化方案)

## 背景

OAI MCP 项目当前支持三种特征提取方法（Radiomics ~107-1316维、MEAS ~30维、LDSE ~250维）和两种分类方法（LASSO、Filter），但特征提取和分类之间缺少质量把关环节：

- 无低方差特征过滤
- 无共线性检测
- OnlineAugmentor 产出的 `_std` 列（特征不确定性）未被下游使用
- Agent 不会主动建议质量检查

本设计是整体优化方案的第一阶段，目标是在分类前建立特征质量门控。

## 目标

1. 新增 `assess_feature_quality` MCP 工具，对特征 CSV 进行自动化质量评估
2. 增强 `FeatureExtractionAgent`，使其主动建议使用在线增强和质量检查
3. 增强 `ClassificationAgent`，使其在分类前自动检查并使用质量过滤后的特征
4. 新增 `FeatureQualityAssessor` 算法类，与 MCP 层解耦

## 架构

### 新增模块

```
src/algorithm/feature_quality/
├── __init__.py               # 导出 FeatureQualityAssessor
└── quality_assessor.py       # 核心评估逻辑
```

### 数据流

```
用户请求特征提取
       │
       ▼
FeatureExtractionAgent
  ├─ 建议使用 augment_online_extract_features（鲁棒特征+不确定性）
  ├─ 调用 extract_radiomics_features / meas_features_extraction / ldse_features_extraction
  └─ 建议调用 assess_feature_quality
       │
       ▼
assess_feature_quality (MCP工具)
  ├─ 基础统计（样本数、缺失值、KL分布）
  ├─ 低方差过滤（variance_threshold=0.01）
  ├─ 共线性检测（correlation_threshold=0.95）
  ├─ 稳定性评估（CV > cv_threshold, 仅当有 _std 列）
  └─ 输出：quality_report.txt + filtered_features.csv + quality_metrics.json
       │
       ▼
ClassificationAgent
  ├─ 检查 filtered_features.csv 是否存在
  ├─ 有 → 使用过滤后特征进行分类
  └─ 无 → 建议先做质量评估
       │
       ▼
koa_lasso_classification / koa_filter_classification
```

## 详细设计

### 1. FeatureQualityAssessor 类

文件: `src/algorithm/feature_quality/quality_assessor.py`

```python
class FeatureQualityAssessor:
    def __init__(
        self,
        variance_threshold: float = 0.01,
        correlation_threshold: float = 0.95,
        cv_threshold: float = 0.3,
    ): ...

    def assess(self, df: pd.DataFrame, label_col: str = "KLGrade") -> QualityReport:
        """执行完整质量评估"""
        # 1. 基础统计
        # 2. 低方差过滤
        # 3. 共线性检测（保留与标签更相关的特征）
        # 4. 稳定性评估（检测 _std 列）
        # 返回 QualityReport

    def filter_features(self, df: pd.DataFrame, report: QualityReport) -> pd.DataFrame:
        """根据评估报告过滤特征"""

class QualityReport:
    """质量报告数据结构"""
    total_samples: int
    total_features: int
    missing_count: int
    kl_distribution: dict
    low_variance_features: list[str]
    high_correlation_pairs: list[tuple[str, str, float]]  # (feat1, feat2, r)
    unstable_features: list[tuple[str, float]]  # (feat, cv)
    quality_score: int  # 0-100
    recommendations: list[str]
```

### 2. MCP 工具: assess_feature_quality

文件: `src/server/auto_experiment_server.py`

```python
@mcp.tool()
def assess_feature_quality(
    feature_csv: str,              # 特征 CSV 路径
    label_col: str = "KLGrade",    # 标签列名
    variance_threshold: float = 0.01,
    correlation_threshold: float = 0.95,
    cv_threshold: float = 0.3,
    output_dir: str,               # 输出目录
) -> dict:
    """
    返回:
    {
        "status": "success" | "failed",
        "report_path": "quality_report.txt",
        "filtered_csv": "filtered_features.csv",
        "metrics_json": "quality_metrics.json",
        "quality_score": 85,
        "summary": "..."
    }
    """
```

### 3. FeatureExtractionAgent 增强

文件: `src/agents/feature_extraction_agent.py`

在 system_prompt 末尾添加:
```
FEATURE QUALITY GUIDELINES:
- When user intends to use extracted features for subsequent classification,
  suggest using augment_online_extract_features for robust features with
  per-feature uncertainty estimates (_std columns)
- After feature extraction completes, suggest running assess_feature_quality
  to detect and filter low-variance, collinear, or unstable features
- If user requests both training and test set extraction, remind them
  that quality should be assessed on the training set first
```

在可用工具列表中添加 `assess_feature_quality`。

### 4. ClassificationAgent 增强

文件: `src/agents/classification_agent.py`

在 system_prompt 中的分类流程中插入质量检查步骤:
```
1. Identify classification method
2. Extract train/test paths
3. [NEW] Check if filtered_features.csv exists for training set
   → If yes, use filtered features for classification
   → If no, suggest running assess_feature_quality first
4. Call classification tool(s)
5. Report results
```

在可用工具列表中添加 `assess_feature_quality`，并在 process_input 中传入检测到的质量报告路径。

## 涉及文件

| 文件 | 改动类型 | 说明 |
|------|---------|------|
| `src/algorithm/feature_quality/__init__.py` | 新增 | 模块入口 |
| `src/algorithm/feature_quality/quality_assessor.py` | 新增 | 核心评估逻辑 |
| `src/server/auto_experiment_server.py` | 修改 | 新增 `assess_feature_quality` 工具 |
| `src/agents/feature_extraction_agent.py` | 修改 | 增强 system prompt，加入质量指南 |
| `src/agents/classification_agent.py` | 修改 | 增加质量感知逻辑 |

## 测试策略

| 层级 | 内容 |
|------|------|
| 单元测试 | `test_quality_assessor.py`: 测试方差过滤、共线性检测、CV 计算、质量评分 |
| 集成测试 | 端到端流程: 特征提取 → 质量评估 → 分类，验证 filtered_features.csv 被正确使用 |
| 回归测试 | 确保现有 MEAS/LDSE/radiomics 提取和 LASSO/Filter 分类不受影响 |

## 验收标准

1. `assess_feature_quality` 工具能正确识别低方差、高共线性和不稳定特征
2. 输出 `quality_report.txt` 包含所有评估指标和建议
3. `filtered_features.csv` 剔除问题特征后仍保持正确格式（CaseID/KLGrade 列完整）
4. `FeatureExtractionAgent` 在分类场景下会主动建议质量评估
5. `ClassificationAgent` 会检测并使用已有的过滤后特征
6. 质量评分逻辑合理：无问题=100，每类问题按严重程度扣分
