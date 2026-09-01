# KL 分级目标驱动自动规划

## 1. 本阶段实现范围

系统现在支持用户只给出研究目标，而不必指定 MEAS、LDSE、Radiomics、LASSO 或 Filter 等步骤。例如：

```text
完成当前数据的 KL 分级任务
```

在本项目专用场景中，极简输入“完成任务”也会映射到默认 KL 目标。

系统现在采用两阶段选择：先用能力、历史可靠性和成本形成候选排序，再在当前训练集内部对预算化 Top-K 候选做患者级嵌套交叉验证。历史测试集上的 AUC/F1 不会跨不匹配数据集直接参与自动选择，当前测试集的标签和指标也不会参与候选排序。

## 2. 三种任务模式

| 模式 | 示例 | 行为 |
|---|---|---|
| `auto` | `完成KL分级任务` | 自动选择特征与分类方法 |
| `partially_constrained` | `使用MEAS完成KL分级任务` | 固定 MEAS，自动选择分类方法 |
| `manual` | `提取MEAS特征并使用LASSO分类` | 保持用户指定的执行约束 |

`GoalSpec` 与原有执行 `TaskSpec` 分离。大模型理解目标并提出候选 ID，确定性控制层检查候选是否可执行、是否满足用户约束，以及是否处于最高分候选的允许近似并列范围内。

## 3. 数据画像

自动规划前会对当前请求中的 CSV 进行只读画像。数据来源优先级为：当前请求显式路径、当前用户与当前会话的 `user_files` 登记记录、限定的 `user_upload_file` 默认路径。不会从聊天文本中猜测其他任务的文件路径。

画像包含 train/test 角色、列集合、行数、KLGrade 聚合分布、文件 SHA-256、组合数据集指纹和 schema 问题。画像不保存病例行、CaseID、ImagePath 或 MaskPath 的具体内容。

## 4. 候选能力注册表

版本化注册表位于 `src/evaluation/pipeline_capabilities.v1.json`。首批候选包括 MEAS、LDSE、Radiomics 分别与 LASSO、Filter 组合的 6 条流水线。注册表版本 `1.1.0` 开始，每条候选还必须声明可执行的 `model_recipe` 与 `recipe_config`：LASSO 映射到 `L1_LOGISTIC`，Filter 映射到 `ANOVA_TOPK_L2`。

每个候选声明必要列、Agent 序列、工具序列、填补/标准化/选择器/估计器/超参数空间、冷启动先验与成本等级。缺少必要列、配方与旧方法名不兼容、目标类型不受支持或违反用户部分约束的候选会在执行前被拒绝。

## 5. 历史排序

排序读取现有轨迹中的任务级可靠性、同任务指纹下的成对 Trial 证据和工具失败率：

- `task_success_rate`；
- `false_completion_rate`；
- 工具近 30 天失败率；
- 候选训练内 `validation_score`、AUC 置信下界与特征选择稳定性；
- 版本化冷启动先验；
- 计算成本等级。

系统使用 Beta(1,1) 平滑，避免单次历史成功或失败支配选择。历史数据不可用时自动回退到版本化冷启动排序，不会阻断任务。

大模型可在综合分差距不超过 `llm_tie_tolerance` 的合法候选之间打破近似并列；不能选择已被前置条件淘汰的候选，也不能用明显低分候选覆盖控制层选择。

## 6. 执行与审计

自动目标会被物化为：

```text
PIPELINE_OPTIMIZATION_AGENT
  -> CLASSIFICATION_AGENT（胜出方案的唯一最终分类 + 读取本次报告）
```

冷启动默认试跑 3 个多样化候选，至少覆盖不同特征族和不同分类方法；当当前领先方案积累不少于 5 次匹配历史后，预算收缩为“领先方案 + 1 个挑战方案”。同一特征族在一次搜索中只提取一次并复用。

候选验证工具的接口只有 `train_path`，没有 `test_path`。LASSO、Filter 的插补、标准化、特征选择和调参都位于训练折内部。每个候选额外输出患者级 bootstrap 95% CI、外层折特征集合的平均两两 Jaccard 稳定性，以及由外层折汇总的推荐参数。近似并列时依次比较 AUC 置信下界、特征稳定性和原有先验。

胜出方案确定后，分类智能体才执行一次最终测试集评估，且不得根据测试结果重新选方案。`Filter` 的最终执行已固定为同一个 `ANOVA_TOPK_L2` 配方，并强制接收胜出 Trial 的 `selector__k` 与 `model__C`；不会再在测试阶段比较 Chi²、互信息或全特征后改选。

源 CSV 使用绝对路径传递；最终分类只能使用当前 `run_id` 下胜出方案的 train/test 特征工件。每个尝试过的候选（包括失败候选）会先生成不可变的 `pipeline_trials` 记录，包含候选配置哈希、训练数据哈希、验证器版本、OOF 指标、区间、稳定性、耗时、推荐参数与失败原因。初始候选写入 `pipeline_candidates_generated`，试跑摘要写入 `pipeline_trial_completed`，最终选择写入 `pipeline_selected`。搜索清单注册到 `artifact_registry`，交叉验证指标以 `split_type=cross_validation` 写入 `evaluation_results`。

API 返回新增：`goal_spec`、`dataset_profile`、`pipeline_candidates`、`selected_pipeline` 和 `planning_decision`。

## 7. 当前能力边界

现有 `koa_lasso_classification` 和 `koa_filter_classification` 将 `KLGrade` 转换成 `KLGrade < 2` 与 `KLGrade >= 2`。

因此当前自动目标实际是基于 KLGrade 的 KOA 二分类，不是 KLGrade 0–4 五分类。用户明确要求 `0-4`、五分类、多分类或 ordinal 分类时，自动规划会拒绝现有候选并报告能力边界，而不会错误调用二分类工具后宣称完成五分类。

## 8. 验证方式

重启 Agent 服务后发送“完成当前数据的 KL 分级任务”，检查：

1. 路由进入 `ORCHESTRATOR_AGENT`；
2. `dataset_profile.status` 为 `ready`；
3. `pipeline_candidates` 包含合法候选与拒绝原因；
4. `planning_decision.trial_portfolio` 冷启动预算为 3；
5. 执行步骤为 `PIPELINE_OPTIMIZATION_AGENT` 后接一次 `CLASSIFICATION_AGENT`；
6. 分类失败时报告读取被阻断；
7. `evaluation_results` 出现候选的 `cross_validation` 指标和最终方案的测试指标；
8. `trajectory_events` 中依次出现 `pipeline_candidates_generated`、`pipeline_trial_completed`、`pipeline_selected`；
9. `pipeline_search_manifest.json` 明确记录 `test_labels_or_metrics_used_for_selection=false`。
10. `pipeline_trials` 对每个已尝试候选存在一条记录，且 `test_data_used=0`。

## 9. 后续增强

当前已完成预算化 Top-K 训练集验证、原子配方、候选级 Trial、bootstrap CI 和稳定性排序。下一阶段建议依次增加：

1. 在现有两种原子配方稳定运行后，再按受控变更加入 Elastic Net、MI-TopK-L2 和无选择 L2 基线；每个新配方必须同时实现训练验证与最终执行，不能只扩充注册表；
2. 引入冻结测试集登记和一次性使用门，令 `test_is_frozen` 从 `unknown` 变为可验证；
3. 增加周期性探索率和漂移检测，防止历史领先方案长期垄断；
4. 为同一数据快照增加确定性的多随机种子重复验证，而不是把重复运行误当成新任务；
5. 实现真正的 KLGrade 0–4 有序分类候选与二次加权 Kappa 评价。
