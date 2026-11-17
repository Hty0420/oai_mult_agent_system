# 项目目录导读

本文按当前本地项目目录整理。源码、配置和测试逐项列出；Git 内部文件、第三方环境包、缓存、数据库数据和运行日志不逐项展开。

## 1. 根目录

```text
oai_multi_agent_system_new/
|-- src/                            应用源码
|-- tests/                          自动化测试及 MCP 冒烟验证脚本
|-- docs/                           架构、设计、实施计划和本导读
|-- assets/                         页面使用的图片资源
|-- oai_multi_agent_system/          本次创建的 Python 环境和本地 MySQL 数据
|-- .venv/                          原有 Python 环境
|-- output/                         本地运行输出、日志
|-- tmp/                            临时脚本和文件
|-- user_upload_file/               根目录启动时可能使用的上传目录
|-- .git/                           Git 历史和内部数据
|-- .idea/                          PyCharm 配置
|-- .pytest_cache/                  pytest 缓存
|-- .env                            本地密钥，已被 Git 忽略
|-- .env.example                    环境变量模板，不应填写真实密钥
|-- .gitignore                      Git 忽略规则
|-- llm_config.json                 统一的模型、接口和密钥变量名配置
|-- pyproject.toml                  Python 版本要求和依赖声明
|-- uv.lock                         锁定的依赖版本和来源
|-- start_oai_multi_agent_system.ps1 本地启动脚本，按需启动 MySQL 后启动应用
|-- README.md                       项目介绍、安装和使用说明
|-- PROJECT_SUMMARY.md              项目总结文档
|-- prompt.txt                      示例研究任务输入
`-- preview_csv.py                  CSV 预览辅助脚本
```

当前源码目录内没有随项目提供的 `datasets/` 数据集。README 中提到的数据路径需要使用者另行准备，不应把文档中的路径当成已存在的文件。

## 2. src：应用入口和配套文件

```text
src/
|-- __init__.py
|-- app.py                   FastAPI 路由、Gradio 页面、登录注册和任务界面
|-- oai_client.py            OAI_ChatBot，管理对话、LLM 客户端、MCP 连接和工具映射
|-- llm_config.py            读取根目录 llm_config.json 和 .env，创建统一 LLM 客户端
|-- server_config.json       定义需要启动的 MCP 工具服务器
|-- mesh_visualization.py    三维网格及相关工件的前端展示辅助
|-- system_capabilities.py   组织和展示系统能力、智能体与工具状态
|-- augment_data.py          数据增强辅助脚本，含示例数据路径
|-- agents/                 智能体及编排逻辑
|-- server/                 MCP 工具服务入口
|-- algorithm/              实际科研计算
|-- db_utils/               持久化、任务轨迹、评估及优化支持
|-- evaluation/             评估和能力配置 JSON
|-- env_utils/              环境变量辅助代码
|-- user_upload_file/       从 src 启动时使用的上传目录
`-- __pycache__/            Python 生成的缓存
```

学习时先看 `app.py` 的页面、事件绑定和启动入口，再追踪 `OAI_ChatBot`。不必第一遍逐行读完较大的界面文件。

## 3. agents：决定做什么、怎样组织任务

```text
agents/
|-- __init__.py
|-- config.py                       各智能体配置
|-- state.py                        多智能体共享状态和路由决策结构
|-- orchestrator.py                 LangGraph 工作流总调度器
|-- orchestrator_agent.py           复杂多步骤请求的规划与协调智能体
|-- router_agent.py                 判断请求应交给哪个智能体
|-- conversation_agent.py           通用对话和解释
|-- segmentation_agent.py           分割任务的智能体层
|-- feature_extraction_agent.py     特征提取任务的智能体层
|-- feature_combination_agent.py    特征组合任务的智能体层
|-- classification_agent.py         分类任务的智能体层
|-- web_search_agent.py             搜索任务的智能体层
|-- pipeline_optimization_agent.py  流程优化智能体
|-- biomarker_discovery_agent.py    生物标志物发现智能体
|-- biomarker_verifier_agent.py     生物标志物核验智能体
|-- biomarker_reporter_agent.py     生物标志物报告智能体
|-- tool_manager.py                 MCP 工具管理和调用
|-- tool_policy.py                  工具调用约束
|-- tool_progress.py                工具执行进度
|-- tool_result_utils.py            工具结果处理
|-- task_contracts.py               任务规格及计划约束
|-- pipeline_planner.py             生成、排序和选择流程候选
|-- dataset_profiler.py             数据集结构分析
|-- feature_plan_utils.py           特征任务计划辅助
|-- path_utils.py                   路径和工具结果路径处理
|-- llm_json.py                     模型 JSON 输出解析
|-- scientific_trajectory_graph.py  科研任务轨迹图和报告
|-- biomarker_summary.py            生物标志物结果摘要
`-- biomarker_reporting.py          生物标志物报告辅助
```

注意两个名字相近的文件：`orchestrator.py` 负责运行整个图和调度节点；`orchestrator_agent.py` 是图中处理复杂任务的一个智能体。智能体层规划或调用工具，不等于算法本身。

## 4. server：把计算功能暴露为 MCP 工具

```text
server/
|-- __init__.py
|-- auto_experiment_server.py       实验、特征提取、分类等工具服务
|-- lasso_class_features_server.py  特征组合和 LASSO 分类工具服务
|-- cartilage_analysis_server.py    软骨厚度分析工具服务
|-- biomarker_discovery_server.py   生物标志物分析工具服务
|-- web_search_server.py            智谱搜索工具服务
`-- ldse_tool_handlers.py           LDSE 工具处理辅助代码
```

`server_config.json` 配置的是五个服务入口；`ldse_tool_handlers.py` 是辅助模块。MCP 在这里是应用与工具之间的协议，项目通过标准输入输出与 Python 子进程通信。

## 5. algorithm：实际计算实现

```text
algorithm/
|-- __init__.py
|-- klgrade_policy.py                        KL 分级相关规则
|-- KOA_feature_extraction/
|   |-- __init__.py
|   |-- meas_feature_extraction.py           MEAS 特征提取
|   `-- ldse_feature_extraction.py           LDSE 特征提取
|-- feature_class_algorithm/
|   |-- __init__.py
|   |-- koa_lasso_classifier.py              LASSO 分类流程
|   |-- universal_filter_classifier.py       Filter 特征选择和分类
|   |-- leakage_safe_l1_logistic.py          防数据泄漏的 L1 Logistic 实现
|   |-- patient_feature_table.py             患者级特征表处理
|   |-- grouped_validation.py                分组验证
|   |-- training_only_validation.py          仅使用训练数据的候选验证
|   |-- classification_artifacts.py          分类输出工件
|   `-- classification_output_contract.py   分类输出检查
|-- cartilage_thickness/
|   |-- __init__.py
|   |-- service.py                          分析流程入口
|   |-- thickness.py                        厚度计算
|   |-- subregion.py                        亚区处理
|   `-- mesh.py                             网格处理
|-- biomarker_discovery/
|   |-- __init__.py
|   |-- service.py                          生物标志物分析流程入口
|   `-- statistics.py                       统计分析
|-- data_augmentation/
|   |-- __init__.py
|   |-- offline_augmentor.py                离线增强
|   |-- online_augmentor.py                 在线增强
|   |-- transforms.py                       变换操作
|   `-- utils.py                            增强辅助函数
`-- feature_quality/
    |-- __init__.py
    `-- quality_assessor.py                 特征质量评估
```

影像组学工具也直接在 `server/auto_experiment_server.py` 中调用 PyRadiomics，因此算法逻辑并非全部集中在 `algorithm/`。分割智能体的存在也不表示所有分割模型和权重都包含在当前目录中，具体实现和资源要沿调用链确认。

## 6. db_utils：保存数据、评估结果和积累经验

```text
db_utils/
|-- db_config.example.json        数据库配置模板
|-- db_config.json                本地数据库凭据，已被 Git 忽略
|-- db_utils.py                   DatabaseManager、建表及数据访问
|-- workspace_memory.py           工作区记忆
|-- trajectory.py                 执行轨迹和审计数据结构
|-- task_catalog.py               任务目录处理
|-- task_evaluator.py             任务结果评估
|-- task_learning.py              任务结果与学习记录
|-- scientific_quality.py         科研质量检查
|-- error_categorizer.py           错误分类
|-- pipeline_trials.py            流程试验记录
|-- offline_optimizer.py          离线优化
|-- self_optimization_control.py   自优化控制
|-- experiment_etl.py             实验数据转换和整理
|-- analysis_queries.py           分析查询
`-- fix_database_charset.py       数据库字符集维护辅助脚本
```

这个目录包含数据库访问，也包含依赖持久化结果的评估和优化逻辑。MySQL 程序本身不在这里；这里只放连接配置和 Python 代码。

## 7. evaluation 和 env_utils

```text
evaluation/
|-- task_catalog.v1.json                      任务目录
|-- pipeline_capabilities.v1.json             流程能力目录
|-- self_optimization_constitution.v1.json    自优化规则
|-- biomarker_task_catalog.v1.json            生物标志物任务目录
`-- biomarker_pipeline_capabilities.v1.json   生物标志物流程能力目录

env_utils/
`-- env_util.py                              环境变量辅助代码
```

JSON 文件是程序读取的规则和配置，不是训练数据。主聊天与多智能体的统一模型配置入口是 `llm_config.py`；不要把 `env_util.py` 中列出的旧模型环境变量当成当前统一配置入口。

## 8. tests：可执行的行为说明

```text
tests/
|-- conftest.py
|-- test_llm_config.py
|-- test_tool_connection_status.py
|-- test_tool_progress.py
|-- test_trajectory_report_refresh.py
|-- test_system_capabilities.py
|-- test_biomarker_contracts.py
|-- test_biomarker_policy_and_evaluation.py
|-- test_biomarker_reporting.py
|-- test_biomarker_service.py
|-- test_biomarker_statistics.py
|-- test_biomarker_summary.py
`-- mcp_biomarker_smoke.py
```

`conftest.py` 提供测试公共设置。`test_*.py` 是 pytest 测试；`mcp_biomarker_smoke.py` 是 MCP 冒烟验证脚本。阅读某个功能时，可以同时阅读相应测试，了解预期行为和边界。

## 9. docs 和 assets

```text
docs/
|-- project-directory-guide.md
|-- oai_mcp_architecture.svg
|-- autonomous-kl-planning.md
|-- trajectory-optimization.md
|-- self-optimization-control.md
|-- multi-agent-optimization-summary-and-roadmap.md
`-- superpowers/
    |-- specs/
    |   |-- 2026-06-13-data-augmentation-design.md
    |   |-- 2026-06-15-radiomics-feature-quality-layer-design.md
    |   |-- 2026-08-01-ldse-train-fitted-pca-design.md
    |   |-- 2026-08-01-meas-lasso-metadata-compatibility-design.md
    |   |-- 2026-08-02-patient-grouped-l1-logistic-design.md
    |   `-- 2026-08-02-unlabeled-klgrade-filtering-design.md
    `-- plans/
        |-- 2026-06-13-data-augmentation-implementation.md
        |-- 2026-06-15-radiomics-feature-quality-layer.md
        |-- 2026-08-01-ldse-train-fitted-pca.md
        |-- 2026-08-01-meas-lasso-metadata-compatibility.md
        |-- 2026-08-02-patient-grouped-l1-logistic.md
        `-- 2026-08-02-unlabeled-klgrade-filtering.md

assets/
`-- default_placeholder.png
```

`specs/` 保存设计说明，`plans/` 保存实施计划；它们有助于理解设计缘由，但运行行为应以当前代码和测试为准。

## 10. 运行环境与本地数据

```text
oai_multi_agent_system/
|-- Scripts/                Python 解释器、激活脚本及命令工具
|-- Lib/site-packages/      安装的第三方 Python 包
|-- Include/               Python 环境相关头文件目录
|-- pyvenv.cfg              虚拟环境信息
`-- mysql_runtime/
    |-- data/              MySQL 数据文件，不能当作普通缓存删除
    |-- admin.json         本地数据库管理员凭据
    `-- server.log         数据库启动及运行日志
```

当前这个环境不仅含 Python 包，还存放了本地 MySQL 数据。删除整个环境目录会连数据库数据一起删除。学习源码时跳过这里，但管理环境时要区别对待软件包和业务数据。

`output/`、`tmp/`、上传目录及缓存目录是本地运行辅助目录。上传文件也可能被数据库记录引用，不应一概视为可随意清空的缓存。

## 11. 建议阅读路线

1. `README.md` 和根目录配置：弄清项目目标、依赖和启动方式。
2. `src/app.py`：找到启动入口、页面事件以及发起任务的位置。
3. `src/oai_client.py`：理解对话、模型调用、MCP 连接和数据库的关系。
4. `src/agents/state.py`、`config.py`、`orchestrator.py`、`router_agent.py`：理解任务在图中如何流转。
5. 选择一个专用智能体，再追踪 `tool_manager.py` 和对应 `server/` 工具。
6. 追踪 `algorithm/` 的实际计算，结合相关测试理解输入输出。
7. 阅读 `db_utils/` 和 `evaluation/`，理解历史记录、质量检查和优化反馈。

典型调用方向：页面事件 → 聊天客户端 → 智能体编排 → MCP 工具服务 → 科研计算 → 返回结果。数据库记录和评估逻辑贯穿其中，并非只在流程末尾执行。
