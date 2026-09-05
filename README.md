# OAI Agent System — 膝骨关节炎多智能体医学影像分析系统

基于 **LangGraph 多智能体架构** 和 **MCP（Model Context Protocol）** 工具服务器的膝骨关节炎（KOA）MRI 智能分析系统。通过 Gradio 聊天界面提供 MRI 影像分割、特征提取、分类评估和网络搜索等功能。

## 系统架构

```
用户 (Gradio UI) → FastAPI Session → OAI_ChatBot.process_query()
  → MultiAgentOrchestrator (LangGraph StateGraph)
    → RouterAgent（意图分类 → 选择智能体）
    → 专用智能体（分割 / 特征提取 / 分类 / ...）
    → 智能体通过 MCPToolManager 调用 MCP 工具（stdio 子进程）
    → 工具执行 ML 算法（pyradiomics、3D-MN、scikit-learn、SHAP、MONAI）
  → 结果流式返回 Gradio UI（含图表与文件下载）
```

### 智能体系统

| 智能体 | 职责 |
|---|---|
| 🧭 **路由智能体 (Router)** | 分析用户意图，将请求路由到合适的专用智能体 |
| 🧬 **分割智能体 (Segmentation)** | 通过 U-Mamba、LKM-UNet、LightM-UNet 进行 MRI 影像分割 |
| 📊 **特征提取智能体 (Feature Extraction)** | 影像组学、MEAS、LDSE、3D-MN/16亚区软骨厚度特征提取；数据增强；特征质量评估 |
| 🎯 **分类智能体 (Classification)** | LASSO / Filter 特征选择 + KL 分级二分类 |
| 🔗 **特征组合智能体 (Feature Combination)** | 多区域影像组学特征组合 + 批量 LASSO 分类 |
| 💬 **对话智能体 (Conversation)** | 一般对话、结果解释、流程指导 |
| 🔍 **网络搜索智能体 (Web Search)** | 基于智谱 AI 的实时网络搜索 |
| 🎼 **编排智能体 (Orchestrator)** | 多步骤任务规划与跨智能体协调 |

### MCP 工具服务器

当前 `src/server_config.json` 配置了 5 个 MCP 服务器，通过 stdio 子进程连接：

| 服务器 | 主要工具 |
|---|---|
| `auto_experiment_server` | 3 种分割模型、影像组学/MEAS/LDSE 特征提取、LASSO/Filter 分类、数据增强、特征质量评估、报告读取 |
| `lasso_class_features_server` | 影像组学特征组合（按区域 / 不按区域）、组合特征批量 LASSO |
| `cartilage_analysis_server` | 六分类分割校验、3D-MN/16 亚区分析、患者级厚度特征批量提取、三维表面网格导出 |
| `biomarker_discovery_server` | KL 0–4 候选生物标志物发现、训练/测试确认和独立审计 |
| `web_search_server` | 智谱 AI 网络搜索 |

## 功能概览

### MRI 影像分割
- 支持 **U-Mamba** / **LKM-UNet** / **LightM-UNet** 三种分割模型
- 可分割股骨、胫骨、软骨等膝关节结构
- 输出 .nii.gz 分割结果文件，支持下载

### 特征提取
- **影像组学特征 (Radiomics)**：基础约 107 个特征（形状、纹理、一阶统计），启用图像滤波器后约 1316 个
- **MEAS 特征**：形态学与几何特征（体积、表面积、厚度等）
- **LDSE 特征**：低维形状编码（默认 50 维）
- **CartilageThickness 特征**：3D-MN 双向表面法线、面积加权统计，以及 Wirth/Eckstein 16 亚区厚度
- 支持 5 个膝关节区域分别提取
- 支持训练集/测试集批量处理

### 软骨厚度与 16 亚区分析

- `validate_cartilage_segmentation`：校验 `.nii/.nii.gz` 是否满足 OAI-ZIB 六分类标签和 `[D,H,W]` 空间要求。
- `analyze_cartilage_thickness`：一次计算并复用顶点数据，输出股骨软骨、内外侧胫骨软骨全局统计和 16 亚区统计。
- `extract_cartilage_thickness_features`：读取含 `CaseID`、`MaskPath`，可选 `SourceCaseID`、`KLGrade` 的 CSV，输出患者级分类特征表。
- `generate_cartilage_3d_mesh`：从六标签分割生成股骨、胫骨及三类软骨的三角网格，支持 PLY、STL、NPZ；顶点使用 ITK LPS 毫米物理坐标。
- 网格任务完成后，右侧“结果与运行”区域会自动打开交互式三维视图；可勾选股骨、胫骨及三类软骨，独立控制各结构的显示，并支持旋转、缩放和平移。
- 右侧预览会从 NPZ 几何生成临时 GLB：使用参考项目一致的二次表面平滑、平滑点法线和 PBR 参数（`metallic=0`、`roughness=0.55`），获得连续高光、体积阴影和更强的实体感；原始 PLY/STL/NPZ 科学计算产物不受影响。
- 单病例输出包括全局 CSV、16 亚区 CSV、JSON，以及可选的三类厚度 NIfTI 和 1–16 亚区标签 NIfTI。
- 算法输入标签固定为：0 背景、1 股骨、2 股骨软骨、3 胫骨、4 内侧胫骨软骨、5 外侧胫骨软骨。

#### 软骨厚度工具任务验收

从 `src` 目录启动系统后，可直接输入以下自然语言任务：

```text
请检查 D:\data\oaizib_001.nii.gz 是否能做软骨厚度分析。
请计算 D:\data\oaizib_001.nii.gz 的 3D-MN 软骨厚度和 16 亚区厚度，并导出亚区掩膜。
请根据 D:\data\labels_train.csv 批量提取 3D-MN 软骨厚度特征和 16 亚区特征。
请将 D:\data\oaizib_001.nii.gz 的全部标签重建为平滑三维网格，并导出 PLY。
请为我上传的六标签分割生成三类软骨的 3D surface mesh，并导出 STL。
```

界面执行过程应分别显示调用 `validate_cartilage_segmentation`、
`analyze_cartilage_thickness`、`extract_cartilage_thickness_features`。
三维网格请求应显示调用 `generate_cartilage_3d_mesh`。
批量 CSV 至少需要 `CaseID`、`MaskPath` 两列；用于分类时建议同时提供
`SourceCaseID` 和 `KLGrade`。

### 分类与 KL 分级
- **LASSO 分类**：真正的 L1 正则化 Logistic Regression 二分类与特征选择（不是回归模型 `LassoCV`）
- **Filter 分类**：固定原子配方 `ANOVA_TOPK_L2`；候选期在患者级嵌套交叉验证内选择 Top-K 与 C，最终测试只执行胜出参数，不再比较测试集上的多种 Filter
- 二分类任务（KL < 2 vs KL ≥ 2）
- 输出：患者级 ROC/PR 曲线、`L1_Logistic_CV_AUC*.png`、入选特征、系数和可用时的 SHAP 可解释性分析
- 评估指标：AUC、准确率、精确率、召回率/灵敏度、特异度、F1 分数

#### 患者级 L1 Logistic 分类与迁移说明

- 阳性定义为 `KLGrade >= 2`。主模型是经 L1 惩罚的 Logistic Regression；全特征模型仅作为显式的探索性、分组 L2 Logistic 基线。
- Radiomics 在 `by_region=False` 时，会将同一 `CaseID` 的 Region 行透视为一行，并按确定性顺序生成 `R{region}__{feature}` 特征列。
- 可用的 `SourceCaseID` 定义来源患者分组（否则使用 `CaseID`）。同一来源患者的增强样本不能跨越训练/测试集或分组交叉验证折；仅测试集允许重复来源患者行，并先平均其预测概率，再计算患者级指标。
- 中位数填补、标准化、C 调优和 L1 特征选择均封装在训练折内，并通过嵌套分层分组交叉验证拟合；测试数据不参与 schema、预处理或调参。
- 训练 schema 是唯一权威：测试集缺少训练特征会失败；仅测试集存在的额外特征或 Region 会被忽略并记录。训练/测试分组重叠、缺失或空白 `CaseID`、标签冲突、缺失锁定特征、无效/非有限特征及每类患者组不足都会给出明确失败。数值特征缺失值可以保留，由每个训练折内的中位数填补处理。
- 主测试指标按患者/来源患者计算，`metrics_summary.csv` 使用 `NumPatients`；`nested_cv_summary*.json` 另含由外层折 OOF 概率先按来源患者平均、再以固定 0.5 阈值计算的训练 `oof_metrics`。现有脚本不得再把 LASSO alpha、MSE/CV error 或原始行数解释为分类结果；请改用最终 C、C-AUC 和 `NumPatients`。旧的回归 MSE/CV-error 图已由 `L1_Logistic_CV_AUC*.png` 取代。
- 保留输出包括 `metrics_summary.csv`、患者级文本报告 `metrics_LASSO*.txt`、ROC/PR 图、`LASSO_selected_features*.txt` 和可用时的 SHAP 图；新增可复现产物包括 `patient_feature_schema*.json`、`nested_cv_summary*.json`、`model_coefficients*.csv` 和 `classification_run_manifest*.json`。分类流程不会写出 pickle/joblib 可执行模型。
- `output_dir` 必须不存在或为空；非空目录会被拒绝，以防混入陈旧结果。一次完整运行先在同一父目录暂存，只有所有请求的 Region 和模型都成功后才整体发布。
- MCP 输入参数保持不变；分类结果中的 JSON 可复现产物会与 TXT/CSV/图片一起作为可下载数据文件返回。

### 数据增强
- **离线增强**：基于 MONAI 的几何 + 强度增强，将增强后的 .nii.gz 保存到磁盘
- **在线增强**：特征提取时实时增强，生成带 `_std` 列的鲁棒特征
- 支持训练集和测试集同时增强

### 特征质量评估
- 自动检测低方差、高共线性、不稳定特征
- 输出过滤后的特征 CSV 文件，提升分类效果

### 特征组合
- 按区域 / 不按区域组合影像组学特征
- 支持二二组合、三三组合、四四组合
- 对组合特征批量执行 LASSO 分类

### 对话与交互
- 用户注册/登录（SHA-256 密码哈希）
- MySQL 持久化对话历史，支持记忆摘要
- 文件上传（CSV、.nii.gz）与下载
- 图表实时展示（ROC、PR、SHAP 等）
- 流式输出，显示分步执行进度；支持 MCP 工具的实时百分比进度条

#### 工具任务进度

批量工具通过统一的 MCP progress 通道上报进度，Agent 编排层将进度转发到后台任务记录，前端每秒轮询并同时显示在当前对话回复和任务状态栏。进度行采用 20 格百分比条，并按工具覆盖旧值，避免处理数百个病例时堆积大量历史进度。当前已接入软骨厚度、MEAS、Radiomics、LDSE、离线数据增强、在线增强特征提取和候选生物标志物统计分析；以后任何发送标准 MCP progress notification 的工具会自动获得相同的前端展示，无需再次修改页面。

### 科学轨迹图与报告

任务进入终态后，系统根据结构化执行记录生成科学轨迹产物，包括任务判定、数据划分、候选方案、Agent/工具执行、产物来源、科学门禁和复现信息。执行失败或校验未通过的任务也会尝试生成报告，报告中会标明失败或待复核状态。

默认目录为 `output/scientific_trajectory_graphs/<workspace_id>/<trace_id>/`：

| 文件 | 用途 |
|---|---|
| `scientific_trajectory.png` | 右侧结果区域的轨迹概览图 |
| `scientific_trajectory.md` | 右侧“科学轨迹报告”和可下载的 Markdown 报告 |
| `scientific_trajectory.json` | 保存结构化证据、图节点、连线和产物哈希的清单 |

可在 `.env` 中设置 `SCIENTIFIC_TRAJECTORY_DIR` 调整存储目录，建议使用绝对路径。生成的报告属于本地运行产物，默认不提交到 Git。

报告仍会在任务结束后生成。前端轮询只检查后台状态，内容变化时才更新报告组件，避免每秒进入等待状态造成明暗闪烁。新任务运行时隐藏旧报告，完成后显示新报告；重新打开历史会话时会恢复该会话最新任务的报告。Gradio 5.x 的报告状态轮询使用队列事件，确保首次显示和后续更新均能触发。

图表生成异常会记录为 `scientific_trajectory_graph_error`，不会改变任务本身的完成判定。轨迹报告的详细字段和登记方式见 [轨迹与离线优化说明](docs/trajectory-optimization.md)。

## 日常任务驱动的离线优化

每个请求会生成独立的 `task_fingerprint`、确定性完成门禁、规则反馈和
可选的人工反馈。历史可靠性只在相同任务指纹内参与规划，Agent 不能在
门禁运行前把任务标记为完成。

自动 KL 任务会为每个候选生成独立的 `pipeline_trials` 记录。记录包含
原子配方哈希、训练内 OOF 指标、患者级 bootstrap 置信区间、特征选择
稳定性、推荐参数、耗时和失败原因；测试集数据不进入候选排序。即使当前
只有一种 KL 分级任务，同一次任务里的候选也能形成有意义的成对证据。

从 `src` 目录生成每日策略预览：

```powershell
python -m db_utils.offline_optimizer `
  --output task_results_history/offline_optimizer/latest_policy_preview.json
```

该命令默认只生成预览。加上 `--register-draft` 也只会注册 `draft`，不会
启用生产策略；仍需历史回放、Shadow 验证和人工批准。详细说明见
[`docs/trajectory-optimization.md`](docs/trajectory-optimization.md)。

## 项目结构

```
oai_multi_agent_system_new/
├── src/
│   ├── app.py                          # FastAPI + Gradio 应用入口
│   ├── oai_client.py                   # OAI_ChatBot：会话管理、历史、记忆
│   ├── llm_config.py                   # 统一读取配置并创建 LLM 客户端
│   ├── server_config.json              # MCP 服务器启动配置
│   ├── agents/
│   │   ├── orchestrator.py             # LangGraph StateGraph 编排定义
│   │   ├── scientific_trajectory_graph.py # 科学轨迹图、报告与结构化清单
│   │   ├── orchestrator_agent.py       # 多步骤任务规划智能体
│   │   ├── router_agent.py             # 查询路由智能体
│   │   ├── segmentation_agent.py       # MRI 分割智能体
│   │   ├── feature_extraction_agent.py # 特征提取智能体
│   │   ├── feature_combination_agent.py# 特征组合智能体
│   │   ├── classification_agent.py     # 分类智能体
│   │   ├── conversation_agent.py       # 通用对话智能体
│   │   ├── web_search_agent.py         # 网络搜索智能体
│   │   ├── tool_manager.py            # MCP 工具连接管理器
│   │   ├── state.py                    # AgentState 类型定义
│   │   └── config.py                   # 各智能体参数，共用统一 LLM 配置
│   ├── server/
│   │   ├── auto_experiment_server.py   # 主 MCP 服务器（分割、提取、分类）
│   │   ├── lasso_class_features_server.py # 特征组合 MCP 服务器
│   │   ├── cartilage_analysis_server.py # 3D-MN/16亚区软骨厚度 MCP 服务
│   │   ├── biomarker_discovery_server.py # 候选标志物发现与审计
│   │   └── web_search_server.py        # 网络搜索 MCP 服务器
│   ├── algorithm/
│   │   ├── cartilage_thickness/         # 厚度核心算法、16亚区算法和文件服务边界
│   │   ├── KOA_feature_extraction/     # MEAS 和 LDSE 特征提取器
│   │   ├── feature_class_algorithm/    # LASSO 和 Filter 分类器
│   │   ├── feature_quality/            # 特征质量评估器
│   │   └── data_augmentation/          # 离线和在线数据增强器
│   ├── db_utils/
│   │   ├── db_utils.py                 # MySQL 数据库管理器
│   │   └── db_config.json              # MySQL 连接配置
│   └── env_utils/
│       └── env_util.py                 # 环境变量加载器
├── llm_config.json                     # 模型、接口、密钥变量名和请求参数
├── .env.example                        # 密钥及其他环境变量模板
├── tests/                              # 自动化回归测试
├── .gitignore
├── pyproject.toml                      # 项目配置与依赖
└── README.md
```

## 环境配置

### 前置条件

- **Python** >= 3.12
- **MySQL** 8.0+（用于用户账户和对话历史存储）
- **Microsoft Visual C++ Build Tools**（仅 Windows，pyradiomics 编译需要）
- **DeepSeek** 或其他兼容 OpenAI API 的 LLM 端点及对应密钥

### 1. 克隆项目并安装依赖

```bash
git clone https://github.com/Hccccy2002/oai_multi_agent_system_new.git
cd oai_multi_agent_system_new

# 使用 uv（推荐）
uv sync --locked
```

依赖以 `pyproject.toml` 和 `uv.lock` 为准，其中 PyRadiomics 固定到源码标签 `v3.1.0`。请使用同一个虚拟环境安装依赖、启动应用和运行测试。

### 2. 配置 LLM 和环境变量

```bash
cp .env.example .env
# 编辑 .env 文件，填入你的 API Key
```

项目根目录的 `llm_config.json` 是单智能体客户端和全部 9 个智能体的统一模型配置入口，默认沿用 DeepSeek：

```json
{
  "model": "deepseek-v4-pro",
  "base_url": "https://api.deepseek.com",
  "api_key_env": "DEEPSEEK_API_KEY",
  "timeout": 600,
  "max_retries": 2
}
```

- `model`：模型名称；`base_url`：对应的 OpenAI 兼容 API 地址。
- `api_key_env`：保存密钥的环境变量名称，实际密钥写在项目根目录 `.env` 中，例如 `DEEPSEEK_API_KEY=你的密钥`。进程环境中的同名变量优先。
- `timeout`：每次请求的超时秒数；`max_retries`：失败后的最大重试次数。

切换模型时，在这个 JSON 文件中同时修改 `model`、`base_url` 和 `api_key_env`，并确保对应密钥已配置。**修改后重启应用和后台工作进程。** 配置按进程加载一次，从根目录或 `src/` 启动都会读取同一文件。

旧的 `MODEL_NAME`、`OPENAI_BASE_URL`、`DEEPSEEK_BASE_URL` 不再控制这些 LLM 调用；`OAI_ChatBot(model=...)` 也不再允许指定与统一配置不同的模型。各智能体原有的 temperature 保持不变。

环境变量：

| 变量 | 说明 |
|---|---|
| `DEEPSEEK_API_KEY` | 默认配置所需的 DeepSeek 密钥 |
| `OPENAI_API_KEY` | 使用该密钥时，将 JSON 中的 `api_key_env` 改为此名称 |
| `ZHIPU_API_KEY` | 智谱 AI 密钥（网络搜索需要） |
| `LANGSMITH_API_KEY` | LangSmith 追踪（可选） |

### 3. 配置 MySQL 数据库

先复制数据库配置模板，再编辑 `src/db_utils/db_config.json`：

```bash
cp src/db_utils/db_config.example.json src/db_utils/db_config.json
```

```json
{
  "host": "localhost",
  "port": 3306,
  "user": "root",
  "password": "你的密码",
  "database": "mcp_chatbot"
}
```

创建数据库：

```sql
CREATE DATABASE mcp_chatbot CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
```

数据表会在首次运行时自动创建。

### 4. Windows 下安装 PyRadiomics

Windows 系统需要安装 Visual Studio Build Tools 才能编译 pyradiomics：

1. 访问 [Microsoft Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)。
2. 下载并安装，勾选 **"使用 C++ 的桌面开发"** 工作负载
3. 确保选中 MSVC v143 和 Windows 10/11 SDK
4. 安装完成后重启终端，执行：

```bash
uv sync --locked
```

## 使用方式

### 启动应用

```powershell
cd src
..\.venv\Scripts\python.exe app.py
```

Linux/macOS 在 `src` 目录下使用 `../.venv/bin/python app.py`。

应用启动后访问 [本地工作台](http://127.0.0.1:7860)：

| 路径 | 页面 |
|---|---|
| `/` | 首页（登录/注册入口） |
| `/login` | 用户登录 |
| `/register` | 用户注册 |
| `/chat` | 聊天主界面 |

首次进入工作区时，页面最多等待工具连接 1.5 秒；超过这段等待时间会显示“工具正在后台连接”，工具继续在后台接入，成功后状态栏自动更新为“工具已连接”。这段页面等待时间不代表 MCP 连接超时。正在运行或已完成的任务状态优先显示，工具列表也可通过“刷新连接状态”按钮重新获取。

### 工作流示例

**MRI 分割：**
> 上传 MRI .nii.gz 文件 → "使用 U-Mamba 分割 MRI"

**特征提取：**
> 上传标签 CSV 文件 → "提取所有病例的影像组学特征"

**多步骤任务（训练集 + 测试集）：**
> 上传训练集和测试集标签文件 → "分别对训练集和测试集提取 MEAS 特征，然后使用 LASSO 分类并读取分类报告"

**数据增强：**
> 上传训练集和测试集标签文件 → "对训练集和测试集进行数据增强，提取 LDSE 特征，然后使用 LASSO 分类"

**特征质量评估：**
> 上传特征 CSV 文件 → "评估特征质量，然后使用过滤后的特征进行 LASSO 分类"

**读取报告：**
> "读取 LASSO 分类报告"

**特征组合：**
> 上传多区域影像组学特征文件 → "按区域组合影像组学特征，然后使用 LASSO 分类"

## LDSE 安全训练/测试提取

配对提取请使用 MCP 工具 `ldse_train_test_features_extraction`，参数为 `output_dir`、`train_csv_path`、`test_csv_path`、`n_components` 和 `random_seed`。该工具只在训练掩膜上拟合 scaler/PCA；测试集仅使用冻结的训练产物进行 transform。

输出目录结构如下：

```text
<output_dir>/
├── model/light_ldse_models_<uuid>.npz
├── train/ldse_features_summary.csv
├── train/ldse_features_summary.pkl
├── test/ldse_features_summary.csv
├── test/ldse_features_summary.pkl
└── ldse_run_manifest.json
```

`ldse_run_manifest.json` 用于记录输出路径、文件哈希、随机种子、特征维度、样本计数，以及训练/测试特征 schema 是否一致。

如需单独调用，请严格按以下生命周期执行：

```text
# 训练
ldse_features_extraction(..., mode="fit_transform", dataset_role="train")

# 测试（model_path 指向训练阶段产生的 artifact）
ldse_features_extraction(..., mode="transform", dataset_role="test", model_path=<training artifact>)
```

测试集拟合，以及分别为训练集和测试集建立独立 PCA 模型，都会被拒绝。为避免覆盖已有结果，已存在的输出目标不会被覆盖；请使用新的 `output_dir`。

这是一次破坏性安全迁移：旧的隐式单 CSV 调用已被拒绝，必须显式提供 `mode` 和相应的数据集角色；旧版 pickled LDSE 模型也会被拒绝。模型交接必须使用版本 2 的压缩 `.npz` 数值产物，该产物以 `np.load(..., allow_pickle=False)` 读取并在重建已知的 scaler/PCA 类之前完成元数据、数组类型、尺寸和有限值验证。

### 未标注 KLGrade

- `KLGrade=-1` 表示未标注；MEAS、LDSE 和 LASSO 会在各自流程开始时动态排除这些行，不会将其映射为阴性或阳性。除 `-1` 外，0–4 以外的其他标签值仍会报错。
- 样本数量不是代码限制：`有效数 = 输入数 - KLGrade=-1 数`。当前上传数据的验收计数为训练集 383 例、测试集 98 例，仅用于本批数据的核验，不是固定上限或固定常量。
- 新增训练病例后，应使用更新后的已标注训练集重新执行 LDSE 训练/测试配对提取（重新拟合 scaler/PCA），并在新的空输出目录重新训练 LASSO。
- 仅新增测试病例时，LDSE 必须复用训练集已拟合的工件进行 transform，不能在测试集重新拟合；MEAS 可直接提取新增病例，但训练集变化后仍须重新训练 LASSO。
- 分类失败时，Agent 会直接返回原始失败信息，不再尝试读取不存在的分类报告目录。

## 技术说明

- MCP 服务器以 Python 子进程方式启动，**必须在 `src/` 目录下运行 `app.py`**
- 软骨厚度 MCP 服务使用 SimpleITK 读取 NIfTI，并显式将 `[z,y,x]` 转为算法要求的 `[D,H,W]`；不要绕过服务边界直接传 SimpleITK 数组。
- 系统使用 `nest_asyncio` 支持 LangGraph 同步执行模型内的异步 MCP 调用
- 记忆摘要机制：保留最近 2 轮对话的完整内容，更早的消息被 LLM 压缩为 JSON 摘要并持久化到 MySQL
- 所有智能体通过 `ChatOpenAI` 接入 LLM，并与单智能体客户端共用根目录 `llm_config.json` 中的模型、接口地址和密钥来源

## 回归测试

在项目根目录运行本次配置与界面修复的聚焦测试：

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_llm_config.py tests/test_trajectory_report_refresh.py tests/test_tool_connection_status.py
```

三组测试共 28 项，覆盖单智能体与多智能体配置一致性、密钥来源与配置校验、科学轨迹报告首次显示/更新/会话切换，以及工具慢连接、真实失败和状态自动更新。测试使用模拟请求、连接状态和数据库，不要求调用真实 LLM 或启动真实 MCP 工具。

历史用例 `tests/test_biomarker_contracts.py::test_default_dataset_and_fixed_four_role_plan_are_materialized` 依赖 `src/datasets/labels_train.csv` 和 `labels_test.csv`，并固定断言原批次的行数。该数据集未随仓库分发，因此在干净检出的仓库运行全量测试时，这一用例会因缺少数据而失败。本次上传内容的全量检查结果为 50 项通过、上述 1 项失败。

## 常见问题

| 现象 | 检查与处理 |
|---|---|
| 修改模型后仍调用旧接口 | 同时检查根目录 `llm_config.json` 的 `model`、`base_url`、`api_key_env`，确认对应密钥已设置，重启应用后再提交任务。 |
| 刚进入工作区显示工具正在连接 | 等待后台接入完成，状态栏会自动更新；如最终显示连接失败，再检查控制台中的具体异常。 |
| 科学轨迹报告暂时不可见 | 当前任务等待或运行时会隐藏报告；任务结束后检查输出文件中是否有 `scientific_trajectory.md`，以及控制台是否记录轨迹生成异常。 |
| 更新代码后仍闪烁或报告不显示 | 完整重启应用并刷新浏览器，确保加载新的报告状态轮询与显示事件。已生成的文件会保留。 |
| 模型接口报 `Connection error` 或 SSL EOF | 检查 `llm_config.json` 中的接口地址、启动进程的代理和网络连通性；模型切换与网络连接分别排查。 |

## 2026-09-05 更新

- 将单智能体客户端和 9 个 LLM 智能体统一到 `llm_config.json`，默认沿用 `deepseek-v4-pro`；密钥从 `.env` 或进程环境读取。
- 增加科学轨迹 PNG、Markdown 和 JSON 产物，以及右侧报告展示和下载支持。
- 修复报告持续明暗闪烁，并补齐 Gradio 5.x 队列状态事件，保证报告首次显示和后续更新。
- 修复 Python 3.10 中页面等待超时被误判为工具连接失败的问题；后台连接完成后自动更新状态。
- 补充会话中间件依赖 `itsdangerous`。
- 上述三组聚焦回归测试共 28 项通过；慢连接提示与报告显示流程另在 Python 3.10.20 / Gradio 5.49.0 既有环境中做了兼容性验证。新安装环境仍以项目声明的 Python >= 3.12 为准。

以下为历史更新记录；其中测试数量和环境描述对应各记录当时的版本。

## 2026-08-24 本次会话更新总结

- 工作区隔离：支持自定义名称新建工作区、重命名和删除工作区；修复新增或重命名后界面仍显示“新工作区”的问题。每个工作区分别保存工作区记忆、历史会话和任务上下文。
- 历史会话：新增类似 Codex/ChatGPT 的“最近会话”列表，可恢复并查看完整聊天记录；会话悬停时显示删除按钮，删除前使用页面内确认弹窗。
- 后台任务持久化：任务在前端关闭、刷新或切换会话后仍由后台继续执行；重新进入对应会话时可恢复执行状态、过程信息和 Agent 最终回复。任务运行期间输入区会保持禁用，避免重复提交。
- 工作台交互：整体调整为更紧凑、整洁的工作台布局，增大自动化任务对话区、任务文件区和图表展示区，减少无效留白；移除当前自动化流程中不需要的人工输入区域，并增强按钮灰色边框等视觉状态。
- 消息输入：支持按 `Enter` 发送任务、按 `Alt+Enter` 换行，并在输入提示中明确操作方式。
- 页面弹窗：新增工作区、重命名、删除工作区以及新增/删除会话等操作不再使用浏览器顶部弹窗，统一改为页面中间的确认或输入弹窗。
- Gradio 兼容性：移除 `gr.HTML` 不支持的 `scale` 参数，修复旧版 Gradio 启动时的 `TypeError`。
- 任务数据与自优化：完善 train、validation、test 轨迹分区及任务评估流程，补充工作区级记忆、任务反馈、流水线能力注册表、自优化约束和开发期评估任务目录。
- 通用异步执行修复：LangGraph 编排改为在持久事件循环中异步执行，不再为 Agent 节点创建短生命周期事件循环；MCP 管理器改为按事件循环隔离，并由单一生命周期任务负责连接的创建、使用和关闭。
- MCP 稳定性：修复 `There is no current event loop` 和 `anyio.ClosedResourceError` 导致所有工具任务失败的问题；连接意外关闭时自动重连一次并重试当前工具。MCP 配置使用绝对路径，Python 服务统一使用当前运行环境的解释器启动。
- 自动化流水线：补齐 MEAS、LDSE、Radiomics 与 LASSO、Filter 的候选流水线配置、训练集内部验证规则、报告读取顺序和工件范围校验，避免针对单个工具做特殊修补。
- 验证结果：真实启动并发现 4 个 MCP 服务、22 个工具；新增软骨厚度服务已通过 stdio 工具发现、真实 NIfTI 调用、自然语言计划修复和全量回归测试（`551 passed`）。

--- 修复图表重复出现的问题

## 2026-08-29
请检查我上传的文件是否能做软骨厚度分析
请计算我上传MRI影像的3D-MN 软骨厚度和 16 亚区厚度，并导出亚区掩膜
请根据我上传的文件批量提取 3D-MN 软骨厚度特征和 16 亚区特征。
添加了软骨厚度提取Tool。

## 2026-08-30
三维预览升级为参考项目风格的 GLB/PBR 渲染，增加预览专用表面平滑、点法线、独立结构材质和浅灰蓝背景；保持结构勾选交互以及原始 PLY/STL/NPZ 产物不变。

### KL 0–4 横断面 MRI 候选生物标志物发现

系统新增研究用途的 `BIOMARKER_DISCOVERY` 任务。该任务不是原有 KL 二分类的别名，而是四阶段多智能体流程：

1. `FEATURE_EXTRACTION_AGENT` 分别从训练集、测试集分割掩膜提取患者/膝关节级 3D-MN 全局软骨厚度与 Wirth/Eckstein 16 亚区特征。
2. `BIOMARKER_DISCOVERY_AGENT` 仅在训练集执行 proportional-odds 有序 Logistic 关联分析，使用 Benjamini–Hochberg FDR、相关性去冗余和 bootstrap 方向稳定性筛选并冻结候选，再在测试集用 Holm 校正确认。
3. `BIOMARKER_VERIFIER_AGENT` 独立复核候选集合哈希、训练/测试隔离、多重检验结果、产物哈希和报告计数。
4. `BIOMARKER_REPORTER_AGENT` 只接收审计通过的结构化事实，根据本次样本量、候选数量和每个确认特征的实际统计量，自然组织面向用户的中文解释。

推荐请求：`自动完成 KL 0–4 横断面关联的 MRI 候选生物标志物发现`。若请求未显式给出数据路径，系统会在当前工作区有界查找 `src/datasets/labels_train.csv` 和 `src/datasets/labels_test.csv`。输出位于 `src/biomarker_runs/<run_id>/`，包括数据质控、分析计划、全特征发现表、冻结候选清单、测试集确认表、研究报告、运行清单和独立审计报告。

“没有训练集候选”或“没有测试集确认候选”是有效的阴性科学结局，不会被当成工具故障，也不会通过放宽阈值重做。V1 只能支持横断面候选关联，不能声称因果、生物机制、纵向进展预测、临床替代终点或独立外部验证。

独立审计通过后，`BIOMARKER_REPORTER_AGENT` 会基于经过哈希和数值复核的事实生成非固定话术的中文讲解，前端对话框直接展示该回复。讲解会针对本次实际纳入/排除样本量、分析特征数、冻结与确认候选数逐项说明其含义，并为每个确认候选解释效应方向、训练集 OR/SD、BH q、bootstrap 方向稳定性、测试集 OR/SD 和 Holm P。生成后还要经过确定性的数字覆盖、效应方向、统计解释与越界声明校验；未通过会自动重写一次，连续两次失败才启用保守的固定摘要作为安全兜底。因此，同一组科学事实可以有更自然的表达，但 Agent 不能改动数字、候选或把横断面关联夸大成因果与临床结论。

方法依据：McCullagh 的有序回归（DOI `10.1111/j.2517-6161.1980.tb01109.x`）、Benjamini–Hochberg FDR（DOI `10.1111/j.2517-6161.1995.tb02031.x`）、IBSI（DOI `10.1148/radiol.2020191145`）、CLEAR（DOI `10.1186/s13244-023-01415-8`）、Wirth/Eckstein 解剖亚区方法（DOI `10.1109/TMI.2007.907323`）、横断面 KL 阶段与区域软骨厚度研究（DOI `10.1002/acr.20262`），以及膝 OA 软骨厚度进展影像生物标志物研究（DOI `10.1002/art.39324`）。
