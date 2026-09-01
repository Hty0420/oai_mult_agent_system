"""User-facing presentation model for the Agent/Tool capability panel.

The execution allowlist remains the source of truth for Agent-to-Tool
permissions. This module only adds concise names, descriptions and status UI.
"""

from __future__ import annotations

from collections import defaultdict
from html import escape
from typing import Any, Iterable, Mapping

from agents.tool_policy import AGENT_TOOL_ALLOWLIST


AGENT_META: Mapping[str, dict[str, Any]] = {
    "ROUTER_AGENT": {
        "name": "任务路由 Agent", "short": "RT", "stage": "入口", "stage_key": "entry",
        "summary": "理解当前请求，判断由单个专用 Agent 处理，还是进入多步骤编排。",
        "capabilities": ("意图识别", "任务分流", "路由置信度"),
    },
    "ORCHESTRATOR_AGENT": {
        "name": "任务编排 Agent", "short": "OR", "stage": "编排", "stage_key": "orchestration",
        "summary": "拆解复杂研究目标，安排多个 Agent 的执行顺序、输入依赖与结果汇总。",
        "capabilities": ("任务拆解", "依赖编排", "过程留痕"),
    },
    "SEGMENTATION_AGENT": {
        "name": "影像分割 Agent", "short": "SG", "stage": "执行", "stage_key": "execution",
        "summary": "调用三类分割模型，从膝关节 MRI 中生成后续分析所需的结构掩膜。",
        "capabilities": ("U-Mamba", "LKM-UNet", "LightM-UNet"),
    },
    "FEATURE_EXTRACTION_AGENT": {
        "name": "特征提取 Agent", "short": "FE", "stage": "执行", "stage_key": "execution",
        "summary": "提取 radiomics、MEAS、LDSE 与软骨厚度特征，并支持增强和质量检查。",
        "capabilities": ("影像组学", "形态与形状", "软骨定量"),
    },
    "FEATURE_COMBINATION_AGENT": {
        "name": "特征组合 Agent", "short": "FC", "stage": "执行", "stage_key": "execution",
        "summary": "按区域或全局合并多类 radiomics 特征，并批量比较组合特征表现。",
        "capabilities": ("区域组合", "全局组合", "批量比较"),
    },
    "CLASSIFICATION_AGENT": {
        "name": "分级分类 Agent", "short": "CL", "stage": "执行", "stage_key": "execution",
        "summary": "执行 KOA/KL 分类、特征选择和结果读取，输出可复核的指标与图表。",
        "capabilities": ("LASSO", "Filter", "结果解读"),
    },
    "PIPELINE_OPTIMIZATION_AGENT": {
        "name": "流程优化 Agent", "short": "PO", "stage": "优化", "stage_key": "optimization",
        "summary": "在训练数据内搜索并验证候选特征流程，不触碰冻结测试集。",
        "capabilities": ("候选搜索", "嵌套验证", "防止泄漏"),
    },
    "BIOMARKER_DISCOVERY_AGENT": {
        "name": "标志物发现 Agent", "short": "BD", "stage": "发现", "stage_key": "discovery",
        "summary": "在训练集发现 KL 相关候选 MRI 标志物，并在冻结测试集进行确认。",
        "capabilities": ("候选筛选", "多重检验", "冻结确认"),
    },
    "BIOMARKER_VERIFIER_AGENT": {
        "name": "标志物复核 Agent", "short": "BV", "stage": "验证", "stage_key": "validation",
        "summary": "独立核验运行清单、候选冻结、文件哈希、统计校正与报告计数。",
        "capabilities": ("独立审计", "哈希核验", "统计复核"),
    },
    "BIOMARKER_REPORTER_AGENT": {
        "name": "标志物报告 Agent", "short": "BR", "stage": "报告", "stage_key": "reporting",
        "summary": "只基于审计通过的结构化事实生成结果说明，不重新计算统计。",
        "capabilities": ("事实约束", "数字讲解", "研究报告"),
    },
    "WEB_SEARCH_AGENT": {
        "name": "网页检索 Agent", "short": "WS", "stage": "检索", "stage_key": "retrieval",
        "summary": "按关键词、来源域名和时间范围检索实时网页信息。",
        "capabilities": ("实时搜索", "域名过滤", "时效筛选"),
    },
    "CONVERSATION_AGENT": {
        "name": "研究对话 Agent", "short": "CO", "stage": "交互", "stage_key": "interaction",
        "summary": "回答系统、流程和结果相关问题，为不需要调用工具的请求提供说明。",
        "capabilities": ("能力介绍", "流程指导", "结果解释"),
    },
}


TOOL_META: Mapping[str, dict[str, str]] = {
    "umamba_segment_mri": {"name": "U-Mamba MRI 分割", "category": "segmentation", "summary": "调用 U-Mamba 服务生成膝关节 MRI 结构分割。"},
    "lkm_unet_segment_mri": {"name": "LKM-UNet MRI 分割", "category": "segmentation", "summary": "调用 LKM-UNet 服务生成膝关节 MRI 结构分割。"},
    "lightm_unet_segment_mri": {"name": "LightM-UNet MRI 分割", "category": "segmentation", "summary": "调用 LightM-UNet 服务生成膝关节 MRI 结构分割。"},
    "extract_radiomics_features": {"name": "Radiomics 特征提取", "category": "features", "summary": "从影像和掩膜提取一阶、纹理、形状等影像组学特征。"},
    "meas_features_extraction": {"name": "MEAS 形态特征提取", "category": "features", "summary": "提取体积、表面积、厚度等形态与几何测量特征。"},
    "ldse_train_test_features_extraction": {"name": "训练/测试 LDSE 提取", "category": "features", "summary": "仅在训练集拟合 PCA/缩放器，再用冻结模型转换测试集。"},
    "ldse_features_extraction": {"name": "LDSE 特征提取", "category": "features", "summary": "按显式 fit/transform 生命周期提取低维形状编码特征。"},
    "augment_dataset_offline": {"name": "离线数据增强", "category": "features", "summary": "同步增强 MRI 与掩膜并保存增强数据及标签清单。"},
    "augment_online_extract_features": {"name": "在线增强与特征聚合", "category": "features", "summary": "生成多个增强视图，提取特征后按样本聚合。"},
    "assess_feature_quality": {"name": "特征质量评估", "category": "features", "summary": "检查低方差、共线性与稳定性，并生成质量报告。"},
    "validate_cartilage_segmentation": {"name": "软骨分割校验", "category": "cartilage", "summary": "校验标签、几何方向、spacing 与结构是否适合厚度分析。"},
    "analyze_cartilage_thickness": {"name": "软骨厚度分析", "category": "cartilage", "summary": "执行 3D-MN 面积加权厚度与 Wirth/Eckstein 16 亚区分析。"},
    "extract_cartilage_thickness_features": {"name": "批量软骨厚度特征", "category": "cartilage", "summary": "批量生成可用于分类的患者级全局与 16 亚区厚度特征。"},
    "generate_cartilage_3d_mesh": {"name": "膝关节三维网格", "category": "cartilage", "summary": "从分割掩膜生成物理空间中的分标签三角网格。"},
    "combine_radiomics_features_by_region": {"name": "按区域组合 Radiomics", "category": "fusion", "summary": "按 Region 对多个特征表生成两两、三元和全量组合。"},
    "combine_radiomics_features_all": {"name": "全局组合 Radiomics", "category": "fusion", "summary": "不区分区域地合并多个影像组学特征表。"},
    "lasso_comb_radiomics_features_by_region": {"name": "区域组合批量 LASSO", "category": "fusion", "summary": "对各 Region 的组合特征批量执行 LASSO 分类比较。"},
    "lasso_comb_radiomics_features_all": {"name": "全局组合批量 LASSO", "category": "fusion", "summary": "对全局组合特征批量执行 LASSO 分类比较。"},
    "koa_lasso_classification": {"name": "KOA LASSO 分类", "category": "modeling", "summary": "用患者分组、嵌套验证的 L1 Logistic 完成 KOA 二分类。"},
    "koa_filter_classification": {"name": "KOA Filter 分类", "category": "modeling", "summary": "用固定 ANOVA Top-K 与 L2 Logistic 配方完成 KOA 二分类。"},
    "read_classification_results": {"name": "分类结果读取", "category": "modeling", "summary": "汇总分类结果目录中的指标和特征选择报告。"},
    "validate_kl_pipeline_candidate": {"name": "KL 流程候选验证", "category": "modeling", "summary": "仅用训练数据的患者级嵌套验证评估候选分类流程。"},
    "run_biomarker_discovery": {"name": "MRI 标志物发现", "category": "biomarker", "summary": "发现 KL 0–4 候选 MRI 标志物并在冻结测试集确认。"},
    "audit_biomarker_discovery": {"name": "标志物独立审计", "category": "biomarker", "summary": "核验运行清单、候选冻结、多重检验与文件完整性。"},
    "web_search": {"name": "实时网页搜索", "category": "retrieval", "summary": "按查询词、域名与时间范围检索网页并返回摘要。"},
}


TOOL_CATEGORIES: tuple[tuple[str, str], ...] = (
    ("segmentation", "影像分割"),
    ("features", "特征工程与质量"),
    ("cartilage", "软骨定量与三维"),
    ("fusion", "特征组合"),
    ("modeling", "建模与验证"),
    ("biomarker", "生物标志物"),
    ("retrieval", "信息检索"),
    ("other", "其他已注册工具"),
)


def _runtime_tool_map(available_tools: Iterable[Mapping[str, Any]] | None) -> dict[str, str]:
    runtime_tools: dict[str, str] = {}
    for tool in available_tools or ():
        function = tool.get("function", {}) if isinstance(tool, Mapping) else {}
        name = str(function.get("name") or "").strip()
        if name:
            runtime_tools[name] = str(function.get("description") or "").strip()
    return runtime_tools


def registered_tool_names() -> frozenset[str]:
    """Return every tool referenced by the fail-closed Agent policy."""
    return frozenset(
        tool_name
        for tool_names in AGENT_TOOL_ALLOWLIST.values()
        for tool_name in tool_names
    )


def tool_owners() -> dict[str, tuple[str, ...]]:
    owners: defaultdict[str, list[str]] = defaultdict(list)
    for agent_name, tool_names in AGENT_TOOL_ALLOWLIST.items():
        for tool_name in tool_names:
            owners[tool_name].append(agent_name)
    return {name: tuple(agent_names) for name, agent_names in owners.items()}


def _connection_state(initialized: bool, ready_count: int, total_count: int) -> tuple[str, str]:
    if not initialized:
        return "等待系统连接", "idle"
    if ready_count >= total_count:
        return "全部能力已就绪", "ready"
    if ready_count:
        return f"{ready_count} 项工具已就绪", "partial"
    return "工具连接中", "connecting"


def render_capability_summary(
    available_tools: Iterable[Mapping[str, Any]] | None = None,
    *, initialized: bool = False, multi_agent: bool = True,
) -> str:
    runtime_tools = _runtime_tool_map(available_tools)
    known_tools = registered_tool_names()
    ready_tools = len(known_tools.intersection(runtime_tools))
    status_label, status_key = _connection_state(initialized, ready_tools, len(known_tools))
    architecture = "多智能体协作" if multi_agent else "单智能体兼容模式"
    return f"""
    <section class="capability-overview" aria-label="系统能力概览">
      <div class="capability-overview-heading">
        <div><span class="capability-eyebrow">CAPABILITY MAP</span><h3>Agent × Tool 能力图谱</h3>
          <p>{architecture} · Agent 负责理解与决策，Tool 负责执行可复核的具体操作。</p></div>
        <span class="capability-live-status is-{status_key}"><i></i>{escape(status_label)}</span>
      </div>
      <div class="capability-metrics">
        <div><strong>{len(AGENT_META)}</strong><span>Agent 角色</span></div>
        <div><strong>{len(known_tools)}</strong><span>受控 Tool</span></div>
        <div><strong>{ready_tools}</strong><span>当前已就绪</span></div>
      </div>
      <div class="capability-flow" aria-label="任务执行路径">
        <span>理解任务</span><b>→</b><span>路由 / 编排</span><b>→</b><span>专用 Agent</span><b>→</b><span>MCP Tool</span><b>→</b><span>结果与审计</span>
      </div>
    </section>"""


def render_agent_capabilities(
    available_tools: Iterable[Mapping[str, Any]] | None = None,
    *, initialized: bool = False,
) -> str:
    ready_names = frozenset(_runtime_tool_map(available_tools))
    cards: list[str] = []
    for agent_name, meta in AGENT_META.items():
        allowed_tools = AGENT_TOOL_ALLOWLIST.get(agent_name, frozenset())
        ready_count, total_count = len(allowed_tools & ready_names), len(allowed_tools)
        if not initialized:
            state_key, state_label = "idle", "待连接"
        elif not total_count:
            state_key, state_label = "ready", "已启用 · 不直接调用 Tool"
        elif ready_count == total_count:
            state_key, state_label = "ready", f"{ready_count}/{total_count} Tool 就绪"
        elif ready_count:
            state_key, state_label = "partial", f"{ready_count}/{total_count} Tool 就绪"
        else:
            state_key, state_label = "connecting", f"0/{total_count} Tool 待就绪"
        tags = "".join(f'<span>{escape(str(item))}</span>' for item in meta["capabilities"])
        cards.append(f"""
        <article class="capability-agent-card">
          <div class="capability-card-heading">
            <span class="capability-agent-mark stage-{meta['stage_key']}">{escape(meta['short'])}</span>
            <div><h4>{escape(meta['name'])}</h4><code>{escape(agent_name)}</code></div>
            <span class="capability-stage stage-{meta['stage_key']}">{escape(meta['stage'])}</span>
          </div>
          <p>{escape(meta['summary'])}</p><div class="capability-tags">{tags}</div>
          <div class="capability-card-footer"><span class="capability-mini-status is-{state_key}"><i></i>{escape(state_label)}</span></div>
        </article>""")
    return f"""
    <section class="capability-list-section" aria-label="Agent 角色">
      <div class="capability-list-intro"><div><strong>职责边界</strong><span>每个 Agent 只处理自己的研究环节，并按白名单调用 Tool。</span></div><span class="capability-count">{len(AGENT_META)} 个角色</span></div>
      <div class="capability-agent-grid">{''.join(cards)}</div>
    </section>"""


def render_tool_capabilities(
    available_tools: Iterable[Mapping[str, Any]] | None = None,
    *, initialized: bool = False,
) -> str:
    runtime_tools, owners = _runtime_tool_map(available_tools), tool_owners()
    known_names = registered_tool_names()
    all_names = known_names.union(runtime_tools)
    grouped: defaultdict[str, list[str]] = defaultdict(list)
    for tool_name in all_names:
        grouped[TOOL_META.get(tool_name, {}).get("category", "other")].append(tool_name)

    groups: list[str] = []
    for category_key, category_name in TOOL_CATEGORIES:
        tool_names = sorted(grouped.get(category_key, ()), key=lambda item: TOOL_META.get(item, {}).get("name", item))
        if not tool_names:
            continue
        rows: list[str] = []
        category_ready = sum(name in runtime_tools for name in tool_names)
        for tool_name in tool_names:
            meta = TOOL_META.get(tool_name, {})
            display_name = meta.get("name", tool_name)
            summary = meta.get("summary") or runtime_tools.get(tool_name) or "已由系统注册的执行工具。"
            if tool_name in runtime_tools:
                state_key, state_label = "ready", "就绪"
            elif initialized:
                state_key, state_label = "connecting", "待就绪"
            else:
                state_key, state_label = "idle", "待连接"
            owner_chips = []
            for owner_name in owners.get(tool_name, ()):
                owner_meta = AGENT_META.get(owner_name, {})
                owner_label = owner_meta.get("name", owner_name).replace(" Agent", "")
                owner_chips.append(f'<span title="{escape(owner_name)}">{escape(owner_label)}</span>')
            rows.append(f"""
            <div class="capability-tool-row">
              <span class="capability-tool-dot is-{state_key}" aria-hidden="true"></span>
              <div class="capability-tool-copy"><div><strong>{escape(display_name)}</strong><code>{escape(tool_name)}</code></div>
                <p>{escape(summary)}</p><div class="capability-tool-owners"><b>授权给</b>{''.join(owner_chips) or '<span>运行时注册</span>'}</div></div>
              <span class="capability-tool-state is-{state_key}">{state_label}</span>
            </div>""")
        groups.append(f"""
        <details class="capability-tool-group" open>
          <summary><span>{escape(category_name)}</span><small>{category_ready}/{len(tool_names)} 就绪</small></summary>
          <div class="capability-tool-rows">{''.join(rows)}</div>
        </details>""")

    ready_count = len(known_names.intersection(runtime_tools))
    return f"""
    <section class="capability-list-section" aria-label="Tool 能力">
      <div class="capability-list-intro"><div><strong>执行能力</strong><span>状态来自 MCP 实时注册；Agent 只能调用被明确授权的 Tool。</span></div><span class="capability-count">{ready_count}/{len(known_names)} 已就绪</span></div>
      <div class="capability-tool-groups">{''.join(groups)}</div>
    </section>"""


def render_system_capabilities(
    available_tools: Iterable[Mapping[str, Any]] | None = None,
    *, initialized: bool = False, multi_agent: bool = True,
) -> tuple[str, str, str]:
    """Return overview, Agent view and Tool view markup for Gradio HTML blocks."""
    return (
        render_capability_summary(available_tools, initialized=initialized, multi_agent=multi_agent),
        render_agent_capabilities(available_tools, initialized=initialized),
        render_tool_capabilities(available_tools, initialized=initialized),
    )
