"""Constrained writer role for natural, result-specific biomarker explanations."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from .biomarker_reporting import (
    append_verified_report_links,
    build_biomarker_reporting_packet,
    validate_biomarker_narrative,
)
from .biomarker_summary import build_biomarker_user_summary
from .state import AgentState


_STYLE_INSTRUCTIONS = (
    "先点明最值得关注的结论，再解释数字如何支持或限制这个结论。",
    "沿着样本筛选、训练集发现、测试集确认的证据链自然讲述，不要套固定栏目。",
    "像向研究团队口头汇报一样，重点比较候选数与确认数，并逐项解释确认特征。",
    "从读者最可能误解的数字切入，解释效应大小、校正显著性和横断面边界。",
)


class BiomarkerReporterAgent:
    """Writes fluent prose from an immutable, independently audited fact packet."""

    system_prompt = """你是医学影像研究结果讲解 Agent。你的输入只有已经通过独立审计的结构化事实。
请用自然、针对本次实验的简体中文向研究人员解释结果，不使用固定套话或机械模板。

硬性规则：
1. 必须逐一解释输入中的样本量、排除数、检验特征数、冻结候选数和确认候选数意味着什么。
2. 对每个 confirmed candidate，必须原样写出 feature 名称和提供的所有显示数值，并解释：OR/SD 是效应大小，BH q 是发现阶段多重校正，bootstrap support 是方向稳定性，Holm P 是测试集确认显著性。
3. OR>1 只能解释为特征升高与更高 KL 等级相关；OR<1 只能解释为与更低 KL 等级相关。
4. 如果没有确认候选，要分析训练发现与测试确认之间的差异，不得建议放宽阈值或挑选未确认特征。
5. 明确这是横断面关联，不能推断因果、未来进展、个体诊断或临床替代终点。
6. 不得添加输入中没有的数字、候选、文献、临床建议或机制解释。
7. 不输出内部审计过程、JSON、固定格式表格或千篇一律的开场白。可以自由组织段落，使两次相同实验的表达也更自然。
"""

    def __init__(self, config=None):
        self.config = config
        self.llm = config.biomarker_reporter_agent.llm

    @staticmethod
    def _style_for_run(state: AgentState) -> str:
        run_id = str(state.get("task_run_id") or state.get("trace_id") or "run")
        index = int(hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:8], 16)
        return _STYLE_INSTRUCTIONS[index % len(_STYLE_INSTRUCTIONS)]

    def _invoke(self, packet: dict[str, Any], style: str, correction: str = "") -> str:
        prompt = (
            f"本次表达侧重点：{style}\n"
            "以下 JSON 是唯一事实来源。所有字符串形式的显示数值必须原样出现在回复中：\n"
            f"{json.dumps(packet, ensure_ascii=False, indent=2)}"
        )
        if correction:
            prompt += (
                "\n上一次草稿未通过事实校验，请完整重写。校验问题："
                f"{correction}"
            )
        response = self.llm.invoke(
            [SystemMessage(content=self.system_prompt), HumanMessage(content=prompt)]
        )
        return str(getattr(response, "content", response) or "").strip()

    async def process(self, state: AgentState) -> AgentState:
        audit = dict(state.get("biomarker_audit", {}) or {})
        if audit.get("valid") is not True:
            message = "结果讲解被阻断：本次生物标志物运行尚未通过独立审计。"
            return {
                **state,
                "agent_name": "BIOMARKER_REPORTER_AGENT",
                "output": message,
                "success": False,
                "error_category": "unverified_biomarker_result",
                "error_message": message,
                "tool_calls": [],
                "tool_results": [],
            }

        packet = build_biomarker_reporting_packet(audit)
        style = self._style_for_run(state)
        validation_errors: list[str] = []
        for attempt in range(2):
            try:
                draft = self._invoke(
                    packet,
                    style,
                    "; ".join(validation_errors) if attempt else "",
                )
            except Exception as exc:
                validation_errors = [f"writer_exception:{type(exc).__name__}"]
                continue
            validation_errors = validate_biomarker_narrative(draft, packet)
            if not validation_errors:
                narrative = append_verified_report_links(draft, packet)
                return {
                    **state,
                    "agent_name": "BIOMARKER_REPORTER_AGENT",
                    "output": narrative,
                    "success": True,
                    "tool_calls": [],
                    "tool_results": [],
                    "biomarker_narrative": narrative,
                    "biomarker_reporting": {
                        "mode": "verified_llm_narrative",
                        "validation_status": "pass",
                        "attempts": attempt + 1,
                    },
                }

        # This is an exceptional safety fallback, not the normal response path.
        narrative = build_biomarker_user_summary(audit)
        return {
            **state,
            "agent_name": "BIOMARKER_REPORTER_AGENT",
            "output": narrative,
            "success": True,
            "tool_calls": [],
            "tool_results": [],
            "biomarker_narrative": narrative,
            "biomarker_reporting": {
                "mode": "deterministic_safety_fallback",
                "validation_status": "fallback",
                "validation_errors": validation_errors,
                "attempts": 2,
            },
        }
