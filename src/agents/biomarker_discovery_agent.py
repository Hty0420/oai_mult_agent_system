"""Deterministic analyst role for MRI candidate biomarker discovery."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

from .path_utils import collect_csv_paths, select_train_test_feature_paths
from .state import AgentState
from .tool_manager import get_tool_manager
from .tool_result_utils import (
    parse_structured_tool_result,
    tool_error_message,
    tool_result_succeeded,
)


class BiomarkerDiscoveryAgent:
    """Calls the approved deterministic discovery tool; it does no statistics."""

    system_prompt = (
        "Research-only analyst. Use training data for candidate discovery and the "
        "frozen test set only for confirmation. Never invent or relax thresholds."
    )

    def __init__(self, config=None):
        self.config = config
        self.llm = None

    @staticmethod
    def _run_output_dir(state: AgentState) -> str:
        run_id = str(state.get("task_run_id") or state.get("trace_id") or "run")
        safe_run_id = re.sub(r"[^A-Za-z0-9_.-]", "_", run_id)
        src_root = Path(__file__).resolve().parents[1]
        return str(src_root / "biomarker_runs" / safe_run_id / "analysis")

    @staticmethod
    def _feature_paths(state: AgentState) -> tuple[str | None, str | None]:
        artifact_paths = [
            str(item.get("path"))
            for item in state.get("current_artifacts", []) or []
            if item.get("path") and str(item.get("path")).lower().endswith(".csv")
        ]
        current_input = state.get("current_input")
        text = current_input if isinstance(current_input, str) else str(current_input or "")
        paths = artifact_paths + collect_csv_paths(text)
        return select_train_test_feature_paths(paths)

    async def process(self, state: AgentState) -> AgentState:
        stream_callback = state.get("stream_callback")
        train_path, test_path = self._feature_paths(state)
        if not train_path or not test_path:
            message = "生物标志物发现缺少当前运行生成的训练/测试患者级特征 CSV。"
            return {
                **state,
                "agent_name": "BIOMARKER_DISCOVERY_AGENT",
                "output": message,
                "success": False,
                "error_category": "missing_feature_artifact",
                "error_message": message,
                "tool_calls": [],
                "tool_results": [],
            }

        arguments = {
            "train_feature_path": train_path,
            "test_feature_path": test_path,
            "output_dir": self._run_output_dir(state),
            "endpoint": "KLGrade",
            "id_column": "CaseID",
            "covariates": "Age,Gender,BMI,KneeSide",
            "fdr_alpha": 0.05,
            "confirmation_alpha": 0.05,
            "correlation_threshold": 0.90,
            "maximum_candidates": 20,
            "bootstrap_iterations": 100,
            "bootstrap_support_threshold": 0.70,
            "random_seed": 42,
        }
        if stream_callback:
            stream_callback("   正在训练集发现候选，并在冻结测试集中执行预设确认。")
        manager = await get_tool_manager()
        raw_result = await manager.call_tool(
            "run_biomarker_discovery",
            arguments,
            stream_callback=stream_callback,
        )
        parsed = parse_structured_tool_result(raw_result)
        if not tool_result_succeeded(raw_result) or not isinstance(parsed, dict):
            message = tool_error_message(raw_result)
            return {
                **state,
                "agent_name": "BIOMARKER_DISCOVERY_AGENT",
                "output": f"生物标志物发现失败：{message}",
                "success": False,
                "error_category": "biomarker_analysis_failure",
                "error_message": message,
                "tool_calls": ["run_biomarker_discovery"],
                "tool_results": [raw_result],
            }

        outcome = str(parsed.get("scientific_outcome") or "completed")
        output = (
            f"已完成 KL 0–4 MRI 候选生物标志物发现。"
            f"训练集冻结候选 {parsed.get('candidate_count', 0)} 个，"
            f"测试集确认 {parsed.get('confirmed_count', 0)} 个；"
            f"科学结局为 {outcome}。\n"
            f"报告：{parsed.get('report_path', '')}\n"
            f"运行清单：{parsed.get('manifest_path', '')}"
        )
        return {
            **state,
            "agent_name": "BIOMARKER_DISCOVERY_AGENT",
            "output": output,
            "success": True,
            "tool_calls": ["run_biomarker_discovery"],
            "tool_results": [raw_result],
            "biomarker_result": parsed,
        }
