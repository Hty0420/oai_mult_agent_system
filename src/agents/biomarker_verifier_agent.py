"""Independent read-only verifier role for biomarker discovery runs."""

from __future__ import annotations

from pathlib import Path
import re

from .path_utils import parse_tool_result_paths
from .state import AgentState
from .tool_manager import get_tool_manager
from .tool_result_utils import (
    parse_structured_tool_result,
    tool_error_message,
    tool_result_succeeded,
)


class BiomarkerVerifierAgent:
    """Verifies hashes and statistics without changing the candidate set."""

    system_prompt = (
        "Independent research verifier. Recompute deterministic gates and fail "
        "closed on leakage, hash mismatch, multiplicity mismatch, or count mismatch."
    )

    def __init__(self, config=None):
        self.config = config
        self.llm = None

    @staticmethod
    def _manifest_path(state: AgentState) -> str | None:
        current_input = state.get("current_input")
        text = current_input if isinstance(current_input, str) else str(current_input or "")
        for key, path in parse_tool_result_paths(text):
            if key == "manifest_path" and Path(path).name == "biomarker_manifest.json":
                return path
        for item in state.get("current_artifacts", []) or []:
            path = str(item.get("path") or "")
            if Path(path).name == "biomarker_manifest.json":
                return path
        return None

    @staticmethod
    def _audit_output_dir(state: AgentState) -> str:
        run_id = str(state.get("task_run_id") or state.get("trace_id") or "run")
        safe_run_id = re.sub(r"[^A-Za-z0-9_.-]", "_", run_id)
        src_root = Path(__file__).resolve().parents[1]
        return str(src_root / "biomarker_runs" / safe_run_id / "audit")

    async def process(self, state: AgentState) -> AgentState:
        manifest_path = self._manifest_path(state)
        if not manifest_path:
            message = "独立审计缺少当前运行的 biomarker_manifest.json。"
            return {
                **state,
                "agent_name": "BIOMARKER_VERIFIER_AGENT",
                "output": message,
                "success": False,
                "error_category": "missing_biomarker_manifest",
                "error_message": message,
                "tool_calls": [],
                "tool_results": [],
            }

        arguments = {
            "manifest_path": manifest_path,
            "output_dir": self._audit_output_dir(state),
        }
        stream_callback = state.get("stream_callback")
        if stream_callback:
            stream_callback("   正在独立复核候选冻结、数据泄漏、多重检验和产物哈希。")
        manager = await get_tool_manager()
        raw_result = await manager.call_tool(
            "audit_biomarker_discovery",
            arguments,
            stream_callback=stream_callback,
        )
        parsed = parse_structured_tool_result(raw_result)
        if (
            not tool_result_succeeded(raw_result)
            or not isinstance(parsed, dict)
            or parsed.get("valid") is not True
        ):
            message = tool_error_message(raw_result)
            return {
                **state,
                "agent_name": "BIOMARKER_VERIFIER_AGENT",
                "output": f"生物标志物运行审计未通过：{message}",
                "success": False,
                "error_category": "biomarker_verification_failure",
                "error_message": message,
                "tool_calls": ["audit_biomarker_discovery"],
                "tool_results": [raw_result],
            }

        output = (
            "独立审计通过；已将经过复核的结构化事实交给结果讲解智能体。"
        )
        return {
            **state,
            "agent_name": "BIOMARKER_VERIFIER_AGENT",
            "output": output,
            "success": True,
            "tool_calls": ["audit_biomarker_discovery"],
            "tool_results": [raw_result],
            "biomarker_audit": parsed,
        }
