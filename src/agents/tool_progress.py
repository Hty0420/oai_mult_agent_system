"""Uniform MCP tool-progress rendering for the persisted front-end stream."""

from __future__ import annotations

import re
from typing import Any, Callable


TOOL_PROGRESS_PREFIX = "📈 工具进度"

_TOOL_LABELS = {
    "extract_cartilage_thickness_features": "软骨厚度批量特征提取",
    "extract_radiomics_features": "Radiomics 批量特征提取",
    "meas_features_extraction": "MEAS 批量特征提取",
    "ldse_train_test_features_extraction": "LDSE 训练/测试特征提取",
    "ldse_features_extraction": "LDSE 批量特征提取",
    "augment_dataset_offline": "离线数据增强",
    "augment_online_extract_features": "在线增强与特征提取",
    "koa_lasso_classification": "LASSO 分类",
    "koa_filter_classification": "Filter 分类",
    "run_biomarker_discovery": "候选生物标志物统计分析",
    "audit_biomarker_discovery": "候选生物标志物独立审计",
}


def tool_progress_label(tool_name: str) -> str:
    return _TOOL_LABELS.get(str(tool_name), str(tool_name).replace("_", " "))


def _one_line(value: Any, max_chars: int = 240) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text[:max_chars]


def build_tool_progress_callback(
    tool_name: str,
    stream_callback: Callable[[str], Any] | None,
    *,
    percent_step: int = 5,
):
    """Create an MCP callback that emits one replaceable Markdown progress line."""
    if stream_callback is None:
        return None
    label = tool_progress_label(tool_name)
    state = {"bucket": -1, "indeterminate_sent": False}
    step = max(1, min(25, int(percent_step)))

    async def report_progress(
        progress: float,
        total: float | None,
        message: str | None,
    ) -> None:
        detail = _one_line(message)
        try:
            current = float(progress)
            maximum = float(total) if total is not None else 0.0
        except (TypeError, ValueError):
            current, maximum = 0.0, 0.0

        if maximum <= 0:
            if state["indeterminate_sent"] and not detail:
                return
            state["indeterminate_sent"] = True
            line = f"{TOOL_PROGRESS_PREFIX} · {label} [准备中]"
            if detail:
                line += f" · {detail}"
            stream_callback(line)
            return

        percent = max(0.0, min(100.0, current / maximum * 100.0))
        bucket = min(100 // step, int(percent // step))
        if bucket == state["bucket"] and current < maximum:
            return
        state["bucket"] = bucket
        filled = min(20, int(percent // 5))
        bar = "█" * filled + "░" * (20 - filled)
        line = f"{TOOL_PROGRESS_PREFIX} · {label} [{bar}] {percent:5.1f}%"
        if detail:
            line += f" · {detail}"
        stream_callback(line)

    return report_progress


def progress_line_key(line: str) -> str | None:
    """Return the stable prefix used to replace an earlier update for one tool."""
    stripped = str(line or "").strip()
    if stripped.startswith(TOOL_PROGRESS_PREFIX):
        return stripped.split("[", 1)[0].strip()
    # Retain compatibility with runs started by the previous cartilage-only path.
    if stripped.startswith("📈 软骨厚度批量提取"):
        return "📈 软骨厚度批量提取"
    return None


def append_progress_text(
    existing: str | None,
    content: Any,
    *,
    max_chars: int,
) -> str:
    """Append a stream chunk while retaining only the latest bar per tool."""
    chunk = str(content or "").strip()
    if not chunk:
        return str(existing or "")
    chunk_key = progress_line_key(chunk)
    if chunk_key:
        retained_lines = [
            line
            for line in str(existing or "").splitlines()
            if progress_line_key(line) != chunk_key
        ]
        combined = "\n".join([*retained_lines, chunk]).strip()
    else:
        combined = f"{existing}\n{chunk}".strip() if existing else chunk
    if len(combined) > max_chars:
        combined = "…（较早的执行信息已折叠）\n" + combined[-max_chars:]
    return combined
