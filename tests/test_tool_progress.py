from __future__ import annotations

import asyncio

from agents.tool_progress import (
    TOOL_PROGRESS_PREFIX,
    append_progress_text,
    build_tool_progress_callback,
    progress_line_key,
)


def test_mcp_progress_is_rendered_as_throttled_frontend_bar() -> None:
    messages: list[str] = []
    callback = build_tool_progress_callback(
        "meas_features_extraction",
        messages.append,
    )
    assert callback is not None

    asyncio.run(callback(0, None, "正在读取病例 CSV"))
    asyncio.run(callback(0, 20, "已处理 0/20"))
    asyncio.run(callback(1, 20, "已处理 1/20"))
    asyncio.run(callback(2, 20, "已处理 2/20"))
    asyncio.run(callback(20, 20, "已处理 20/20"))

    assert messages[0].startswith(f"{TOOL_PROGRESS_PREFIX} · MEAS")
    assert "[准备中]" in messages[0]
    assert any(" 10.0%" in message for message in messages)
    assert messages[-1].count("█") == 20
    assert "100.0%" in messages[-1]


def test_persisted_progress_keeps_only_latest_bar_for_each_tool() -> None:
    meas_10 = (
        f"{TOOL_PROGRESS_PREFIX} · MEAS 批量特征提取 "
        "[██░░░░░░░░░░░░░░░░░░]  10.0%"
    )
    meas_20 = (
        f"{TOOL_PROGRESS_PREFIX} · MEAS 批量特征提取 "
        "[████░░░░░░░░░░░░░░░░]  20.0%"
    )
    ldse_5 = (
        f"{TOOL_PROGRESS_PREFIX} · LDSE 批量特征提取 "
        "[█░░░░░░░░░░░░░░░░░░░]   5.0%"
    )

    text = append_progress_text("步骤开始", meas_10, max_chars=2000)
    text = append_progress_text(text, ldse_5, max_chars=2000)
    text = append_progress_text(text, meas_20, max_chars=2000)

    assert meas_10 not in text
    assert meas_20 in text
    assert ldse_5 in text
    assert text.count(TOOL_PROGRESS_PREFIX) == 2
    assert progress_line_key(meas_10) == progress_line_key(meas_20)
