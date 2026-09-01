from __future__ import annotations

from agents.tool_policy import AGENT_TOOL_ALLOWLIST
from system_capabilities import (
    AGENT_META,
    TOOL_META,
    registered_tool_names,
    render_system_capabilities,
    tool_owners,
)


def _tool(name: str, description: str = "runtime description") -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {},
        },
    }


def test_capability_catalog_covers_the_execution_policy():
    assert set(AGENT_META) == set(AGENT_TOOL_ALLOWLIST)
    assert registered_tool_names() == frozenset(TOOL_META)
    assert len(AGENT_META) == 12
    assert len(registered_tool_names()) == 25


def test_agent_to_tool_ownership_is_exposed():
    owners = tool_owners()
    assert owners["umamba_segment_mri"] == ("SEGMENTATION_AGENT",)
    assert set(owners["assess_feature_quality"]) == {
        "FEATURE_EXTRACTION_AGENT",
        "CLASSIFICATION_AGENT",
    }


def test_rendered_panel_separates_agents_tools_and_live_state():
    overview, agents, tools = render_system_capabilities(
        [_tool("umamba_segment_mri"), _tool("web_search")],
        initialized=True,
        multi_agent=True,
    )

    assert "12" in overview and "25" in overview
    assert "2 项工具已就绪" in overview
    assert "任务路由 Agent" in agents
    assert "1/3 Tool 就绪" in agents
    assert "U-Mamba MRI 分割" in tools
    assert "实时网页搜索" in tools
    assert "待就绪" in tools


def test_unknown_runtime_tool_is_escaped_and_still_visible():
    _, _, tools = render_system_capabilities(
        [_tool("future_tool", "<script>alert('x')</script>")],
        initialized=True,
    )

    assert "future_tool" in tools
    assert "&lt;script&gt;" in tools
    assert "<script>" not in tools
