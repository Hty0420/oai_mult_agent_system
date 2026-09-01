"""Manual smoke check for MCP server discovery (not collected by pytest)."""

from __future__ import annotations

import asyncio
from pathlib import Path
import sys


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from agents.tool_manager import MCPToolManager


async def main() -> None:
    manager = MCPToolManager()
    await manager.initialize()
    try:
        names = sorted(
            item["function"]["name"] for item in manager.get_available_tools()
        )
        print("TOOL_COUNT", len(names))
        print("BIOMARKER_TOOLS", [name for name in names if "biomarker" in name])
    finally:
        await manager.close()


if __name__ == "__main__":
    asyncio.run(main())
