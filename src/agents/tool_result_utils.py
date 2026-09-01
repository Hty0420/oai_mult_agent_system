"""Dependency-light helpers for interpreting MCP tool results."""

from __future__ import annotations

import ast
import json
from typing import Any


_ERROR_PREFIXES = (
    "tool execution failed:",
    "error executing tool",
    "error:",
    "工具执行失败",
    "执行失败",
)


def parse_structured_tool_result(result: Any) -> Any:
    """Parse JSON/Python-literal tool payloads while preserving plain text."""
    if not isinstance(result, str):
        return result
    stripped = result.strip()
    if not stripped.startswith(("{", "[")):
        return result
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(stripped)
        except (ValueError, SyntaxError):
            return result


def tool_result_succeeded(result: Any) -> bool:
    """Return False for structured failures and common MCP error strings."""
    parsed = parse_structured_tool_result(result)
    if isinstance(parsed, dict):
        status = str(parsed.get("status", "")).strip().lower()
        if status in {"failed", "failure", "error"}:
            return False
        if parsed.get("success") is False:
            return False
        error = parsed.get("error") or parsed.get("error_message")
        if error:
            return False
        return True
    if isinstance(parsed, str):
        lowered = parsed.strip().lower()
        return not any(lowered.startswith(prefix) for prefix in _ERROR_PREFIXES)
    return parsed is not None


def tool_error_message(result: Any) -> str:
    """Extract the most useful failure explanation from a tool result."""
    parsed = parse_structured_tool_result(result)
    if isinstance(parsed, dict):
        message = str(
            parsed.get("message")
            or parsed.get("error_message")
            or parsed.get("error")
            or parsed
        )
        qc = parsed.get("qc")
        qc_errors = qc.get("errors", []) if isinstance(qc, dict) else []
        if isinstance(qc_errors, list):
            details = [
                str(error).strip()
                for error in qc_errors
                if str(error).strip() and str(error).strip() not in message
            ]
            if details:
                message = f"{message} 具体原因：{'；'.join(details)}"
        return message
    return str(result)
