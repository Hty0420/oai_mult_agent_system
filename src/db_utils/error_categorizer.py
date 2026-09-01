"""Pure-function error categorizer for tool and agent errors.

No database dependency — usable from any module without circular imports.
"""

from __future__ import annotations

import re
from typing import Optional

# Ordered: first match wins. More specific patterns rank above generic ones.
_RULES: list[tuple[str, str]] = [
    # --- placeholder_path (most specific, before generic "not found") ---
    (
        "placeholder_path",
        r"path/to/|path\\to\\|placeholder|占位路径|占位",
    ),
    # --- path_not_found ---
    (
        "path_not_found",
        r"no such file|filenotfounderror|cannot find the path|does not exist"
        r"|not exist|路径不存在|文件不存在|目录不存在|找不到文件"
        r"|directory does not exist|no such file or directory"
        r"|must be an existing regular file",
    ),
    # --- tool_not_found ---
    (
        "tool_not_found",
        r"tool .* not found|not registered|not initialized|available tools:"
        r"|工具未找到|未找到工具|未注册",
    ),
    # --- tool_timeout ---
    (
        "tool_timeout",
        r"timed out|timeout|timeouterror|超时|连接超时",
    ),
    # --- external_service_down ---
    (
        "external_service_down",
        r"connection refused|connectionerror|failed to connect"
        r"|connection reset|econnrefused|无法连接|连接失败|连接断开"
        r"|error code: 500|internalservererror|internal server error"
        r"|new_api_error|database error.*contact the administrator",
    ),
    # --- permission_error ---
    (
        "permission_error",
        r"permission denied|permissionerror|access denied"
        r"|权限不足|没有权限",
    ),
    # --- missing_model_path ---
    (
        "missing_model_path",
        r"requires the exact model_path|model_path.*required"
        r"|missing.*model_path|ldse_model_path",
    ),
    # --- llm_parse_error ---
    (
        "llm_parse_error",
        r"no valid json|jsondecodeerror|json\.decoder"
        r"|expecting value|解析失败|invalid json"
        r"|parse error|unable to parse|no valid json object",
    ),
    # --- ml_fitting_error ---
    (
        "ml_fitting_error",
        r"did not converge|convergencewarning|singular matrix"
        r"|不收敛|convergence|matrix is singular",
    ),
    # --- invalid_params ---
    (
        "invalid_params",
        r"invalid|must be|typeerror|valueerror|missing.*argument"
        r"|invalid.*parameter|argument|参数错误|参数缺失"
        r"|refusing to call|missing required",
    ),
]

CATEGORIES: tuple[str, ...] = tuple(
    dict.fromkeys(cat for cat, _ in _RULES + [("unknown", "")])
)


def categorize_error(
    error_message: Optional[str],
    tool_name: Optional[str] = None,
) -> str:
    """Classify an error message into one of the known categories.

    Args:
        error_message: The error text to classify.  ``None`` or empty → ``"unknown"``.
        tool_name: Optional tool name for context-aware classification.
                   ``NaN``/``inf`` in a classification-tool error → ``ml_fitting_error``.

    Returns:
        One of: ``placeholder_path``, ``path_not_found``, ``tool_not_found``,
        ``tool_timeout``, ``external_service_down``, ``permission_error``,
        ``missing_model_path``, ``llm_parse_error``, ``ml_fitting_error``,
        ``invalid_params``, ``unknown``.
    """
    if not error_message or not error_message.strip():
        return "unknown"

    lower = error_message.strip().lower()

    # Tool-aware: NaN / inf in classification context
    if tool_name and any(kw in (tool_name or "").lower() for kw in ("classif", "lasso")):
        if re.search(r"\bnan\b|\binf\b|not finite|non-finite", lower):
            return "ml_fitting_error"

    for category, pattern in _RULES:
        if re.search(pattern, lower):
            return category

    return "unknown"
