"""Helpers for parsing JSON returned by chat models."""

from __future__ import annotations

import json
import re
from typing import Any


_FENCED_JSON_RE = re.compile(
    r"```(?:json|JSON)?\s*(.*?)\s*```",
    re.DOTALL,
)


def parse_llm_json(content: str) -> Any:
    """Parse JSON from an LLM response.

    Models sometimes wrap valid JSON in Markdown fences or add a short
    explanatory sentence. This helper keeps strict JSON as the fast path, then
    falls back to extracting the first valid JSON object or array.
    """
    text = content.strip()
    if not text:
        raise ValueError("LLM response is empty")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fence_match = _FENCED_JSON_RE.search(text)
    if fence_match:
        fenced_text = fence_match.group(1).strip()
        try:
            return json.loads(fenced_text)
        except json.JSONDecodeError:
            text = fenced_text

    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char not in "{[":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[index:])
            return parsed
        except json.JSONDecodeError:
            continue

    raise ValueError("No valid JSON object or array found in LLM response")
