"""Parse a MedTool submission into normalized calculator rows.

Accept JSON objects or lists, Markdown fences and Python literal representations.
Normalize calculator-name and value keys. Empty, missing and malformed outputs
remain distinct submission failures for the task oracle.
"""
from __future__ import annotations

import ast
import json
import re
from typing import Any

#: Submission failures are reported as distinct outcome categories.
NO_ANSWER = "no_answer"
EMPTY_SUBMISSION = "empty_submission"
INVALID_FORMAT = "invalid_response_format"

_FENCE = re.compile(r"^\s*```(?:json|python)?\s*|\s*```\s*$", re.IGNORECASE)
_NAME_KEYS = ("name", "calculator", "calculator_name")
_VALUE_KEYS = ("value", "final_answer", "answer", "result")


def _decode(text: str, start: str) -> Any | None:
    """Return the first substring from ``start`` that parses as JSON or a Python literal.

    Scan candidate starting positions in order to locate structured output within prose.
    """
    dec = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch != start:
            continue
        try:
            return dec.raw_decode(text[i:])[0]
        except Exception:
            pass
        try:
            return ast.literal_eval(text[i:])
        except Exception:
            continue
    return None


def _row(obj: Any) -> dict[str, Any] | None:
    if not isinstance(obj, dict):
        return None
    name = next((obj[k] for k in _NAME_KEYS if obj.get(k) not in (None, "")), None)
    if name is None:
        return None
    value = next((obj[k] for k in _VALUE_KEYS if k in obj), None)
    row: dict[str, Any] = {"name": str(name), "value": value}
    episode = obj.get("episode_id")
    if episode not in (None, ""):
        row["episode_id"] = str(episode)
    return row


def recover_submission(answer: str | None) -> tuple[dict[str, Any] | None, str | None]:
    """(prediction, None) on success, or (None, reason) with one of the classified failures."""
    text = (answer or "").strip()
    if not text:
        return None, NO_ANSWER
    text = _FENCE.sub("", text).strip()

    payload = _decode(text, "[")
    if payload is None:
        obj = _decode(text, "{")
        if isinstance(obj, dict) and isinstance(obj.get("calculators"), list):
            payload = obj["calculators"]
        elif isinstance(obj, dict):
            payload = [obj]
    if payload is None:
        return None, INVALID_FORMAT
    if not isinstance(payload, list):
        return None, INVALID_FORMAT
    if not payload:
        return None, EMPTY_SUBMISSION

    rows = [_row(o) for o in payload]
    if any(r is None for r in rows):
        return None, INVALID_FORMAT
    return {"calculators": rows}, None
