"""Recover the complete part of a JSON reply that was cut off mid-generation.

A reply that hits max_tokens ends mid-entry, so strict parsing rejects the
WHOLE reply -- in a class_proposal run that discards every class the model
had already written correctly. `salvage_truncated_json` keeps exactly the
entries that were fully written before the cut: it drops the partial entry
and closes the containers that were still open. It never invents content.
"""
from __future__ import annotations

import json
from typing import Any


def salvage_truncated_json(text: str) -> dict[str, Any] | None:
    """Parse the longest prefix of `text` that ends on a complete value.

    Returns None when no complete entry exists. A reply that is already valid
    JSON is returned as parsed.
    """
    start = text.find("{")
    if start == -1:
        return None
    try:
        whole = json.loads(text[start:])
        return whole if isinstance(whole, dict) else None
    except json.JSONDecodeError:
        pass

    stack: list[str] = []
    in_str = escape = False
    # (cut index, snapshot of open containers) after each complete container
    # value -- the only places where everything before is well-formed.
    safe: tuple[int, list[str]] | None = None
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append(ch)
            if ch == "[":  # an opened array is a valid (empty) value
                safe = (i + 1, list(stack))
        elif ch in "}]":
            if not stack:
                break
            stack.pop()
            if not stack:
                safe = (i + 1, [])
                break
            safe = (i + 1, list(stack))
    if safe is None:
        return None
    cut, open_stack = safe
    closers = "".join("}" if c == "{" else "]" for c in reversed(open_stack))
    candidate = text[start:cut].rstrip().rstrip(",") + closers
    try:
        out = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return out if isinstance(out, dict) else None
