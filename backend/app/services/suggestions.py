"""User-supplied class suggestions.

Users can hand-author a JSON file listing classes they want added to the
ontology, in addition to whatever the LLM proposes from documents. This
module loads + normalizes that file and converts it to the same shape the
LLM uses for proposed classes (MATCH NOT FOUND entries), so the rest of the
pipeline doesn't need to special-case them.

Input format (list of objects):
    [
      {
        "CLASS_TYPE": "Adverse Events",
        "CLASS_DESCRIPTION": "Adverse Events listed in a Drug or in a Study",
        "PARENT_CLASS_TYPE": "NONE"
      },
      ...
    ]

PARENT_CLASS_TYPE may be "NONE" (no parent — class roots at owl:Thing) or
the LABEL of another class in the same suggestion list / existing ontology.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_suggested_classes(path: Path | None) -> list[dict[str, Any]]:
    """Read the suggestions JSON, returning a list of dicts. Empty list if
    the path is None or the file is missing."""
    if path is None:
        return []
    if not path.exists():
        raise FileNotFoundError(f"suggested-new-classes file not found: {path}")
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError(
            f"{path} must contain a JSON list of "
            f"{{CLASS_TYPE, CLASS_DESCRIPTION, PARENT_CLASS_TYPE}} objects"
        )
    out: list[dict[str, Any]] = []
    for i, entry in enumerate(data):
        if not isinstance(entry, dict):
            raise ValueError(f"{path}[{i}] is not an object")
        if "CLASS_TYPE" not in entry:
            raise ValueError(f"{path}[{i}] missing CLASS_TYPE")
        out.append(
            {
                "CLASS_TYPE": str(entry["CLASS_TYPE"]).strip(),
                "CLASS_DESCRIPTION": str(entry.get("CLASS_DESCRIPTION", "")).strip(),
                "PARENT_CLASS_TYPE": str(entry.get("PARENT_CLASS_TYPE", "NONE")).strip() or "NONE",
            }
        )
    return out


def to_match_not_found_entries(suggested: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert user-suggested classes to the {LABEL, DESCRIPTION} shape that
    the deterministic `add_new_classes_from_match_not_found` consumes.

    The parent-class hint is preserved as an extra `PARENT_LABEL` field; the
    add-helper uses it only if we explicitly pass a parent resolver. Phase 1
    keeps things simple: PARENT_CLASS_TYPE == "NONE" falls back to the
    default_parent_iri configured in app config.
    """
    out: list[dict[str, Any]] = []
    for entry in suggested:
        out.append(
            {
                "LABEL": entry["CLASS_TYPE"],
                "DESCRIPTION": entry.get("CLASS_DESCRIPTION", ""),
                "PARENT_LABEL": entry.get("PARENT_CLASS_TYPE", "NONE"),
                "SOURCE": "user_suggestion",
            }
        )
    return out


def merge_suggestions_into_results(
    match_results: dict[str, Any],
    suggested: list[dict[str, Any]],
) -> dict[str, Any]:
    """Add user-suggested classes to the deduplicated MATCH NOT FOUND list.

    Skip ones whose label already appears (case-insensitive) so a user
    suggestion isn't double-added when the LLM has already proposed the
    same concept. We DON'T touch MATCHES FOUND.
    """
    if not suggested:
        return match_results
    existing_labels = {
        str(e.get("LABEL", "")).strip().lower()
        for e in match_results.get("MATCH NOT FOUND", [])
        if isinstance(e, dict)
    }
    new_entries = []
    for entry in to_match_not_found_entries(suggested):
        if entry["LABEL"].lower() in existing_labels:
            continue
        new_entries.append(entry)
        existing_labels.add(entry["LABEL"].lower())

    merged = {
        "MATCHES FOUND": list(match_results.get("MATCHES FOUND", [])),
        "MATCH NOT FOUND": list(match_results.get("MATCH NOT FOUND", [])) + new_entries,
        # Pass through any LLM-proposed relations untouched -- user
        # suggestions don't currently include relations (the JSON file
        # format is class-only), so there's nothing to merge here.
        "MATCH NOT FOUND RELATIONS": list(match_results.get("MATCH NOT FOUND RELATIONS", [])),
    }
    return merged
