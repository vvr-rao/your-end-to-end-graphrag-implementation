"""User-supplied class suggestions: load + normalize + inject into match results."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.app.services import suggestions


def _write_json(tmp_path: Path, data) -> Path:
    p = tmp_path / "sug.json"
    p.write_text(json.dumps(data))
    return p


def test_load_none_returns_empty() -> None:
    assert suggestions.load_suggested_classes(None) == []


def test_load_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        suggestions.load_suggested_classes(tmp_path / "missing.json")


def test_load_normalizes_entries(tmp_path: Path) -> None:
    p = _write_json(
        tmp_path,
        [
            {"CLASS_TYPE": "  Foo  ", "CLASS_DESCRIPTION": "desc"},
            {"CLASS_TYPE": "Bar", "PARENT_CLASS_TYPE": "Foo"},
        ],
    )
    out = suggestions.load_suggested_classes(p)
    assert out == [
        {"CLASS_TYPE": "Foo", "CLASS_DESCRIPTION": "desc", "PARENT_CLASS_TYPE": "NONE"},
        {"CLASS_TYPE": "Bar", "CLASS_DESCRIPTION": "", "PARENT_CLASS_TYPE": "Foo"},
    ]


def test_load_rejects_non_list(tmp_path: Path) -> None:
    p = _write_json(tmp_path, {"not": "a list"})
    with pytest.raises(ValueError):
        suggestions.load_suggested_classes(p)


def test_load_rejects_missing_class_type(tmp_path: Path) -> None:
    p = _write_json(tmp_path, [{"CLASS_DESCRIPTION": "only desc"}])
    with pytest.raises(ValueError):
        suggestions.load_suggested_classes(p)


def test_merge_suggestions_adds_new_to_match_not_found() -> None:
    base = {"MATCHES FOUND": [{"IRI": "http://x/A"}], "MATCH NOT FOUND": []}
    suggested = [{"CLASS_TYPE": "NewThing", "CLASS_DESCRIPTION": "d", "PARENT_CLASS_TYPE": "NONE"}]
    merged = suggestions.merge_suggestions_into_results(base, suggested)
    labels = [e["LABEL"] for e in merged["MATCH NOT FOUND"]]
    assert "NewThing" in labels
    assert merged["MATCHES FOUND"] == base["MATCHES FOUND"]


def test_merge_suggestions_skips_duplicates_case_insensitive() -> None:
    base = {
        "MATCHES FOUND": [],
        "MATCH NOT FOUND": [{"LABEL": "newthing", "DESCRIPTION": "existing"}],
    }
    suggested = [{"CLASS_TYPE": "NewThing", "CLASS_DESCRIPTION": "d", "PARENT_CLASS_TYPE": "NONE"}]
    merged = suggestions.merge_suggestions_into_results(base, suggested)
    labels = [e["LABEL"].lower() for e in merged["MATCH NOT FOUND"]]
    assert labels.count("newthing") == 1


def test_merge_suggestions_with_empty_list_is_passthrough() -> None:
    base = {"MATCHES FOUND": [{"IRI": "http://x/A"}], "MATCH NOT FOUND": [{"LABEL": "X"}]}
    assert suggestions.merge_suggestions_into_results(base, []) is base
