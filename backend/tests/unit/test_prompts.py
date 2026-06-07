"""Prompt builders return non-empty (system, user) tuples with the right JSON shape hints."""

from __future__ import annotations

from backend.app.services.prompts import (
    PROMPTS,
    chunk_classification,
    class_identification_and_expansion,
    match_dedup,
)


def test_chunk_classification_shape() -> None:
    sys_, user = chunk_classification(
        top_level_branches=[{"iri": "http://x/A", "label": "A"}], text_chunk="hello world"
    )
    assert "relevant_iris" in sys_
    assert "hello world" in user
    assert "BRANCHES" in user


def test_class_identification_and_expansion_shape() -> None:
    sys_, user = class_identification_and_expansion(
        ontology_slice={"http://x/A": {"name": "A", "labels": ["A"]}},
        text_chunk="some passage",
    )
    assert "MATCHES FOUND" in sys_
    assert "MATCH NOT FOUND" in sys_
    assert "DATA_CLASSES" in user
    assert "some passage" in user


def test_class_identification_with_suggestions() -> None:
    sys_, _user = class_identification_and_expansion(
        ontology_slice={},
        text_chunk="text",
        suggested_new_classes={"X": "y"},
    )
    assert "SUGGESTED NEW CLASSES" in sys_


def test_match_dedup_skips_when_no_proposals_is_not_in_prompt() -> None:
    sys_, user = match_dedup({"MATCHES FOUND": [], "MATCH NOT FOUND": []})
    assert "MATCH NOT FOUND" in sys_
    assert '"MATCHES FOUND"' in user or "MATCHES FOUND" in user


def test_prompts_registry_covers_four_tasks() -> None:
    assert set(PROMPTS) == {
        "chunk_classification",
        "class_proposal",
        "match_dedup",
        "concept_grouping",
    }
