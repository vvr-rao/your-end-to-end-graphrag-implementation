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


def test_prompts_registry_covers_core_phase1_tasks() -> None:
    # Phase 1 LLM-pipeline tasks must always be present. Phase 2 added
    # many more (artifact extraction, QA, judging) -- those are covered
    # by a superset check below.
    phase1_tasks = {
        "chunk_classification",
        "class_proposal",
        "match_dedup",
        "concept_grouping",
        "compact_description",
        "document_summarize",
        "classification_audit",
    }
    assert phase1_tasks.issubset(set(PROMPTS))


def test_classification_audit_prompt_shape() -> None:
    """Layer H prompt covers KEEP / RE_HOME / CONVERT_TO_INSTANCE,
    lists the allowed buckets including Person + Organization + Event +
    Role, and embeds the items in the user message."""
    from backend.app.services.prompts import classification_audit

    items = [{"LABEL": "Donald Trump", "CURRENT_PARENT": "Person", "DESCRIPTION": ""}]
    sys_, user = classification_audit(items)
    assert "KEEP" in sys_ and "RE_HOME" in sys_ and "CONVERT_TO_INSTANCE" in sys_
    for bucket in ("Person", "Organization", "Event", "Role", "Infrastructure"):
        assert bucket in sys_, f"bucket {bucket} missing"
    assert "DECISIONS" in sys_
    assert "Donald Trump" in user


def test_class_proposal_prompt_has_entity_type_rules() -> None:
    """Stage 2 prompt explicitly covers the people/org/event rules
    introduced for the misclassification fix."""
    from backend.app.services.prompts import class_identification_and_expansion

    sys_, _user = class_identification_and_expansion({}, "some passage")
    assert "ENTITY-TYPE RULES" in sys_
    assert "foaf:Person" in sys_ or "Person" in sys_
    assert "Organization" in sys_
    assert "Event" in sys_
    assert "Strait of Hormuz crisis" in sys_  # negative example for geo->event


def test_document_summarize_prompt_shape() -> None:
    """The document_summarize prompt builder returns non-empty (system,
    user); system mentions entities + relationships; user contains the
    input text. NOT JSON-mode; the downstream chunker eats raw prose."""
    from backend.app.services.prompts import document_summarize

    sys_, user = document_summarize("Iran exports oil through the Strait of Hormuz.")
    assert isinstance(sys_, str) and sys_
    assert isinstance(user, str) and user
    # The system prompt covers the key things we want preserved.
    sys_lower = sys_.lower()
    assert "entit" in sys_lower      # 'entities' / 'entity'
    assert "relationship" in sys_lower
    assert "do not use bullet" in sys_lower
    # The user message includes the source text.
    assert "Iran exports oil through the Strait of Hormuz." in user


def test_concept_grouping_prompt_mentions_industry_and_domainconcept() -> None:
    """Regression check for the user-reported 'Agriculture under
    Organization' bug. The concept_grouping prompt MUST mention
    Industry and DomainConcept as buckets and explicitly say
    Agriculture should NOT land under Organization."""
    from backend.app.services.prompts import concept_grouping

    sys_, _user = concept_grouping([{"LABEL": "Agriculture", "DESCRIPTION": ""}])
    assert "Industry" in sys_
    assert "DomainConcept" in sys_
    assert "Agriculture" in sys_
    assert "Organization" in sys_
