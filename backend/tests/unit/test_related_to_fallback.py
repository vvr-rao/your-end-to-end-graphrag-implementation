"""graphrag#relatedTo: a stated relationship the ontology has no predicate for
is kept with its phrase instead of discarded; the per-chunk relationship cap
scales with how many entities a chunk holds."""
from __future__ import annotations

from backend.app.services.db_entity_extract import (
    _clean_relation,
    _relationship_cap,
    _types_fit,
)
from backend.app.services.predicates import (
    _BUILTIN_ALLOWLIST,
    GRAPHRAG_RELATED_TO,
)
from backend.app.services.prompts import GENERIC_PREDICATE_IRIS


def test_related_to_is_reserved_and_last_resort() -> None:
    assert GRAPHRAG_RELATED_TO in _BUILTIN_ALLOWLIST
    assert GRAPHRAG_RELATED_TO in GENERIC_PREDICATE_IRIS


def test_related_to_skips_only_the_type_check() -> None:
    pred = {"iri": GRAPHRAG_RELATED_TO, "domain_iris": [], "range_iris": []}
    assert _types_fit(pred, "Person", "City", {})


def test_relation_phrase_is_required_and_short() -> None:
    assert _clean_relation("was born in") == "was born in"
    assert _clean_relation("  alleges fraud against. ") == "alleges fraud against"
    assert _clean_relation(None) is None
    assert _clean_relation("") is None
    assert _clean_relation("related to") is None        # says nothing
    assert _clean_relation("a b c d e f g h i") is None  # a sentence, not a relation


def test_relationship_cap_is_one_per_entity() -> None:
    assert _relationship_cap(4) == 10      # floor
    assert _relationship_cap(19) == 19
    assert _relationship_cap(80) == 80     # a full lineup chunk
    assert _relationship_cap(500) == 100   # token-safety ceiling


def test_extract_prompt_offers_relation_field_and_the_cap() -> None:
    from backend.app.services.prompts import relationship_extract

    ents = [{"canonical_name": "Jim Ratcliffe", "class_label": "Person"},
            {"canonical_name": "Failsworth", "class_label": "City"}]
    preds = [{"iri": GRAPHRAG_RELATED_TO, "label": "relatedTo",
              "domain_label": "Thing", "range_label": "Thing"}]
    system, user = relationship_extract("text", ents, preds, max_relationships=27)
    assert "relation:" in system and "LAST resort" in system
    assert "merely appear together" in system
    assert "At most 27 relationships" in system
    assert "LAST-RESORT PREDICATES" in user
    assert '"relation"' in user
    # Default unchanged for callers that pass no cap.
    assert "At most 10 relationships" in relationship_extract("t", ents, preds)[0]


def test_retrieval_shows_and_ranks_the_phrase_preferring_specific_on_ties() -> None:
    import inspect

    from backend.app.services import retrieval_sql
    from backend.app.services.retrieval import _rank_relationships
    src = inspect.getsource(retrieval_sql.fetch_relationships_among_entities)
    assert "extra_metadata ->> 'relation'" in src

    rels = [
        {"subject": "A", "predicate": "was born in", "object": "B",
         "predicate_iri": GRAPHRAG_RELATED_TO, "support": 1},
        {"subject": "C", "predicate": "bornIn", "object": "D",
         "predicate_iri": "https://x#bornIn", "support": 1},
        {"subject": "E", "predicate": "playsFor", "object": "F",
         "predicate_iri": "https://x#playsFor", "support": 5},
    ]
    ranked = _rank_relationships(rels, "where was he born?")
    # Both "born" edges beat the unrelated one; the typed edge wins the tie.
    assert [r["subject"] for r in ranked] == ["C", "A", "E"]


def test_merged_names_rename_relationship_endpoints_too() -> None:
    """The rename bug: entities were renamed by name collapse but the chunk's
    relationships kept the old spelling and were dropped as unresolved."""
    from backend.app.services.db_entity_extract import _apply_merged_names

    merges = {"Alameda Research": "Alameda Research LLC", "FTX": "FTX Trading Ltd."}
    kept = [{"canonical_name": "Alameda Research"}, {"canonical_name": "FTX"},
            {"canonical_name": "Caroline Ellison"}]
    rels = [{"subject": "Caroline Ellison", "object": "Alameda Research"},
            {"subject": "Alameda Research", "object": "FTX"}]
    results = [("chunk", "iri", "doc", kept, rels), None]

    renamed, endpoints = _apply_merged_names(results, merges.get)

    assert [e["canonical_name"] for e in kept] == [
        "Alameda Research LLC", "FTX Trading Ltd.", "Caroline Ellison"]
    assert rels == [
        {"subject": "Caroline Ellison", "object": "Alameda Research LLC"},
        {"subject": "Alameda Research LLC", "object": "FTX Trading Ltd."}]
    assert endpoints == 3
    assert renamed == {"Alameda Research LLC": {"Alameda Research"},
                       "FTX Trading Ltd.": {"FTX"}}
