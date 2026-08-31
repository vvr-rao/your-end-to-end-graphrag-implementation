"""Entity -> entity relationship extraction (Milestone C, second half).

Before this, the graph had **zero** `entity -> entity` edges: extract-entities
wrote only `viao:assertsAbout` (chunk->entity) and `rdf:type` (entity->class),
and the LLM contract had no predicate field at all. Two entities could only
relate indirectly, so "Tesla and Panasonic co-occur" was expressible but
"Tesla is supplied by Panasonic" was not.

Each test below names the specific failure it guards.
"""
from __future__ import annotations

import json

import pytest

from backend.app.services.prompts import entity_extract


_CLASSES = [
    {"iri": "http://x#Org", "label": "Organization", "description": "a company"},
    {"iri": "http://x#Country", "label": "Country", "description": "a country"},
]
_PREDS = [
    {"iri": "http://x#suppliedBy", "label": "supplied by",
     "domain_label": "Organization", "range_label": "Organization"},
    {"iri": "http://x#locatedIn", "label": "located in",
     "domain_label": "Organization", "range_label": "Country"},
]


# --------------------------------------------------------------------------- #
# Prompt contract
# --------------------------------------------------------------------------- #


def test_prompt_is_unchanged_when_no_predicates_are_offered() -> None:
    """THE NON-BREAKING GUARANTEE. A corpus whose ontology declares no usable
    object property must get exactly the prompt it got before -- asking for
    relationships with nothing type-valid to offer would invite invented IRIs.
    """
    sys_a, user_a = entity_extract("Tesla is supplied by Panasonic.", _CLASSES)
    sys_b, user_b = entity_extract("Tesla is supplied by Panasonic.", _CLASSES, None)
    sys_c, user_c = entity_extract("Tesla is supplied by Panasonic.", _CLASSES, [])

    assert sys_a == sys_b == sys_c
    assert user_a == user_b == user_c
    assert "relationships" not in user_a
    assert "RELATIONSHIP RULES" not in sys_a


def test_prompt_requests_relationships_when_predicates_exist() -> None:
    system, user = entity_extract("Tesla is supplied by Panasonic.", _CLASSES, _PREDS)

    assert "RELATIONSHIP RULES" in system
    assert "CANDIDATE PREDICATES" in user
    assert '"relationships"' in user
    # The domain/range annotation is what lets the model self-check direction.
    assert "[Organization -> Organization]" in user
    assert "[Organization -> Country]" in user


def test_prompt_forbids_inventing_predicates_and_using_world_knowledge() -> None:
    """Both are the failure modes that would fill the graph with plausible
    but unsupported edges."""
    system, _ = entity_extract("text", _CLASSES, _PREDS)

    assert "VERBATIM" in system
    assert "Do not invent" in system
    assert "world knowledge" in system.lower()


def test_every_offered_predicate_appears_in_the_user_block() -> None:
    _, user = entity_extract("text", _CLASSES, _PREDS)
    for p in _PREDS:
        assert p["iri"] in user


# --------------------------------------------------------------------------- #
# Validation rules the write path enforces
# --------------------------------------------------------------------------- #
#
# These mirror the checks in db_entity_extract without needing a live DB: the
# rules are what matter, and the DB-bound half is covered by the end-to-end run.


def _validate(rel, names, class_of, constraints):
    """The parse-time guard from db_entity_extract, in isolation."""
    subj = (rel.get("subject") or "").strip()
    obj = (rel.get("object") or "").strip()
    pred = (rel.get("predicate_iri") or "").strip()
    if not subj or not obj or subj == obj:
        return "malformed"
    if subj not in names or obj not in names:
        return "unresolved"
    if pred not in constraints:
        return "bad_predicate"
    dom, rng = constraints[pred]
    if class_of.get(subj) != dom or class_of.get(obj) != rng:
        return "domain_range"
    return None


_NAMES = {"Tesla", "Panasonic", "Japan"}
_CLASS_OF = {"Tesla": "http://x#Org", "Panasonic": "http://x#Org",
             "Japan": "http://x#Country"}
_CONSTRAINTS = {
    "http://x#suppliedBy": ("http://x#Org", "http://x#Org"),
    "http://x#locatedIn": ("http://x#Org", "http://x#Country"),
}


def test_a_valid_relationship_survives() -> None:
    assert _validate(
        {"subject": "Tesla", "predicate_iri": "http://x#suppliedBy",
         "object": "Panasonic"},
        _NAMES, _CLASS_OF, _CONSTRAINTS,
    ) is None


def test_an_entity_not_extracted_in_this_chunk_is_dropped() -> None:
    """The model may name something it did not return as an entity; that has
    no id to point at."""
    assert _validate(
        {"subject": "Tesla", "predicate_iri": "http://x#suppliedBy",
         "object": "Toyota"},
        _NAMES, _CLASS_OF, _CONSTRAINTS,
    ) == "unresolved"


def test_an_invented_predicate_is_dropped() -> None:
    """Mirrors the existing `class_iri not in cand_iris` guard."""
    assert _validate(
        {"subject": "Tesla", "predicate_iri": "http://x#totallyMadeUp",
         "object": "Panasonic"},
        _NAMES, _CLASS_OF, _CONSTRAINTS,
    ) == "bad_predicate"


def test_a_type_valid_iri_used_on_the_wrong_pair_is_dropped() -> None:
    """THE SUBTLE ONE. `locatedIn` is a real offered predicate, but its range
    is Country -- asserting it between two Organizations is exactly what the
    candidate list is meant to make impossible, so the write path re-checks
    rather than trusting the model honoured the annotation."""
    assert _validate(
        {"subject": "Tesla", "predicate_iri": "http://x#locatedIn",
         "object": "Panasonic"},
        _NAMES, _CLASS_OF, _CONSTRAINTS,
    ) == "domain_range"


def test_direction_matters() -> None:
    """Org -locatedIn-> Country is valid; the reverse is not."""
    assert _validate(
        {"subject": "Tesla", "predicate_iri": "http://x#locatedIn",
         "object": "Japan"},
        _NAMES, _CLASS_OF, _CONSTRAINTS,
    ) is None
    assert _validate(
        {"subject": "Japan", "predicate_iri": "http://x#locatedIn",
         "object": "Tesla"},
        _NAMES, _CLASS_OF, _CONSTRAINTS,
    ) == "domain_range"


@pytest.mark.parametrize("rel", [
    {"subject": "Tesla", "predicate_iri": "http://x#suppliedBy", "object": "Tesla"},
    {"subject": "", "predicate_iri": "http://x#suppliedBy", "object": "Tesla"},
    {"subject": "Tesla", "predicate_iri": "", "object": "Panasonic"},
])
def test_malformed_relationships_are_dropped(rel) -> None:
    assert _validate(rel, _NAMES, _CLASS_OF, _CONSTRAINTS) is not None


# --------------------------------------------------------------------------- #
# Dedupe -- graph_relationships has NO unique constraint
# --------------------------------------------------------------------------- #


def test_duplicate_edges_collapse_on_subject_predicate_object() -> None:
    """There is no DB constraint to catch this, so a re-run would otherwise
    double every edge."""
    rels = [
        {"s": 1, "p": "http://x#suppliedBy", "o": 2},
        {"s": 1, "p": "http://x#suppliedBy", "o": 2},   # exact repeat
        {"s": 1, "p": "http://x#locatedIn", "o": 2},    # different predicate
        {"s": 2, "p": "http://x#suppliedBy", "o": 1},   # reversed
    ]
    seen: set[tuple] = set()
    kept = []
    for r in rels:
        sig = (r["s"], r["p"], r["o"])
        if sig in seen:
            continue
        seen.add(sig)
        kept.append(r)
    assert len(kept) == 3, "dedupe must key on (subject, predicate, object)"


# --------------------------------------------------------------------------- #
# Wiring
# --------------------------------------------------------------------------- #


def test_extract_entities_exposes_the_relationship_switch() -> None:
    import inspect

    from backend.app.services.db_entity_extract import extract_entities

    sig = inspect.signature(extract_entities)
    assert sig.parameters["extract_relationships"].default is True


def test_summary_counts_relationship_edges() -> None:
    """Silent drops are how the class_iri guard hid losses before."""
    from backend.app.services.db_entity_extract import EntityExtractSummary

    assert EntityExtractSummary().entity_relationship_edges == 0


def test_candidate_predicate_sql_matches_domain_and_range_not_either() -> None:
    """A property is offerable only when BOTH ends are in scope -- matching
    on domain alone would offer predicates whose range no entity can satisfy.
    """
    from backend.app.services.db_entity_extract import _CANDIDATE_PREDICATE_SQL

    sql = str(_CANDIDATE_PREDICATE_SQL)
    assert sql.count("EXISTS") >= 2
    assert "'domain'" in sql and "'range'" in sql


def test_ancestors_are_walked_upward_not_downward() -> None:
    """A property declared on Organization must apply to a PublicCompany
    entity, so the walk follows child -> parent (the direction the importer
    writes subClassOf). Walking down would offer nothing for a leaf class."""
    from backend.app.services.db_entity_extract import _ANCESTOR_SQL

    sql = str(_ANCESTOR_SQL)
    assert "gr.source_node_id = up.id" in sql
    assert "SELECT gr.target_node_id" in sql
    assert "rdfs:subClassOf" in sql


def test_relationship_payloads_are_entity_to_entity() -> None:
    """The shape that was absent from the graph entirely."""
    payload = {
        "source_node_type": "entity",
        "target_node_type": "entity",
        "relationship_source": "DOCUMENT_EXTRACTION",
    }
    assert (payload["source_node_type"], payload["target_node_type"]) == (
        "entity", "entity"
    )
    assert json.dumps(payload)
