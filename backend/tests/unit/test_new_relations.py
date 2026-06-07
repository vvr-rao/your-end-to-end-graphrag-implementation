"""LLM-proposed relations land as new object properties in the ontology.

Covers add_new_relations_from_match_results and _apply_expand wiring:

- DOMAIN/RANGE resolution by IRI (exact key of classes_dict).
- DOMAIN/RANGE resolution by case-insensitive label lookup.
- Resolution against classes proposed in the SAME LLM run (label match
  works because add_new_classes_from_match_not_found runs first).
- Unresolved endpoints land in the `skipped` list with a reason; the
  property is NOT created.
- Re-proposing the same property merges domain/range (no duplicate
  property IRIs).
- _apply_expand returns the new four-tuple shape.
"""

from __future__ import annotations

from backend.app.helpers.ontology_pruning import (
    add_new_relations_from_match_results,
    create_new_object_property_entry,
    make_property_iri,
)
from backend.app.services.pipeline_llm import _apply_expand


BASE_IRI = "http://example.org/ontology/"


def _cls(iri: str, label: str) -> tuple[str, dict]:
    return iri, {
        "iri": iri,
        "name": iri.rsplit("/", 1)[-1],
        "labels": [label],
        "superclasses": [],
        "restrictions_and_class_constructs": [],
    }


def test_make_property_iri_uses_slug() -> None:
    iri = make_property_iri(BASE_IRI, "Treats Effectively")
    assert iri.startswith(BASE_IRI)
    assert iri.lower() == iri  # slug is lowercase
    assert " " not in iri


def test_create_new_object_property_entry_has_canonical_shape() -> None:
    iri, entry = create_new_object_property_entry(
        label="treats",
        description="A Drug treats a Disease.",
        domain_iri="http://example.org/Drug",
        range_iri="http://example.org/Disease",
        new_property_base_iri=BASE_IRI,
    )
    assert entry["property_kind"] == "object_property"
    assert entry["labels"] == ["treats"]
    assert entry["descriptions"] == ["A Drug treats a Disease."]
    assert entry["domain"][0]["iri"] == "http://example.org/Drug"
    assert entry["range"][0]["iri"] == "http://example.org/Disease"
    assert entry["annotations"]["review_status"] == ["proposed"]


def test_resolves_endpoint_by_iri_and_by_label() -> None:
    classes = dict([
        _cls("http://example.org/Drug", "Drug"),
        _cls("http://example.org/Disease", "Disease"),
    ])
    ontology = {
        "classes_dict": classes,
        "object_properties_dict": {},
        "data_properties_dict": {},
        "instances_dict": {},
    }
    results = {
        "MATCH NOT FOUND RELATIONS": [
            # By exact IRI.
            {"LABEL": "treats", "DESCRIPTION": "",
             "DOMAIN": "http://example.org/Drug",
             "RANGE": "http://example.org/Disease"},
            # By case-insensitive label.
            {"LABEL": "is treated by", "DESCRIPTION": "",
             "DOMAIN": "DISEASE", "RANGE": "drug"},
        ]
    }
    extended, created, skipped, _auto_minted = add_new_relations_from_match_results(
        ontology, results, new_property_base_iri=BASE_IRI
    )
    assert len(created) == 2, (created, skipped)
    assert not skipped
    treats = extended["object_properties_dict"][created[0]]
    assert treats["domain"][0]["iri"] == "http://example.org/Drug"
    assert treats["range"][0]["iri"] == "http://example.org/Disease"


def test_resolves_endpoint_against_just_proposed_classes_via_apply_expand() -> None:
    """End-to-end: a relation whose DOMAIN/RANGE are LABELS of classes
    proposed in the SAME run must resolve, because _apply_expand creates
    classes BEFORE creating relations."""
    ontology = {
        "classes_dict": {},
        "object_properties_dict": {},
        "data_properties_dict": {},
        "instances_dict": {},
    }
    results = {
        "MATCHES FOUND": [],
        "MATCH NOT FOUND": [
            {"LABEL": "Drug", "DESCRIPTION": "A pharmaceutical."},
            {"LABEL": "Disease", "DESCRIPTION": "An illness."},
        ],
        "MATCH NOT FOUND RELATIONS": [
            {"LABEL": "treats", "DESCRIPTION": "",
             "DOMAIN": "Drug", "RANGE": "Disease"},
        ],
    }
    extended, created_classes, created_props, skipped = _apply_expand(
        ontology, results, base_iri=BASE_IRI, default_parent_iri=None,
    )
    assert len(created_classes) == 2
    assert len(created_props) == 1
    assert not skipped
    prop = extended["object_properties_dict"][created_props[0]]
    # Both endpoints resolve to the just-created class IRIs.
    domain_iri = prop["domain"][0]["iri"]
    range_iri = prop["range"][0]["iri"]
    assert domain_iri in extended["classes_dict"]
    assert range_iri in extended["classes_dict"]


def test_unresolved_endpoints_get_auto_minted() -> None:
    """Updated behavior: unresolved endpoints get AUTO-MINTED as new
    classes (with `auto_created_from_relation` annotation) rather than
    skipped. The relation always lands."""
    classes = dict([_cls("http://example.org/Drug", "Drug")])
    ontology = {
        "classes_dict": classes,
        "object_properties_dict": {},
        "data_properties_dict": {},
        "instances_dict": {},
    }
    results = {
        "MATCH NOT FOUND RELATIONS": [
            {"LABEL": "treats", "DESCRIPTION": "",
             "DOMAIN": "Drug", "RANGE": "Unobtainium"},
        ]
    }
    extended, created, skipped, auto_minted = add_new_relations_from_match_results(
        ontology, results,
        new_property_base_iri=BASE_IRI,
        new_class_base_iri=BASE_IRI,
    )
    # Relation lands.
    assert len(created) == 1
    # No skips for unresolvable endpoints (only genuine garbage skips now).
    assert not skipped
    # Unobtainium got auto-minted as a class.
    assert len(auto_minted) == 1
    assert any("unobtainium" in iri.lower() for iri in auto_minted)
    # The minted class carries the audit annotation.
    auto_iri = auto_minted[0]
    auto_rec = extended["classes_dict"][auto_iri]
    assert auto_rec["annotations"].get("auto_created_from_relation") == ["treats"]


def test_duplicate_proposed_relation_merges_domain_range() -> None:
    """Two chunks each propose the same relation with overlapping but
    not-identical endpoints. We don't create two properties -- the second
    one extends the first's domain/range lists."""
    classes = dict([
        _cls("http://example.org/Drug", "Drug"),
        _cls("http://example.org/AlternativeTherapy", "AlternativeTherapy"),
        _cls("http://example.org/Disease", "Disease"),
        _cls("http://example.org/Symptom", "Symptom"),
    ])
    ontology = {
        "classes_dict": classes,
        "object_properties_dict": {},
        "data_properties_dict": {},
        "instances_dict": {},
    }
    results = {
        "MATCH NOT FOUND RELATIONS": [
            {"LABEL": "treats", "DESCRIPTION": "",
             "DOMAIN": "Drug", "RANGE": "Disease"},
            {"LABEL": "treats", "DESCRIPTION": "",
             "DOMAIN": "AlternativeTherapy", "RANGE": "Symptom"},
        ]
    }
    extended, created, skipped, _auto_minted = add_new_relations_from_match_results(
        ontology, results, new_property_base_iri=BASE_IRI
    )
    # Only ONE property IRI -- the slug "treats" collapses to the same
    # key. The second occurrence merges new endpoints in.
    assert len(created) == 1
    prop = extended["object_properties_dict"][created[0]]
    domain_iris = {d["iri"] for d in prop["domain"]}
    range_iris = {r["iri"] for r in prop["range"]}
    assert domain_iris == {"http://example.org/Drug", "http://example.org/AlternativeTherapy"}
    assert range_iris == {"http://example.org/Disease", "http://example.org/Symptom"}


def test_missing_relations_section_is_a_noop() -> None:
    """Backward compatibility: callers (or tests) that don't pass
    MATCH NOT FOUND RELATIONS still work."""
    ontology = {
        "classes_dict": {},
        "object_properties_dict": {},
        "data_properties_dict": {},
        "instances_dict": {},
    }
    extended, created, skipped, _auto_minted = add_new_relations_from_match_results(
        ontology, {"MATCH NOT FOUND": []}, new_property_base_iri=BASE_IRI
    )
    assert created == []
    assert skipped == []
    assert extended["object_properties_dict"] == {}
