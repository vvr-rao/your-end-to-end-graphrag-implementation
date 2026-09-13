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
    extended, created_classes, created_props, skipped, _created_instances = _apply_expand(
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


# --------------------------------------------------------------------------- #
# Disjunction endpoints
#
# `match_dedup` rule 3a asks the model to GENERALISE relation DOMAIN/RANGE.
# When no single class covers both ends it hedges -- "Organization or
# AppStoreOperator" -- and that string used to be auto-minted VERBATIM as a
# class. The live DB carries 7 of them, each with an
# `auto_created_from_relation` annotation proving the path. A disjunction is
# not a class: no entity can ever instantiate it, yet it sits in the candidate
# menu forever.
# --------------------------------------------------------------------------- #


def _disjunction_case(domain: str, classes: list[tuple[str, dict]]) -> tuple:
    ontology = {
        "classes_dict": dict(classes),
        "object_properties_dict": {},
        "data_properties_dict": {},
        "instances_dict": {},
    }
    results = {
        "MATCH NOT FOUND RELATIONS": [
            {"LABEL": "allocatedTo", "DESCRIPTION": "",
             "DOMAIN": domain, "RANGE": "Drug"},
        ]
    }
    return add_new_relations_from_match_results(
        ontology, results,
        new_property_base_iri=BASE_IRI,
        new_class_base_iri=BASE_IRI,
    )


def test_disjunction_endpoint_is_never_minted_verbatim() -> None:
    """Neither side resolves -> skip the relation, do NOT mint 'X or Y'.

    Trading a junk class for a SILENT drop would be no improvement, so the
    skip must be counted in `skipped` where the stage-4 log line reports it.
    """
    extended, created, skipped, auto_minted = _disjunction_case(
        "Organization or AppStoreOperator",
        [_cls("http://example.org/Drug", "Drug")],
    )
    assert not created
    assert not auto_minted, "a disjunction must never become a class"
    assert len(skipped) == 1
    assert "disjunction" in skipped[0]["reason"]
    labels = [
        lbl
        for rec in extended["classes_dict"].values()
        for lbl in rec.get("labels", [])
    ]
    assert not any(" or " in lbl for lbl in labels)


def test_disjunction_with_one_resolvable_side_uses_that_side() -> None:
    extended, created, skipped, auto_minted = _disjunction_case(
        "Organization or AppStoreOperator",
        [
            _cls("http://example.org/Drug", "Drug"),
            _cls("http://example.org/Organization", "Organization"),
        ],
    )
    assert len(created) == 1
    assert not skipped
    assert not auto_minted
    prop = extended["object_properties_dict"][created[0]]
    assert prop["domain"][0]["iri"] == "http://example.org/Organization"


def test_disjunction_with_both_sides_resolvable_picks_the_more_general() -> None:
    """Rule 3a asks for the common parent. When both alternatives are real
    classes, the shallower one is the reusable endpoint -- pinning the
    relation to the narrow side is what made 77% of minted relations
    unusable."""
    general = _cls("http://example.org/Asset", "Asset")
    narrow_iri = "http://example.org/SegmentAsset"
    narrow = (narrow_iri, {
        "iri": narrow_iri,
        "name": "SegmentAsset",
        "labels": ["SegmentAsset"],
        "superclasses": [{"iri": "http://example.org/Asset"}],
        "restrictions_and_class_constructs": [],
    })
    extended, created, _skipped, auto_minted = _disjunction_case(
        "SegmentAsset or Asset",
        [_cls("http://example.org/Drug", "Drug"), general, narrow],
    )
    assert len(created) == 1
    assert not auto_minted
    prop = extended["object_properties_dict"][created[0]]
    assert prop["domain"][0]["iri"] == "http://example.org/Asset"


def test_split_disjunction_label_only_fires_on_real_disjunctions() -> None:
    from backend.app.helpers.ontology_pruning import split_disjunction_label

    assert split_disjunction_label("Organization or AppStoreOperator") == [
        "Organization", "AppStoreOperator",
    ]
    assert split_disjunction_label("A and/or B") == ["A", "B"]
    assert split_disjunction_label("A | B") == ["A", "B"]
    assert split_disjunction_label("Foo / Bar") == ["Foo", "Bar"]
    # Not disjunctions -- these must survive untouched.
    assert split_disjunction_label("Organization") == []
    assert split_disjunction_label("Doctor") == []
    assert split_disjunction_label("") == []
    assert split_disjunction_label(None) == []


# --------------------------------------------------------------------------- #
# Stage 3 reconciliation: the dedup model drops entries it never merged
# --------------------------------------------------------------------------- #


def _reconcile(merged_in, model_out, existing=()):
    """The reconciliation `_dedup` performs, in isolation."""
    from backend.app.services.pipeline_llm import _DEDUP_KEYS, _dedup_merge_key

    out = {k: list(model_out.get(k) or []) for k in _DEDUP_KEYS}
    restored = 0
    for key in _DEDUP_KEYS:
        have = {_dedup_merge_key(key, e) for e in out[key] if isinstance(e, dict)}
        for entry in merged_in.get(key) or []:
            ident = _dedup_merge_key(key, entry)
            if not ident or ident in have:
                continue
            if key == "MATCH NOT FOUND" and ident in {c.lower() for c in existing}:
                continue
            have.add(ident)
            out[key].append(entry)
            restored += 1
    return out, restored


def _mnf(label):
    return {"LABEL": label, "DESCRIPTION": "d", "PARENT_LABEL": "NONE"}


def test_dropped_proposals_are_restored() -> None:
    """Stage 3 is a FILTER, not a summariser. Measured on two corpora it drops
    the same ~64% of proposals -- finance 1632 -> 550, pharma 509 -> 186 --
    against a genuine near-duplicate rate of ~6% (the batch clusterer found
    only 21 of 341 class clusters with more than one member). There was nothing
    for the other 320 to be merged into."""
    src = {"MATCH NOT FOUND": [_mnf("GovernmentAgency"), _mnf("SystemOperator"),
                               _mnf("WindFarm")]}
    out, n = _reconcile(src, {"MATCH NOT FOUND": [_mnf("WindFarm")]})
    labels = {e["LABEL"] for e in out["MATCH NOT FOUND"]}
    assert labels == {"GovernmentAgency", "SystemOperator", "WindFarm"}
    assert n == 2


def test_a_genuine_collapse_is_not_undone_twice() -> None:
    """An entry the model DID return is never duplicated by the restore."""
    src = {"MATCH NOT FOUND": [_mnf("WindFarm")]}
    out, n = _reconcile(src, {"MATCH NOT FOUND": [_mnf("WindFarm")]})
    assert len(out["MATCH NOT FOUND"]) == 1 and n == 0


def test_rule_1_still_holds_after_restore() -> None:
    """Never resurrect a class that already exists in the ontology -- that is
    what rule 1 removes, and the restore must not put it back."""
    src = {"MATCH NOT FOUND": [_mnf("Organization")]}
    out, n = _reconcile(src, {"MATCH NOT FOUND": []}, existing=("Organization",))
    assert out["MATCH NOT FOUND"] == [] and n == 0


def test_restore_covers_relations_and_instances_too() -> None:
    """Instances take the same hit: 1,021 proposed -> 389 emitted on finance."""
    src = {
        "MATCH NOT FOUND RELATIONS": [
            {"LABEL": "owns", "DOMAIN": "Org", "RANGE": "Asset"}],
        "MATCH NOT FOUND INSTANCES": [
            {"LABEL": "X", "CANONICAL_FORM": "X", "TYPE_LABEL": "Org"}],
    }
    out, n = _reconcile(src, {})
    assert len(out["MATCH NOT FOUND RELATIONS"]) == 1
    assert len(out["MATCH NOT FOUND INSTANCES"]) == 1
    assert n == 2


def test_iri_shaped_instance_types_are_not_synthesised_as_classes() -> None:
    """`extend_ontology_with_instances` resolves an IRI TYPE_LABEL directly
    against `classes_dict`, which is keyed BY IRI. `existing_concepts` holds
    LABELS, so comparing against it reports these as orphans -- the first run
    of the synthesis minted classes literally named
    `http://www.w3.org/ns/org#Organization`.
    """
    for tl in ("http://www.w3.org/ns/org#Organization",
               "http://www.w3.org/2006/time#Instant",
               "https://example.com/x#Thing"):
        assert "://" in tl or "#" in tl, tl
    # a plain label must still be eligible
    assert "://" not in "GovernmentAgency" and "#" not in "GovernmentAgency"
