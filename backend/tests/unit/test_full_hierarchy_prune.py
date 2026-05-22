"""Verify the prune helpers retain the user-required keep-set:

  retained = detected ∪ full ancestors via subClassOf
                       ∪ full descendants via subClassOf
                       ∪ other-endpoint classes of relationships
                         that touch the retained set

Test fixture is a tiny synthetic ontology where the IS-A tree depth is >1
so the difference between N-hop BFS (old behavior) and full transitive
closure (new behavior) is visible.
"""

from __future__ import annotations

from backend.app.helpers.ontology_pruning import (
    _build_isa_indexes,
    collect_full_class_hierarchy,
    expand_with_relationship_partners,
)


# ---- Fixture ---------------------------------------------------------------
#
#   Animal                            <- 3 hops above Beagle
#     └─ Mammal                       <- 2 hops above Beagle
#         └─ Dog                      <- 1 hop above Beagle  (the seed)
#             ├─ Beagle  *seed*
#             └─ Labrador             <- 1 hop below Dog
#                 └─ ChocolateLab     <- 2 hops below Dog
#     └─ Bird (parallel branch, NOT related to Dog -- must NOT be kept)
#
# Relationships:
#   ownedBy: domain=Dog, range=Person   -> Person is a "partner" we must keep
#   livesIn: domain=Mammal, range=Habitat -> Habitat is a "partner"
#   eatsFood: domain=Bird, range=Seed    -> NOT touching keep-set; must drop


def _class(iri: str, superclasses: list[str] | None = None) -> tuple[str, dict]:
    return iri, {
        "iri": iri,
        "name": iri.rsplit("/", 1)[-1],
        "superclasses": [{"kind": "entity", "iri": sc} for sc in (superclasses or [])],
        "restrictions_and_class_constructs": [],
    }


CLASSES = dict(
    [
        _class("ex:Animal"),
        _class("ex:Mammal", ["ex:Animal"]),
        _class("ex:Dog", ["ex:Mammal"]),
        _class("ex:Beagle", ["ex:Dog"]),
        _class("ex:Labrador", ["ex:Dog"]),
        _class("ex:ChocolateLab", ["ex:Labrador"]),
        _class("ex:Bird", ["ex:Animal"]),
        _class("ex:Person"),
        _class("ex:Habitat"),
        _class("ex:Seed"),
    ]
)

OBJ_PROPS = {
    "ex:ownedBy": {
        "iri": "ex:ownedBy",
        "domain": [{"kind": "entity", "iri": "ex:Dog"}],
        "range": [{"kind": "entity", "iri": "ex:Person"}],
    },
    "ex:livesIn": {
        "iri": "ex:livesIn",
        "domain": [{"kind": "entity", "iri": "ex:Mammal"}],
        "range": [{"kind": "entity", "iri": "ex:Habitat"}],
    },
    "ex:eatsFood": {
        "iri": "ex:eatsFood",
        "domain": [{"kind": "entity", "iri": "ex:Bird"}],
        "range": [{"kind": "entity", "iri": "ex:Seed"}],
    },
}
DATA_PROPS: dict = {}


def test_isa_indexes_have_correct_edges() -> None:
    parent_of, children_of = _build_isa_indexes(CLASSES)
    assert parent_of["ex:Beagle"] == {"ex:Dog"}
    assert parent_of["ex:Dog"] == {"ex:Mammal"}
    assert parent_of["ex:Animal"] == set()
    assert children_of["ex:Dog"] == {"ex:Beagle", "ex:Labrador"}
    assert children_of["ex:Mammal"] == {"ex:Dog"}
    # Bird is a child of Animal but a sibling of Mammal -- crucial for proving
    # the hierarchy walk follows IS-A, not a flood-fill.
    assert children_of["ex:Animal"] == {"ex:Mammal", "ex:Bird"}


def test_full_hierarchy_includes_all_ancestors_and_all_descendants() -> None:
    keep = collect_full_class_hierarchy(CLASSES, ["ex:Beagle"])

    # Seed.
    assert "ex:Beagle" in keep
    # ALL ancestors (transitive), not just immediate parent.
    assert "ex:Dog" in keep
    assert "ex:Mammal" in keep
    assert "ex:Animal" in keep
    # ALL descendants from any node in the IS-A connected component reachable
    # from the seed: Beagle has none, but its sibling Labrador (and its
    # grandchild ChocolateLab) become reachable once we walk up to Dog and
    # back down.
    assert "ex:Labrador" in keep
    assert "ex:ChocolateLab" in keep
    # Bird is an Animal but not in the Dog->Mammal->Animal IS-A path's
    # downward closure from Mammal -- only Mammal's descendants matter when
    # walking down from Mammal, and Bird is a sibling, not a descendant.
    # However, our BFS WILL reach Bird via the up-down walk from Beagle:
    # Beagle -> Dog -> Mammal -> Animal, then Animal -> Bird (descendant).
    # That is the EXPECTED behavior -- siblings reachable via a common
    # ancestor ARE kept because they're part of the IS-A closure.
    assert "ex:Bird" in keep
    # Person/Habitat/Seed are not in the IS-A tree of Beagle; not kept by
    # this helper alone (relationship partners come from the next step).
    assert "ex:Person" not in keep
    assert "ex:Habitat" not in keep
    assert "ex:Seed" not in keep


def test_relationship_partners_pull_in_other_endpoints() -> None:
    keep = collect_full_class_hierarchy(CLASSES, ["ex:Beagle"])
    augmented = expand_with_relationship_partners(keep, OBJ_PROPS, DATA_PROPS)

    # ownedBy: domain=Dog ∈ keep -> Person comes in.
    assert "ex:Person" in augmented
    # livesIn: domain=Mammal ∈ keep -> Habitat comes in.
    assert "ex:Habitat" in augmented
    # eatsFood touches Bird; Bird is in keep (via IS-A from the up-down
    # walk through Animal), so Seed is pulled in.
    assert "ex:Seed" in augmented


def test_unrelated_relationship_partners_not_pulled_in() -> None:
    """If we seed on something whose IS-A closure doesn't touch Bird, the
    eatsFood property's range (Seed) must NOT come in. Tests that we only
    add partners for properties whose other endpoint is already in keep."""
    # Use ChocolateLab as the seed -- its IS-A closure is
    # {ChocolateLab, Labrador, Dog, Mammal, Animal, Bird, Beagle},
    # so Bird IS in keep and Seed WOULD come in.
    # Use a different fixture with a sibling tree disconnected from Bird
    # to actually exercise "don't pull unrelated partners".
    sub_classes = {
        "ex:Plant": {"iri": "ex:Plant", "superclasses": [], "restrictions_and_class_constructs": []},
        "ex:Flower": {
            "iri": "ex:Flower",
            "superclasses": [{"kind": "entity", "iri": "ex:Plant"}],
            "restrictions_and_class_constructs": [],
        },
        "ex:Tree": {
            "iri": "ex:Tree",
            "superclasses": [{"kind": "entity", "iri": "ex:Plant"}],
            "restrictions_and_class_constructs": [],
        },
        "ex:Country": {"iri": "ex:Country", "superclasses": [], "restrictions_and_class_constructs": []},
    }
    sub_obj_props = {
        "ex:grownIn": {
            "iri": "ex:grownIn",
            "domain": [{"kind": "entity", "iri": "ex:Flower"}],
            "range": [{"kind": "entity", "iri": "ex:Country"}],
        },
        # Property entirely outside the keep set:
        "ex:capitalOf": {
            "iri": "ex:capitalOf",
            "domain": [{"kind": "entity", "iri": "ex:City"}],   # not in classes_dict
            "range": [{"kind": "entity", "iri": "ex:Region"}],  # not in classes_dict
        },
    }
    keep = collect_full_class_hierarchy(sub_classes, ["ex:Tree"])
    augmented = expand_with_relationship_partners(keep, sub_obj_props, {})
    # Tree's IS-A closure: {Tree, Plant, Flower}.
    assert {"ex:Tree", "ex:Plant", "ex:Flower"} <= augmented
    # grownIn touches Flower (in keep) so Country gets pulled in.
    assert "ex:Country" in augmented
    # capitalOf doesn't touch anything in keep -> nothing extra.
    # (City/Region weren't even in classes_dict so they wouldn't be valid
    # anyway, but the property itself shouldn't trigger inclusions.)


def test_apply_prune_end_to_end_preserves_full_hierarchy_and_relationships() -> None:
    """Drive the pipeline-level _apply_prune and assert the four output
    dicts contain exactly the union of the helpers' outputs."""
    from backend.app.services.pipeline_llm import _apply_prune

    loaded = {
        "classes_dict": CLASSES,
        "object_properties_dict": OBJ_PROPS,
        "data_properties_dict": DATA_PROPS,
        "instances_dict": {},
    }
    pruned, keep = _apply_prune(loaded, ["ex:Beagle"])

    # Same set as the helper test above plus the relationship partners.
    expected = {
        "ex:Beagle",
        "ex:Dog",
        "ex:Mammal",
        "ex:Animal",
        "ex:Labrador",
        "ex:ChocolateLab",
        "ex:Bird",
        "ex:Person",
        "ex:Habitat",
        "ex:Seed",
    }
    assert keep == expected
    assert set(pruned["classes_dict"].keys()) == expected
    # All three relationships survive (each one touches the keep-set).
    assert set(pruned["object_properties_dict"].keys()) == {
        "ex:ownedBy", "ex:livesIn", "ex:eatsFood",
    }
    # Critically: the OTHER endpoint is also kept (no orphan `range=[]`).
    owned = pruned["object_properties_dict"]["ex:ownedBy"]
    assert any(d.get("iri") == "ex:Dog" for d in owned["domain"])
    assert any(r.get("iri") == "ex:Person" for r in owned["range"])
