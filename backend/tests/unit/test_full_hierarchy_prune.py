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


def test_full_hierarchy_includes_all_ancestors_and_seed_descendants_only() -> None:
    """Seeding on Beagle:
      - keeps the seed.
      - keeps Beagle's ancestors transitively (Dog, Mammal, Animal).
      - keeps Beagle's descendants transitively (none -- Beagle is a leaf).

    Crucially does NOT keep siblings of Beagle (Labrador, ChocolateLab) or
    siblings of ancestors (Bird, sibling of Mammal under Animal). Otherwise
    an ontology with a single common root explodes back to its full size.
    """
    keep = collect_full_class_hierarchy(CLASSES, ["ex:Beagle"])

    # Seed.
    assert "ex:Beagle" in keep
    # ALL ancestors (transitive), not just immediate parent.
    assert "ex:Dog" in keep
    assert "ex:Mammal" in keep
    assert "ex:Animal" in keep

    # Beagle's siblings + their descendants must NOT come along just
    # because they share a parent with Beagle.
    assert "ex:Labrador" not in keep
    assert "ex:ChocolateLab" not in keep
    # Bird shares an ancestor (Animal) with Beagle but is not an
    # ancestor OR descendant of Beagle. Must NOT be kept.
    assert "ex:Bird" not in keep

    # Person/Habitat/Seed are not in the IS-A tree of Beagle.
    assert "ex:Person" not in keep
    assert "ex:Habitat" not in keep
    assert "ex:Seed" not in keep


def test_full_hierarchy_with_internal_seed_includes_descendants() -> None:
    """Seeding on Dog (an internal node) keeps:
      - the seed.
      - all Dog's ancestors (Mammal, Animal).
      - all Dog's descendants (Beagle, Labrador, ChocolateLab).
    But NOT Bird (Mammal sibling)."""
    keep = collect_full_class_hierarchy(CLASSES, ["ex:Dog"])
    assert "ex:Dog" in keep
    assert "ex:Mammal" in keep
    assert "ex:Animal" in keep
    assert "ex:Beagle" in keep
    assert "ex:Labrador" in keep
    assert "ex:ChocolateLab" in keep
    # Sibling of an ancestor must NOT be kept.
    assert "ex:Bird" not in keep


def test_relationship_partners_pull_in_other_endpoints() -> None:
    keep = collect_full_class_hierarchy(CLASSES, ["ex:Beagle"])
    augmented = expand_with_relationship_partners(keep, OBJ_PROPS, DATA_PROPS)

    # ownedBy: domain=Dog ∈ keep -> Person comes in.
    assert "ex:Person" in augmented
    # livesIn: domain=Mammal ∈ keep -> Habitat comes in.
    assert "ex:Habitat" in augmented
    # eatsFood touches Bird only; Bird is NOT in keep (sibling of Mammal
    # under Animal, not on Beagle's IS-A chain). So Seed must NOT come in.
    assert "ex:Seed" not in augmented


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
    # Seed on Flower (NOT Tree) so the relationship endpoint we care
    # about (Flower) is actually in keep. Tree's IS-A closure under the
    # ancestors-and-descendants-of-seed rule is just {Tree, Plant}
    # because Tree is a leaf -- Flower is a SIBLING of Tree, not an
    # ancestor or descendant.
    keep = collect_full_class_hierarchy(sub_classes, ["ex:Flower"])
    augmented = expand_with_relationship_partners(keep, sub_obj_props, {})
    assert {"ex:Flower", "ex:Plant"} <= augmented
    # Tree is a sibling of Flower -- must NOT come in.
    assert "ex:Tree" not in augmented
    # grownIn has domain=Flower (in keep) so Country is a relationship
    # partner.
    assert "ex:Country" in augmented
    # capitalOf doesn't touch anything in keep -> nothing extra.


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

    # Beagle's strict ancestor chain (Dog -> Mammal -> Animal) plus the
    # seed itself plus relationship partners reachable from that chain.
    # ownedBy: domain=Dog (in keep) -> Person added.
    # livesIn: domain=Mammal (in keep) -> Habitat added.
    # eatsFood touches Bird only; Bird is NOT in keep (sibling of Mammal
    # under Animal) so this property and Seed must not appear.
    expected = {
        "ex:Beagle",
        "ex:Dog",
        "ex:Mammal",
        "ex:Animal",
        "ex:Person",
        "ex:Habitat",
    }
    assert keep == expected, sorted(keep)
    assert set(pruned["classes_dict"].keys()) == expected
    # Only relationships whose endpoints stay in the keep-set survive.
    assert set(pruned["object_properties_dict"].keys()) == {"ex:ownedBy", "ex:livesIn"}
    # Critically: the OTHER endpoint is also kept (no orphan `range=[]`).
    owned = pruned["object_properties_dict"]["ex:ownedBy"]
    assert any(d.get("iri") == "ex:Dog" for d in owned["domain"])
    assert any(r.get("iri") == "ex:Person" for r in owned["range"])
