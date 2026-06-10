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


def test_apply_prune_protected_iri_prefixes_force_preservation() -> None:
    """Classes whose IRI starts with a configured protected prefix survive
    prune even when no detection or IS-A path would otherwise reach them.

    Fixture extends the standard one with a parallel `viao:` namespace
    that has zero IS-A connection to the Beagle seed. With no protection,
    seeding on Beagle drops every viao class. With the viao prefix
    protected, the same prune keeps every viao class regardless.

    Also asserts a property whose domain/range both live in the protected
    namespace survives -- the standard prune helper already does this for
    free because it only drops a property when BOTH domain and range fall
    outside the keep-set.
    """
    from backend.app.services.pipeline_llm import _apply_prune

    classes = dict(CLASSES)
    classes["viao:InformationSource"] = {
        "iri": "viao:InformationSource",
        "name": "InformationSource",
        "superclasses": [],
        "restrictions_and_class_constructs": [],
    }
    classes["viao:Document"] = {
        "iri": "viao:Document",
        "name": "Document",
        "superclasses": [{"kind": "entity", "iri": "viao:InformationSource"}],
        "restrictions_and_class_constructs": [],
    }
    classes["viao:Chunk"] = {
        "iri": "viao:Chunk",
        "name": "Chunk",
        "superclasses": [{"kind": "entity", "iri": "viao:Document"}],
        "restrictions_and_class_constructs": [],
    }
    obj_props = dict(OBJ_PROPS)
    obj_props["viao:hasChunk"] = {
        "iri": "viao:hasChunk",
        "domain": [{"kind": "entity", "iri": "viao:Document"}],
        "range": [{"kind": "entity", "iri": "viao:Chunk"}],
    }
    loaded = {
        "classes_dict": classes,
        "object_properties_dict": obj_props,
        "data_properties_dict": {},
        "instances_dict": {},
    }

    # 1. Unprotected baseline: seeding on Beagle drops every viao:* class.
    pruned_unprotected, _ = _apply_prune(loaded, ["ex:Beagle"])
    surviving = set(pruned_unprotected["classes_dict"].keys())
    assert "viao:InformationSource" not in surviving
    assert "viao:Document" not in surviving
    assert "viao:Chunk" not in surviving
    assert "viao:hasChunk" not in pruned_unprotected["object_properties_dict"]

    # 2. With viao: protected, every viao class survives the same prune.
    pruned_protected, keep = _apply_prune(
        loaded, ["ex:Beagle"], protected_iri_prefixes=("viao:",)
    )
    surviving = set(pruned_protected["classes_dict"].keys())
    assert "viao:InformationSource" in surviving
    assert "viao:Document" in surviving
    assert "viao:Chunk" in surviving
    # The property whose domain/range live entirely in the protected
    # namespace also survives (both endpoints now in keep-set).
    assert "viao:hasChunk" in pruned_protected["object_properties_dict"]
    # The Beagle/Dog/Mammal/Animal/Person/Habitat keep-set from the
    # unprotected case is unchanged -- protection adds, never removes.
    assert {"ex:Beagle", "ex:Dog", "ex:Mammal", "ex:Animal", "ex:Person", "ex:Habitat"} <= keep


def test_apply_prune_protected_prefixes_with_no_detection_still_runs() -> None:
    """When detected_iris is empty but protected_iri_prefixes is set, the
    function must still execute the prune path (returning only protected
    classes) -- the early-return short-circuit only fires when BOTH inputs
    are empty."""
    from backend.app.services.pipeline_llm import _apply_prune

    classes = {
        "viao:Document": {
            "iri": "viao:Document",
            "name": "Document",
            "superclasses": [],
            "restrictions_and_class_constructs": [],
        },
        "other:Irrelevant": {
            "iri": "other:Irrelevant",
            "name": "Irrelevant",
            "superclasses": [],
            "restrictions_and_class_constructs": [],
        },
    }
    loaded = {
        "classes_dict": classes,
        "object_properties_dict": {},
        "data_properties_dict": {},
        "instances_dict": {},
    }
    pruned, keep = _apply_prune(loaded, [], protected_iri_prefixes=("viao:",))
    assert set(pruned["classes_dict"].keys()) == {"viao:Document"}
    assert "other:Irrelevant" not in pruned["classes_dict"]
    assert keep == {"viao:Document"}


def test_top_level_branches_excludes_owl_thing_and_surfaces_thing_children() -> None:
    """Stage 1's root detection used to treat any class with `owl:Thing`
    as super as non-root, because `owl:Thing` lives in classes_dict
    alongside the user's domain classes (owlready2 loads it that way).
    That caused VIAO / geography / W3C-time roots to disappear from the
    Stage-1 classifier's branch list, since every one of them declares
    `owl:Thing` as super. After the fix, `owl:Thing` is treated as
    outside-the-ontology for containment checks, AND `owl:Thing` itself
    is excluded from the returned root list."""
    from backend.app.services.pipeline_llm import _top_level_branches

    OWL_THING = "http://www.w3.org/2002/07/owl#Thing"
    classes = {
        OWL_THING: {
            "iri": OWL_THING,
            "name": "Thing",
            "labels": [],
            "superclasses": [],
        },
        "viao:InformationSource": {
            "iri": "viao:InformationSource",
            "name": "InformationSource",
            "labels": ["Information Source"],
            "superclasses": [{"kind": "entity", "iri": OWL_THING}],
        },
        "geo:GeographicEntity": {
            "iri": "geo:GeographicEntity",
            "name": "GeographicEntity",
            "labels": ["Geographic Entity"],
            "superclasses": [{"kind": "entity", "iri": OWL_THING}],
        },
        "geo:Country": {
            "iri": "geo:Country",
            "name": "Country",
            "labels": ["Country"],
            # NOT a root -- has a domain parent inside the dict.
            "superclasses": [{"kind": "entity", "iri": "geo:GeographicEntity"}],
        },
        "time:DayOfWeek": {
            "iri": "time:DayOfWeek",
            "name": "DayOfWeek",
            "labels": ["Day of week"],
            "superclasses": [{"kind": "entity", "iri": OWL_THING}],
        },
        # A class with NO superclass declared at all -- should also surface.
        "x:Standalone": {
            "iri": "x:Standalone",
            "name": "Standalone",
            "labels": ["Standalone"],
            "superclasses": [],
        },
    }
    loaded = {
        "classes_dict": classes,
        "object_properties_dict": {},
        "data_properties_dict": {},
        "instances_dict": {},
    }
    roots = _top_level_branches(loaded)
    root_iris = {r["iri"] for r in roots}

    # Domain roots surface even though their super is owl:Thing.
    assert "viao:InformationSource" in root_iris
    assert "geo:GeographicEntity" in root_iris
    assert "time:DayOfWeek" in root_iris
    assert "x:Standalone" in root_iris

    # geo:Country is NOT a root -- it has a genuine domain parent in the dict.
    assert "geo:Country" not in root_iris

    # owl:Thing itself is excluded -- pointless to ask Stage 1 to classify
    # against the universal type.
    assert OWL_THING not in root_iris


# ============================================================================
# Layer C / D / E tests: smarter class placement, fuzzy dedup, auto-mint,
# stem-based relation inference.
# ============================================================================


def _geo_class(local: str, parent_local: str | None = None) -> tuple[str, dict]:
    iri = f"https://veerla-ramrao.ai/ontology/geography#{local}"
    supers = []
    if parent_local:
        supers.append({
            "kind": "entity",
            "iri": f"https://veerla-ramrao.ai/ontology/geography#{parent_local}",
        })
    return iri, {
        "iri": iri,
        "name": local,
        "labels": [local],
        "superclasses": supers,
        "restrictions_and_class_constructs": [],
    }


def test_add_classes_places_landform_in_parent_namespace_when_resolved() -> None:
    """When PARENT_LABEL resolves to a geography class AND the new class
    carries a landform-keyword signal, the new IRI lands in the geography
    namespace under that parent. (Use 'Bering Strait' under 'Strait' --
    'strait' is in landform vocabulary so the safety guard allows it.)

    Note: a class like 'Washington' under 'Country' would now NOT land
    in geography ns -- the guard rejects geographic parents for class
    labels with no landform signal. That's by design to prevent the
    cascade described in the prior session. Layer G concept_grouping
    handles non-landform place names downstream."""
    from backend.app.helpers.ontology_pruning import add_new_classes_from_match_not_found

    classes = dict([
        _geo_class("Strait"),
    ])
    ontology = {
        "classes_dict": classes,
        "object_properties_dict": {},
        "data_properties_dict": {},
        "instances_dict": {},
    }
    results = {
        "MATCH NOT FOUND": [
            {"LABEL": "Bering Strait", "DESCRIPTION": "Strait between Russia and Alaska.",
             "PARENT_LABEL": "Strait"},
        ]
    }
    extended, created = add_new_classes_from_match_not_found(
        ontology, results,
        new_class_base_iri="http://default.example/ontology/",
        default_parent_iri="http://www.w3.org/2002/07/owl#Thing",
    )
    assert len(created) == 1
    bering_iri = created[0]
    # IRI is in the geography namespace.
    assert bering_iri.startswith("https://veerla-ramrao.ai/ontology/geography#")
    assert "default.example" not in bering_iri
    # Parent IRI is geography:Strait.
    record = extended["classes_dict"][bering_iri]
    parent_iri = record["superclasses"][0]["iri"]
    assert parent_iri == "https://veerla-ramrao.ai/ontology/geography#Strait"


def test_add_classes_fuzzy_dedups_token_variants() -> None:
    """'Washington DC' and 'Washington D.C.' tokenize to the same set
    {washington, dc}; the second proposal reuses the first's IRI."""
    from backend.app.helpers.ontology_pruning import add_new_classes_from_match_not_found

    ontology = {
        "classes_dict": dict([_geo_class("Country")]),
        "object_properties_dict": {},
        "data_properties_dict": {},
        "instances_dict": {},
    }
    results = {
        "MATCH NOT FOUND": [
            {"LABEL": "Washington DC", "DESCRIPTION": "Capital.",
             "PARENT_LABEL": "Country"},
            {"LABEL": "Washington D.C.", "DESCRIPTION": "Same place.",
             "PARENT_LABEL": "Country"},
        ]
    }
    extended, created = add_new_classes_from_match_not_found(
        ontology, results,
        new_class_base_iri="http://default.example/ontology/",
        default_parent_iri=None,
    )
    # Exactly ONE new class minted -- the second proposal got deduped in.
    assert len(created) == 1
    survivor = extended["classes_dict"][created[0]]
    # Both descriptions are preserved.
    descs = survivor.get("descriptions", [])
    assert any("Capital." in d for d in descs)
    assert any("Same place." in d for d in descs)


def test_add_classes_fuzzy_dedups_substring_with_same_parent() -> None:
    """Two proposals with overlapping labels and the SAME (post-guard)
    parent collapse into one class. Two proposals where the labels
    overlap but the parents differ (one a landform, one not) do not
    collapse.

    Note: after the geo-parent safety guard, non-landform labels like
    'Washington' have their geography-class PARENT_LABEL stripped
    regardless of what the LLM proposed. Use landform labels ('Bering
    Strait' / 'Bering Strait Channel') under Strait/Channel here to
    keep both proposals in the geography namespace and exercise the
    same-parent dedup path."""
    from backend.app.helpers.ontology_pruning import add_new_classes_from_match_not_found

    classes = dict([
        _geo_class("Strait"),
        _geo_class("Bridge"),
    ])
    ontology = {
        "classes_dict": classes,
        "object_properties_dict": {},
        "data_properties_dict": {},
        "instances_dict": {},
    }
    # Same-parent case: both proposals share landform-keyword + Strait parent.
    results_same_parent = {
        "MATCH NOT FOUND": [
            {"LABEL": "Bering Strait", "DESCRIPTION": "",
             "PARENT_LABEL": "Strait"},
            {"LABEL": "Bering Strait NW", "DESCRIPTION": "",
             "PARENT_LABEL": "Strait"},
        ]
    }
    _, created = add_new_classes_from_match_not_found(
        ontology, results_same_parent,
        new_class_base_iri="http://default.example/ontology/",
        default_parent_iri=None,
    )
    assert len(created) == 1

    # Different-parent case: one strait, one bridge -- expect TWO classes
    # because the resolved parents differ.
    results_diff_parent = {
        "MATCH NOT FOUND": [
            {"LABEL": "Bering Strait", "DESCRIPTION": "",
             "PARENT_LABEL": "Strait"},
            {"LABEL": "Bering Strait Bridge", "DESCRIPTION": "",
             "PARENT_LABEL": "Bridge"},
        ]
    }
    _, created_diff = add_new_classes_from_match_not_found(
        ontology, results_diff_parent,
        new_class_base_iri="http://default.example/ontology/",
        default_parent_iri=None,
    )
    assert len(created_diff) == 2


def test_add_relations_auto_mints_unresolved_endpoints() -> None:
    """Both DOMAIN and RANGE unresolved -> auto-mint both classes, then
    create the relation. Nothing in `skipped`."""
    from backend.app.helpers.ontology_pruning import add_new_relations_from_match_results

    ontology = {
        "classes_dict": {},
        "object_properties_dict": {},
        "data_properties_dict": {},
        "instances_dict": {},
    }
    results = {
        "MATCH NOT FOUND RELATIONS": [
            {"LABEL": "targets", "DESCRIPTION": "A potential move.",
             "DOMAIN": "Washington", "RANGE": "Kharg Island"},
        ]
    }
    extended, created_rels, skipped, auto_minted = add_new_relations_from_match_results(
        ontology, results,
        new_property_base_iri="http://default.example/ontology/",
        new_class_base_iri="http://default.example/ontology/",
        default_parent_iri="http://www.w3.org/2002/07/owl#Thing",
    )
    assert len(created_rels) == 1
    assert len(auto_minted) == 2
    assert not skipped
    # Both endpoints exist as classes.
    rel = extended["object_properties_dict"][created_rels[0]]
    domain_iri = rel["domain"][0]["iri"]
    range_iri = rel["range"][0]["iri"]
    assert domain_iri in extended["classes_dict"]
    assert range_iri in extended["classes_dict"]
    # Both classes carry the audit annotation.
    for iri in auto_minted:
        ann = extended["classes_dict"][iri].get("annotations", {})
        assert ann.get("auto_created_from_relation") == ["targets"]


def test_add_relations_auto_mint_inherits_landform_parent_from_other_endpoint() -> None:
    """When DOMAIN resolves to a class with a non-Thing parent AND the
    auto-minted RANGE's label carries a landform-keyword signal, the
    safety guard allows the inherited geography parent. So 'Bering
    Strait' (with 'strait' in its label) next to `Kharg Island`
    parented at `GeographicEntity` lands in geography namespace.

    Without a landform signal in the auto-minted label (e.g.
    'Washington'), the guard would reject the geography inheritance
    and drop the new class to default ns + owl:Thing -- that path is
    covered by `test_geo_parent_guard_in_relation_automint`."""
    from backend.app.helpers.ontology_pruning import add_new_relations_from_match_results

    geo_iri, geo_rec = _geo_class("GeographicEntity")
    kharg_iri = "https://veerla-ramrao.ai/ontology/geography#KhargIsland"
    classes = {
        geo_iri: geo_rec,
        kharg_iri: {
            "iri": kharg_iri,
            "name": "KhargIsland",
            "labels": ["Kharg Island"],
            "superclasses": [{"kind": "entity", "iri": geo_iri}],
            "restrictions_and_class_constructs": [],
        },
    }
    ontology = {
        "classes_dict": classes,
        "object_properties_dict": {},
        "data_properties_dict": {},
        "instances_dict": {},
    }
    results = {
        "MATCH NOT FOUND RELATIONS": [
            {"LABEL": "borders", "DESCRIPTION": "",
             "DOMAIN": "Bering Strait", "RANGE": "Kharg Island"},
        ]
    }
    extended, _, skipped, auto_minted = add_new_relations_from_match_results(
        ontology, results,
        new_property_base_iri="http://default.example/ontology/",
        new_class_base_iri="http://default.example/ontology/",
        default_parent_iri="http://www.w3.org/2002/07/owl#Thing",
    )
    assert not skipped
    assert len(auto_minted) == 1
    bering_iri = auto_minted[0]
    # Bering Strait carries landform signal, so guard allows the
    # inherited geography parent. Lands in geography ns.
    assert bering_iri.startswith("https://veerla-ramrao.ai/ontology/geography#")
    # Parent is GeographicEntity (inherited from Kharg Island).
    parent = extended["classes_dict"][bering_iri]["superclasses"][0]["iri"]
    assert parent == geo_iri


def test_add_relations_still_skips_garbage_endpoints() -> None:
    """Empty endpoints, sentence-length endpoints (>80 chars), and
    newline-containing endpoints all still skip. Auto-mint isn't a
    free-for-all."""
    from backend.app.helpers.ontology_pruning import add_new_relations_from_match_results

    ontology = {
        "classes_dict": {},
        "object_properties_dict": {},
        "data_properties_dict": {},
        "instances_dict": {},
    }
    results = {
        "MATCH NOT FOUND RELATIONS": [
            {"LABEL": "rel1", "DOMAIN": "", "RANGE": "X"},
            {"LABEL": "rel2", "DOMAIN": "X",
             "RANGE": "platforms and mechanisms that supervise the safety status of electric vehicles in regulated urban environments per chapter 12"},
            {"LABEL": "rel3", "DOMAIN": "good", "RANGE": "has\nnewline"},
        ]
    }
    _, _, skipped, _ = add_new_relations_from_match_results(
        ontology, results,
        new_property_base_iri="http://default.example/ontology/",
        new_class_base_iri="http://default.example/ontology/",
        default_parent_iri=None,
    )
    assert len(skipped) == 3


def test_stem_relation_inference_creates_has_x_properties() -> None:
    """helium / helium_market / helium_supply / helium_price all share
    namespace -> infer `helium has_market helium_market` etc."""
    from backend.app.helpers.ontology_pruning import infer_stem_relations

    ns = "http://example.com/onto/"
    classes = {
        ns + "helium": {"iri": ns + "helium", "name": "helium",
                        "labels": ["helium"], "superclasses": []},
        ns + "helium_market": {"iri": ns + "helium_market", "name": "helium_market",
                                "labels": ["helium_market"], "superclasses": []},
        ns + "helium_supply": {"iri": ns + "helium_supply", "name": "helium_supply",
                                "labels": ["helium_supply"], "superclasses": []},
        ns + "helium_price": {"iri": ns + "helium_price", "name": "helium_price",
                               "labels": ["helium_price"], "superclasses": []},
    }
    obj_props: dict = {}
    _, created = infer_stem_relations(classes, obj_props, new_property_base_iri=ns)
    assert len(created) == 3
    # Find the has_market property.
    market_prop = next(
        (p for p in created if "has_market" in p), None
    )
    assert market_prop is not None
    rec = obj_props[market_prop]
    domain_iris = {d["iri"] for d in rec["domain"]}
    range_iris = {r["iri"] for r in rec["range"]}
    assert ns + "helium" in domain_iris
    assert ns + "helium_market" in range_iris
    assert rec["annotations"]["auto_created_via"] == ["stem_inference"]


def test_stem_relation_inference_skips_when_property_exists() -> None:
    """If `helium has_market helium_market` already exists with the same
    endpoints, the inference pass leaves it alone."""
    from backend.app.helpers.ontology_pruning import infer_stem_relations, make_property_iri

    ns = "http://example.com/onto/"
    classes = {
        ns + "helium": {"iri": ns + "helium", "name": "helium",
                        "labels": ["helium"], "superclasses": []},
        ns + "helium_market": {"iri": ns + "helium_market", "name": "helium_market",
                                "labels": ["helium_market"], "superclasses": []},
    }
    existing_iri = make_property_iri(ns, "has_market")
    obj_props = {
        existing_iri: {
            "iri": existing_iri,
            "name": "has_market",
            "labels": ["has_market"],
            "domain": [{"kind": "entity", "iri": ns + "helium"}],
            "range": [{"kind": "entity", "iri": ns + "helium_market"}],
            "annotations": {"existing": True},
        }
    }
    _, created = infer_stem_relations(classes, obj_props, new_property_base_iri=ns)
    # No new property created (existing one already covers this pair).
    assert created == []
    # Existing annotation untouched.
    assert obj_props[existing_iri]["annotations"] == {"existing": True}


def test_stem_relation_inference_respects_namespace_boundary() -> None:
    """If helium is in geography ns and helium_market is in default ns,
    NO relation is inferred (they belong to different ontologies)."""
    from backend.app.helpers.ontology_pruning import infer_stem_relations

    classes = {
        "http://geo.example/#helium": {
            "iri": "http://geo.example/#helium", "name": "helium",
            "labels": ["helium"], "superclasses": [],
        },
        "http://default.example/onto/helium_market": {
            "iri": "http://default.example/onto/helium_market",
            "name": "helium_market", "labels": ["helium_market"],
            "superclasses": [],
        },
    }
    obj_props: dict = {}
    _, created = infer_stem_relations(classes, obj_props,
                                       new_property_base_iri="http://default.example/onto/")
    assert created == []


# ============================================================================
# Phase 1 tests: time-as-instances feature (MATCH NOT FOUND INSTANCES).
# ============================================================================


def _time_class(local: str) -> tuple[str, dict]:
    iri = f"http://www.w3.org/2006/time#{local}"
    return iri, {
        "iri": iri,
        "name": local,
        "labels": [local],
        "superclasses": [],
        "restrictions_and_class_constructs": [],
    }


def test_add_instances_creates_named_individual_with_type() -> None:
    """Given a Year class and a MATCH NOT FOUND INSTANCES entry with
    CANONICAL_FORM='January 2004' + TYPE_LABEL='Year', the helper mints
    a single named individual typed to Year, placed in the time
    namespace (derived from Year's namespace)."""
    from backend.app.helpers.ontology_pruning import add_new_instances_from_match_results

    year_iri, year_rec = _time_class("Year")
    ontology = {
        "classes_dict": {year_iri: year_rec},
        "object_properties_dict": {},
        "data_properties_dict": {},
        "instances_dict": {},
    }
    results = {
        "MATCH NOT FOUND INSTANCES": [
            {"LABEL": "Jan 2004", "CANONICAL_FORM": "January 2004",
             "TYPE_LABEL": "Year",
             "DESCRIPTION": "Reference month from the corpus."},
        ]
    }
    extended, created = add_new_instances_from_match_results(
        ontology, results,
        new_instance_base_iri="http://default.example/onto/",
        default_type_iri="http://www.w3.org/2002/07/owl#Thing",
    )
    assert len(created) == 1
    iri = created[0]
    # IRI lands in the default namespace (NOT the W3C `time:` namespace).
    # Per Fix A: minting new entities into standard upper ontologies
    # is forbidden -- those namespaces are owned by W3C/FOAF/SKOS.
    # Instances follow the same rule as classes.
    assert iri.startswith("http://default.example/onto/")
    # Slugged from canonical form, NOT the surface form.
    assert iri.endswith("january_2004")
    inst = extended["instances_dict"][iri]
    # Typed to Year on both `types` and `direct_types`.
    assert any(t["iri"] == year_iri for t in inst["types"])
    assert any(t["iri"] == year_iri for t in inst["direct_types"])
    # Both labels preserved.
    assert "January 2004" in inst["labels"]
    assert "Jan 2004" in inst["labels"]
    # Audit annotation.
    assert inst["annotations"]["canonical_form"] == ["January 2004"]


def test_add_instances_dedups_same_canonical_form() -> None:
    """Three entries with the SAME CANONICAL_FORM but different surface
    LABELs produce ONE minted instance. All surface forms get folded
    into the survivor's labels list."""
    from backend.app.helpers.ontology_pruning import add_new_instances_from_match_results

    year_iri, year_rec = _time_class("Year")
    ontology = {
        "classes_dict": {year_iri: year_rec},
        "object_properties_dict": {},
        "data_properties_dict": {},
        "instances_dict": {},
    }
    results = {
        "MATCH NOT FOUND INSTANCES": [
            {"LABEL": "Jan 2004", "CANONICAL_FORM": "January 2004",
             "TYPE_LABEL": "Year", "DESCRIPTION": "First mention."},
            {"LABEL": "Jan04", "CANONICAL_FORM": "January 2004",
             "TYPE_LABEL": "Year", "DESCRIPTION": "Abbrev form."},
            {"LABEL": "January 2004", "CANONICAL_FORM": "January 2004",
             "TYPE_LABEL": "Year", "DESCRIPTION": "Full form."},
        ]
    }
    extended, created = add_new_instances_from_match_results(
        ontology, results,
        new_instance_base_iri="http://default.example/onto/",
        default_type_iri=None,
    )
    assert len(created) == 1
    inst = extended["instances_dict"][created[0]]
    # All three surface forms in labels.
    assert set(inst["labels"]) == {"January 2004", "Jan 2004", "Jan04"}
    # All three descriptions preserved.
    assert len(inst["descriptions"]) == 3


def test_add_instances_falls_back_to_default_type_when_unresolved() -> None:
    """TYPE_LABEL that doesn't resolve to any class -> instance is typed
    at default_type_iri (owl:Thing) and lands in the default namespace."""
    from backend.app.helpers.ontology_pruning import add_new_instances_from_match_results

    ontology = {
        "classes_dict": {},
        "object_properties_dict": {},
        "data_properties_dict": {},
        "instances_dict": {},
    }
    results = {
        "MATCH NOT FOUND INSTANCES": [
            {"LABEL": "Mystery Event", "CANONICAL_FORM": "Mystery Event",
             "TYPE_LABEL": "Nonexistent Class",
             "DESCRIPTION": "Unresolvable type."},
        ]
    }
    OWL_THING = "http://www.w3.org/2002/07/owl#Thing"
    extended, created = add_new_instances_from_match_results(
        ontology, results,
        new_instance_base_iri="http://default.example/onto/",
        default_type_iri=OWL_THING,
    )
    assert len(created) == 1
    iri = created[0]
    # Lands in default namespace (owl:Thing -> fallback).
    assert iri.startswith("http://default.example/onto/")
    inst = extended["instances_dict"][iri]
    assert inst["types"][0]["iri"] == OWL_THING


def test_add_instances_places_in_type_namespace() -> None:
    """End-to-end: type resolves to time:Year, so instance IRI uses
    time#... namespace; type resolves to geography:Country, so instance
    uses geography#... namespace."""
    from backend.app.helpers.ontology_pruning import add_new_instances_from_match_results

    year_iri, year_rec = _time_class("Year")
    country_iri = "https://veerla-ramrao.ai/ontology/geography#Country"
    country_rec = {
        "iri": country_iri, "name": "Country", "labels": ["Country"],
        "superclasses": [],
    }
    ontology = {
        "classes_dict": {year_iri: year_rec, country_iri: country_rec},
        "object_properties_dict": {},
        "data_properties_dict": {},
        "instances_dict": {},
    }
    results = {
        "MATCH NOT FOUND INSTANCES": [
            {"LABEL": "2030", "CANONICAL_FORM": "2030",
             "TYPE_LABEL": "Year", "DESCRIPTION": ""},
            {"LABEL": "Kingdom of Saudi Arabia", "CANONICAL_FORM": "Saudi Arabia",
             "TYPE_LABEL": "Country", "DESCRIPTION": ""},
        ]
    }
    extended, created = add_new_instances_from_match_results(
        ontology, results,
        new_instance_base_iri="http://default.example/onto/",
        default_type_iri=None,
    )
    assert len(created) == 2
    # Year is in the W3C time namespace (a standard upper ontology),
    # so per Fix A its instances land in default. Country lives in the
    # user's own geography ontology, so its instances DO inherit.
    year_inst = next(i for i in created if "year" in i.lower() or "2030" in i)
    country_inst = next(i for i in created if "saudi" in i.lower())
    assert year_inst.startswith("http://default.example/onto/")  # NOT time:
    assert country_inst.startswith("https://veerla-ramrao.ai/ontology/geography#")
    # Slugs are from canonical forms.
    assert year_inst.endswith("2030")
    assert country_inst.endswith("saudi_arabia")


# ============================================================================
# Retry-after parser tests
# ============================================================================


def test_parse_retry_after_seconds_handles_ms_suffix() -> None:
    """Groq Dev-tier TPM bursts often surface '207.2ms' / '61ms' hints.
    Our parser was previously matching only second-suffix patterns and
    silently dropping ms-suffix retries -- now it converts to seconds."""
    from backend.app.services.pipeline_llm import _parse_retry_after_seconds

    class _Err(Exception):
        pass

    e1 = _Err("rate_limit_exceeded ... Please try again in 207.2ms.")
    assert abs(_parse_retry_after_seconds(e1) - 0.2072) < 1e-6

    e2 = _Err("Error code: 429 ... try again in 7.5s.")
    assert _parse_retry_after_seconds(e2) == 7.5

    e3 = _Err("rate_limit ... try again in 61ms.")
    assert abs(_parse_retry_after_seconds(e3) - 0.061) < 1e-6

    # Non-rate-limit error -> None.
    assert _parse_retry_after_seconds(_Err("connection refused")) is None


# ============================================================================
# Layer F: deterministic geographic-entity inference + IRI rewriting.
# ============================================================================


GEO_NS = "https://veerla-ramrao.ai/ontology/geography#"
DEFAULT_NS = "http://default.example/ontology/"
OWL_THING = "http://www.w3.org/2002/07/owl#Thing"


def _make_default_class(local: str, parent_iri: str | None = None,
                         labels: list[str] | None = None) -> tuple[str, dict]:
    iri = DEFAULT_NS + local
    supers = []
    if parent_iri:
        supers.append({"kind": "entity", "iri": parent_iri})
    return iri, {
        "iri": iri,
        "name": local,
        "labels": labels or [local],
        "superclasses": supers,
        "restrictions_and_class_constructs": [],
    }


def _make_geo_class(local: str, parent_iri: str | None = None,
                     labels: list[str] | None = None) -> tuple[str, dict]:
    iri = GEO_NS + local
    supers = []
    if parent_iri:
        supers.append({"kind": "entity", "iri": parent_iri})
    return iri, {
        "iri": iri,
        "name": local,
        "labels": labels or [local],
        "superclasses": supers,
        "restrictions_and_class_constructs": [],
    }


def test_geo_inference_landform_keyword_in_local_name_rehomes() -> None:
    """Class 'strait_of_hormuz' in default namespace with parent owl:Thing.
    The 'strait' token matches an existing Strait class in geography ns.
    The class IS re-homed into geography ns with Strait as parent."""
    from backend.app.helpers.ontology_pruning import infer_geographic_placement

    strait_iri, strait_rec = _make_geo_class("Strait", labels=["Strait"])
    hormuz_iri, hormuz_rec = _make_default_class(
        "strait_of_hormuz", parent_iri=OWL_THING,
        labels=["Strait of Hormuz"],
    )
    classes_dict = {strait_iri: strait_rec, hormuz_iri: hormuz_rec}

    audit = infer_geographic_placement(
        classes_dict=classes_dict,
        obj_props_dict={},
        data_props_dict={},
        instances_dict={},
    )

    assert len(audit) == 1
    rec = audit[0]
    new_iri = rec["new_iri"]
    # IRI moved into geography namespace.
    assert new_iri.startswith(GEO_NS)
    assert new_iri.endswith("strait_of_hormuz")
    # Parent is Strait.
    assert classes_dict[new_iri]["superclasses"][0]["iri"] == strait_iri
    # Old IRI removed.
    assert hormuz_iri not in classes_dict
    # Audit signal includes the keyword.
    assert "strait" in rec["signal"]


def test_geo_inference_falls_back_to_generic_geo_root_when_no_specific_match() -> None:
    """If the landform keyword has no specific class (e.g. no Island class
    exists), but a generic geographic root (GeographicEntity) exists, fall
    back to it."""
    from backend.app.helpers.ontology_pruning import infer_geographic_placement

    geo_root_iri, geo_root_rec = _make_geo_class(
        "GeographicEntity", labels=["GeographicEntity"]
    )
    kharg_iri, kharg_rec = _make_default_class(
        "kharg_island", parent_iri=OWL_THING, labels=["Kharg Island"],
    )
    classes_dict = {geo_root_iri: geo_root_rec, kharg_iri: kharg_rec}

    audit = infer_geographic_placement(
        classes_dict=classes_dict,
        obj_props_dict={}, data_props_dict={}, instances_dict={},
    )

    assert len(audit) == 1
    new_iri = audit[0]["new_iri"]
    assert new_iri.startswith(GEO_NS)
    assert classes_dict[new_iri]["superclasses"][0]["iri"] == geo_root_iri


def test_geo_inference_skips_non_landform_classes() -> None:
    """A class like 'us_iran_conflict' has no landform keyword in its
    local name or labels -- it must NOT be re-homed even though there
    are geography classes available."""
    from backend.app.helpers.ontology_pruning import infer_geographic_placement

    geo_root_iri, geo_root_rec = _make_geo_class("GeographicEntity")
    conflict_iri, conflict_rec = _make_default_class(
        "us_iran_conflict", parent_iri=OWL_THING,
        labels=["US-Iran Conflict"],
    )
    classes_dict = {geo_root_iri: geo_root_rec, conflict_iri: conflict_rec}

    audit = infer_geographic_placement(
        classes_dict=classes_dict,
        obj_props_dict={}, data_props_dict={}, instances_dict={},
    )

    assert audit == []
    # conflict class unchanged in its original namespace.
    assert conflict_iri in classes_dict
    assert classes_dict[conflict_iri]["superclasses"][0]["iri"] == OWL_THING


def test_geo_inference_does_not_use_is_part_of_relation_signal() -> None:
    """Regression / negative test for the Issue-1 fix.

    Prior behavior: a non-landform class like 'helium' would get re-homed
    into the geography namespace if it was the DOMAIN of an 'is_part_of'
    (or 'within' / 'partof' / ...) relation whose RANGE was in geography
    -- because the LLM uses those predicates non-spatially ('Helium
    is_part_of Qatar's gas output') and the cascading false positives
    swept in dozens of non-geographic classes (Logic Chips, Foundries,
    Memory Components, Electronics Supply Chain, etc.).

    Current behavior: the relation-context signal has been removed
    entirely. Only the landform-keyword signal remains. So 'helium'
    (no landform keyword) must STAY where it is, even when the relation
    points to a class in geography ns.
    """
    from backend.app.helpers.ontology_pruning import infer_geographic_placement

    qatar_iri, qatar_rec = _make_geo_class("Qatar", labels=["Qatar"])
    helium_iri, helium_rec = _make_default_class(
        "helium", parent_iri=OWL_THING, labels=["Helium"],
    )
    classes_dict = {qatar_iri: qatar_rec, helium_iri: helium_rec}
    obj_props = {
        "ex:isPartOf": {
            "iri": "ex:isPartOf",
            "name": "is_part_of",
            "labels": ["is_part_of"],
            "domain": [{"kind": "entity", "iri": helium_iri}],
            "range": [{"kind": "entity", "iri": qatar_iri}],
        }
    }
    audit = infer_geographic_placement(
        classes_dict=classes_dict,
        obj_props_dict=obj_props,
        data_props_dict={}, instances_dict={},
    )
    # Helium is NOT re-homed -- no landform keyword and no Signal 2 path.
    assert audit == []
    assert helium_iri in classes_dict
    assert classes_dict[helium_iri]["superclasses"][0]["iri"] == OWL_THING


def test_geo_inference_rewrites_iri_references_in_relations_and_instances() -> None:
    """When a class's IRI moves from default to geography ns, every
    reference to the old IRI in obj props, data props, and instances must
    be rewritten so the graph stays connected."""
    from backend.app.helpers.ontology_pruning import infer_geographic_placement

    strait_iri, strait_rec = _make_geo_class("Strait")
    hormuz_iri, hormuz_rec = _make_default_class(
        "strait_of_hormuz", parent_iri=OWL_THING, labels=["Strait of Hormuz"],
    )
    other_iri, other_rec = _make_default_class("oil_export_hub", parent_iri=OWL_THING)
    classes_dict = {
        strait_iri: strait_rec,
        hormuz_iri: hormuz_rec,
        other_iri: other_rec,
    }
    obj_props = {
        "ex:disrupts": {
            "iri": "ex:disrupts", "name": "disrupts", "labels": ["disrupts"],
            "domain": [{"kind": "entity", "iri": hormuz_iri}],
            "range": [{"kind": "entity", "iri": other_iri}],
        }
    }
    instances = {
        "ex:closure2025": {
            "iri": "ex:closure2025", "name": "closure_2025", "labels": ["closure 2025"],
            "types": [{"kind": "entity", "iri": hormuz_iri}],
            "direct_types": [{"kind": "entity", "iri": hormuz_iri}],
        }
    }

    audit = infer_geographic_placement(
        classes_dict=classes_dict,
        obj_props_dict=obj_props,
        data_props_dict={},
        instances_dict=instances,
    )
    assert len(audit) == 1
    new_iri = audit[0]["new_iri"]
    assert new_iri.startswith(GEO_NS)

    # Relation's domain now references the NEW IRI.
    assert obj_props["ex:disrupts"]["domain"][0]["iri"] == new_iri
    # Instance's types reference the NEW IRI on both lists.
    assert instances["ex:closure2025"]["types"][0]["iri"] == new_iri
    assert instances["ex:closure2025"]["direct_types"][0]["iri"] == new_iri
    # Other class unaffected.
    assert other_iri in classes_dict


def test_geo_inference_does_not_disturb_classes_already_in_real_namespace() -> None:
    """A class that's already in geography or another real ontology
    namespace is never touched, even if it matches a keyword."""
    from backend.app.helpers.ontology_pruning import infer_geographic_placement

    strait_iri, strait_rec = _make_geo_class("Strait")
    # This one is already in geography ns -- should be left alone.
    bering_iri, bering_rec = _make_geo_class(
        "bering_strait", parent_iri=strait_iri, labels=["Bering Strait"],
    )
    classes_dict = {strait_iri: strait_rec, bering_iri: bering_rec}

    audit = infer_geographic_placement(
        classes_dict=classes_dict,
        obj_props_dict={}, data_props_dict={}, instances_dict={},
    )
    # No re-homing -- bering_strait is already where it belongs.
    assert audit == []
    assert bering_iri in classes_dict


# ============================================================================
# Layer G: top-level concept grouping (LLM-driven proposals applied
# deterministically). LLM-side is mocked; these tests assert the apply
# helper's behavior on synthetic LLM responses.
# ============================================================================


_CG_DEFAULT_BASE = "http://default.example/ontology/"


def _make_orphan(local: str, label: str = None,
                  description: str = "") -> tuple[str, dict]:
    iri = _CG_DEFAULT_BASE + local
    return iri, {
        "iri": iri,
        "name": local,
        "labels": [label or local],
        "descriptions": [description] if description else [],
        "superclasses": [{"kind": "entity", "iri": OWL_THING}],
        "restrictions_and_class_constructs": [],
    }


def test_collect_orphan_classes_picks_default_ns_thing_parented() -> None:
    """Only classes whose IRI lives under default_base_iri AND whose
    first parent is owl:Thing show up as orphans."""
    from backend.app.helpers.ontology_pruning import _collect_orphan_classes

    helium_iri, helium_rec = _make_orphan(
        "helium", label="Helium", description="A noble gas."
    )
    chips_iri, chips_rec = _make_orphan("logic_chips", label="Logic Chips")
    classes = {helium_iri: helium_rec, chips_iri: chips_rec}
    orphans = _collect_orphan_classes(classes, _CG_DEFAULT_BASE)
    labels = {o["label"] for o in orphans}
    assert {"Helium", "Logic Chips"} <= labels
    helium = next(o for o in orphans if o["label"] == "Helium")
    assert helium["description"] == "A noble gas."
    assert helium["iri"] == helium_iri


def test_collect_orphan_classes_skips_real_namespace() -> None:
    """A class in a 'real' ontology namespace (geography, time, OntoCAPE,
    etc.) is NOT an orphan, even if parented at owl:Thing."""
    from backend.app.helpers.ontology_pruning import _collect_orphan_classes

    helium_iri, helium_rec = _make_orphan("helium", label="Helium")
    # Class in geography ns at owl:Thing -- still NOT an orphan.
    foreign_iri = "https://veerla-ramrao.ai/ontology/geography#Country"
    foreign_rec = {
        "iri": foreign_iri,
        "name": "Country",
        "labels": ["Country"],
        "superclasses": [{"kind": "entity", "iri": OWL_THING}],
    }
    classes = {helium_iri: helium_rec, foreign_iri: foreign_rec}
    orphans = _collect_orphan_classes(classes, _CG_DEFAULT_BASE)
    labels = {o["label"] for o in orphans}
    assert "Helium" in labels
    assert "Country" not in labels


def test_collect_orphan_classes_skips_non_thing_parent() -> None:
    """Default-namespace class whose parent is something OTHER than
    owl:Thing is NOT an orphan."""
    from backend.app.helpers.ontology_pruning import _collect_orphan_classes

    material_iri, material_rec = _make_orphan("Material")
    helium_iri = _CG_DEFAULT_BASE + "helium"
    helium_rec = {
        "iri": helium_iri, "name": "helium", "labels": ["Helium"],
        "superclasses": [{"kind": "entity", "iri": material_iri}],
    }
    classes = {material_iri: material_rec, helium_iri: helium_rec}
    orphans = _collect_orphan_classes(classes, _CG_DEFAULT_BASE)
    labels = {o["label"] for o in orphans}
    assert "Helium" not in labels
    # Material itself is still an orphan (owl:Thing parent).
    assert "Material" in labels


def test_apply_concept_grouping_mints_concepts_and_reparents() -> None:
    """Given a Helium orphan and an LLM result proposing Material + an
    assignment of Helium->Material, the helper mints `Material` (with the
    `auto_created_via` annotation) and re-parents Helium under Material."""
    from backend.app.helpers.ontology_pruning import apply_concept_grouping

    helium_iri, helium_rec = _make_orphan("helium", label="Helium")
    classes = {helium_iri: helium_rec}
    llm_result = {
        "TOP_LEVEL_CONCEPTS": [
            {"LABEL": "Material", "DESCRIPTION": "A physical substance."},
        ],
        "ASSIGNMENTS": [
            {"CLASS_LABEL": "Helium", "CONCEPT_LABEL": "Material"},
        ],
    }
    concept_iris, audit = apply_concept_grouping(
        classes_dict=classes,
        default_base_iri=_CG_DEFAULT_BASE,
        llm_result=llm_result,
        default_parent_iri=OWL_THING,
    )
    assert len(concept_iris) == 1
    material_iri = concept_iris[0]
    assert material_iri.startswith(_CG_DEFAULT_BASE)
    # Material is in classes_dict with the audit annotation.
    material_rec = classes[material_iri]
    assert material_rec["annotations"]["auto_created_via"] == ["concept_grouping"]
    # Helium is re-parented under Material.
    assert classes[helium_iri]["superclasses"][0]["iri"] == material_iri
    # Helium's audit annotation records the concept.
    assert classes[helium_iri]["annotations"]["inferred_concept_grouping"] == ["Material"]
    assert len(audit) == 1


def test_apply_concept_grouping_reuses_existing_concept() -> None:
    """If Material already exists as a class (label match), the helper
    doesn't double-mint -- it just re-parents the orphan to the existing
    Material IRI."""
    from backend.app.helpers.ontology_pruning import apply_concept_grouping

    existing_material_iri = _CG_DEFAULT_BASE + "Material"
    existing_material_rec = {
        "iri": existing_material_iri, "name": "Material",
        "labels": ["Material"], "superclasses": [{"kind": "entity", "iri": OWL_THING}],
    }
    helium_iri, helium_rec = _make_orphan("helium", label="Helium")
    classes = {existing_material_iri: existing_material_rec, helium_iri: helium_rec}

    llm_result = {
        "TOP_LEVEL_CONCEPTS": [
            {"LABEL": "Material", "DESCRIPTION": "Same as existing."},
        ],
        "ASSIGNMENTS": [
            {"CLASS_LABEL": "Helium", "CONCEPT_LABEL": "Material"},
        ],
    }
    concept_iris, audit = apply_concept_grouping(
        classes_dict=classes,
        default_base_iri=_CG_DEFAULT_BASE,
        llm_result=llm_result,
        default_parent_iri=OWL_THING,
    )
    # No new IRI minted -- the existing Material was reused.
    assert concept_iris == []
    # Helium parented under the pre-existing Material.
    assert classes[helium_iri]["superclasses"][0]["iri"] == existing_material_iri


def test_apply_concept_grouping_skips_unresolvable_assignment() -> None:
    """LLM proposes an assignment for an unknown class label. The helper
    skips it cleanly with no error / no mutation for that class."""
    from backend.app.helpers.ontology_pruning import apply_concept_grouping

    helium_iri, helium_rec = _make_orphan("helium", label="Helium")
    classes = {helium_iri: helium_rec}
    llm_result = {
        "TOP_LEVEL_CONCEPTS": [
            {"LABEL": "Material", "DESCRIPTION": ""},
        ],
        "ASSIGNMENTS": [
            {"CLASS_LABEL": "NonexistentClass", "CONCEPT_LABEL": "Material"},
            {"CLASS_LABEL": "Helium", "CONCEPT_LABEL": "Material"},
        ],
    }
    concept_iris, audit = apply_concept_grouping(
        classes_dict=classes,
        default_base_iri=_CG_DEFAULT_BASE,
        llm_result=llm_result,
        default_parent_iri=OWL_THING,
    )
    # Material got minted; Helium got re-parented; NonexistentClass skipped.
    assert len(concept_iris) == 1
    assert len(audit) == 1
    assert audit[0]["concept_label"] == "Material"


def test_apply_concept_grouping_handles_empty_llm_result() -> None:
    """An empty LLM result is a no-op (purely additive failure mode)."""
    from backend.app.helpers.ontology_pruning import apply_concept_grouping

    helium_iri, helium_rec = _make_orphan("helium", label="Helium")
    classes = {helium_iri: helium_rec}
    concept_iris, audit = apply_concept_grouping(
        classes_dict=classes,
        default_base_iri=_CG_DEFAULT_BASE,
        llm_result={"TOP_LEVEL_CONCEPTS": [], "ASSIGNMENTS": []},
        default_parent_iri=OWL_THING,
    )
    assert concept_iris == []
    assert audit == []
    # Helium's parent unchanged.
    assert classes[helium_iri]["superclasses"][0]["iri"] == OWL_THING


def test_apply_concept_grouping_concept_class_carries_audit_annotation() -> None:
    """Newly-minted concept class records `auto_created_via:
    ['concept_grouping']` in annotations -- so post-run audit can
    distinguish concept classes from regular Stage-2 proposals."""
    from backend.app.helpers.ontology_pruning import apply_concept_grouping

    helium_iri, helium_rec = _make_orphan("helium")
    classes = {helium_iri: helium_rec}
    llm_result = {
        "TOP_LEVEL_CONCEPTS": [
            {"LABEL": "Material", "DESCRIPTION": "A physical substance."},
        ],
        "ASSIGNMENTS": [
            {"CLASS_LABEL": "Helium", "CONCEPT_LABEL": "Material"},
        ],
    }
    concept_iris, _ = apply_concept_grouping(
        classes_dict=classes,
        default_base_iri=_CG_DEFAULT_BASE,
        llm_result=llm_result,
        default_parent_iri=OWL_THING,
    )
    rec = classes[concept_iris[0]]
    assert "auto_created_via" in rec["annotations"]
    assert rec["annotations"]["auto_created_via"] == ["concept_grouping"]
    # Description was carried into the concept class.
    assert "A physical substance." in (rec.get("descriptions") or [])


# ============================================================================
# compact_description (one-time class-metadata compression) tests.
# ============================================================================


def test_summarize_class_descriptions_writes_compact_description_field() -> None:
    """`summarize_class_descriptions_async` should call the LLM (mocked
    here) and write a `compact_description` field on each candidate
    class. Classes that already have one are skipped (idempotent)."""
    import asyncio
    from backend.app.services.pipeline_llm import (
        _has_useful_text,
        summarize_class_descriptions_async,
    )

    # Sanity check on the helper used by the orchestrator.
    assert _has_useful_text({"descriptions": ["A long description."]})
    assert not _has_useful_text({"descriptions": [], "comments": []})
    assert not _has_useful_text({"descriptions": ["x"]})  # too short

    classes_dict = {
        "ex:Helium": {
            "iri": "ex:Helium", "name": "Helium", "labels": ["Helium"],
            "descriptions": ["A noble gas used as a coolant and lifting gas."],
        },
        "ex:Material": {
            "iri": "ex:Material", "name": "Material", "labels": ["Material"],
            "descriptions": ["Generic substance class for chemical and physical materials."],
        },
        "ex:Trivial": {
            "iri": "ex:Trivial", "name": "Trivial", "labels": ["Trivial"],
            "descriptions": [],  # nothing to summarize -> skipped
        },
        "ex:AlreadyDone": {
            "iri": "ex:AlreadyDone", "name": "AlreadyDone",
            "labels": ["AlreadyDone"],
            "descriptions": ["Some long description text here."],
            "compact_description": "Already summarised.",  # skipped
        },
    }

    # Build a fake LLM router that returns a deterministic JSON response
    # containing compact_description for each iri in the batch.
    class _FakeChatResult:
        def __init__(self, text: str) -> None:
            self.text = text
            self.prompt_tokens = 100
            self.completion_tokens = 50
            self.cost_usd = 0.0001

    class _FakeRouter:
        def __init__(self) -> None:
            self.calls: list[list[dict]] = []
            self.total_cost_usd = 0.0

        async def chat(self, task: str, *, system: str, user: str):
            assert task == "compact_description"
            # Parse the batch out of the user message (JSON after CLASSES:).
            import json as _json
            payload = _json.loads(user.split("CLASSES:\n", 1)[1].split("\n\nReturn JSON:", 1)[0])
            self.calls.append(payload)
            results = [
                {"iri": c["iri"], "compact_description": f"Compact for {c['name']}"}
                for c in payload
            ]
            return _FakeChatResult(_json.dumps({"results": results}))

    router = _FakeRouter()
    summary = asyncio.run(
        summarize_class_descriptions_async(
            classes_dict=classes_dict,
            router=router,
            max_cost_usd=1.0,
            batch_size=10,
            concurrency=2,
        )
    )

    # ex:Helium and ex:Material were summarized.
    assert classes_dict["ex:Helium"]["compact_description"] == "Compact for Helium"
    assert classes_dict["ex:Material"]["compact_description"] == "Compact for Material"
    # ex:Trivial got nothing (skipped on _has_useful_text).
    assert "compact_description" not in classes_dict["ex:Trivial"]
    # ex:AlreadyDone untouched.
    assert classes_dict["ex:AlreadyDone"]["compact_description"] == "Already summarised."
    # Summary record matches.
    assert summary["classes_summarized"] == 2
    assert summary["classes_total"] == 4
    assert summary["llm_calls"] == 1


def test_slice_ontology_uses_compact_description_when_present() -> None:
    """When a class has `compact_description`, `_slice_ontology` ships
    it INSTEAD of the verbose descriptions+comments. Without it, falls
    back to the originals (backward-compatible)."""
    from backend.app.services.pipeline_llm import _slice_ontology

    classes = {
        "ex:A": {
            "iri": "ex:A", "name": "A", "labels": ["A"],
            "descriptions": ["A long description we want to compress."],
            "comments": ["Likewise a long comment."],
            "compact_description": "Short A.",
            "superclasses": [],
        },
        "ex:B": {
            "iri": "ex:B", "name": "B", "labels": ["B"],
            "descriptions": ["Verbose for B."],
            "comments": ["Comment for B."],
            # no compact_description -- fallback path
            "superclasses": [],
        },
    }
    loaded = {"classes_dict": classes}
    sliced = _slice_ontology(loaded, ["ex:A", "ex:B"], max_hops=1)

    # ex:A ships compact_description and NOT the verbose fields.
    assert sliced["ex:A"]["compact_description"] == "Short A."
    assert "descriptions" not in sliced["ex:A"]
    assert "comments" not in sliced["ex:A"]

    # ex:B falls back to the verbose fields (no compact_description).
    assert sliced["ex:B"]["descriptions"] == ["Verbose for B."]
    assert sliced["ex:B"]["comments"] == ["Comment for B."]
    assert "compact_description" not in sliced["ex:B"]


# ============================================================================
# Geographic-parent safety guard (`_is_safe_geo_parent_for`).
# Prevents non-geographic classes from cascading into the geography
# namespace via Stage 2's PARENT_LABEL or Layer D's auto-mint inheritance.
# ============================================================================


def test_geo_parent_guard_filters_non_landform_class() -> None:
    """LLM proposes a non-geographic class (EV bus) with PARENT_LABEL
    pointing at a geography class (GeographicEntity). The deterministic
    guard rejects the parent assignment -- EV bus lands in default ns
    parented at owl:Thing instead of in the geography ns."""
    from backend.app.helpers.ontology_pruning import add_new_classes_from_match_not_found

    geo_iri = "https://veerla-ramrao.ai/ontology/geography#GeographicEntity"
    geo_rec = {
        "iri": geo_iri, "name": "GeographicEntity",
        "labels": ["GeographicEntity"], "superclasses": [],
    }
    ontology = {
        "classes_dict": {geo_iri: geo_rec},
        "object_properties_dict": {},
        "data_properties_dict": {},
        "instances_dict": {},
    }
    results = {
        "MATCH NOT FOUND": [
            {"LABEL": "EV bus", "DESCRIPTION": "An electric bus.",
             "PARENT_LABEL": "GeographicEntity"},
        ]
    }
    extended, created = add_new_classes_from_match_not_found(
        ontology, results,
        new_class_base_iri="http://default.example/ontology/",
        default_parent_iri=OWL_THING,
    )
    assert len(created) == 1
    ev_bus_iri = created[0]
    # IRI is in DEFAULT namespace -- guard rejected the geography parent.
    assert ev_bus_iri.startswith("http://default.example/ontology/")
    assert "geography" not in ev_bus_iri.lower()
    # Parent is owl:Thing, not GeographicEntity.
    rec = extended["classes_dict"][ev_bus_iri]
    parent = rec["superclasses"][0]["iri"]
    assert parent == OWL_THING


def test_geo_parent_guard_allows_landform_class_under_geo_parent() -> None:
    """LLM proposes a landform class (Bering Strait) with parent Strait.
    Both the class label AND the parent's local name contain the landform
    keyword 'strait', so the guard ALLOWS the parent -- Bering Strait
    lands in the geography ns under Strait."""
    from backend.app.helpers.ontology_pruning import add_new_classes_from_match_not_found

    strait_iri = "https://veerla-ramrao.ai/ontology/geography#Strait"
    strait_rec = {
        "iri": strait_iri, "name": "Strait",
        "labels": ["Strait"], "superclasses": [],
    }
    ontology = {
        "classes_dict": {strait_iri: strait_rec},
        "object_properties_dict": {},
        "data_properties_dict": {},
        "instances_dict": {},
    }
    results = {
        "MATCH NOT FOUND": [
            {"LABEL": "Bering Strait", "DESCRIPTION": "A strait between Russia and Alaska.",
             "PARENT_LABEL": "Strait"},
        ]
    }
    extended, created = add_new_classes_from_match_not_found(
        ontology, results,
        new_class_base_iri="http://default.example/ontology/",
        default_parent_iri=OWL_THING,
    )
    assert len(created) == 1
    bering_iri = created[0]
    # Lands in geography namespace.
    assert bering_iri.startswith("https://veerla-ramrao.ai/ontology/geography#")
    # Parent is Strait.
    parent = extended["classes_dict"][bering_iri]["superclasses"][0]["iri"]
    assert parent == strait_iri


def test_geo_parent_guard_in_relation_automint() -> None:
    """Relation 'Subsidy receivedBy Iran' where Iran is in geography ns
    with parent Country (also in geography ns). Layer D would normally
    auto-mint Subsidy inheriting Iran's parent -- Subsidy would cascade
    into geography. The guard rejects that inheritance; Subsidy lands at
    owl:Thing in the default namespace."""
    from backend.app.helpers.ontology_pruning import add_new_relations_from_match_results

    country_iri = "https://veerla-ramrao.ai/ontology/geography#Country"
    country_rec = {
        "iri": country_iri, "name": "Country",
        "labels": ["Country"], "superclasses": [],
    }
    iran_iri = "https://veerla-ramrao.ai/ontology/geography#Iran"
    iran_rec = {
        "iri": iran_iri, "name": "Iran", "labels": ["Iran"],
        "superclasses": [{"kind": "entity", "iri": country_iri}],
    }
    ontology = {
        "classes_dict": {country_iri: country_rec, iran_iri: iran_rec},
        "object_properties_dict": {},
        "data_properties_dict": {},
        "instances_dict": {},
    }
    results = {
        "MATCH NOT FOUND RELATIONS": [
            {"LABEL": "received_by", "DESCRIPTION": "Iran receives subsidies.",
             "DOMAIN": "Subsidy", "RANGE": "Iran"},
        ]
    }
    extended, _, _, auto_minted = add_new_relations_from_match_results(
        ontology, results,
        new_property_base_iri="http://default.example/ontology/",
        new_class_base_iri="http://default.example/ontology/",
        default_parent_iri=OWL_THING,
    )
    # Subsidy was auto-minted.
    assert len(auto_minted) == 1
    subsidy_iri = auto_minted[0]
    # CRITICAL: it landed in DEFAULT ns, NOT geography.
    assert subsidy_iri.startswith("http://default.example/ontology/")
    assert "geography" not in subsidy_iri.lower()
    # Parent is owl:Thing (the guard rejected Country inheritance).
    parent = extended["classes_dict"][subsidy_iri]["superclasses"][0]["iri"]
    assert parent == OWL_THING


# ============================================================================
# Pre-pipeline document summarization (`summarize_long_documents_async`).
# ============================================================================


def test_summarize_long_documents_skips_short_docs() -> None:
    """Documents under the threshold pass through unchanged; only the
    over-threshold docs trigger an LLM call."""
    import asyncio
    from pathlib import Path
    from backend.app.services.document_io import LoadedDocument
    from backend.app.services.pipeline_llm import summarize_long_documents_async

    short = LoadedDocument(path=Path("short.txt"), text="Short doc, just a few words.")
    long_words = "Long document with many entities. " * 800  # ~5000 tokens
    long_a = LoadedDocument(path=Path("long_a.txt"), text=long_words)
    long_b = LoadedDocument(path=Path("long_b.txt"), text=long_words)

    class _FakeResult:
        def __init__(self, text: str) -> None:
            self.text = text
            self.prompt_tokens = 100
            self.completion_tokens = 50
            self.cost_usd = 0.0001

    class _FakeRouter:
        def __init__(self) -> None:
            self.calls: list[str] = []
            self.total_cost_usd = 0.0

        async def chat(self, task: str, *, system: str, user: str):
            assert task == "document_summarize"
            self.calls.append(user)
            # Return a much shorter summary.
            return _FakeResult("Summarized text mentioning the entities.")

    router = _FakeRouter()
    out = asyncio.run(
        summarize_long_documents_async(
            documents=[short, long_a, long_b],
            router=router,
            threshold_tokens=2000,
            concurrency=2,
            use_cache=False,  # tests should not pollute the user cache dir
        )
    )

    # Order preserved.
    assert out[0].path.name == "short.txt"
    assert out[1].path.name == "long_a.txt"
    assert out[2].path.name == "long_b.txt"

    # Short doc unchanged.
    assert out[0].text == "Short doc, just a few words."

    # Long docs replaced by the summary.
    assert out[1].text == "Summarized text mentioning the entities."
    assert out[2].text == "Summarized text mentioning the entities."

    # Exactly TWO LLM calls fired (one per long doc; short skipped).
    assert len(router.calls) == 2


def test_summarize_long_documents_falls_back_on_llm_failure() -> None:
    """A failed summarization call preserves the original document text
    (purely additive failure mode -- pipeline never breaks)."""
    import asyncio
    from pathlib import Path
    from backend.app.services.document_io import LoadedDocument
    from backend.app.services.pipeline_llm import summarize_long_documents_async

    long_text = "Long document with many entities. " * 800  # ~5000 tokens
    doc = LoadedDocument(path=Path("doc.txt"), text=long_text)

    class _ExplodingRouter:
        total_cost_usd = 0.0

        async def chat(self, task: str, *, system: str, user: str):
            raise RuntimeError("simulated provider failure")

    out = asyncio.run(
        summarize_long_documents_async(
            documents=[doc],
            router=_ExplodingRouter(),
            threshold_tokens=2000,
            concurrency=1,
            use_cache=False,  # this test must exercise the LLM-failure fallback
        )
    )
    # Original text preserved.
    assert out[0].text == long_text


def test_summarize_long_documents_disabled_when_threshold_zero() -> None:
    """threshold_tokens <= 0 short-circuits the entire pass: no
    LLM calls, no list mutation."""
    import asyncio
    from pathlib import Path
    from backend.app.services.document_io import LoadedDocument
    from backend.app.services.pipeline_llm import summarize_long_documents_async

    long_text = "x " * 5000
    docs = [LoadedDocument(path=Path("a.txt"), text=long_text)]

    class _NoChat:
        total_cost_usd = 0.0

        async def chat(self, *_args, **_kwargs):  # pragma: no cover
            raise AssertionError("must not be called when threshold is 0")

    out = asyncio.run(
        summarize_long_documents_async(
            documents=docs,
            router=_NoChat(),
            threshold_tokens=0,
            concurrency=1,
        )
    )
    assert out[0].text == long_text


def test_doc_summary_cache_hit_skips_llm_call(monkeypatch, tmp_path) -> None:
    """When a cache entry exists for a doc's (text, model) hash, the
    cached summary is returned directly without calling the LLM."""
    import asyncio
    from pathlib import Path
    from backend.app.services import pipeline_llm
    from backend.app.services.document_io import LoadedDocument
    from backend.app.services.pipeline_llm import (
        summarize_long_documents_async,
        _doc_summary_cache_key,
        _doc_summary_cache_path,
        _doc_summary_cache_save,
    )

    # Redirect the cache to a per-test tmp dir.
    monkeypatch.setattr(
        pipeline_llm,
        "_doc_summary_cache_dir",
        lambda: tmp_path,
    )

    long_text = "Long document with entities. " * 800
    doc = LoadedDocument(path=Path("doc.txt"), text=long_text)

    # Pre-populate the cache with a known summary.
    key = _doc_summary_cache_key(long_text, "gpt-4o-mini")
    cache_file = _doc_summary_cache_path(tmp_path, key)
    _doc_summary_cache_save(cache_file, "CACHED summary text.")

    class _ExplodingRouter:
        total_cost_usd = 0.0

        async def chat(self, *_args, **_kwargs):  # pragma: no cover
            raise AssertionError("LLM must not be called on cache hit")

    out = asyncio.run(
        summarize_long_documents_async(
            documents=[doc],
            router=_ExplodingRouter(),
            threshold_tokens=2000,
            concurrency=1,
            use_cache=True,
        )
    )
    assert out[0].text == "CACHED summary text."


def test_doc_summary_cache_miss_writes_cache_for_future_runs(
    monkeypatch, tmp_path,
) -> None:
    """A cache miss triggers the LLM, and the response is written to
    disk under the deterministic hash key."""
    import asyncio
    from pathlib import Path
    from backend.app.services import pipeline_llm
    from backend.app.services.document_io import LoadedDocument
    from backend.app.services.pipeline_llm import (
        summarize_long_documents_async,
        _doc_summary_cache_key,
        _doc_summary_cache_path,
    )

    monkeypatch.setattr(
        pipeline_llm,
        "_doc_summary_cache_dir",
        lambda: tmp_path,
    )

    long_text = "Long document with named entities and relationships. " * 500
    doc = LoadedDocument(path=Path("doc.txt"), text=long_text)

    class _FakeResult:
        def __init__(self, text: str) -> None:
            self.text = text
            self.prompt_tokens = 100
            self.completion_tokens = 50
            self.cost_usd = 0.0001

    class _FakeRouter:
        total_cost_usd = 0.0
        calls = 0

        async def chat(self, task, *, system, user):
            type(self).calls += 1
            return _FakeResult("Fresh LLM summary text.")

    out = asyncio.run(
        summarize_long_documents_async(
            documents=[doc],
            router=_FakeRouter(),
            threshold_tokens=2000,
            concurrency=1,
            use_cache=True,
        )
    )
    # The summary made it through.
    assert out[0].text == "Fresh LLM summary text."
    # Exactly one LLM call.
    assert _FakeRouter.calls == 1
    # The cache file now exists with the summary text.
    key = _doc_summary_cache_key(long_text, "gpt-4o-mini")
    cache_file = _doc_summary_cache_path(tmp_path, key)
    assert cache_file.exists()
    assert cache_file.read_text(encoding="utf-8").strip() == "Fresh LLM summary text."


def test_doc_summary_cache_invalidates_when_text_changes(
    monkeypatch, tmp_path,
) -> None:
    """Editing the doc text changes the hash, so the prior cache entry
    is NOT used and the LLM is called for the new content."""
    import asyncio
    from pathlib import Path
    from backend.app.services import pipeline_llm
    from backend.app.services.document_io import LoadedDocument
    from backend.app.services.pipeline_llm import (
        summarize_long_documents_async,
        _doc_summary_cache_key,
        _doc_summary_cache_path,
        _doc_summary_cache_save,
    )

    monkeypatch.setattr(
        pipeline_llm,
        "_doc_summary_cache_dir",
        lambda: tmp_path,
    )

    original_text = "Original long doc. " * 400
    edited_text = "Edited long doc with new fact. " * 400

    # Pre-populate cache with the ORIGINAL hash.
    original_key = _doc_summary_cache_key(original_text, "gpt-4o-mini")
    _doc_summary_cache_save(
        _doc_summary_cache_path(tmp_path, original_key),
        "Stale summary from original text.",
    )

    doc = LoadedDocument(path=Path("doc.txt"), text=edited_text)

    class _FakeResult:
        def __init__(self, text):
            self.text = text
            self.prompt_tokens = 100
            self.completion_tokens = 50
            self.cost_usd = 0.0001

    class _FakeRouter:
        total_cost_usd = 0.0
        calls = 0

        async def chat(self, task, *, system, user):
            type(self).calls += 1
            return _FakeResult("Fresh summary for edited text.")

    out = asyncio.run(
        summarize_long_documents_async(
            documents=[doc],
            router=_FakeRouter(),
            threshold_tokens=2000,
            concurrency=1,
            use_cache=True,
        )
    )
    # The stale entry was NOT used; the LLM was called fresh.
    assert out[0].text == "Fresh summary for edited text."
    assert _FakeRouter.calls == 1


def test_doc_summary_cache_disabled_via_use_cache_false(
    monkeypatch, tmp_path,
) -> None:
    """When use_cache=False, neither cache lookup nor cache write
    happens -- every run hits the LLM and nothing accumulates on disk."""
    import asyncio
    from pathlib import Path
    from backend.app.services import pipeline_llm
    from backend.app.services.document_io import LoadedDocument
    from backend.app.services.pipeline_llm import (
        summarize_long_documents_async,
        _doc_summary_cache_key,
        _doc_summary_cache_path,
        _doc_summary_cache_save,
    )

    monkeypatch.setattr(
        pipeline_llm,
        "_doc_summary_cache_dir",
        lambda: tmp_path,
    )

    long_text = "Long doc to summarize. " * 500
    # Pre-populate cache so we can prove use_cache=False ignores it.
    key = _doc_summary_cache_key(long_text, "gpt-4o-mini")
    _doc_summary_cache_save(
        _doc_summary_cache_path(tmp_path, key),
        "Existing cached text we should NOT see.",
    )

    doc = LoadedDocument(path=Path("doc.txt"), text=long_text)

    class _FakeResult:
        def __init__(self, text):
            self.text = text
            self.prompt_tokens = 100
            self.completion_tokens = 50
            self.cost_usd = 0.0001

    class _FakeRouter:
        total_cost_usd = 0.0

        async def chat(self, task, *, system, user):
            return _FakeResult("Fresh LLM output ignoring cache.")

    out = asyncio.run(
        summarize_long_documents_async(
            documents=[doc],
            router=_FakeRouter(),
            threshold_tokens=2000,
            concurrency=1,
            use_cache=False,
        )
    )
    # Pre-existing cache value was ignored; LLM result used.
    assert out[0].text == "Fresh LLM output ignoring cache."


# ============================================================================
# Oversize-doc ceiling: docs above max_doc_input_tokens must be DROPPED
# (text replaced with empty string), not passed through to the LLM or the
# chunker. Prevents Stage 2 cost explosion on multi-million-token PDFs.
# ============================================================================


def test_summarize_oversize_doc_uses_hierarchical_path(monkeypatch, tmp_path) -> None:
    """A document over the 100K-token ceiling is split into sub-chunks
    and each sub-chunk gets its own LLM call; the per-sub-chunk
    summaries are concatenated into a combined summary."""
    import asyncio
    from pathlib import Path
    from backend.app.services import pipeline_llm
    from backend.app.services.document_io import LoadedDocument
    from backend.app.services.pipeline_llm import summarize_long_documents_async

    monkeypatch.setattr(pipeline_llm, "_doc_summary_cache_dir", lambda: tmp_path)

    # ~110K-ish tokens to force hierarchical (above 100K ceiling).
    oversize_text = "word " * 110_000
    normal_text = "Long doc within ceiling. " * 800  # ~5K tokens

    oversize = LoadedDocument(path=Path("huge.txt"), text=oversize_text)
    normal = LoadedDocument(path=Path("normal.txt"), text=normal_text)

    class _FakeResult:
        def __init__(self, text):
            self.text = text
            self.prompt_tokens = 100
            self.completion_tokens = 50
            self.cost_usd = 0.0001

    class _FakeRouter:
        total_cost_usd = 0.0
        calls: list[str] = []

        async def chat(self, task, *, system, user):
            type(self).calls.append(user[:60])
            return _FakeResult("SUB-SUMMARY.")

    out = asyncio.run(
        summarize_long_documents_async(
            documents=[oversize, normal],
            router=_FakeRouter(),
            threshold_tokens=2000,
            concurrency=4,
            use_cache=False,
            max_doc_input_tokens=100_000,
            oversize_doc_sub_chunk_tokens=80_000,
        )
    )

    # Normal doc summarized once with the standard single-call path.
    assert out[1].path.name == "normal.txt"
    assert out[1].text == "SUB-SUMMARY."

    # Oversize doc has a COMBINED summary (multiple sub-summaries joined).
    assert out[0].path.name == "huge.txt"
    assert "SUB-SUMMARY." in out[0].text
    # 110K tokens / 80K per sub-chunk = at least 2 sub-chunk calls.
    # Plus 1 call for the normal doc = at least 3 total.
    assert len(_FakeRouter.calls) >= 3
    # The combined summary contains the sub-summary token at least
    # twice (one per sub-chunk of the oversize doc).
    assert out[0].text.count("SUB-SUMMARY.") >= 2


def test_summarize_oversize_partial_failure_keeps_what_works(
    monkeypatch, tmp_path,
) -> None:
    """If SOME sub-chunks fail, the combined summary contains only the
    successful ones. The doc is NOT entirely lost."""
    import asyncio
    from pathlib import Path
    from backend.app.services import pipeline_llm
    from backend.app.services.document_io import LoadedDocument
    from backend.app.services.pipeline_llm import summarize_long_documents_async

    monkeypatch.setattr(pipeline_llm, "_doc_summary_cache_dir", lambda: tmp_path)

    oversize_text = "word " * 200_000  # forces multiple sub-chunks
    doc = LoadedDocument(path=Path("huge.txt"), text=oversize_text)

    class _FakeResult:
        def __init__(self, text):
            self.text = text
            self.prompt_tokens = 100
            self.completion_tokens = 50
            self.cost_usd = 0.0001

    class _FlakyRouter:
        total_cost_usd = 0.0
        call_n = 0

        async def chat(self, task, *, system, user):
            type(self).call_n += 1
            # Fail every odd-numbered call.
            if type(self).call_n % 2 == 1:
                raise RuntimeError("simulated transient failure")
            return _FakeResult(f"OK-SUMMARY-{type(self).call_n}.")

    out = asyncio.run(
        summarize_long_documents_async(
            documents=[doc],
            router=_FlakyRouter(),
            threshold_tokens=2000,
            concurrency=1,
            use_cache=False,
            max_doc_input_tokens=100_000,
            oversize_doc_sub_chunk_tokens=80_000,
        )
    )
    # The combined text exists (not empty) and contains at least one
    # successful sub-summary.
    assert out[0].text != ""
    assert "OK-SUMMARY-" in out[0].text


def test_summarize_oversize_caches_combined_summary(monkeypatch, tmp_path) -> None:
    """The combined summary from hierarchical summarization is cached
    under the ORIGINAL doc text's hash key. A re-run hits the cache and
    fires zero LLM calls."""
    import asyncio
    from pathlib import Path
    from backend.app.services import pipeline_llm
    from backend.app.services.document_io import LoadedDocument
    from backend.app.services.pipeline_llm import (
        summarize_long_documents_async,
        _doc_summary_cache_key,
        _doc_summary_cache_path,
    )

    monkeypatch.setattr(pipeline_llm, "_doc_summary_cache_dir", lambda: tmp_path)

    oversize_text = "word " * 110_000
    doc = LoadedDocument(path=Path("huge.txt"), text=oversize_text)

    class _FakeResult:
        def __init__(self, text):
            self.text = text
            self.prompt_tokens = 100
            self.completion_tokens = 50
            self.cost_usd = 0.0001

    class _CountingRouter:
        total_cost_usd = 0.0
        calls = 0

        async def chat(self, task, *, system, user):
            type(self).calls += 1
            return _FakeResult("SUB.")

    # First run: cache empty -> sub-chunks summarized -> combined cached.
    out1 = asyncio.run(
        summarize_long_documents_async(
            documents=[doc],
            router=_CountingRouter(),
            threshold_tokens=2000,
            concurrency=2,
            use_cache=True,
            max_doc_input_tokens=100_000,
            oversize_doc_sub_chunk_tokens=80_000,
        )
    )
    first_call_count = _CountingRouter.calls
    assert first_call_count >= 2  # at least 2 sub-chunks fired

    # Cache file exists under the original doc's hash.
    key = _doc_summary_cache_key(oversize_text, "gpt-4o-mini")
    assert _doc_summary_cache_path(tmp_path, key).exists()

    # Second run with a router that explodes if called: cache hit only.
    class _ExplodingRouter:
        total_cost_usd = 0.0

        async def chat(self, *_args, **_kwargs):
            raise AssertionError("LLM must not be called on warm cache")

    out2 = asyncio.run(
        summarize_long_documents_async(
            documents=[doc],
            router=_ExplodingRouter(),
            threshold_tokens=2000,
            concurrency=2,
            use_cache=True,
            max_doc_input_tokens=100_000,
            oversize_doc_sub_chunk_tokens=80_000,
        )
    )
    # Same combined summary served from cache.
    assert out2[0].text == out1[0].text


def test_summarize_default_max_doc_input_tokens_is_100k() -> None:
    """The function signature exposes max_doc_input_tokens with a default
    of 100_000 -- below gpt-4o-mini's 128K context, with headroom for
    the system + user prompt overhead."""
    import inspect
    from backend.app.services.pipeline_llm import summarize_long_documents_async

    sig = inspect.signature(summarize_long_documents_async)
    param = sig.parameters.get("max_doc_input_tokens")
    assert param is not None, "max_doc_input_tokens parameter missing"
    assert param.default == 100_000, (
        f"default ceiling should be 100K, got {param.default}"
    )


def test_summarize_default_oversize_sub_chunk_is_80k() -> None:
    """Default sub-chunk size for hierarchical path is 80K tokens --
    leaves comfortable headroom under gpt-4o-mini's 128K input limit."""
    import inspect
    from backend.app.services.pipeline_llm import summarize_long_documents_async

    sig = inspect.signature(summarize_long_documents_async)
    param = sig.parameters.get("oversize_doc_sub_chunk_tokens")
    assert param is not None
    assert param.default == 80_000


# ============================================================================
# OWL-export control-char sanitizer (XML 1.0 compliance).
# ============================================================================


def test_owl_export_strips_xml_forbidden_control_chars(tmp_path) -> None:
    """A class label containing \\x13 (Device Control 3) must NOT
    appear verbatim in the exported OWL -- XML 1.0 forbids it."""
    import xml.etree.ElementTree as ET
    from backend.app.services import ontology_export

    loaded = {
        "classes_dict": {
            "http://x/A": {
                "iri": "http://x/A",
                "name": "Bol\x13var",
                "labels": ["Bol\x13var fuerte"],
                "comments": ["Mojibake from PDF\x13 extraction"],
                "descriptions": [],
                "superclasses": [],
            }
        },
        "object_properties_dict": {},
        "data_properties_dict": {},
        "instances_dict": {},
    }
    out_path = tmp_path / "merged.owl"
    ontology_export.write_owl(loaded, out_path)

    # 1) the file is well-formed XML.
    tree = ET.parse(out_path)
    assert tree.getroot().tag.endswith("RDF")
    # 2) raw \x13 doesn't appear in the file bytes.
    data = out_path.read_bytes()
    assert b"\x13" not in data


# ============================================================================
# Layer F geographic-inference stop-words.
# ============================================================================


def test_geo_inference_skips_event_named_after_strait() -> None:
    """'Strait of Hormuz crisis' must not be parented under Strait --
    the event keyword overrides the landform substring match."""
    from backend.app.helpers.ontology_pruning import infer_geographic_placement

    OWL_THING = "http://www.w3.org/2002/07/owl#Thing"
    classes = {
        "http://geo/strait": {
            "iri": "http://geo/strait",
            "name": "strait",
            "labels": ["Strait"],
            "superclasses": [],
        },
        "http://default/strait_of_hormuz_crisis": {
            "iri": "http://default/strait_of_hormuz_crisis",
            "name": "strait_of_hormuz_crisis",
            "labels": ["Strait of Hormuz crisis"],
            "superclasses": [{"iri": OWL_THING, "name": "Thing"}],
        },
        "http://default/strait_of_hormuz": {
            "iri": "http://default/strait_of_hormuz",
            "name": "strait_of_hormuz",
            "labels": ["Strait of Hormuz"],
            "superclasses": [{"iri": OWL_THING, "name": "Thing"}],
        },
    }
    infer_geographic_placement(classes, {}, {}, {})

    # Crisis stays unparented (still owl:Thing); will go to Layer G.
    crisis = classes.get("http://default/strait_of_hormuz_crisis")
    assert crisis is not None, "crisis class shouldn't be deleted"
    crisis_parent = (crisis.get("superclasses") or [{}])[0].get("iri")
    assert crisis_parent == OWL_THING, (
        f"crisis class was re-homed under {crisis_parent!r} -- the "
        f"event keyword should have blocked geo inference"
    )

    # The pure place (no event modifier) IS re-homed under Strait.
    # (The re-home moves it to the geo namespace, so look at any
    # remaining key with 'strait_of_hormuz' as local name.)
    rehomed = None
    for k, v in classes.items():
        if k != "http://geo/strait" and "strait_of_hormuz" in k.lower() and "crisis" not in k.lower():
            rehomed = v
            break
    assert rehomed is not None
    rehomed_parent = (rehomed.get("superclasses") or [{}])[0].get("iri")
    assert rehomed_parent == "http://geo/strait"


# ============================================================================
# Layer H classification audit: suspicious-detection + apply logic.
# ============================================================================


def test_layer_h_is_suspicious_flags_corporate_label_under_helium() -> None:
    """A LABEL like 'Air Products and Chemicals Inc' currently parented
    under Helium IS suspicious -- corporate suffix + non-Org parent."""
    from backend.app.services.pipeline_llm import _is_suspicious

    classes = {
        "http://x/helium": {
            "iri": "http://x/helium",
            "name": "Helium",
            "labels": ["Helium"],
            "superclasses": [],
        },
        "http://x/air_products": {
            "iri": "http://x/air_products",
            "name": "air_products",
            "labels": ["Air Products and Chemicals Inc"],
            "superclasses": [{"iri": "http://x/helium", "name": "Helium"}],
            "semantic_role": {"reasons": ["Created from MATCH NOT FOUND"]},
        },
    }
    rec = classes["http://x/air_products"]
    assert _is_suspicious("http://x/air_products", rec, classes) is True


def test_layer_h_is_suspicious_flags_event_label_under_strait() -> None:
    """'Strait of Hormuz crisis' under Strait is suspicious."""
    from backend.app.services.pipeline_llm import _is_suspicious

    classes = {
        "http://x/strait": {
            "iri": "http://x/strait",
            "name": "Strait",
            "labels": ["Strait"],
            "superclasses": [],
        },
        "http://x/crisis": {
            "iri": "http://x/crisis",
            "name": "crisis",
            "labels": ["Strait of Hormuz crisis"],
            "superclasses": [{"iri": "http://x/strait", "name": "Strait"}],
            "semantic_role": {"reasons": ["Created from MATCH NOT FOUND"]},
        },
    }
    assert _is_suspicious("http://x/crisis", classes["http://x/crisis"], classes) is True


def test_layer_h_apply_rehome_rewrites_superclasses() -> None:
    """RE_HOME action rewrites the class's superclasses to point at the
    new parent (looked up by label) and tags the class as
    'classification_audit'-touched."""
    from backend.app.services.pipeline_llm import _apply_audit_decision

    classes = {
        "http://x/helium": {
            "iri": "http://x/helium",
            "name": "Helium",
            "labels": ["Helium"],
            "superclasses": [],
        },
        "http://x/org_bucket": {
            "iri": "http://x/org_bucket",
            "name": "Organization",
            "labels": ["Organization"],
            "superclasses": [],
        },
        "http://x/air_products": {
            "iri": "http://x/air_products",
            "name": "air_products",
            "labels": ["Air Products"],
            "superclasses": [{"iri": "http://x/helium", "name": "Helium"}],
        },
    }
    rec = classes["http://x/air_products"]
    instances: dict = {}
    outcome = _apply_audit_decision(
        "http://x/air_products", rec,
        {"LABEL": "Air Products", "ACTION": "RE_HOME", "NEW_PARENT": "Organization"},
        classes, instances,
        "https://veerla-ramrao.ai/ontology/merged#",
    )
    assert outcome == "rehomed"
    parent = rec["superclasses"][0]["iri"]
    assert parent == "http://x/org_bucket"
    assert "classification_audit" in (rec.get("auto_created_via") or [])


def test_layer_h_apply_convert_to_instance_moves_to_instances_dict() -> None:
    """CONVERT_TO_INSTANCE moves the entity out of classes_dict and
    creates an instance under the proposed parent class."""
    from backend.app.services.pipeline_llm import _apply_audit_decision

    classes = {
        "http://x/person_bucket": {
            "iri": "http://x/person_bucket",
            "name": "Person",
            "labels": ["Person"],
            "superclasses": [],
        },
        "http://x/putin": {
            "iri": "http://x/putin",
            "name": "Vladimir_Putin",
            "labels": ["Vladimir Putin"],
            "superclasses": [{"iri": "http://x/person_bucket", "name": "Person"}],
        },
    }
    rec = classes["http://x/putin"]
    instances: dict = {}
    outcome = _apply_audit_decision(
        "http://x/putin", rec,
        {"LABEL": "Vladimir Putin", "ACTION": "CONVERT_TO_INSTANCE",
         "NEW_PARENT": "Person"},
        classes, instances,
        "https://veerla-ramrao.ai/ontology/merged#",
    )
    assert outcome == "converted"
    # Class entry deleted; instance entry created.
    assert "http://x/putin" not in classes
    assert "http://x/putin" in instances
    inst = instances["http://x/putin"]
    assert inst["types"][0]["iri"] == "http://x/person_bucket"


# ============================================================================
# Streaming load+summarize+chunk: memory-bounded pipeline path.
# ============================================================================


def test_stream_summarize_and_chunk_processes_in_batches(monkeypatch, tmp_path) -> None:
    """50 short docs with batch_size=10 -> the summarizer is called
    5 times (once per batch). Each batch is independent and the chunks
    accumulate."""
    import asyncio
    from backend.app.services import pipeline_llm
    from backend.app.services.pipeline_llm import stream_summarize_and_chunk_async

    # Redirect cache to per-test dir so we don't pollute the user cache.
    monkeypatch.setattr(pipeline_llm, "_doc_summary_cache_dir", lambda: tmp_path)

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    # 50 short docs, each ~3000 tokens (> 2000 threshold), so each triggers
    # a summarization call.
    for i in range(50):
        (docs_dir / f"doc_{i:03d}.txt").write_text(
            "word " * 3000, encoding="utf-8"
        )

    call_counts = {"summarize": 0, "chat": 0}

    # Patch summarize_long_documents_async to count invocations and
    # return docs unchanged (so chunking still works).
    real_summarize = pipeline_llm.summarize_long_documents_async

    async def counting_summarize(*args, **kwargs):
        call_counts["summarize"] += 1
        # Don't actually call the LLM. Just return the docs unchanged.
        return list(kwargs.get("documents") or args[0])

    monkeypatch.setattr(pipeline_llm, "summarize_long_documents_async", counting_summarize)

    class _NoChat:
        total_cost_usd = 0.0

        async def chat(self, *_args, **_kwargs):
            raise AssertionError("LLM must not be called via the chat() path here")

    chunks = asyncio.run(stream_summarize_and_chunk_async(
        documents_dir=docs_dir,
        router=_NoChat(),
        chunk_size=2000,
        chunk_overlap=150,
        encoding_name="o200k_base",
        threshold_tokens=2000,
        max_doc_input_tokens=100_000,
        oversize_doc_sub_chunk_tokens=80_000,
        use_cache=False,
        concurrency=2,
        batch_size=10,
    ))

    # 50 docs / batch_size=10 = 5 batches => summarize called 5 times.
    assert call_counts["summarize"] == 5
    # Chunks accumulated across all batches; each ~3000-token doc -> 2
    # chunks at chunk_size=2000 overlap=150 -> ~100 chunks total.
    assert len(chunks) >= 50


def test_stream_summarize_and_chunk_preserves_doc_order(monkeypatch, tmp_path) -> None:
    """Chunks come out in source-doc filename order (sorted by
    iter_documents)."""
    import asyncio
    from backend.app.services import pipeline_llm
    from backend.app.services.pipeline_llm import stream_summarize_and_chunk_async

    monkeypatch.setattr(pipeline_llm, "_doc_summary_cache_dir", lambda: tmp_path)

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    # Write under-threshold docs so no LLM call is needed and we can
    # verify chunk source order purely.
    for i in range(20):
        (docs_dir / f"doc_{i:02d}.txt").write_text(
            f"Doc {i:02d}: a small chunk of content. ",
            encoding="utf-8",
        )

    class _NoChat:
        total_cost_usd = 0.0

        async def chat(self, *_args, **_kwargs):
            raise AssertionError("LLM must not be called for under-threshold docs")

    chunks = asyncio.run(stream_summarize_and_chunk_async(
        documents_dir=docs_dir,
        router=_NoChat(),
        chunk_size=2000,
        chunk_overlap=150,
        encoding_name="o200k_base",
        threshold_tokens=2000,
        max_doc_input_tokens=100_000,
        oversize_doc_sub_chunk_tokens=80_000,
        use_cache=False,
        concurrency=1,
        batch_size=5,  # 4 batches
    ))

    # Each under-threshold doc -> 1 chunk. Source order preserved.
    sources = [c.source_name for c in chunks]
    expected = [f"doc_{i:02d}.txt" for i in range(20)]
    assert sources == expected


def test_stream_summarize_and_chunk_falls_back_to_load_all_when_batch_size_zero(
    monkeypatch, tmp_path,
) -> None:
    """batch_size=0 is the legacy 'load everything at once' path.
    Useful for tests + small-corpus runs on machines with plenty of RAM."""
    import asyncio
    from backend.app.services import pipeline_llm
    from backend.app.services.pipeline_llm import stream_summarize_and_chunk_async

    monkeypatch.setattr(pipeline_llm, "_doc_summary_cache_dir", lambda: tmp_path)

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    for i in range(8):
        (docs_dir / f"doc_{i}.txt").write_text(
            f"Short doc {i}. ", encoding="utf-8"
        )

    class _NoChat:
        total_cost_usd = 0.0

        async def chat(self, *_args, **_kwargs):
            raise AssertionError("no LLM expected for tiny docs")

    chunks = asyncio.run(stream_summarize_and_chunk_async(
        documents_dir=docs_dir,
        router=_NoChat(),
        chunk_size=2000,
        chunk_overlap=150,
        encoding_name="o200k_base",
        threshold_tokens=2000,
        max_doc_input_tokens=100_000,
        oversize_doc_sub_chunk_tokens=80_000,
        use_cache=False,
        concurrency=1,
        batch_size=0,  # legacy "all-at-once"
    ))
    assert len(chunks) == 8


# ============================================================================
# Bootstrap domain-concepts ontology (Fix B): the .owl file ships with the
# repo and must parse cleanly into the 12 bucket classes the LLM anchors to.
# ============================================================================


def test_domain_concepts_owl_parses_with_12_buckets() -> None:
    """source_ontologies/core_ontologies/domain_concepts.owl declares 12
    owl:Class buckets, each ALSO typed as skos:Concept (OWL Punning), and
    one skos:ConceptScheme grouping them. The pipeline expects these
    classes to land in classes_dict via the standard merge path."""
    from pathlib import Path
    from rdflib import Graph, RDF, OWL
    import rdflib
    SKOS = rdflib.Namespace("http://www.w3.org/2004/02/skos/core#")

    p = Path("source_ontologies/core_ontologies/domain_concepts.owl")
    assert p.exists(), f"missing bootstrap file: {p}"

    g = Graph()
    g.parse(p, format="xml")

    owl_classes = list(g.subjects(RDF.type, OWL.Class))
    skos_concepts = list(g.subjects(RDF.type, SKOS.Concept))
    schemes = list(g.subjects(RDF.type, SKOS.ConceptScheme))

    assert len(owl_classes) == 12, f"expected 12 owl:Class, got {len(owl_classes)}"
    assert len(skos_concepts) == 12, (
        f"expected 12 skos:Concept (dual-typed via OWL Punning), "
        f"got {len(skos_concepts)}"
    )
    assert len(schemes) == 1, f"expected 1 skos:ConceptScheme, got {len(schemes)}"

    # The 12 bucket names the Stage 2 LLM uses as PARENT_LABEL.
    expected_locals = {
        "Industry", "EconomicConcept", "PolicyConcept", "Event", "Process",
        "Material", "NaturalResource", "TechnologyConcept", "Infrastructure",
        "SupplyChainConcept", "GeographicFeature", "DomainConcept",
    }
    got_locals = {str(iri).rsplit("#", 1)[-1] for iri in owl_classes}
    assert got_locals == expected_locals, (
        f"missing or extra buckets. expected={expected_locals} got={got_locals}"
    )


# ============================================================================
# Rebrand regression: the codebase no longer hardcodes the .local placeholder
# IRI. Catches accidental reintroduction of the old branding.
# ============================================================================


def test_rebrand_default_ontology_iri_uses_veerla_ramrao_ai() -> None:
    """ontology_export.DEFAULT_ONTOLOGY_IRI is the IRI written into the
    `<owl:Ontology rdf:about=...>` header of every emitted merged.owl.
    Must be on the user's domain, not the .local placeholder."""
    from backend.app.services.ontology_export import DEFAULT_ONTOLOGY_IRI

    assert DEFAULT_ONTOLOGY_IRI == "https://veerla-ramrao.ai/ontology/merged"
    assert "your-personal-ontologist.local" not in DEFAULT_ONTOLOGY_IRI


def test_rebrand_export_emits_veerla_namespace_for_annotations(tmp_path) -> None:
    """Custom annotation predicates (semantic_role, sources, etc.) must
    use the veerla-ramrao.ai/ontology/ann# namespace, not the old
    .local placeholder."""
    from backend.app.services import ontology_export

    loaded = {
        "classes_dict": {
            "https://veerla-ramrao.ai/ontology/merged#TestClass": {
                "iri": "https://veerla-ramrao.ai/ontology/merged#TestClass",
                "name": "TestClass",
                "labels": ["Test Class"],
                "semantic_role": {"role": "test"},
                "superclasses": [],
            }
        },
        "object_properties_dict": {},
        "data_properties_dict": {},
        "instances_dict": {},
    }
    out_path = tmp_path / "merged.owl"
    ontology_export.write_owl(loaded, out_path)

    data = out_path.read_text(encoding="utf-8")
    assert "https://veerla-ramrao.ai/ontology/ann#" in data
    assert "your-personal-ontologist.local" not in data


# ============================================================================
# Fix A: standard-upper-ontology namespace guard in
# _derive_namespace_from_parent_iri. A class parented under foaf:Organization
# (or org:Role, skos:Concept, etc.) MUST mint in default_base_iri, not in
# the upper-ontology's namespace.
# ============================================================================


def test_derive_namespace_skips_foaf_org_skos_namespaces() -> None:
    """The Fix A guard. Without it, 'Samsung Electronics' parented under
    foaf:Organization gets the IRI http://xmlns.com/foaf/0.1/samsung_electronics
    -- polluting FOAF + accidentally protecting the squatter via
    protected_iri_prefixes. Guard: parent in FOAF/ORG/SKOS/time/OWL/RDFS
    -> fall back to default_base_iri."""
    from backend.app.helpers.ontology_pruning import _derive_namespace_from_parent_iri

    base = "https://veerla-ramrao.ai/ontology/merged#"

    # FOAF parent -> base.
    assert _derive_namespace_from_parent_iri(
        "http://xmlns.com/foaf/0.1/Organization", base
    ) == base
    # ORG parent -> base.
    assert _derive_namespace_from_parent_iri(
        "http://www.w3.org/ns/org#Role", base
    ) == base
    # SKOS parent -> base.
    assert _derive_namespace_from_parent_iri(
        "http://www.w3.org/2004/02/skos/core#Concept", base
    ) == base
    # owl:Thing -> base (existing behavior preserved).
    assert _derive_namespace_from_parent_iri(
        "http://www.w3.org/2002/07/owl#Thing", base
    ) == base
    # Domain ontology (geography) -> inherit parent's namespace.
    assert _derive_namespace_from_parent_iri(
        "https://veerla-ramrao.ai/ontology/geography#Country", base
    ) == "https://veerla-ramrao.ai/ontology/geography#"


# ============================================================================
# Fix C: force-convert person-shape classes parented under
# Person/PersonRole/Role to instances of foaf:Person, overriding any LLM
# decision. Catches Elon-Musk-style entities the audit LLM marked KEEP.
# ============================================================================


def test_force_convert_overrides_llm_keep_for_person_under_person_class() -> None:
    """LLM says KEEP for Elon Musk under Person. The force-convert
    pre-check overrides to CONVERT_TO_INSTANCE."""
    from backend.app.services.pipeline_llm import _apply_audit_decision

    classes = {
        "https://x/Person": {
            "iri": "https://x/Person",
            "name": "Person",
            "labels": ["Person"],
            "superclasses": [],
        },
        "https://x/elon_musk": {
            "iri": "https://x/elon_musk",
            "name": "elon_musk",
            "labels": ["Elon Musk"],
            "superclasses": [{"iri": "https://x/Person", "name": "Person"}],
        },
    }
    rec = classes["https://x/elon_musk"]
    instances: dict = {}
    outcome = _apply_audit_decision(
        "https://x/elon_musk", rec,
        # LLM says KEEP -- but the deterministic guard should override.
        {"LABEL": "Elon Musk", "ACTION": "KEEP"},
        classes, instances,
        "https://veerla-ramrao.ai/ontology/merged#",
        current_parent_label="Person",
    )
    assert outcome == "converted"
    assert "https://x/elon_musk" not in classes
    assert "https://x/elon_musk" in instances


def test_force_convert_under_personorrole_label() -> None:
    """Same override for entities parented under the awkward
    'PersonOrRole' label."""
    from backend.app.services.pipeline_llm import _apply_audit_decision

    classes = {
        "https://x/personorrole": {
            "iri": "https://x/personorrole",
            "name": "PersonOrRole",
            "labels": ["PersonOrRole"],
            "superclasses": [],
        },
        "https://x/trump": {
            "iri": "https://x/trump",
            "name": "trump",
            "labels": ["Donald Trump"],
            "superclasses": [{"iri": "https://x/personorrole", "name": "PersonOrRole"}],
        },
    }
    rec = classes["https://x/trump"]
    instances: dict = {}
    outcome = _apply_audit_decision(
        "https://x/trump", rec,
        {"LABEL": "Donald Trump", "ACTION": "KEEP"},
        classes, instances,
        "https://veerla-ramrao.ai/ontology/merged#",
        current_parent_label="PersonOrRole",
    )
    assert outcome == "converted"
    assert "https://x/trump" in instances


def test_force_convert_does_not_fire_for_non_person_parent() -> None:
    """Person-shape labels under non-person parents (e.g. Organization)
    are NOT auto-converted -- the LLM verdict stands."""
    from backend.app.services.pipeline_llm import _apply_audit_decision

    classes = {
        "https://x/Organization": {
            "iri": "https://x/Organization",
            "name": "Organization",
            "labels": ["Organization"],
            "superclasses": [],
        },
        "https://x/coromandel": {
            "iri": "https://x/coromandel",
            "name": "coromandel",
            "labels": ["Coromandel International"],
            "superclasses": [{"iri": "https://x/Organization", "name": "Organization"}],
        },
    }
    rec = classes["https://x/coromandel"]
    instances: dict = {}
    outcome = _apply_audit_decision(
        "https://x/coromandel", rec,
        {"LABEL": "Coromandel International", "ACTION": "KEEP"},
        classes, instances,
        "https://veerla-ramrao.ai/ontology/merged#",
        current_parent_label="Organization",
    )
    # Stays a class (KEEP); no force-convert.
    assert outcome == "kept"
    assert "https://x/coromandel" in classes
    assert "https://x/coromandel" not in instances


# ============================================================================
# repair-output deterministic passes: FOAF cleanup + brand rewrite +
# person-convert. Verifies each pass independently against a tiny fixture.
# ============================================================================


def test_repair_output_moves_foaf_squatters_to_project_namespace(tmp_path) -> None:
    """A class squatted at http://xmlns.com/foaf/0.1/archerdanielsmidland
    gets moved to default_base_iri/archerdanielsmidland. Canonical FOAF
    classes (foaf:Person, foaf:Organization, etc.) are left alone."""
    import asyncio
    from backend.app.services.pipeline import run_repair_output
    from backend.app.services import folder_io

    folder = tmp_path / "v_test"
    folder.mkdir()
    loaded = {
        "classes_dict": {
            # Canonical FOAF -- must stay.
            "http://xmlns.com/foaf/0.1/Person": {
                "iri": "http://xmlns.com/foaf/0.1/Person",
                "name": "Person",
                "labels": ["Person"],
                "superclasses": [],
            },
            # Squatter -- must move.
            "http://xmlns.com/foaf/0.1/archerdanielsmidland": {
                "iri": "http://xmlns.com/foaf/0.1/archerdanielsmidland",
                "name": "archerdanielsmidland",
                "labels": ["Archer Daniels Midland"],
                "superclasses": [{
                    "iri": "http://xmlns.com/foaf/0.1/Person",
                    "name": "Person",
                }],
            },
        },
        "object_properties_dict": {},
        "data_properties_dict": {},
        "instances_dict": {},
    }
    folder_io.write_merged_json(folder, loaded)
    # write_owl needs a target -- use a dummy
    (folder / "merged.owl").write_text(
        '<?xml version="1.0"?><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"/>',
        encoding="utf-8",
    )

    asyncio.run(run_repair_output(
        input_folder=folder,
        do_foaf_cleanup=True,
        do_rebrand=False,
        do_person_convert=False,
    ))

    after = folder_io.load_version_folder(folder)
    classes = after["classes_dict"]
    # Canonical FOAF intact.
    assert "http://xmlns.com/foaf/0.1/Person" in classes
    # Squatter moved out of FOAF.
    foaf_squatters = [
        iri for iri in classes
        if iri.startswith("http://xmlns.com/foaf/0.1/")
        and iri.rsplit("/", 1)[-1] not in {
            "Agent", "Person", "Organization", "Group", "Document", "Image",
            "OnlineAccount", "OnlineChatAccount", "OnlineEcommerceAccount",
            "OnlineGamingAccount", "PersonalProfileDocument", "LabelProperty",
            "Project",
        }
    ]
    assert foaf_squatters == [], (
        f"FOAF squatters remained after cleanup: {foaf_squatters}"
    )


def test_repair_output_rebrands_local_iris_to_veerla_ramrao_ai(tmp_path) -> None:
    """Every class IRI starting with the old .local placeholder is
    rewritten to veerla-ramrao.ai. Cross-references update too."""
    import asyncio
    from backend.app.services.pipeline import run_repair_output
    from backend.app.services import folder_io

    folder = tmp_path / "v_test"
    folder.mkdir()
    loaded = {
        "classes_dict": {
            "http://your-personal-ontologist.local/ontology/widget": {
                "iri": "http://your-personal-ontologist.local/ontology/widget",
                "name": "widget",
                "labels": ["Widget"],
                "superclasses": [{
                    "iri": "http://your-personal-ontologist.local/ontology/thing",
                    "name": "thing",
                }],
            },
            "http://your-personal-ontologist.local/ontology/thing": {
                "iri": "http://your-personal-ontologist.local/ontology/thing",
                "name": "thing",
                "labels": ["Thing"],
                "superclasses": [],
            },
        },
        "object_properties_dict": {},
        "data_properties_dict": {},
        "instances_dict": {},
    }
    folder_io.write_merged_json(folder, loaded)
    (folder / "merged.owl").write_text(
        '<?xml version="1.0"?><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"/>',
        encoding="utf-8",
    )

    asyncio.run(run_repair_output(
        input_folder=folder,
        do_foaf_cleanup=False,
        do_rebrand=True,
        do_person_convert=False,
    ))

    after = folder_io.load_version_folder(folder)
    classes = after["classes_dict"]
    # No .local IRIs left.
    leftover = [iri for iri in classes if "your-personal-ontologist.local" in iri]
    assert leftover == [], f"leftover .local IRIs: {leftover}"
    # New IRIs present.
    veerla_keys = [iri for iri in classes if "veerla-ramrao.ai" in iri]
    assert len(veerla_keys) == 2
    # Cross-reference updated (widget's parent points at the new thing IRI).
    widget_iri = next(iri for iri in classes if iri.endswith("widget"))
    parent_iri = classes[widget_iri]["superclasses"][0]["iri"]
    assert "veerla-ramrao.ai" in parent_iri
    assert "thing" in parent_iri
