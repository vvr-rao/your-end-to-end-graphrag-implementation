"""extract_ontology_to_dicts() must keep per-call state out of the global
owlready2 default_world when an isolated World is passed in.

This is the load-bearing fix for the multi-file merge hang: large bundles
(FIBO ~100 .rdf, OntoCAPE ~64 .owl) accumulate triples in default_world
across sequential extract calls, making each per-call rdf_graph snapshot
grow O(N) and total work O(N^2). Per-file Worlds keep snapshots bounded.
"""

from __future__ import annotations

from pathlib import Path

from owlready2 import World

from backend.app.helpers.ontology_parsing import extract_ontology_to_dicts

REPO_ROOT = Path(__file__).resolve().parents[3]
SKOS = REPO_ROOT / "source_ontologies" / "core_ontologies" / "skos.rdf"
INDEX = REPO_ROOT / "source_ontologies" / "general_ontologies" / "index.rdf"


def _class_iris(ontology_dict: dict) -> set[str]:
    return set(ontology_dict["classes_dict"].keys())


def test_isolated_worlds_do_not_bleed_entities() -> None:
    """Loading two different ontologies into two separate Worlds must produce
    disjoint class sets -- i.e. World A's load doesn't leak triples into
    World B's rdf_graph snapshot."""
    assert SKOS.exists(), f"missing test fixture: {SKOS}"
    assert INDEX.exists(), f"missing test fixture: {INDEX}"

    world_a = World()
    world_b = World()

    dict_a = extract_ontology_to_dicts(
        str(SKOS),
        load_imported=False,
        local_only=True,
        local_ontology_dir=str(SKOS.parent),
        world=world_a,
    )
    dict_b = extract_ontology_to_dicts(
        str(INDEX),
        load_imported=False,
        local_only=True,
        local_ontology_dir=str(INDEX.parent),
        world=world_b,
    )

    iris_a = _class_iris(dict_a)
    iris_b = _class_iris(dict_b)

    # Both files must have parsed at least one class.
    assert iris_a, "skos.rdf produced no classes"

    # The two sets must be disjoint -- World A's classes must not appear in
    # World B's extraction, and vice versa. If they overlap, a class from one
    # file leaked into the other's rdf_graph snapshot.
    overlap = iris_a & iris_b
    assert not overlap, (
        f"World isolation broken: {len(overlap)} class IRI(s) shared "
        f"between separate Worlds. Example: {next(iter(overlap))}"
    )


def test_reloading_same_file_in_fresh_world_returns_same_classes() -> None:
    """Loading the same file twice into two fresh Worlds must yield the same
    class set. Proves the per-file isolation is stable, not order-dependent."""
    assert SKOS.exists()

    first = extract_ontology_to_dicts(
        str(SKOS),
        load_imported=False,
        local_only=True,
        local_ontology_dir=str(SKOS.parent),
        world=World(),
    )
    second = extract_ontology_to_dicts(
        str(SKOS),
        load_imported=False,
        local_only=True,
        local_ontology_dir=str(SKOS.parent),
        world=World(),
    )
    assert _class_iris(first) == _class_iris(second)
