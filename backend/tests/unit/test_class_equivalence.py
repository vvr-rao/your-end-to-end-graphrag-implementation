"""Named owl:equivalentClass must survive loading, and the type check must use it.

The W3C ORG ontology declares `org:Organization owl:equivalentClass
foaf:Organization`. owlready2 cannot load a cross-ontology named equivalence --
it raises `TypeError: issubclass() arg 1 must be a class` -- so the Turtle loader
stripped every owl:equivalentClass triple. With the equivalence gone the two
classes became unrelated siblings under `foaf:Agent`, and relationship
type-checking rejected valid edges. Measured on a 40-document news build:

    `Caroline Ellison --controls--> Alameda Research LLC`  -> DOMAIN_RANGE
    (`controls` is declared Person -> foaf:Organization;
     Alameda Research LLC was typed org:Organization)

with 81% of person-organization co-mentions involving `org:Organization` while
140 of 170 organization predicates declare `foaf:Organization`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ORG = "http://www.w3.org/ns/org#Organization"
FOAF = "http://xmlns.com/foaf/0.1/Organization"
CORE = Path("source_ontologies/core_ontologies")


def _load(*names: str) -> dict:
    from backend.app.services.ontology_io import enumerate_inputs, load_ontology_files

    bundle = enumerate_inputs([CORE / n for n in names])
    with bundle:
        return load_ontology_files(bundle.sources)["classes_dict"]


@pytest.mark.skipif(not (CORE / "org.ttl").exists(), reason="core ontologies not present")
def test_named_equivalence_survives_the_turtle_loader() -> None:
    cd = _load("org.ttl", "foaf.rdf")
    eq = [e.get("iri") for e in cd[ORG].get("equivalent_to") or []]
    assert FOAF in eq


@pytest.mark.skipif(not (CORE / "org.ttl").exists(), reason="core ontologies not present")
def test_loading_still_does_not_crash_owlready2_and_keeps_superclasses() -> None:
    """The reason the triples were stripped must still be honoured: owlready2 is
    never handed an owl:equivalentClass it cannot model."""
    cd = _load("org.ttl", "foaf.rdf")
    assert ORG in cd
    assert "http://xmlns.com/foaf/0.1/Agent" in [
        s.get("iri") for s in cd[ORG].get("superclasses") or []
    ]


@pytest.mark.skipif(not (CORE / "org.ttl").exists(), reason="core ontologies not present")
def test_anonymous_equivalences_are_still_dropped() -> None:
    """org.ttl also has a blank-node `owl:equivalentClass [ owl:intersectionOf ... ]`.
    It carries no class identity, and must not leak in as an empty or bnode entry."""
    cd = _load("org.ttl", "foaf.rdf")
    for rec in cd.values():
        for e in rec.get("equivalent_to") or []:
            iri = e.get("iri") if isinstance(e, dict) else None
            assert iri and not str(iri).startswith("_:"), e


def test_ancestor_closure_crosses_equivalence_in_both_directions() -> None:
    """An equivalent class is the same class: an ancestor AND a descendant.
    UNION rather than UNION ALL keeps the symmetric edge from looping."""
    from backend.app.services.db_entity_extract import _ANCESTOR_SQL, _DESCENDANT_SQL

    for sql in (str(_ANCESTOR_SQL), str(_DESCENDANT_SQL)):
        assert "owl:equivalentClass" in sql
        assert "rdfs:subClassOf" in sql, "subClassOf must still be walked"
        assert "gr.source_node_id = " in sql and "gr.target_node_id = " in sql
        assert "UNION\n" in sql and "UNION ALL" not in sql
