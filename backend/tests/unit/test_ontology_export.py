"""OWL export — round-trip a small synthetic dict-of-dicts through rdflib."""

from __future__ import annotations

from pathlib import Path

import rdflib
from rdflib.namespace import OWL, RDF, RDFS

from backend.app.services import ontology_export


def _tiny_loaded() -> dict:
    return {
        "classes_dict": {
            "http://x/A": {
                "iri": "http://x/A",
                "name": "A",
                "labels": ["A"],
                "comments": ["the A class"],
                "superclasses": [],
                "raw_axiom_triples": [],
            },
            "http://x/B": {
                "iri": "http://x/B",
                "name": "B",
                "labels": ["B"],
                "comments": [],
                "superclasses": [{"kind": "entity", "iri": "http://x/A", "name": "A"}],
                "raw_axiom_triples": [],
            },
        },
        "object_properties_dict": {
            "http://x/hasThing": {
                "iri": "http://x/hasThing",
                "name": "hasThing",
                "labels": ["hasThing"],
                "domain": [{"kind": "entity", "iri": "http://x/A", "name": "A"}],
                "range": [{"kind": "entity", "iri": "http://x/B", "name": "B"}],
                "characteristics": {"transitive": True},
            },
        },
        "data_properties_dict": {
            "http://x/score": {
                "iri": "http://x/score",
                "name": "score",
                "labels": ["score"],
                "domain": [{"kind": "entity", "iri": "http://x/A", "name": "A"}],
            },
        },
        "instances_dict": {
            "http://x/a1": {
                "iri": "http://x/a1",
                "name": "a1",
                "labels": ["a1"],
                "classes": [{"kind": "entity", "iri": "http://x/A", "name": "A"}],
            },
        },
    }


def test_round_trip_through_rdflib(tmp_path: Path) -> None:
    out = tmp_path / "merged.owl"
    ontology_export.write_owl(_tiny_loaded(), out)
    assert out.exists()
    g = rdflib.Graph()
    g.parse(str(out), format="xml")

    # Class declarations
    classes = set(g.subjects(RDF.type, OWL.Class))
    assert rdflib.URIRef("http://x/A") in classes
    assert rdflib.URIRef("http://x/B") in classes

    # subClassOf B -> A
    assert (rdflib.URIRef("http://x/B"), RDFS.subClassOf, rdflib.URIRef("http://x/A")) in g

    # Object property + transitive characteristic
    obj_props = set(g.subjects(RDF.type, OWL.ObjectProperty))
    assert rdflib.URIRef("http://x/hasThing") in obj_props
    transitives = set(g.subjects(RDF.type, OWL.TransitiveProperty))
    assert rdflib.URIRef("http://x/hasThing") in transitives

    # Data property
    data_props = set(g.subjects(RDF.type, OWL.DatatypeProperty))
    assert rdflib.URIRef("http://x/score") in data_props

    # Individual a1 typed to A
    individuals = set(g.subjects(RDF.type, OWL.NamedIndividual))
    assert rdflib.URIRef("http://x/a1") in individuals
    assert (rdflib.URIRef("http://x/a1"), RDF.type, rdflib.URIRef("http://x/A")) in g


def test_streaming_export_on_1000_classes(tmp_path: Path) -> None:
    """Regression test for the OOM-during-write_owl bug. Verifies the
    streaming export produces a syntactically valid RDF/XML file with
    the correct class count when the input ontology has ~1000 classes.
    Each class declares a subClassOf to a 'root' and a raw_axiom_triple
    so the export touches the same code paths as the HP/FIBO scale runs.
    """
    base = "http://example.org/big/"
    root_iri = base + "Root"
    classes = {
        root_iri: {
            "iri": root_iri,
            "name": "Root",
            "labels": ["Root"],
            "comments": [],
            "descriptions": [],
            "superclasses": [],
            "raw_axiom_triples": [],
        }
    }
    n = 1000
    for i in range(n):
        iri = f"{base}C{i:04d}"
        classes[iri] = {
            "iri": iri,
            "name": f"C{i:04d}",
            "labels": [f"Class {i}"],
            "comments": [],
            "descriptions": [f"Auto-generated class number {i}."],
            "superclasses": [{"kind": "entity", "iri": root_iri, "name": "Root"}],
            "raw_axiom_triples": [
                {"predicate": "http://example.org/ann/seq",
                 "object": {"kind": "literal", "value": i, "datatype": "http://www.w3.org/2001/XMLSchema#integer", "lang": None}},
            ],
        }
    loaded = {
        "classes_dict": classes,
        "object_properties_dict": {},
        "data_properties_dict": {},
        "instances_dict": {},
    }
    out = tmp_path / "big.owl"
    ontology_export.write_owl(loaded, out)
    assert out.exists() and out.stat().st_size > 0

    g = rdflib.Graph()
    g.parse(str(out), format="xml")
    declared = set(g.subjects(RDF.type, OWL.Class))
    # All N + 1 (root) classes round-trip.
    assert len(declared) == n + 1
    assert rdflib.URIRef(root_iri) in declared
    # Spot-check one subClassOf link.
    assert (rdflib.URIRef(f"{base}C0042"), RDFS.subClassOf, rdflib.URIRef(root_iri)) in g
    # Raw-triple replay survived.
    assert any(
        str(p) == "http://example.org/ann/seq"
        for _, p, _ in g.triples((rdflib.URIRef(f"{base}C0042"), None, None))
    )
