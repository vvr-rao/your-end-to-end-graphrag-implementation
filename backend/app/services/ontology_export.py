"""Write a loaded ontology dict-of-dicts to a Protégé-readable .owl file.

Extends reference/pickle_to_owl.py to be lossless across the parts that
`extract_ontology_to_dicts` actually preserves:
  - classes with labels/comments, subclass-of relationships, equivalence,
    and disjointness
  - object properties + data properties with domain/range and characteristics
    (Transitive/Symmetric/Functional/etc.)
  - individuals with type assertions
  - raw_axiom_triples replay as a fallback so anything we did not handle
    structurally still shows up in the output (Protégé will read them)

Output format: RDF/XML (the conventional .owl serialization).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rdflib import OWL, RDF, RDFS, XSD, Graph, Literal, Namespace, URIRef

DEFAULT_ONTOLOGY_IRI = "http://your-personal-ontologist.local/ontology/merged"


def _uri(value: str | dict | None) -> URIRef | None:
    """Coerce a string IRI or an entity_ref dict {'iri': ..., 'name': ...} to a URIRef."""
    if value is None:
        return None
    if isinstance(value, dict):
        iri = value.get("iri") or value.get("IRI")
        if not iri:
            return None
        return URIRef(iri)
    if isinstance(value, str) and value.strip():
        return URIRef(value)
    return None


def _add_literals(graph: Graph, subject: URIRef, predicate: URIRef, values: Any) -> None:
    if values is None:
        return
    if isinstance(values, (str, int, float, bool)):
        text = str(values).strip()
        if text:
            graph.add((subject, predicate, Literal(text)))
        return
    if isinstance(values, list):
        for v in values:
            _add_literals(graph, subject, predicate, v)


def _add_annotations(graph: Graph, subject: URIRef, record: dict[str, Any]) -> None:
    _add_literals(graph, subject, RDFS.label, record.get("labels"))
    _add_literals(graph, subject, RDFS.label, record.get("name"))
    _add_literals(graph, subject, RDFS.comment, record.get("comments"))
    _add_literals(graph, subject, RDFS.comment, record.get("descriptions"))

    # Custom annotation properties (e.g. semantic_role, review_status, sources, generated)
    for ann_pred in ("semantic_role", "review_status", "generated", "sources"):
        if ann_pred in record and record[ann_pred] is not None:
            pred_uri = URIRef(f"http://your-personal-ontologist.local/ann#{ann_pred}")
            value = record[ann_pred]
            if isinstance(value, list):
                for v in value:
                    graph.add((subject, pred_uri, Literal(str(v), datatype=XSD.string)))
            else:
                graph.add((subject, pred_uri, Literal(str(value), datatype=XSD.string)))


def _add_typed(graph: Graph, subject: URIRef, type_uri: URIRef) -> None:
    graph.add((subject, RDF.type, type_uri))


def _add_subclass_of(graph: Graph, class_uri: URIRef, superclasses: Any) -> None:
    if not isinstance(superclasses, list):
        return
    for parent in superclasses:
        parent_uri = _uri(parent)
        if parent_uri is not None:
            graph.add((class_uri, RDFS.subClassOf, parent_uri))


def _add_equivalence_and_disjoint(graph: Graph, class_uri: URIRef, record: dict[str, Any]) -> None:
    for eq in record.get("equivalent_to") or []:
        eq_uri = _uri(eq)
        if eq_uri is not None:
            graph.add((class_uri, OWL.equivalentClass, eq_uri))
    for dj in record.get("disjoints") or []:
        dj_uri = _uri(dj)
        if dj_uri is not None:
            graph.add((class_uri, OWL.disjointWith, dj_uri))


def _add_property_domain_range(graph: Graph, prop_uri: URIRef, record: dict[str, Any]) -> None:
    for domain in record.get("domain") or []:
        d_uri = _uri(domain)
        if d_uri is not None:
            graph.add((prop_uri, RDFS.domain, d_uri))
    for range_ in record.get("range") or []:
        r_uri = _uri(range_)
        if r_uri is not None:
            graph.add((prop_uri, RDFS.range, r_uri))


def _add_property_characteristics(graph: Graph, prop_uri: URIRef, record: dict[str, Any]) -> None:
    chars = record.get("characteristics") or {}
    flag_to_type = {
        "functional": OWL.FunctionalProperty,
        "inverse_functional": OWL.InverseFunctionalProperty,
        "transitive": OWL.TransitiveProperty,
        "symmetric": OWL.SymmetricProperty,
        "asymmetric": OWL.AsymmetricProperty,
        "reflexive": OWL.ReflexiveProperty,
        "irreflexive": OWL.IrreflexiveProperty,
    }
    for flag, type_uri in flag_to_type.items():
        if chars.get(flag):
            graph.add((prop_uri, RDF.type, type_uri))


def _replay_raw_triples(graph: Graph, subject: URIRef, raw_triples: Any) -> None:
    """Replay raw triples the extractor preserved as a fallback for constructs
    we don't emit via the named pathways. Each entry is shaped as
        {"predicate": <iri_str>, "object": <entity_ref()_dict_or_primitive>}
    where the object dict has a "kind" field (uri/literal/bnode/entity/...).

    Skip predicates we already emitted to avoid duplication.
    """
    if not isinstance(raw_triples, list):
        return
    skip_predicates = {
        str(RDF.type),
        str(RDFS.label),
        str(RDFS.comment),
        str(RDFS.subClassOf),
        str(RDFS.domain),
        str(RDFS.range),
        str(OWL.equivalentClass),
        str(OWL.disjointWith),
        str(OWL.inverseOf),
        str(RDFS.subPropertyOf),
    }
    for triple in raw_triples:
        if not isinstance(triple, dict):
            continue
        pred = triple.get("predicate")
        obj = triple.get("object")
        if not pred or obj is None:
            continue
        if pred in skip_predicates:
            continue

        # The object is either a primitive (str/int/float/bool) or an
        # entity_ref() dict — branch on shape.
        if isinstance(obj, dict):
            kind = obj.get("kind")
            if kind == "literal":
                value = obj.get("value")
                if value is None:
                    continue
                datatype = obj.get("datatype")
                lang = obj.get("lang")
                if datatype:
                    graph.add((subject, URIRef(pred), Literal(str(value), datatype=URIRef(datatype))))
                elif lang:
                    graph.add((subject, URIRef(pred), Literal(str(value), lang=lang)))
                else:
                    graph.add((subject, URIRef(pred), Literal(str(value))))
            elif kind in ("uri", "entity"):
                iri = obj.get("iri")
                if iri:
                    graph.add((subject, URIRef(pred), URIRef(str(iri))))
            # bnode / python_type / repr: skip — not round-trippable as-is.
        elif isinstance(obj, (str, int, float, bool)):
            # Primitive from extract_*; assume literal.
            text = str(obj).strip()
            if text:
                graph.add((subject, URIRef(pred), Literal(text)))


def build_owl_graph(
    loaded_ontology: dict[str, Any],
    ontology_iri: str = DEFAULT_ONTOLOGY_IRI,
) -> Graph:
    """Convert the loaded dict-of-dicts to an rdflib Graph ready for RDF/XML
    serialization. Lossless across the parts the parser preserves."""
    graph = Graph()
    ont = Namespace(f"{ontology_iri.rstrip('#/')}/")
    graph.bind("owl", OWL)
    graph.bind("rdf", RDF)
    graph.bind("rdfs", RDFS)
    graph.bind("xsd", XSD)
    graph.bind("ont", ont)

    onto_ref = URIRef(ontology_iri)
    graph.add((onto_ref, RDF.type, OWL.Ontology))

    classes_dict = loaded_ontology.get("classes_dict", {})
    object_properties_dict = loaded_ontology.get("object_properties_dict", {})
    data_properties_dict = loaded_ontology.get("data_properties_dict", {})
    instances_dict = loaded_ontology.get("instances_dict", {})

    for class_iri, record in classes_dict.items():
        cls_uri = _uri(class_iri)
        if cls_uri is None:
            continue
        _add_typed(graph, cls_uri, OWL.Class)
        _add_annotations(graph, cls_uri, record)
        _add_subclass_of(graph, cls_uri, record.get("superclasses"))
        _add_equivalence_and_disjoint(graph, cls_uri, record)
        _replay_raw_triples(graph, cls_uri, record.get("raw_axiom_triples"))

    for prop_iri, record in object_properties_dict.items():
        prop_uri = _uri(prop_iri)
        if prop_uri is None:
            continue
        _add_typed(graph, prop_uri, OWL.ObjectProperty)
        _add_annotations(graph, prop_uri, record)
        _add_property_domain_range(graph, prop_uri, record)
        _add_property_characteristics(graph, prop_uri, record)
        for sp in record.get("superproperties") or []:
            sp_uri = _uri(sp)
            if sp_uri is not None:
                graph.add((prop_uri, RDFS.subPropertyOf, sp_uri))
        inv = _uri(record.get("inverse_property"))
        if inv is not None:
            graph.add((prop_uri, OWL.inverseOf, inv))
        _replay_raw_triples(graph, prop_uri, record.get("raw_axiom_triples"))

    for prop_iri, record in data_properties_dict.items():
        prop_uri = _uri(prop_iri)
        if prop_uri is None:
            continue
        _add_typed(graph, prop_uri, OWL.DatatypeProperty)
        _add_annotations(graph, prop_uri, record)
        _add_property_domain_range(graph, prop_uri, record)
        _add_property_characteristics(graph, prop_uri, record)
        for sp in record.get("superproperties") or []:
            sp_uri = _uri(sp)
            if sp_uri is not None:
                graph.add((prop_uri, RDFS.subPropertyOf, sp_uri))
        _replay_raw_triples(graph, prop_uri, record.get("raw_axiom_triples"))

    for inst_iri, record in instances_dict.items():
        inst_uri = _uri(inst_iri)
        if inst_uri is None:
            continue
        _add_typed(graph, inst_uri, OWL.NamedIndividual)
        _add_annotations(graph, inst_uri, record)
        for parent in record.get("classes") or record.get("types") or []:
            parent_uri = _uri(parent)
            if parent_uri is not None:
                graph.add((inst_uri, RDF.type, parent_uri))
        _replay_raw_triples(graph, inst_uri, record.get("raw_axiom_triples"))

    return graph


def write_owl(loaded_ontology: dict[str, Any], output_path: Path, ontology_iri: str = DEFAULT_ONTOLOGY_IRI) -> Path:
    """Serialize a loaded ontology to RDF/XML at output_path."""
    graph = build_owl_graph(loaded_ontology, ontology_iri=ontology_iri)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    graph.serialize(destination=str(output_path), format="xml")
    return output_path
