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

import re
from pathlib import Path
from typing import Any

from rdflib import OWL, RDF, RDFS, XSD, Graph, Literal, Namespace, URIRef

DEFAULT_ONTOLOGY_IRI = "http://your-personal-ontologist.local/ontology/merged"

# XML 1.0 forbids C0 controls (except \t \n \r) and \x7f anywhere in the
# document, in attribute values or text content. PDF text extraction
# occasionally emits these (Bol\x13var from a corrupted ASCII range, etc.),
# and Stage 2 summaries can carry them through into labels / comments. If
# they reach rdflib's serializer the resulting .owl file looks valid but
# any XML 1.0 parser (Protégé, lxml, ElementTree) rejects it with
# "not well-formed (invalid token)". Strip them at the boundary.
_BAD_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _xml_safe(text: str) -> str:
    """Strip C0 control characters that XML 1.0 forbids."""
    return _BAD_CONTROL_CHARS.sub("", text)


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
        text = _xml_safe(str(values)).strip()
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
                    graph.add((subject, pred_uri, Literal(_xml_safe(str(v)), datatype=XSD.string)))
            else:
                graph.add((subject, pred_uri, Literal(_xml_safe(str(value)), datatype=XSD.string)))


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
                safe_value = _xml_safe(str(value))
                if datatype:
                    graph.add((subject, URIRef(pred), Literal(safe_value, datatype=URIRef(datatype))))
                elif lang:
                    graph.add((subject, URIRef(pred), Literal(safe_value, lang=lang)))
                else:
                    graph.add((subject, URIRef(pred), Literal(safe_value)))
            elif kind in ("uri", "entity"):
                iri = obj.get("iri")
                if iri:
                    graph.add((subject, URIRef(pred), URIRef(str(iri))))
            # bnode / python_type / repr: skip — not round-trippable as-is.
        elif isinstance(obj, (str, int, float, bool)):
            # Primitive from extract_*; assume literal.
            text = _xml_safe(str(obj)).strip()
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


def write_owl(
    loaded_ontology: dict[str, Any],
    output_path: Path,
    ontology_iri: str = DEFAULT_ONTOLOGY_IRI,
    *,
    consume_dict: bool = False,
) -> Path:
    """Serialize a loaded ontology to RDF/XML at output_path.

    Uses streaming emission (one entity at a time) so the peak in-memory
    rdflib Graph stays bounded -- previous behaviour built the FULL graph
    in memory before serializing, which OOM-killed HP-scale merges on
    machines with < 4 GiB available RAM.

    Pass `consume_dict=True` when the caller is done with the loaded
    ontology -- entries are popped from the input dict as they're emitted
    so the dict's memory footprint shrinks during the run. Used by
    `pipeline.run_merge` after `write_merged_json` has already persisted
    the data.
    """
    return write_owl_streaming(
        loaded_ontology,
        output_path,
        ontology_iri=ontology_iri,
        consume_dict=consume_dict,
    )


# Stable XML header used by the streaming writer. The prefixes here must
# cover every URIRef we emit inside per-entity fragments (rdflib generates
# `<rdf:Description ...>` blocks using these prefixes).
_STREAMING_HEADER_TPL = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    '<rdf:RDF\n'
    '    xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"\n'
    '    xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"\n'
    '    xmlns:owl="http://www.w3.org/2002/07/owl#"\n'
    '    xmlns:xsd="http://www.w3.org/2001/XMLSchema#"\n'
    '    xmlns:ont="{ontology_iri_slash}">\n'
    '  <owl:Ontology rdf:about="{ontology_iri}"/>\n'
)


# Prefixes that are already declared on the outer streaming <rdf:RDF> tag.
# Any others from per-entity rdflib fragments (rdflib auto-coins ns1, ns2, ...
# for unknown predicate namespaces in raw_axiom_triples) must be re-inlined
# onto the fragment's outer element so they stay in scope after we drop the
# fragment's own <rdf:RDF> wrapper.
_HEADER_PREFIX_NAMES = {"xmlns:rdf", "xmlns:rdfs", "xmlns:owl", "xmlns:xsd", "xmlns:ont"}
_XMLNS_RE = re.compile(r'xmlns:[\w-]+="[^"]*"')


def _strip_rdf_wrapper(xml: str) -> str:
    """Return everything between the opening `<rdf:RDF ...>` tag and the
    closing `</rdf:RDF>` from a per-entity rdflib fragment. Any xmlns
    declarations present on the fragment's wrapper that aren't in the
    streaming header are inlined onto the fragment's outer element so
    references like `ns1:foo` stay resolvable."""
    open_idx = xml.find("<rdf:RDF")
    if open_idx == -1:
        return xml
    open_tag_end = xml.find(">", open_idx)
    close_tag_start = xml.rfind("</rdf:RDF>")
    if open_tag_end == -1 or close_tag_start == -1:
        return xml

    wrapper_tag = xml[open_idx : open_tag_end + 1]
    body = xml[open_tag_end + 1 : close_tag_start].strip("\n")
    if not body:
        return body

    # Pull every xmlns:foo="..." from the fragment's wrapper tag; keep only
    # the ones the streaming header doesn't already declare.
    extras = [
        decl
        for decl in _XMLNS_RE.findall(wrapper_tag)
        if decl.split("=", 1)[0] not in _HEADER_PREFIX_NAMES
    ]
    if not extras:
        return body

    # Inject those xmlns declarations into the FIRST open tag in the body
    # so the prefixes stay in scope for all of its descendants.
    first_open = body.find("<")
    if first_open == -1:
        return body
    first_close = body.find(">", first_open)
    if first_close == -1:
        return body
    first_tag = body[first_open : first_close + 1]
    inject = " " + " ".join(extras)
    if first_tag.endswith("/>"):
        new_tag = first_tag[:-2] + inject + "/>"
    else:
        new_tag = first_tag[:-1] + inject + ">"
    return body[:first_open] + new_tag + body[first_close + 1 :]


def _build_entity_graph(
    entity_iri: str,
    type_uri: URIRef,
    record: dict[str, Any],
    kind: str,
) -> Graph:
    """Build a tiny rdflib Graph holding only the triples for one entity.

    Mirrors the per-entity logic in `build_owl_graph` but in isolation so
    the caller can serialize each fragment and discard it before moving
    to the next entity.
    """
    g = Graph()
    g.bind("owl", OWL)
    g.bind("rdf", RDF)
    g.bind("rdfs", RDFS)
    g.bind("xsd", XSD)
    uri = _uri(entity_iri)
    if uri is None:
        return g
    _add_typed(g, uri, type_uri)
    _add_annotations(g, uri, record)
    if kind == "class":
        _add_subclass_of(g, uri, record.get("superclasses"))
        _add_equivalence_and_disjoint(g, uri, record)
    elif kind in ("object_property", "data_property"):
        _add_property_domain_range(g, uri, record)
        _add_property_characteristics(g, uri, record)
        for sp in record.get("superproperties") or []:
            sp_uri = _uri(sp)
            if sp_uri is not None:
                g.add((uri, RDFS.subPropertyOf, sp_uri))
        inv = _uri(record.get("inverse_property"))
        if inv is not None:
            g.add((uri, OWL.inverseOf, inv))
    elif kind == "instance":
        for parent in record.get("classes") or record.get("types") or []:
            parent_uri = _uri(parent)
            if parent_uri is not None:
                g.add((uri, RDF.type, parent_uri))
    _replay_raw_triples(g, uri, record.get("raw_axiom_triples"))
    return g


def write_owl_streaming(
    loaded_ontology: dict[str, Any],
    output_path: Path,
    ontology_iri: str = DEFAULT_ONTOLOGY_IRI,
    *,
    consume_dict: bool = False,
) -> Path:
    """Stream-serialize a loaded ontology to RDF/XML at output_path.

    Each class/property/instance is serialized through its own tiny
    rdflib.Graph, the wrapper is stripped, and the inner fragment is
    appended to the output file. This keeps memory bounded to one
    entity's triples regardless of total ontology size.

    When `consume_dict=True`, entries are popped from
    `loaded_ontology["..._dict"]` as they're emitted so the in-memory
    dict shrinks during the run. For HP-class merges (32 K entries +
    raw_axiom_triples ≈ 500 MB dict on a 2.7 GB machine) this is the
    difference between OOM and success -- the caller has already written
    `merged.json` and won't reuse the dict, so consuming it is safe.
    """
    import gc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    iri_slash = ontology_iri.rstrip("#/") + "/"
    with output_path.open("w", encoding="utf-8") as fh:
        fh.write(
            _STREAMING_HEADER_TPL.format(
                ontology_iri=ontology_iri,
                ontology_iri_slash=iri_slash,
            )
        )

        def _emit(entity_iri: str, type_uri: URIRef, record: dict[str, Any], kind: str) -> None:
            g = _build_entity_graph(entity_iri, type_uri, record, kind)
            if len(g) == 0:
                return
            xml = g.serialize(format="pretty-xml")
            body = _strip_rdf_wrapper(xml)
            if body:
                fh.write(body)
                fh.write("\n")

        def _drain(dict_name: str, type_uri: URIRef, kind: str) -> None:
            source = loaded_ontology.get(dict_name, {})
            if consume_dict:
                # Iterate by popping so each entity's Python objects can be
                # GC'd as we move on. Iterating by .items() would keep all
                # references alive in the dict until the loop ends.
                keys = list(source.keys())
                processed = 0
                for iri in keys:
                    record = source.pop(iri, None)
                    if record is None:
                        continue
                    _emit(iri, type_uri, record, kind)
                    del record
                    processed += 1
                    # Encourage GC every few thousand entities so RSS
                    # doesn't drift up over the run.
                    if processed % 4096 == 0:
                        gc.collect()
            else:
                for iri, record in source.items():
                    _emit(iri, type_uri, record, kind)

        _drain("classes_dict", OWL.Class, "class")
        _drain("object_properties_dict", OWL.ObjectProperty, "object_property")
        _drain("data_properties_dict", OWL.DatatypeProperty, "data_property")
        _drain("instances_dict", OWL.NamedIndividual, "instance")

        fh.write("</rdf:RDF>\n")
    return output_path
