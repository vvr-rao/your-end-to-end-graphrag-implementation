from __future__ import annotations

import os

from os.path import realpath
from owlready2 import PREDEFINED_ONTOLOGIES, get_ontology, default_world, Thing, onto_path

from typing import Any, Dict, List, Optional
from owlready2 import *
from rdflib import URIRef, BNode, Literal




# -----------------------------
# Utility helpers
# -----------------------------

OWLREADY_INTERNAL_ANNOT_PROPS = {
    "label",
    "comment",
    "python_name",
}

STRUCTURAL_PREDICATES_TO_SKIP = {
    "http://www.w3.org/2000/01/rdf-schema#subClassOf",
    "http://www.w3.org/2002/07/owl#equivalentClass",
    "http://www.w3.org/2000/01/rdf-schema#domain",
    "http://www.w3.org/2000/01/rdf-schema#range",
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
    "http://www.w3.org/2000/01/rdf-schema#subPropertyOf",
    "http://www.w3.org/2002/07/owl#inverseOf",
    "http://www.w3.org/2002/07/owl#sameAs",
}


def safe_iri(x: Any) -> Optional[str]:
    return getattr(x, "iri", None)


def safe_name(x: Any) -> Optional[str]:
    return getattr(x, "name", None)


def safe_namespace(x: Any) -> Optional[str]:
    ns = getattr(x, "namespace", None)
    if ns is None:
        return None
    return getattr(ns, "base_iri", str(ns))


def literal_to_python(x: Any) -> Any:
    if isinstance(x, Literal):
        return x.toPython()
    return x


def entity_ref(x: Any) -> Any:
    """
    Serialize a high-level Owlready2 entity or plain value into a JSON/dict-friendly form.
    """
    if x is None:
        return None

    if isinstance(x, (str, int, float, bool)):
        return x

    # rdflib nodes
    if isinstance(x, URIRef):
        return {"kind": "uri", "iri": str(x)}
    if isinstance(x, BNode):
        return {"kind": "bnode", "id": str(x)}
    if isinstance(x, Literal):
        return {
            "kind": "literal",
            "value": x.toPython(),
            "datatype": str(x.datatype) if x.datatype else None,
            "lang": x.language,
        }

    iri = getattr(x, "iri", None)
    if iri:
        return {
            "kind": "entity",
            "name": getattr(x, "name", None),
            "iri": iri,
            "python_name": getattr(x, "python_name", None),
            "type": type(x).__name__,
        }

    # Python datatypes commonly used as data property ranges
    if isinstance(x, type):
        return {
            "kind": "python_type",
            "name": x.__name__,
            "module": getattr(x, "__module__", None),
        }

    return {
        "kind": "repr",
        "value": repr(x),
        "type": type(x).__name__,
    }


def serialize_construct(expr: Any) -> Any:
    """
    Serialize OWL class expressions / restrictions / logical constructs as best as possible.
    """
    # Named entity / class / property / individual
    if hasattr(expr, "iri"):
        return entity_ref(expr)

    # Common Owlready2 restriction / construct objects expose attributes like:
    # .property, .type, .cardinality, .value, .Classes, .Class, etc.
    out = {
        "kind": type(expr).__name__,
        "repr": repr(expr),
    }

    if hasattr(expr, "property"):
        out["property"] = entity_ref(getattr(expr, "property"))
    if hasattr(expr, "type"):
        # Owlready2 uses numeric/enum restriction codes internally;
        # keep both raw and repr so you do not lose information.
        try:
            out["restriction_type"] = getattr(expr, "type")
        except Exception:
            pass
    if hasattr(expr, "cardinality"):
        out["cardinality"] = getattr(expr, "cardinality")
    if hasattr(expr, "value"):
        out["value"] = serialize_construct(getattr(expr, "value"))
    if hasattr(expr, "Classes"):
        out["classes"] = [serialize_construct(c) for c in getattr(expr, "Classes")]
    if hasattr(expr, "Class"):
        out["class"] = serialize_construct(getattr(expr, "Class"))

    return out


def get_labels(entity: Any) -> List[str]:
    try:
        return [str(x) for x in entity.label]
    except Exception:
        return []


def get_comments(entity: Any) -> List[str]:
    try:
        return [str(x) for x in entity.comment]
    except Exception:
        return []


def get_descriptions(entity: Any) -> List[str]:
    """
    Try common description-like annotations.
    """
    values = []

    for attr_name in ("comment", "definition", "hasDefinition", "IAO_0000115"):
        try:
            attr = getattr(entity, attr_name)
            if isinstance(attr, list):
                values.extend([str(v) for v in attr])
            elif attr is not None:
                values.append(str(attr))
        except Exception:
            continue

    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for v in values:
        if v not in seen:
            seen.add(v)
            deduped.append(v)
    return deduped


def collect_annotations(entity: Any, rdf_graph) -> Dict[str, List[Any]]:
    """
    Collect generic annotations directly from the RDF graph for the entity's IRI.
    This helps capture custom annotation properties not surfaced as friendly attributes.
    """
    annotations: Dict[str, List[Any]] = {}

    iri = safe_iri(entity)
    if not iri:
        return annotations

    subj = URIRef(iri)

    for pred, obj in rdf_graph.predicate_objects(subj):
        pred_str = str(pred)

        if pred_str in STRUCTURAL_PREDICATES_TO_SKIP:
            continue

        annotations.setdefault(pred_str, []).append(entity_ref(obj))

    return annotations


def collect_raw_axiom_triples(entity: Any, rdf_graph) -> List[Dict[str, Any]]:
    """
    Keep raw outgoing triples so you can preserve information not captured
    in the normalized fields. This is helpful for governance / round-tripping.
    """
    triples = []

    iri = safe_iri(entity)
    if not iri:
        return triples

    subj = URIRef(iri)

    for pred, obj in rdf_graph.predicate_objects(subj):
        triples.append(
            {
                "subject": iri,
                "predicate": str(pred),
                "object": entity_ref(obj),
            }
        )

    return triples


def serialize_disjoints(entity: Any) -> List[Any]:
    vals = []
    try:
        for d in entity.disjoints():
            vals.append(serialize_construct(d))
    except Exception:
        pass
    return vals


def serialize_differents(individual: Any) -> List[Any]:
    vals = []
    try:
        for d in individual.differents():
            vals.append(serialize_construct(d))
    except Exception:
        pass
    return vals


def property_characteristics(prop: Any) -> Dict[str, bool]:
    """
    Capture common OWL property characteristics.
    """
    ancestors = set()
    try:
        ancestors = set(prop.ancestors())
    except Exception:
        pass

    return {
        "functional": FunctionalProperty in ancestors,
        "inverse_functional": InverseFunctionalProperty in ancestors,
        "transitive": TransitiveProperty in ancestors,
        "symmetric": SymmetricProperty in ancestors,
        "asymmetric": AsymmetricProperty in ancestors,
        "reflexive": ReflexiveProperty in ancestors,
        "irreflexive": IrreflexiveProperty in ancestors,
    }


def serialize_named_vs_constructs(items: List[Any]) -> Dict[str, List[Any]]:
    named = []
    constructs = []

    for x in items:
        if hasattr(x, "iri"):
            named.append(entity_ref(x))
        else:
            constructs.append(serialize_construct(x))

    return {
        "named": named,
        "constructs": constructs,
    }


# -----------------------------
# Extractors
# -----------------------------

def extract_class(cls: Any, rdf_graph) -> Dict[str, Any]:
    isa_split = serialize_named_vs_constructs(list(getattr(cls, "is_a", [])))
    eq_split = serialize_named_vs_constructs(list(getattr(cls, "equivalent_to", [])))

    direct_instances = []
    try:
        direct_instances = [entity_ref(i) for i in cls.instances()]
    except Exception:
        pass

    return {
        "name": safe_name(cls),
        "iri": safe_iri(cls),
        "namespace": safe_namespace(cls),
        "labels": get_labels(cls),
        "comments": get_comments(cls),
        "descriptions": get_descriptions(cls),
        "annotations": collect_annotations(cls, rdf_graph),

        "superclasses": isa_split["named"],
        "restrictions_and_class_constructs": isa_split["constructs"],

        "equivalent_to": eq_split["named"],
        "equivalent_constructs": eq_split["constructs"],

        "disjoints": serialize_disjoints(cls),

        # Useful extra metadata
        "python_type": type(cls).__name__,
        "direct_instances": direct_instances,

        # Raw fallback for round-trip / governance
        "raw_axiom_triples": collect_raw_axiom_triples(cls, rdf_graph),
    }


def extract_property(prop: Any, rdf_graph, property_kind: str) -> Dict[str, Any]:
    isa_split = serialize_named_vs_constructs(list(getattr(prop, "is_a", [])))

    inverse_prop = None
    try:
        inverse_prop = entity_ref(prop.inverse_property) if prop.inverse_property else None
    except Exception:
        pass

    domains = []
    try:
        domains = [serialize_construct(x) for x in prop.domain]
    except Exception:
        pass

    ranges = []
    try:
        ranges = [serialize_construct(x) for x in prop.range]
    except Exception:
        pass

    return {
        "name": safe_name(prop),
        "iri": safe_iri(prop),
        "namespace": safe_namespace(prop),
        "labels": get_labels(prop),
        "comments": get_comments(prop),
        "descriptions": get_descriptions(prop),
        "annotations": collect_annotations(prop, rdf_graph),

        "property_kind": property_kind,
        "superproperties": isa_split["named"],
        "property_constructs": isa_split["constructs"],

        "domain": domains,
        "range": ranges,
        "inverse_property": inverse_prop,
        "characteristics": property_characteristics(prop),

        "python_name": getattr(prop, "python_name", None),
        "disjoints": serialize_disjoints(prop),

        "raw_axiom_triples": collect_raw_axiom_triples(prop, rdf_graph),
    }


def extract_instance(ind: Any, rdf_graph) -> Dict[str, Any]:
    isa_split = serialize_named_vs_constructs(list(getattr(ind, "is_a", [])))

    prop_assertions = {}
    inverse_assertions = []

    # Direct property assertions
    try:
        for prop in ind.get_properties():
            key = safe_iri(prop) or safe_name(prop) or repr(prop)
            values = []
            try:
                # Property[individual] always returns a list, even for functional props
                values = [entity_ref(v) for v in prop[ind]]
            except Exception:
                pass

            prop_assertions[key] = {
                "property": entity_ref(prop),
                "values": values,
            }
    except Exception:
        pass

    # Inverse assertions
    try:
        for subj, prop in ind.get_inverse_properties():
            inverse_assertions.append(
                {
                    "subject": entity_ref(subj),
                    "property": entity_ref(prop),
                }
            )
    except Exception:
        pass

    eqs = []
    try:
        eqs = [entity_ref(x) for x in ind.equivalent_to]
    except Exception:
        pass

    return {
        "name": safe_name(ind),
        "iri": safe_iri(ind),
        "namespace": safe_namespace(ind),
        "labels": get_labels(ind),
        "comments": get_comments(ind),
        "descriptions": get_descriptions(ind),
        "annotations": collect_annotations(ind, rdf_graph),

        "types": isa_split["named"],
        "type_constructs": isa_split["constructs"],

        "same_as": eqs,
        "differents": serialize_differents(ind),

        "property_assertions": prop_assertions,
        "inverse_property_assertions": inverse_assertions,

        "raw_axiom_triples": collect_raw_axiom_triples(ind, rdf_graph),
    }


# -----------------------------
# Main API
# -----------------------------




def register_local_iri_map(iri_map: Dict[str, str]) -> None:
    """
    Register IRI -> local filesystem path mappings with Owlready2.

    IMPORTANT:
    Use plain absolute file paths in PREDEFINED_ONTOLOGIES,
    not file:// URIs.

    Example:
        {
            "http://purl.org/net/OCRe/statistics.owl": "/abs/path/to/statistics.owl"
        }
    """
    for iri, local_path in iri_map.items():
        abs_path = os.path.abspath(local_path)
        iri_no_hash = iri.rstrip("#")

        if not os.path.exists(abs_path):
            print(f"Warning: mapped file does not exist for {iri_no_hash}: {abs_path}")
            continue

        PREDEFINED_ONTOLOGIES[iri_no_hash] = abs_path
        PREDEFINED_ONTOLOGIES[iri_no_hash + "#"] = abs_path

        print(f"Registered local mapping: {iri_no_hash} -> {abs_path}")


def build_local_iri_map_from_folder(
    local_dir: str,
    base_iri: Optional[str] = None,
) -> Dict[str, str]:
    """
    Build a simple IRI map from all .owl files in a folder.

    If base_iri is provided, each file is mapped like:
        {base_iri.rstrip('/')}/{filename}

    Example:
        statistics.owl -> http://purl.org/net/OCRe/statistics.owl
    """
    mapping: Dict[str, str] = {}
    abs_dir = os.path.abspath(local_dir)

    if not os.path.isdir(abs_dir):
        raise FileNotFoundError(f"Local ontology directory not found: {abs_dir}")

    for fname in os.listdir(abs_dir):
        if fname.lower().endswith(".owl"):
            full_path = os.path.join(abs_dir, fname)

            if base_iri:
                iri = base_iri.rstrip("/") + "/" + fname
                mapping[iri] = full_path

    return mapping


def extract_ontology_to_dicts(
    owl_path_or_iri: str,
    load_imported: bool = True,
    local_only: bool = False,
    local_ontology_dir: Optional[str] = None,
    iri_map: Optional[Dict[str, str]] = None,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Load an ontology with Owlready2 and extract 4 dictionaries:
      - classes_dict
      - object_properties_dict
      - data_properties_dict
      - instances_dict

    Keys are entity IRIs when available, otherwise entity names.

    Additional behavior:
      - resolves imports from local .owl files when possible
      - respects local_only for both root ontology and imports
      - prints unresolved dependencies
      - recursively loads imported ontologies
    """

    # ------------------------------------------------------------
    # 1) Configure local ontology search directory
    # ------------------------------------------------------------
    if local_ontology_dir:
        abs_dir = os.path.abspath(local_ontology_dir)

        if not os.path.isdir(abs_dir):
            raise FileNotFoundError(f"Local ontology directory not found: {abs_dir}")

        if abs_dir not in onto_path:
            onto_path.append(abs_dir)

        print(f"Added ontology search path: {abs_dir}")

    # ------------------------------------------------------------
    # 2) Register explicit IRI -> local file mappings
    # ------------------------------------------------------------
    if iri_map:
        register_local_iri_map(iri_map)

    # ------------------------------------------------------------
    # 3) Normalize root ontology input
    # ------------------------------------------------------------
    # If a plain local file path is passed in, convert it to file://...
    if os.path.exists(owl_path_or_iri):
        root_abs_path = os.path.abspath(owl_path_or_iri)
        owl_source = f"file://{root_abs_path}"
    else:
        owl_source = owl_path_or_iri

    # ------------------------------------------------------------
    # 4) Load root ontology
    # ------------------------------------------------------------
    onto = get_ontology(owl_source).load(only_local=local_only)
    print(f"Ontology loaded from: {onto.base_iri}")

    resolved_imports: Set[str] = set()
    unresolved_imports: Set[str] = set()
    visited: Set[str] = set()

    # ------------------------------------------------------------
    # 5) Recursively load imports
    # ------------------------------------------------------------
    def load_imports_recursive(current_onto) -> None:
        current_iri = (getattr(current_onto, "base_iri", None) or "").rstrip("#")
        if current_iri in visited:
            return
        visited.add(current_iri)

        for imported in current_onto.imported_ontologies:
            imported_iri = (getattr(imported, "base_iri", None) or "").rstrip("#")

            try:
                imported.load(only_local=local_only)
                resolved_imports.add(imported_iri)
                print(f"Resolved import: {imported_iri}")

                load_imports_recursive(imported)

            except Exception as e:
                unresolved_imports.add(imported_iri)
                print(f"Could not resolve import: {imported_iri} ({e})")

    if load_imported:
        load_imports_recursive(onto)

    # ------------------------------------------------------------
    # 6) Owlready2 RDF graph
    # ------------------------------------------------------------
    rdf_graph = default_world.as_rdflib_graph()

    classes_dict: Dict[str, Dict[str, Any]] = {}
    object_properties_dict: Dict[str, Dict[str, Any]] = {}
    data_properties_dict: Dict[str, Dict[str, Any]] = {}
    instances_dict: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------
    # 7) Extract entities from root ontology
    # ------------------------------------------------------------
    for cls in onto.classes():
        key = safe_iri(cls) or safe_name(cls)
        classes_dict[key] = extract_class(cls, rdf_graph)
    print(f"Extracted {len(classes_dict)} classes.")

    for prop in onto.object_properties():
        key = safe_iri(prop) or safe_name(prop)
        object_properties_dict[key] = extract_property(
            prop,
            rdf_graph,
            property_kind="object_property",
        )
    print(f"Extracted {len(object_properties_dict)} object properties.")

    for prop in onto.data_properties():
        key = safe_iri(prop) or safe_name(prop)
        data_properties_dict[key] = extract_property(
            prop,
            rdf_graph,
            property_kind="data_property",
        )
    print(f"Extracted {len(data_properties_dict)} data properties.")

    for ind in onto.individuals():
        key = safe_iri(ind) or safe_name(ind)
        instances_dict[key] = extract_instance(ind, rdf_graph)
    print(f"Extracted {len(instances_dict)} instances.")

    # ------------------------------------------------------------
    # 8) Print dependency summary
    # ------------------------------------------------------------
    if load_imported:
        print("\nDependency summary")
        print("------------------")

        if resolved_imports:
            print("Resolved imports:")
            for iri in sorted(resolved_imports):
                print(f"  - {iri}")
        else:
            print("Resolved imports: none")

        if unresolved_imports:
            print("Unresolved imports:")
            for iri in sorted(unresolved_imports):
                print(f"  - {iri}")
        else:
            print("Unresolved imports: none")

    return {
        "classes_dict": classes_dict,
        "object_properties_dict": object_properties_dict,
        "data_properties_dict": data_properties_dict,
        "instances_dict": instances_dict,
    }


def build_class_relationship_index(classes_dict):
    """
    Returns parent and child class relationships for each class.

    Output format:
    {
        class_iri: {
            "parents": [...],
            "children": [...]
        }
    }
    """

    class_index = {}

    # Initialize structure
    for class_iri in classes_dict:
        class_index[class_iri] = {
            "parents": set(),
            "children": set()
        }

    # Populate parents and children
    for class_iri, class_data in classes_dict.items():

        for parent in class_data.get("superclasses", []):

            parent_iri = None
            if isinstance(parent, dict):
                parent_iri = parent.get("iri")

            if parent_iri and parent_iri in classes_dict:
                class_index[class_iri]["parents"].add(parent_iri)
                class_index[parent_iri]["children"].add(class_iri)

    # Convert sets to lists
    for class_iri in class_index:
        class_index[class_iri]["parents"] = list(class_index[class_iri]["parents"])
        class_index[class_iri]["children"] = list(class_index[class_iri]["children"])

    return class_index

def extract_iris(value):
    """
    Recursively extract entity IRIs from nested dictionary/list structures.
    """

    iris = set()

    if isinstance(value, dict):

        if value.get("kind") == "entity" and value.get("iri"):
            iris.add(value["iri"])

        for v in value.values():
            iris.update(extract_iris(v))

    elif isinstance(value, list):

        for item in value:
            iris.update(extract_iris(item))

    return iris

def find_object_properties_for_classes(classes_dict, object_properties_dict):
    """
    Finds object properties connected to each class based on domain and range.

    Output format:
    {
        class_iri: {
            "as_domain_of": [...],
            "as_range_of": [...]
        }
    }
    """

    results = {}

    # Initialize output
    for class_iri in classes_dict:
        results[class_iri] = {
            "as_domain_of": [],
            "as_range_of": []
        }

    # Process each object property
    for prop_iri, prop_data in object_properties_dict.items():

        prop_name = prop_data.get("name")

        domain_items = prop_data.get("domain", [])
        range_items = prop_data.get("range", [])

        domain_iris = set()
        range_iris = set()

        for d in domain_items:
            domain_iris.update(extract_iris(d))

        for r in range_items:
            range_iris.update(extract_iris(r))

        # Domain relationships
        for class_iri in domain_iris:

            if class_iri in classes_dict:

                results[class_iri]["as_domain_of"].append({
                    "property_iri": prop_iri,
                    "property_name": prop_name,
                    "range": list(range_iris)
                })

        # Range relationships
        for class_iri in range_iris:

            if class_iri in classes_dict:

                results[class_iri]["as_range_of"].append({
                    "property_iri": prop_iri,
                    "property_name": prop_name,
                    "domain": list(domain_iris)
                })

    return results

def find_immediate_edges_for_class(class_iri, classes_dict, object_properties_dict):
    """
    Returns the immediate edges for a class.

    Immediate edges include:
      1. parent_class edges
      2. child_class edges
      3. object_property_domain edges
      4. object_property_range edges
      5. restriction-based edges found in restrictions_and_class_constructs

    Output format:
    {
        "class_iri": ...,
        "edges": [
            {
                "edge_type": "parent_class",
                "source": class_iri,
                "target": parent_iri
            },
            {
                "edge_type": "child_class",
                "source": class_iri,
                "target": child_iri
            },
            {
                "edge_type": "object_property_domain",
                "source": class_iri,
                "property_iri": ...,
                "property_name": ...,
                "target": range_class_iri
            },
            {
                "edge_type": "object_property_range",
                "source": domain_class_iri,
                "property_iri": ...,
                "property_name": ...,
                "target": class_iri
            },
            {
                "edge_type": "restriction",
                "source": class_iri,
                "property_iri": ...,
                "target": related_class_iri,
                "restriction_kind": ...
            }
        ]
    }
    """

    def extract_iris(value):
        iris = set()

        if isinstance(value, dict):
            if value.get("kind") == "entity" and value.get("iri"):
                iris.add(value["iri"])

            for v in value.values():
                iris.update(extract_iris(v))

        elif isinstance(value, list):
            for item in value:
                iris.update(extract_iris(item))

        return iris

    if class_iri not in classes_dict:
        return {
            "class_iri": class_iri,
            "edges": [],
            "error": "class not found in classes_dict"
        }

    edges = []
    class_data = classes_dict[class_iri]

    # ---------------------------------
    # 1. Parent edges
    # ---------------------------------
    for parent in class_data.get("superclasses", []):
        parent_iri = parent.get("iri") if isinstance(parent, dict) else None
        if parent_iri and parent_iri in classes_dict:
            edges.append({
                "edge_type": "parent_class",
                "source": class_iri,
                "target": parent_iri
            })

    # ---------------------------------
    # 2. Child edges
    # ---------------------------------
    for other_class_iri, other_class_data in classes_dict.items():
        if other_class_iri == class_iri:
            continue

        for parent in other_class_data.get("superclasses", []):
            parent_iri = parent.get("iri") if isinstance(parent, dict) else None
            if parent_iri == class_iri:
                edges.append({
                    "edge_type": "child_class",
                    "source": class_iri,
                    "target": other_class_iri
                })

    # ---------------------------------
    # 3. Object property edges via domain/range
    # ---------------------------------
    for prop_iri, prop_data in object_properties_dict.items():
        prop_name = prop_data.get("name")

        domain_iris = set()
        range_iris = set()

        for d in prop_data.get("domain", []):
            domain_iris.update(extract_iris(d))

        for r in prop_data.get("range", []):
            range_iris.update(extract_iris(r))

        # class is in domain -> outgoing edges
        if class_iri in domain_iris:
            for target_iri in range_iris:
                if target_iri in classes_dict:
                    edges.append({
                        "edge_type": "object_property_domain",
                        "source": class_iri,
                        "property_iri": prop_iri,
                        "property_name": prop_name,
                        "target": target_iri
                    })

        # class is in range -> incoming edges from domain classes
        if class_iri in range_iris:
            for source_iri in domain_iris:
                if source_iri in classes_dict:
                    edges.append({
                        "edge_type": "object_property_range",
                        "source": source_iri,
                        "property_iri": prop_iri,
                        "property_name": prop_name,
                        "target": class_iri
                    })

    # ---------------------------------
    # 4. Restriction-based edges from this class
    # ---------------------------------
    for restriction in class_data.get("restrictions_and_class_constructs", []):
        if not isinstance(restriction, dict):
            continue

        prop_info = restriction.get("property")
        prop_iri = prop_info.get("iri") if isinstance(prop_info, dict) else None
        prop_name = prop_info.get("name") if isinstance(prop_info, dict) else None

        related_iris = set()

        # common places related class/value may appear
        for key in ["value", "class", "classes"]:
            if key in restriction:
                related_iris.update(extract_iris(restriction[key]))

        for target_iri in related_iris:
            if target_iri in classes_dict:
                edges.append({
                    "edge_type": "restriction",
                    "source": class_iri,
                    "property_iri": prop_iri,
                    "property_name": prop_name,
                    "target": target_iri,
                    "restriction_kind": restriction.get("kind")
                })

    return {
        "class_iri": class_iri,
        "edges": edges
    }

def build_immediate_edges_index(classes_dict, object_properties_dict):
    """
    Build immediate edges for every class.
    """
    output = {}

    for class_iri in classes_dict:
        output[class_iri] = find_immediate_edges_for_class(
            class_iri,
            classes_dict,
            object_properties_dict
        )["edges"]

    return output


def make_hashable(value):
    """
    Convert nested dict/list/set structures into hashable tuples
    so they can be used for deduplication.
    """
    if isinstance(value, dict):
        return tuple(sorted((k, make_hashable(v)) for k, v in value.items()))
    elif isinstance(value, list):
        return tuple(make_hashable(v) for v in value)
    elif isinstance(value, set):
        return tuple(sorted(make_hashable(v) for v in value))
    else:
        return value


def dedupe_preserve_order(values):
    """
    Remove duplicates while preserving order, even if items are dicts/lists.
    """
    seen = set()
    result = []

    for item in values:
        marker = make_hashable(item)
        if marker not in seen:
            seen.add(marker)
            result.append(item)

    return result


def merge_dicts_recursive(d1, d2):
    result = d1.copy()

    for k, v2 in d2.items():
        if k not in result:
            result[k] = v2
            continue

        v1 = result[k]

        # both dicts -> recurse
        if isinstance(v1, dict) and isinstance(v2, dict):
            result[k] = merge_dicts_recursive(v1, v2)

        # both lists -> merge + dedupe
        elif isinstance(v1, list) and isinstance(v2, list):
            result[k] = dedupe_preserve_order(v1 + v2)

        # one list, one non-list
        elif isinstance(v1, list):
            result[k] = dedupe_preserve_order(v1 + [v2])
        elif isinstance(v2, list):
            result[k] = dedupe_preserve_order([v1] + v2)

        # same scalar
        else:
            if v1 == v2:
                result[k] = v1
            else:
                result[k] = dedupe_preserve_order([v1, v2])

    return result

def import_ontologies(owl_dict):
  imported_ontologies = {}
  for key, value in owl_dict.items():
    print(f"Processing: {key}")
    if value["local_only"]:
        ontology = extract_ontology_to_dicts(value["filename"], 
                                         load_imported=value["load_imported"],
                                         local_only=value["local_only"],
                                         local_ontology_dir=value["local_ontology_dir"],
                                         iri_map=value["iri_map"])
    else:
        ontology = extract_ontology_to_dicts(value["filename"], 
                                         load_imported=value["load_imported"],
                                         local_only=value["local_only"])

    
    if len(imported_ontologies.keys()) == 0:
      imported_ontologies = ontology
    else:
      imported_ontologies = merge_dicts_recursive(imported_ontologies, ontology)
      classes_dict = imported_ontologies["classes_dict"]
      object_properties_dict = imported_ontologies["object_properties_dict"]
      data_properties_dict = imported_ontologies["data_properties_dict"]
      instances_dict = imported_ontologies["instances_dict"]

      print(f"Classes: {len(classes_dict)}")
      print(f"Object properties: {len(object_properties_dict)}")
      print(f"Data properties: {len(data_properties_dict)}")
      print(f"Instances: {len(instances_dict)}")
  
  return imported_ontologies

def get_labels_only_from_classes(classes_dict):
  labels_dict = {}
  for class_iri in classes_dict:
    class_dict = classes_dict[class_iri]
    
    labels_dict[class_iri] = {}
    labels_dict[class_iri]['labels'] = class_dict['labels']
    labels_dict[class_iri]['descriptions'] = "\n".join(class_dict['descriptions'])
    labels_dict[class_iri]['comments'] = class_dict['comments']
    labels_dict[class_iri]['annotations'] = class_dict['annotations']
  

    
  return labels_dict