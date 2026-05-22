import json
import re
from typing import Any, Dict, List, Optional

from copy import deepcopy
from collections import defaultdict, deque
from pathlib import Path

from datetime import datetime



def classify_class_semantic_role(class_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Classify an OWL class into a likely semantic role.

    Returns a dict like:
    {
        "role": "relationship_like_class",
        "confidence": "high",
        "reasons": [...],
        "scores": {...}
    }
    """

    labels = [str(x) for x in class_info.get("labels", [])]
    comments = [str(x) for x in class_info.get("comments", [])]
    descriptions = [str(x) for x in class_info.get("descriptions", [])]

    superclass_names = []
    for sc in class_info.get("superclasses", []):
        if isinstance(sc, dict):
            superclass_names.append(
                " ".join(
                    str(sc.get(k, ""))
                    for k in ["name", "iri", "python_name", "type"]
                    if sc.get(k)
                )
            )
        else:
            superclass_names.append(str(sc))

    restriction_texts = []
    for r in class_info.get("restrictions_and_class_constructs", []):
        if isinstance(r, dict):
            pieces = [str(r.get("kind", "")), str(r.get("repr", ""))]
            prop = r.get("property")
            val = r.get("value")
            if isinstance(prop, dict):
                pieces.append(str(prop.get("name", "")))
                pieces.append(str(prop.get("iri", "")))
                pieces.append(str(prop.get("type", "")))
            if isinstance(val, dict):
                pieces.append(str(val.get("name", "")))
                pieces.append(str(val.get("iri", "")))
                pieces.append(str(val.get("type", "")))
            restriction_texts.append(" ".join(pieces))
        else:
            restriction_texts.append(str(r))

    text_parts = labels + comments + descriptions + superclass_names + restriction_texts
    text = " \n ".join(text_parts).lower()

    name = str(class_info.get("name", "")).lower()
    iri = str(class_info.get("iri", "")).lower()
    python_type = str(class_info.get("python_type", "")).lower()

    def count_matches(patterns: List[str], source_text: str) -> int:
        total = 0
        for p in patterns:
            if re.search(r"\b" + re.escape(p) + r"\b", source_text):
                total += 1
        return total

    relationship_terms = [
        "relationship", "association", "composition", "participation",
        "link", "role", "connection", "mapping", "correlation",
        "dependency", "interaction", "membership", "assignment"
    ]

    specification_terms = [
        "specification", "plan", "planned", "protocol", "definition",
        "template", "design", "instruction", "rule", "criterion",
        "criteria", "guideline", "schedule", "requirement"
    ]

    value_terms = [
        "value", "measurement", "quantity", "unit", "code", "status",
        "score", "result", "observation value", "dimension", "scale",
        "level", "amount", "date", "time", "interval", "duration"
    ]

    event_terms = [
        "event", "process", "procedure", "activity", "act", "encounter",
        "administration", "intervention", "assessment", "visit",
        "screening", "test", "collection", "observation"
    ]

    object_terms = [
        "person", "patient", "study", "protocol", "site", "organization",
        "investigator", "drug", "device", "substance", "specimen",
        "document", "arm", "cohort", "group", "subject", "condition",
        "disease", "treatment", "exposure", "agent", "product"
    ]

    relationship_score = 0
    specification_score = 0
    value_score = 0
    event_score = 0
    object_score = 0
    reasons: List[str] = []

    rel_hits = count_matches(relationship_terms, text)
    spec_hits = count_matches(specification_terms, text)
    val_hits = count_matches(value_terms, text)
    event_hits = count_matches(event_terms, text)
    obj_hits = count_matches(object_terms, text)

    relationship_score += rel_hits * 2
    specification_score += spec_hits * 2
    value_score += val_hits * 2
    event_score += event_hits * 2
    object_score += obj_hits

    if "a relationship between" in text:
        relationship_score += 5
        reasons.append("definition contains 'a relationship between'")

    if "comprised of" in text or "composed of" in text:
        relationship_score += 2
        reasons.append("definition suggests compositional relationship")

    if "specification" in text:
        specification_score += 3
        reasons.append("text contains 'specification'")

    if "planned" in text:
        specification_score += 2
        reasons.append("text contains 'planned'")

    if "objectpropertyclass" in text:
        relationship_score += 1
        reasons.append("restrictions reference object properties")

    if "thingclass" in python_type:
        reasons.append("structurally this is an OWL class (ThingClass)")

    # Strong name/label cues
    strong_label_text = " ".join(labels + [name, iri]).lower()

    if "relationship" in strong_label_text:
        relationship_score += 5
        reasons.append("name/label contains 'relationship'")

    if "specification" in strong_label_text:
        specification_score += 4
        reasons.append("name/label contains 'specification'")

    if "value" in strong_label_text or "measurement" in strong_label_text:
        value_score += 4
        reasons.append("name/label contains value/measurement language")

    if "event" in strong_label_text or "activity" in strong_label_text or "procedure" in strong_label_text:
        event_score += 3
        reasons.append("name/label contains event/process language")

    # Pick top role
    scores = {
        "relationship_like_class": relationship_score,
        "specification_class": specification_score,
        "value_or_measurement_class": value_score,
        "event_or_process_class": event_score,
        "domain_object_class": object_score,
    }

    top_role = max(scores, key=scores.get)
    top_score = scores[top_role]
    second_score = sorted(scores.values(), reverse=True)[1]

    if top_score >= second_score + 4:
        confidence = "high"
    elif top_score >= second_score + 2:
        confidence = "medium"
    else:
        confidence = "low"

    # Fallback if everything is weak
    if top_score == 0:
        top_role = "unclassified_class"
        confidence = "low"
        reasons.append("no strong semantic cues found")

    return {
        "role": top_role,
        "confidence": confidence,
        "reasons": sorted(set(reasons)),
        "scores": scores,
    }


def classify_classes_dict(classes_dict: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Add semantic role classification to every class in classes_dict.

    Returns a new dict with:
      class_info["semantic_role"]
    added to each entry.
    """
    result: Dict[str, Dict[str, Any]] = {}

    for key, class_info in classes_dict.items():
        enriched = dict(class_info)
        enriched["semantic_role"] = classify_class_semantic_role(class_info)
        result[key] = enriched

    return result


def split_classes_by_semantic_role(
    classes_dict: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Split classes into buckets by semantic role.

    Returns:
    {
        "domain_object_class": {...},
        "relationship_like_class": {...},
        "specification_class": {...},
        "value_or_measurement_class": {...},
        "event_or_process_class": {...},
        "unclassified_class": {...},
    }
    """
    classified = classify_classes_dict(classes_dict)

    buckets: Dict[str, Dict[str, Dict[str, Any]]] = {
        "domain_object_class": {},
        "relationship_like_class": {},
        "specification_class": {},
        "value_or_measurement_class": {},
        "event_or_process_class": {},
        "unclassified_class": {},
    }

    for key, class_info in classified.items():
        role = class_info.get("semantic_role", {}).get("role", "unclassified_class")
        if role not in buckets:
            role = "unclassified_class"
        buckets[role][key] = class_info

    return buckets

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

def split_dict_by_size(d, chunk_size):
    items = list(d.items())

    for i in range(0, len(items), chunk_size):
        yield dict(items[i:i + chunk_size])


def append_raw_output_to_log(raw_output: str, log_file: str = "llm_audit_log.txt") -> None:
    """
    Append the full raw LLM output to a text log for auditing.
    """
    timestamp = datetime.utcnow().isoformat()

    with open(log_file, "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"TIMESTAMP: {timestamp} UTC\n")
        f.write(raw_output)
        f.write("\n" + "=" * 80 + "\n")


def extract_json_from_output(raw_output: str) -> Optional[Dict[str, Any]]:
    """
    Extract JSON from LLM output that may contain explanatory text
    before or after the JSON.
    """
    fenced_match = re.search(r"```json\s*(\{.*?\})\s*```", raw_output, re.DOTALL)
    if fenced_match:
        json_str = fenced_match.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

    start = raw_output.find("{")
    if start == -1:
        return None

    brace_count = 0
    end = None

    for i in range(start, len(raw_output)):
        char = raw_output[i]
        if char == "{":
            brace_count += 1
        elif char == "}":
            brace_count -= 1
            if brace_count == 0:
                end = i + 1
                break

    if end is None:
        return None

    json_str = raw_output[start:end]

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None


def process_llm_outputs(
    raw_outputs: List[str],
    log_file: str = "llm_audit_log.txt"
) -> List[Dict[str, Any]]:
    """
    For each raw LLM output:
    1. log the full raw output
    2. extract the JSON
    3. return a list of parsed JSON dictionaries
    """
    parsed_dicts = []

    for i, raw_output in enumerate(raw_outputs):
        append_raw_output_to_log(raw_output, log_file=log_file)

        parsed_json = extract_json_from_output(raw_output)
        if parsed_json is not None:
            parsed_dicts.append(parsed_json)
        else:
            print(f"Warning: Could not extract valid JSON from output #{i}")
            print("RAW OUTPUT START")
            print(raw_output if isinstance(raw_output, str) else raw_output.get("response", ""))
            print("RAW OUTPUT END")
    return parsed_dicts


def make_hashable(obj: Any) -> Any:
    """
    Convert nested dict/list objects into a hashable form.
    """
    if isinstance(obj, dict):
        return tuple(sorted((k, make_hashable(v)) for k, v in obj.items()))
    elif isinstance(obj, list):
        return tuple(make_hashable(x) for x in obj)
    return obj


def unique_list(values: List[Any]) -> List[Any]:
    """
    Deduplicate a list while preserving order.
    Works for nested dict/list values too.
    """
    seen = set()
    result = []

    for value in values:
        marker = make_hashable(value)
        if marker not in seen:
            seen.add(marker)
            result.append(value)

    return result


def merge_scalar_values(v1: Any, v2: Any) -> Any:
    """
    Merge two scalar/non-dict/non-list values.
    If equal, keep one value.
    If different, return a deduplicated list.
    """
    if v1 == v2:
        return v1

    if isinstance(v1, list):
        combined = v1 + ([v2] if not isinstance(v2, list) else v2)
        return unique_list(combined)

    if isinstance(v2, list):
        combined = ([v1] if not isinstance(v1, list) else v1) + v2
        return unique_list(combined)

    return unique_list([v1, v2])


def recursive_merge(v1: Any, v2: Any, key_field: Optional[str] = None) -> Any:
    """
    Recursively merge two values.

    Rules:
    - dict + dict -> merge by keys recursively
    - list + list of dicts with key_field -> merge items by that key_field
    - list + list otherwise -> concatenate and deduplicate
    - scalars -> keep one if same, else convert to unique list
    """
    if isinstance(v1, dict) and isinstance(v2, dict):
        result = dict(v1)
        for key, value2 in v2.items():
            if key in result:
                result[key] = recursive_merge(result[key], value2)
            else:
                result[key] = value2
        return result

    if isinstance(v1, list) and isinstance(v2, list):
        # If a key_field is provided and all relevant items are dicts with that key,
        # merge list items by that key.
        if key_field and all(isinstance(x, dict) and key_field in x for x in v1 + v2):
            merged_by_key = {}

            for item in v1 + v2:
                item_key = item[key_field]
                if item_key in merged_by_key:
                    merged_by_key[item_key] = recursive_merge(merged_by_key[item_key], item)
                else:
                    merged_by_key[item_key] = dict(item)

            return list(merged_by_key.values())

        return unique_list(v1 + v2)

    return merge_scalar_values(v1, v2)


def merge_llm_jsons_recursive(dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merge LLM JSON outputs recursively.

    Special handling:
    - 'MATCHES FOUND' items are merged by 'IRI'
    - 'MATCH NOT FOUND' items are merged by 'LABEL'
    """
    merged: Dict[str, Any] = {}

    for d in dicts:
        for top_key, value in d.items():
            if top_key not in merged:
                merged[top_key] = value
                continue

            if top_key == "MATCHES FOUND":
                merged[top_key] = recursive_merge(merged[top_key], value, key_field="IRI")
            elif top_key == "MATCH NOT FOUND":
                merged[top_key] = recursive_merge(merged[top_key], value, key_field="LABEL")
            else:
                merged[top_key] = recursive_merge(merged[top_key], value)

    return merged


def save_json(data: Any, output_file: str) -> None:
    """
    Save a Python object to a JSON file.
    """
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ============================================================
# BASIC HELPERS
# ============================================================

def slugify(text: str) -> str:
    """
    Convert a label into a safe IRI suffix.
    Example:
        'New Data Class Label' -> 'new_data_class_label'
    """
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unnamed_class"


def normalize_namespace(base_iri: str) -> str:
    """
    Ensure the namespace ends with / or #.
    """
    if base_iri.endswith("/") or base_iri.endswith("#"):
        return base_iri
    return base_iri + "/"


def make_class_iri(base_iri: str, label: str) -> str:
    """
    Create a class IRI from base IRI + label.
    """
    base_iri = normalize_namespace(base_iri)
    return f"{base_iri}{slugify(label)}"


def safe_get_iri(obj):
    """
    Extract iri from structures like:
      {"kind": "entity", "iri": "..."}
    or return None if not present.
    """
    if isinstance(obj, dict):
        return obj.get("iri")
    return None


def safe_name_from_iri(iri: str) -> str:
    """
    Extract a short local name from an IRI.
    """
    if not iri:
        return ""
    if "#" in iri:
        return iri.rsplit("#", 1)[-1]
    return iri.rstrip("/").rsplit("/", 1)[-1]


# ============================================================
# INPUT VALIDATION
# ============================================================

def validate_loaded_ontology_dict(loaded_ontology: dict):
    """
    Ensure ontology dict has expected top-level keys.
    """
    required_keys = [
        "classes_dict",
        "object_properties_dict",
        "data_properties_dict",
        "instances_dict",
    ]

    if not isinstance(loaded_ontology, dict):
        raise ValueError("loaded_ontology must be a dictionary")

    missing = [k for k in required_keys if k not in loaded_ontology]
    if missing:
        raise ValueError(
            f"loaded_ontology is missing required keys: {missing}. "
            f"Expected keys: {required_keys}"
        )


def extract_detected_iris(match_results: dict) -> list[str]:
    """
    Extract IRIs from:
    {
      "MATCHES FOUND": [
        {"IRI": "...", "TEXT_SNIPPET": "..."}
      ],
      "MATCH NOT FOUND": [...]
    }
    """
    iris = []
    for item in match_results.get("MATCHES FOUND", []):
        iri = item.get("IRI")
        if iri:
            iris.append(iri)
    return iris


# ============================================================
# GRAPH BUILDING OVER CLASSES
# ============================================================

def build_class_graph(classes_dict: dict) -> dict[str, set[str]]:
    """
    Build an undirected class graph using:
    - superclass links
    - restriction targets in restrictions_and_class_constructs

    This allows hop-based neighborhood expansion.

    Returns:
        adjacency[class_iri] = {neighbor_iri1, neighbor_iri2, ...}
    """
    adjacency = defaultdict(set)

    for class_iri, class_data in classes_dict.items():
        adjacency[class_iri]  # ensure node exists

        # 1) superclass edges
        for sc in class_data.get("superclasses", []):
            sc_iri = safe_get_iri(sc)
            if sc_iri and sc_iri in classes_dict:
                adjacency[class_iri].add(sc_iri)
                adjacency[sc_iri].add(class_iri)

        # 2) restriction / class construct edges
        for rc in class_data.get("restrictions_and_class_constructs", []):
            if not isinstance(rc, dict):
                continue

            value_iri = safe_get_iri(rc.get("value"))
            if value_iri and value_iri in classes_dict:
                adjacency[class_iri].add(value_iri)
                adjacency[value_iri].add(class_iri)

    return adjacency


def collect_related_class_iris(
    classes_dict: dict,
    detected_class_iris: list[str],
    max_hops: int = 1
) -> set[str]:
    """
    Collect all classes reachable within max_hops from the detected class IRIs.

    hop=0 -> only matched classes
    hop=1 -> direct neighbors
    hop=2 -> neighbors of neighbors
    etc.
    """
    if max_hops < 0:
        raise ValueError("max_hops must be >= 0")

    graph = build_class_graph(classes_dict)
    keep = set()
    queue = deque()

    for iri in detected_class_iris:
        if iri in classes_dict:
            keep.add(iri)
            queue.append((iri, 0))

    while queue:
        current_iri, current_hops = queue.popleft()

        if current_hops >= max_hops:
            continue

        for neighbor_iri in graph.get(current_iri, set()):
            if neighbor_iri not in keep:
                keep.add(neighbor_iri)
                queue.append((neighbor_iri, current_hops + 1))

    return keep


def _build_isa_indexes(classes_dict: dict) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Build (parent_of, children_of) indexes over the IS-A (subClassOf) edges.

    parent_of[iri]   = set of IRIs that are direct parents of `iri`.
    children_of[iri] = set of IRIs that are direct subclasses of `iri`.

    Only includes edges where both endpoints exist in `classes_dict`
    (skips dangling references to external IRIs).
    """
    parent_of: dict[str, set[str]] = {iri: set() for iri in classes_dict}
    children_of: dict[str, set[str]] = {iri: set() for iri in classes_dict}
    for iri, data in classes_dict.items():
        for sc in data.get("superclasses", []):
            sc_iri = safe_get_iri(sc)
            if sc_iri and sc_iri in classes_dict:
                parent_of[iri].add(sc_iri)
                children_of[sc_iri].add(iri)
    return parent_of, children_of


def collect_full_class_hierarchy(
    classes_dict: dict,
    seed_iris: list[str],
) -> set[str]:
    """For each seed IRI, return seeds plus ALL ancestors plus ALL
    descendants via the IS-A (subClassOf) hierarchy.

    The ancestor and descendant walks are SEPARATE traversals starting
    from the seeds, NOT a single BFS over the undirected IS-A graph:

      - Ancestor walk: from each seed, follow `parent_of` only.
      - Descendant walk: from each seed, follow `children_of` only.

    This means siblings of ancestors are NOT pulled in. If A subClassOf
    Mammal, and B subClassOf Mammal, seeding on A retains
    {A, Mammal, Animal, ..., A's descendants} but NOT B and B's
    descendants. Otherwise an ontology with a single common root (HP's
    BFO hierarchy is one example) would expand any single seed to the
    entire ontology.

    Unlike `collect_related_class_iris`, this walks ONLY subClassOf (not
    the undirected restriction graph), and the walk is UNBOUNDED in
    depth -- the full ancestor + descendant transitive closure of the
    seeds.
    """
    parent_of, children_of = _build_isa_indexes(classes_dict)
    seeds = [iri for iri in seed_iris if iri in classes_dict]
    keep: set[str] = set(seeds)

    # Ancestor walk: parents of seeds, parents-of-parents, etc. No
    # descent step here -- we stay on the upward chain.
    upward_queue: deque[str] = deque(seeds)
    upward_visited: set[str] = set(seeds)
    while upward_queue:
        cur = upward_queue.popleft()
        for parent in parent_of.get(cur, ()):
            if parent not in upward_visited:
                upward_visited.add(parent)
                keep.add(parent)
                upward_queue.append(parent)

    # Descendant walk: children of seeds, grandchildren, etc. No ascent
    # step here -- we stay on the downward chain.
    downward_queue: deque[str] = deque(seeds)
    downward_visited: set[str] = set(seeds)
    while downward_queue:
        cur = downward_queue.popleft()
        for child in children_of.get(cur, ()):
            if child not in downward_visited:
                downward_visited.add(child)
                keep.add(child)
                downward_queue.append(child)

    return keep


def expand_with_relationship_partners(
    keep: set[str],
    object_properties_dict: dict,
    data_properties_dict: dict,
) -> set[str]:
    """Augment `keep` with every class that appears as the OTHER endpoint
    of an object/data property whose domain or range already touches
    `keep`.

    Rationale: if class A is being kept and there's a property `worksFor`
    with domain=[A], range=[B], then keeping the property alone (with
    range pruned to []) loses the relationship's meaning. Adding B to
    `keep` preserves it intact.

    Note: this is a single-step extension -- we do NOT recursively walk
    the IS-A hierarchy of newly-added partner classes. Combine with
    `collect_full_class_hierarchy` first if you want hierarchy + relationships.
    """
    extra: set[str] = set()
    for props in (object_properties_dict, data_properties_dict):
        for prop_data in props.values():
            domain_iris = [safe_get_iri(d) for d in prop_data.get("domain", [])]
            range_iris = [safe_get_iri(r) for r in prop_data.get("range", [])]
            touches_keep = any(
                d in keep for d in domain_iris if d
            ) or any(r in keep for r in range_iris if r)
            if not touches_keep:
                continue
            for d in domain_iris:
                if d:
                    extra.add(d)
            for r in range_iris:
                if r:
                    extra.add(r)
    return keep | extra


# ============================================================
# PRUNING CLASSES
# ============================================================

def prune_class_entry(class_data: dict, keep_class_iris: set[str]) -> dict:
    """
    Prune one class entry so that internal references only point
    to retained classes where relevant.
    """
    pruned = deepcopy(class_data)

    # Keep only superclasses that are retained
    pruned["superclasses"] = [
        sc for sc in class_data.get("superclasses", [])
        if safe_get_iri(sc) in keep_class_iris
    ]

    # Keep only restrictions pointing to retained classes,
    # or non-class-valued constructs
    new_restrictions = []
    for rc in class_data.get("restrictions_and_class_constructs", []):
        if not isinstance(rc, dict):
            continue

        value_iri = safe_get_iri(rc.get("value"))
        if value_iri is None:
            new_restrictions.append(deepcopy(rc))
        elif value_iri in keep_class_iris:
            new_restrictions.append(deepcopy(rc))

    pruned["restrictions_and_class_constructs"] = new_restrictions

    return pruned


def prune_classes_dict(classes_dict: dict, keep_class_iris: set[str]) -> dict:
    """
    Keep only retained classes.
    """
    pruned = {}

    for iri, class_data in classes_dict.items():
        if iri in keep_class_iris:
            pruned[iri] = prune_class_entry(class_data, keep_class_iris)

    return pruned


# ============================================================
# PRUNING OBJECT PROPERTIES
# ============================================================

def prune_object_properties_dict(
    object_properties_dict: dict,
    keep_class_iris: set[str]
) -> dict:
    """
    Keep object properties only if their domain/range still touches retained classes.
    Also prune domain/range references.
    """
    pruned = {}

    for prop_iri, prop_data in object_properties_dict.items():
        new_prop = deepcopy(prop_data)

        # Domain
        domain_list = prop_data.get("domain", [])
        new_domain = [d for d in domain_list if safe_get_iri(d) in keep_class_iris]

        # Range
        range_list = prop_data.get("range", [])
        new_range = [r for r in range_list if safe_get_iri(r) in keep_class_iris]

        new_prop["domain"] = new_domain
        new_prop["range"] = new_range

        # Keep property if it still touches retained classes
        if new_domain or new_range:
            pruned[prop_iri] = new_prop

    return pruned


# ============================================================
# PRUNING DATA PROPERTIES
# ============================================================

def prune_data_properties_dict(
    data_properties_dict: dict,
    keep_class_iris: set[str]
) -> dict:
    """
    Keep data properties only if their domain still touches retained classes.
    """
    pruned = {}

    for prop_iri, prop_data in data_properties_dict.items():
        new_prop = deepcopy(prop_data)

        domain_list = prop_data.get("domain", [])
        new_domain = [d for d in domain_list if safe_get_iri(d) in keep_class_iris]

        new_prop["domain"] = new_domain

        if new_domain:
            pruned[prop_iri] = new_prop

    return pruned


# ============================================================
# PRUNING INSTANCES
# ============================================================

def prune_instances_dict(
    instances_dict: dict,
    keep_class_iris: set[str]
) -> dict:
    """
    Keep instances only if they are typed to retained classes.
    Assumes instance entries may contain something like:
      - "types"
      - or "direct_types"

    This function checks both to be safe.
    """
    pruned = {}

    for inst_iri, inst_data in instances_dict.items():
        new_inst = deepcopy(inst_data)

        types_list = inst_data.get("types", [])
        direct_types_list = inst_data.get("direct_types", [])

        new_types = [t for t in types_list if safe_get_iri(t) in keep_class_iris]
        new_direct_types = [t for t in direct_types_list if safe_get_iri(t) in keep_class_iris]

        if "types" in new_inst:
            new_inst["types"] = new_types
        if "direct_types" in new_inst:
            new_inst["direct_types"] = new_direct_types

        if new_types or new_direct_types:
            pruned[inst_iri] = new_inst

    return pruned


# ============================================================
# CREATE NEW PROPOSED CLASSES
# ============================================================

def create_new_class_entry(
    label: str,
    description: str,
    new_class_base_iri: str,
    default_parent_iri: str | None = None
) -> tuple[str, dict]:
    """
    Create a new class entry in the same general format as classes_dict.
    """
    new_iri = make_class_iri(new_class_base_iri, label)

    entry = {
        "name": safe_name_from_iri(new_iri),
        "iri": new_iri,
        "namespace": normalize_namespace(new_class_base_iri),
        "labels": [label],
        "comments": [],
        "descriptions": [description] if description else [],
        "annotations": {
            "generated": [True],
            "review_status": ["proposed"]
        },
        "superclasses": [],
        "restrictions_and_class_constructs": [],
        "equivalent_to": [],
        "equivalent_constructs": [],
        "disjoints": [],
        "python_type": "ThingClass",
        "direct_instances": [],
        "raw_axiom_triples": [],
        "semantic_role": {
            "role": "proposed_data_class",
            "confidence": "rule_based",
            "reasons": ["Created from MATCH NOT FOUND"],
            "scores": {}
        }
    }

    if default_parent_iri:
        entry["superclasses"].append({
            "kind": "entity",
            "name": safe_name_from_iri(default_parent_iri),
            "iri": default_parent_iri,
            "python_name": None,
            "type": "ThingClass"
        })

    return new_iri, entry

def normalize_to_string(value):
    """
    Convert a scalar/list/None into a clean string.
    If list, join unique non-empty string values with ' | '.
    """
    if value is None:
        return ""

    if isinstance(value, str):
        return value.strip()

    if isinstance(value, list):
        cleaned = []
        for v in value:
            if v is None:
                continue
            s = str(v).strip()
            if s and s not in cleaned:
                cleaned.append(s)
        return " | ".join(cleaned)

    return str(value).strip()

def add_new_classes_from_match_not_found(
    loaded_ontology: dict,
    match_results: dict,
    new_class_base_iri: str,
    default_parent_iri: str | None = None
) -> tuple[dict, list[str]]:
    """
    Add new proposed classes from MATCH NOT FOUND into classes_dict.
    """
    result = deepcopy(loaded_ontology)
    created_iris = []

    for item in match_results.get("MATCH NOT FOUND", []):
        label = normalize_to_string(item.get("LABEL"))
        description = normalize_to_string(item.get("DESCRIPTION"))

        if not label:
            continue

        new_iri, entry = create_new_class_entry(
            label=label,
            description=description,
            new_class_base_iri=new_class_base_iri,
            default_parent_iri=default_parent_iri
        )

        if new_iri not in result["classes_dict"]:
            result["classes_dict"][new_iri] = entry
            created_iris.append(new_iri)

    return result, created_iris


def make_property_iri(base_iri: str, label: str) -> str:
    """
    Create an object-property IRI from base IRI + label.
    Mirrors `make_class_iri` so proposed properties live in the same
    user-controlled namespace as proposed classes.
    """
    base_iri = normalize_namespace(base_iri)
    return f"{base_iri}{slugify(label)}"


def create_new_object_property_entry(
    label: str,
    description: str,
    domain_iri: str,
    range_iri: str,
    new_property_base_iri: str,
) -> tuple[str, dict]:
    """
    Create a new object-property entry in the same shape as
    object_properties_dict values. Counterpart to
    `create_new_class_entry`.
    """
    new_iri = make_property_iri(new_property_base_iri, label)

    entry = {
        "name": safe_name_from_iri(new_iri),
        "iri": new_iri,
        "namespace": normalize_namespace(new_property_base_iri),
        "labels": [label],
        "comments": [],
        "descriptions": [description] if description else [],
        "annotations": {
            "generated": [True],
            "review_status": ["proposed"],
        },
        "property_kind": "object_property",
        "superproperties": [],
        "property_constructs": [],
        "domain": [{
            "kind": "entity",
            "name": safe_name_from_iri(domain_iri),
            "iri": domain_iri,
            "python_name": None,
            "type": "ThingClass",
        }],
        "range": [{
            "kind": "entity",
            "name": safe_name_from_iri(range_iri),
            "iri": range_iri,
            "python_name": None,
            "type": "ThingClass",
        }],
        "inverse_property": None,
        "characteristics": [],
        "python_name": None,
        "disjoints": [],
        "raw_axiom_triples": [],
    }
    return new_iri, entry


def _build_label_to_iri_index(classes_dict: dict) -> dict[str, str]:
    """Return a case-insensitive {label_or_name: iri} index over the
    current classes_dict. The FIRST entry wins on collisions (rare; if it
    happens the user can pin a specific IRI via the prompt instead).

    Used to resolve LLM-proposed DOMAIN/RANGE labels to actual class IRIs
    when adding new relations.
    """
    index: dict[str, str] = {}
    for iri, data in classes_dict.items():
        # Index every label.
        for lab in data.get("labels", []) or []:
            if isinstance(lab, str) and lab.strip():
                index.setdefault(lab.strip().lower(), iri)
            elif isinstance(lab, dict):
                v = lab.get("value")
                if isinstance(v, str) and v.strip():
                    index.setdefault(v.strip().lower(), iri)
        # Index the local name (e.g. "Person" from "ex:Person").
        name = data.get("name") or safe_name_from_iri(iri)
        if isinstance(name, str) and name.strip():
            index.setdefault(name.strip().lower(), iri)
    return index


def add_new_relations_from_match_results(
    loaded_ontology: dict,
    match_results: dict,
    new_property_base_iri: str,
) -> tuple[dict, list[str], list[dict]]:
    """
    Add LLM-proposed object properties from `MATCH NOT FOUND RELATIONS`
    into object_properties_dict. Each entry is expected to look like:

        {
            "LABEL": "treats",
            "DESCRIPTION": "A Drug treats a Disease.",
            "DOMAIN": "Drug",           # IRI or label of a class
            "RANGE": "Disease"          # IRI or label of a class
        }

    Resolution order for DOMAIN/RANGE:
      1. If the value is already a known class IRI (key of classes_dict),
         use it directly.
      2. Otherwise, look the value up case-insensitively in the
         label/name index built over classes_dict.
      3. If still unresolved, the relation is skipped and added to the
         returned `skipped` list with a `reason` so the caller can
         surface it for review.

    IMPORTANT: call this AFTER `add_new_classes_from_match_not_found` so
    classes proposed by the same LLM run are already in classes_dict and
    can be referenced as DOMAIN/RANGE by their label.

    Returns:
      (extended_ontology, created_property_iris, skipped_relations)
    """
    result = deepcopy(loaded_ontology)
    classes_dict = result.get("classes_dict", {})
    obj_props_dict = result.setdefault("object_properties_dict", {})
    label_index = _build_label_to_iri_index(classes_dict)

    def _resolve(value: Any) -> tuple[str | None, str | None]:
        """Return (resolved_iri, reason_if_unresolved)."""
        text = normalize_to_string(value)
        if not text:
            return None, "empty"
        if text in classes_dict:
            return text, None
        hit = label_index.get(text.lower())
        if hit:
            return hit, None
        return None, f"could not resolve '{text}' to a class IRI"

    created: list[str] = []
    skipped: list[dict] = []

    for item in match_results.get("MATCH NOT FOUND RELATIONS", []):
        if not isinstance(item, dict):
            continue
        label = normalize_to_string(item.get("LABEL"))
        description = normalize_to_string(item.get("DESCRIPTION"))
        if not label:
            skipped.append({"relation": item, "reason": "missing LABEL"})
            continue
        d_iri, d_reason = _resolve(item.get("DOMAIN"))
        r_iri, r_reason = _resolve(item.get("RANGE"))
        if not d_iri or not r_iri:
            skipped.append({
                "relation": item,
                "reason": d_reason or r_reason or "unresolved DOMAIN/RANGE",
            })
            continue

        new_iri, entry = create_new_object_property_entry(
            label=label,
            description=description,
            domain_iri=d_iri,
            range_iri=r_iri,
            new_property_base_iri=new_property_base_iri,
        )
        if new_iri in obj_props_dict:
            # Property with this slugged IRI already exists. Merge domain/range
            # in case the LLM produced multiple chunks each contributing one
            # endpoint pair for the same property.
            existing = obj_props_dict[new_iri]
            for end in ("domain", "range"):
                seen = {safe_get_iri(d) for d in existing.get(end, [])}
                for d in entry.get(end, []):
                    if safe_get_iri(d) not in seen:
                        existing.setdefault(end, []).append(d)
            continue
        obj_props_dict[new_iri] = entry
        created.append(new_iri)

    return result, created, skipped


# ============================================================
# FULL PIPELINE
# ============================================================

def prune_and_extend_loaded_ontology(
    loaded_ontology: dict,
    match_results: dict,
    max_hops: int,
    new_class_base_iri: str,
    default_parent_iri: str | None = None
) -> dict:
    """
    Full pipeline that expects loaded_ontology with keys:
      - classes_dict
      - object_properties_dict
      - data_properties_dict
      - instances_dict

    Returns:
    {
      "pruned_and_extended_ontology": ...,
      "detected_class_iris": [...],
      "kept_class_iris": [...],
      "created_class_iris": [...]
    }
    """
    validate_loaded_ontology_dict(loaded_ontology)

    classes_dict = loaded_ontology["classes_dict"]
    object_properties_dict = loaded_ontology["object_properties_dict"]
    data_properties_dict = loaded_ontology["data_properties_dict"]
    instances_dict = loaded_ontology["instances_dict"]

    detected_class_iris = extract_detected_iris(match_results)

    keep_class_iris = collect_related_class_iris(
        classes_dict=classes_dict,
        detected_class_iris=detected_class_iris,
        max_hops=max_hops
    )

    pruned_ontology = {
        "classes_dict": prune_classes_dict(classes_dict, keep_class_iris),
        "object_properties_dict": prune_object_properties_dict(object_properties_dict, keep_class_iris),
        "data_properties_dict": prune_data_properties_dict(data_properties_dict, keep_class_iris),
        "instances_dict": prune_instances_dict(instances_dict, keep_class_iris),
    }

    extended_ontology, created_class_iris = add_new_classes_from_match_not_found(
        loaded_ontology=pruned_ontology,
        match_results=match_results,
        new_class_base_iri=new_class_base_iri,
        default_parent_iri=default_parent_iri
    )

    return {
        "pruned_and_extended_ontology": extended_ontology,
        "detected_class_iris": detected_class_iris,
        "kept_class_iris": sorted(keep_class_iris),
        "created_class_iris": created_class_iris,
    }


# ============================================================
# OPTIONAL FILE HELPERS
# ============================================================

def load_json(path: str | Path) -> dict:
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(data: dict, path: str | Path):
    path = Path(path)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


