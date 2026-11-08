"""Keep relation endpoints that would become new classes tied to the chunk text.

A relation in `MATCH NOT FOUND RELATIONS` whose DOMAIN or RANGE is not a known
class makes prune-expand create a class for that endpoint
(`ontology_pruning._automint_endpoint_class`). Runaway replies turn this into
junk: in the dcv2 fin-gen build (2026-11-07) five class_proposal replies listed
119-550 relations each ("highEthics -> SemiconductorHighEthics",
"hasPriceAmbling -> ElectricVehiclePriceAmbling", ...) and produced 866 of the
954 endpoint classes; no entity was ever typed to any of them.

Rule, per chunk, before the reply is merged:
  * an endpoint is KNOWN when it is an IRI, NONE/placeholder, a class or
    instance label proposed in the same reply, or a label of a class already
    in the ontology (case, spacing and punctuation ignored);
  * an unknown endpoint is GROUNDED when every word of its name (camelCase
    split, short words ignored, word endings tolerated) occurs in the chunk
    within a few words of the others;
  * a relation with an unknown, ungrounded endpoint is dropped, and at most
    `max_new` distinct unknown endpoints are kept per chunk (first come).

Relations between known classes are never touched.
"""
from __future__ import annotations

import re
from typing import Any

DEFAULT_MAX_NEW_ENDPOINTS = 15
_MIN_WORD = 3
_SLACK = 4  # extra words allowed between the name's words in the text
_IGNORE = frozenset({"the", "and", "for", "with", "has", "of", "its", "per", "via"})
_NORM_CACHE: dict[int, tuple[int, frozenset[str]]] = {}


def norm_label(text: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text or "").lower())


def name_words(label: Any) -> list[str]:
    """`ElectricVehiclePricePremium` -> [electric, vehicle, price, premium]."""
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(label or ""))
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", s)
    return [w for w in re.findall(r"[a-z0-9]+", s.lower()) if len(w) >= _MIN_WORD and w not in _IGNORE]


def text_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(text or "").lower())


def _stem(word: str) -> str:
    return word[: max(4, len(word) - 3)] if len(word) > 4 else word


def is_grounded(label: Any, words: list[str]) -> bool:
    """Every word of the name occurs in `words`, all within one window."""
    name = name_words(label)
    if not name:
        return False
    positions = []
    for w in name:
        stem = _stem(w)
        hits = [i for i, t in enumerate(words) if t == w or t.startswith(stem)]
        if not hits:
            return False
        positions.append(hits)
    window = len(name) + _SLACK
    return any(all(any(abs(j - i) < window for j in p) for p in positions[1:]) for i in positions[0])


def known_labels(loaded_ontology: dict[str, Any]) -> frozenset[str]:
    """Normalised labels and names of every class in the ontology (cached per
    ontology object, rebuilt when the class count changes)."""
    classes = loaded_ontology.get("classes_dict") or {}
    key = id(loaded_ontology)
    hit = _NORM_CACHE.get(key)
    if hit and hit[0] == len(classes):
        return hit[1]
    out: set[str] = set()
    for iri, rec in classes.items():
        if not isinstance(rec, dict):
            continue
        out.add(norm_label(str(iri).rsplit("#", 1)[-1].rsplit("/", 1)[-1]))
        if rec.get("name"):
            out.add(norm_label(rec["name"]))
        for lab in rec.get("labels") or ():
            out.add(norm_label(lab.get("value") if isinstance(lab, dict) else lab))
    out.discard("")
    labels = frozenset(out)
    _NORM_CACHE[key] = (len(classes), labels)
    return labels


def _is_open_or_iri(value: Any) -> bool:
    t = str(value or "").strip()
    if not t or t.upper() in {"NONE", "NULL", "N/A", "ANY", "THING", "OWL:THING"}:
        return True
    return "://" in t or t.startswith(("urn:", "http_", "https_"))


def ground_relations(
    parsed: dict[str, Any] | None,
    chunk_text: str,
    ontology_labels: frozenset[str] | set[str],
    max_new: int = DEFAULT_MAX_NEW_ENDPOINTS,
) -> tuple[dict[str, Any] | None, dict[str, int]]:
    """Drop relations whose would-be-new endpoint is not in the chunk text.

    Returns (reply, counts) with counts `ungrounded` and `over_cap` (relations
    dropped) and `kept_new` (distinct new endpoints kept). The reply is
    changed in place.
    """
    stats = {"ungrounded": 0, "over_cap": 0, "kept_new": 0}
    rels = (parsed or {}).get("MATCH NOT FOUND RELATIONS")
    if not isinstance(rels, list) or not rels:
        return parsed, stats
    local = set(ontology_labels)
    for section in ("MATCH NOT FOUND", "MATCH NOT FOUND INSTANCES"):
        for item in parsed.get(section) or ():
            if isinstance(item, dict):
                local.add(norm_label(item.get("LABEL")))
    words = text_words(re.sub(r"[*_`#|>]+", " ", chunk_text or ""))
    new_ok: list[str] = []
    verdict: dict[str, bool] = {}
    kept: list[Any] = []
    for rel in rels:
        if not isinstance(rel, dict):
            kept.append(rel)
            continue
        reason = None
        fresh: list[str] = []
        for side in ("DOMAIN", "RANGE"):
            value = rel.get(side)
            key = norm_label(value)
            if _is_open_or_iri(value) or key in local:
                continue
            if key not in verdict:
                verdict[key] = is_grounded(value, words)
            if not verdict[key]:
                reason = "ungrounded"
                break
            if key not in new_ok and key not in fresh:
                fresh.append(key)
        if reason is None and len(new_ok) + len(fresh) > max_new:
            reason = "over_cap"
        if reason:
            stats[reason] += 1
        else:
            new_ok.extend(fresh)
            kept.append(rel)
    parsed["MATCH NOT FOUND RELATIONS"] = kept
    stats["kept_new"] = len(new_ok)
    return parsed, stats
