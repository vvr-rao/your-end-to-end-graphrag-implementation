"""Review classes that prune-expand created from relation endpoints.

When a proposed relation names a DOMAIN or RANGE that is not a known class,
`ontology_pruning._automint_endpoint_class` creates one. Its parent is copied
from the OTHER end of the relation and its definition is the placeholder
"Auto-created from relation '<label>' (<relation description>)". In the dcv2
fin-gen build (2026-11-07) that gave CostOfSales < Agent, TradePayable < Agent,
MineSite < debt, BatterySwapping < Time, and the ontology judge said "no" to
602 of the 954 placeholder definitions.

After grounding (relation_endpoint_grounding) removes the runaway ones, this
step asks one model, in batches, for each remaining endpoint class:
  * keep or drop (opinions, comparisons, value buckets, adjectives),
  * the parent: one root from the Layer-G root menu, or CURRENT,
  * a one-sentence definition replacing the placeholder.

Runs when models.yaml defines task `endpoint_class_review` (unless it sets
`enabled: false`). A failed or skipped answer leaves the class as it was.
"""
from __future__ import annotations

import asyncio
from typing import Any

from backend.app.helpers.ontology_pruning import (
    _DOMAIN_CONCEPTS_NS,
    _build_label_to_iri_index,
    _first_super_iri,
    concept_root_menu,
    safe_name_from_iri,
)

TASK = "endpoint_class_review"
AUTO_KEY = "auto_created_from_relation"
PLACEHOLDER = "Auto-created from relation"
CURRENT = "CURRENT"


def _label(rec: dict[str, Any], iri: str) -> str:
    for lab in rec.get("labels") or ():
        text = lab.get("value") if isinstance(lab, dict) else lab
        if isinstance(text, str) and text.strip():
            return text.strip()
    return str(rec.get("name") or safe_name_from_iri(iri))


def endpoint_classes(classes: dict[str, Any], only: list[str] | None = None) -> list[str]:
    """IRIs of classes minted from relation endpoints that still carry the
    placeholder definition (restricted to `only` when given)."""
    pool = only if only is not None else list(classes)
    out = []
    for iri in pool:
        rec = classes.get(iri)
        if not isinstance(rec, dict) or AUTO_KEY not in (rec.get("annotations") or {}):
            continue
        descs = [d.get("value") if isinstance(d, dict) else d for d in rec.get("descriptions") or ()]
        if not descs or any(isinstance(d, str) and d.startswith(PLACEHOLDER) for d in descs):
            out.append(iri)
    return list(dict.fromkeys(out))


def _relation_lines(iri: str, classes: dict[str, Any], props: dict[str, Any], limit: int = 3) -> str:
    lines: list[str] = []
    for p_iri, prop in props.items():
        if not isinstance(prop, dict):
            continue
        dom = [d.get("iri") if isinstance(d, dict) else d for d in prop.get("domain") or ()]
        rng = [r.get("iri") if isinstance(r, dict) else r for r in prop.get("range") or ()]
        if iri not in dom and iri not in rng:
            continue
        d = _label(classes.get(dom[0], {}), dom[0]) if dom and dom[0] else "any"
        r = _label(classes.get(rng[0], {}), rng[0]) if rng and rng[0] else "any"
        desc = next((x.get("value") if isinstance(x, dict) else x
                     for x in prop.get("descriptions") or () if x), "")
        lines.append(f"{d} --{_label(prop, p_iri)}--> {r}" + (f" ({desc})" if desc else ""))
        if len(lines) >= limit:
            break
    return "; ".join(lines)


def build_items(iris: list[str], classes: dict[str, Any], props: dict[str, Any]) -> list[dict[str, Any]]:
    items = []
    for iri in iris:
        rec = classes[iri]
        parent = _first_super_iri(rec)
        items.append({
            "iri": iri,
            "class": _label(rec, iri),
            "parent": _label(classes.get(parent, {}), parent) if parent else "",
            "relations": _relation_lines(iri, classes, props),
        })
    return items


def root_index(classes: dict[str, Any], menu: list[dict[str, str]]) -> dict[str, str]:
    """{lower-case root label: IRI} for the menu the model answers from."""
    by_label = _build_label_to_iri_index(classes)
    out = {}
    for row in menu:
        label = row["LABEL"]
        iri = _DOMAIN_CONCEPTS_NS + label
        if iri not in classes:
            iri = by_label.get(label.lower(), "")
        if iri:
            out[label.lower()] = iri
    return out


def parse_decisions(raw: Any, n: int) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    if not isinstance(raw, list):
        return out
    for d in raw:
        if not isinstance(d, dict):
            continue
        try:
            k = int(d.get("id"))
        except (TypeError, ValueError):
            continue
        verdict = str(d.get("verdict") or "").strip().lower()
        if 0 <= k < n and verdict in ("keep", "drop"):
            out[k] = {"verdict": verdict, "parent": str(d.get("parent") or "").strip(),
                      "definition": " ".join(str(d.get("definition") or "").split())}
    return out


def _ref(iri: str) -> dict[str, Any]:
    return {"kind": "entity", "name": safe_name_from_iri(iri), "iri": iri,
            "python_name": None, "type": "ThingClass"}


def _drop_class(iri: str, ontology: dict[str, Any], created_props: set[str]) -> int:
    """Remove a class; children and instances move to its parent; relations
    created this run that end at it are removed, older ones lose that end.
    Returns the number of relations removed."""
    classes = ontology.get("classes_dict") or {}
    rec = classes.pop(iri, None) or {}
    parent = _first_super_iri(rec)
    for other in classes.values():
        if not isinstance(other, dict):
            continue
        supers = other.get("superclasses") or []
        if any((s.get("iri") if isinstance(s, dict) else s) == iri for s in supers):
            other["superclasses"] = [s for s in supers if (s.get("iri") if isinstance(s, dict) else s) != iri]
            if parent and not other["superclasses"]:
                other["superclasses"].append(_ref(parent))
    for inst in (ontology.get("instances_dict") or {}).values():
        if not isinstance(inst, dict):
            continue
        for key in ("types", "direct_types"):
            refs = inst.get(key) or []
            if any((t.get("iri") if isinstance(t, dict) else t) == iri for t in refs):
                kept = [t for t in refs if (t.get("iri") if isinstance(t, dict) else t) != iri]
                inst[key] = kept or ([_ref(parent)] if parent else [])
    removed = 0
    props = ontology.get("object_properties_dict") or {}
    for p_iri in list(props):
        prop = props[p_iri]
        if not isinstance(prop, dict):
            continue
        hit = False
        for side in ("domain", "range"):
            refs = prop.get(side) or []
            kept = [r for r in refs if (r.get("iri") if isinstance(r, dict) else r) != iri]
            if len(kept) != len(refs):
                hit = True
                prop[side] = kept
        if hit and p_iri in created_props and (not prop.get("domain") or not prop.get("range")):
            del props[p_iri]
            removed += 1
    return removed


def apply_decisions(ontology: dict[str, Any], items: list[dict[str, Any]],
                    roots: dict[str, str], created_props: set[str] | None = None) -> dict[str, Any]:
    """Apply keep/drop, parent and definition. Items without a decision are
    left untouched."""
    classes = ontology.get("classes_dict") or {}
    stats = {"kept": 0, "dropped": 0, "reparented": 0, "defined": 0,
             "unanswered": 0, "relations_removed": 0, "bad_parent": 0}
    for it in items:
        dec = it.get("decision")
        iri = it["iri"]
        if not dec or iri not in classes:
            stats["unanswered"] += 1
            continue
        if dec["verdict"] == "drop":
            stats["relations_removed"] += _drop_class(iri, ontology, created_props or set())
            stats["dropped"] += 1
            continue
        stats["kept"] += 1
        rec = classes[iri]
        ann = rec.setdefault("annotations", {})
        ann["endpoint_review"] = ["kept"]
        choice = dec["parent"]
        if choice and choice.upper() != CURRENT:
            new_parent = roots.get(choice.lower())
            if new_parent and new_parent != iri:
                if new_parent != _first_super_iri(rec):
                    rec["superclasses"] = [_ref(new_parent)]
                    stats["reparented"] += 1
            else:
                stats["bad_parent"] += 1
        if dec["definition"] and dec["definition"].lower() != it["class"].lower():
            rec["descriptions"] = [dec["definition"]]
            stats["defined"] += 1
    return stats


async def review_endpoint_classes(ontology: dict[str, Any], router: Any, *,
                                  only: list[str] | None = None,
                                  created_props: list[str] | None = None,
                                  log=print) -> dict[str, Any] | None:
    """Run the review. Returns stats (plus per-class decisions for the audit
    log), or None when the task is not configured."""
    try:
        spec = router.task_spec(TASK)
    except (AttributeError, KeyError):
        return None
    if spec.get("enabled") is False:
        return None
    classes = ontology.get("classes_dict") or {}
    iris = endpoint_classes(classes, only)
    if not iris:
        log(f"[{TASK}] no endpoint classes to review")
        return {"items": 0}
    menu = concept_root_menu(classes)
    roots = root_index(classes, menu)
    items = build_items(iris, classes, ontology.get("object_properties_dict") or {})
    if spec.get("prompt_set") == "deepseek":
        from backend.app.services.prompts_deepseek import deepseek_endpoint_class_review as build
    else:
        from backend.app.services.prompts import endpoint_class_review as build
    from backend.app.helpers.ontology_pruning import extract_json_from_output

    size = int(spec.get("items_per_call", 20))
    sem = asyncio.Semaphore(int(spec.get("concurrency", 8)))
    start = router.total_cost_usd
    failed = 0

    async def one(batch: list[dict[str, Any]]) -> None:
        nonlocal failed
        view = [{"id": k, "class": it["class"], "parent": it["parent"], "relations": it["relations"]}
                for k, it in enumerate(batch)]
        system, user = build(view, menu)
        async with sem:
            try:
                res = await router.chat(TASK, system=system, user=user)
                parsed = extract_json_from_output(res.text)
            except Exception as exc:  # one bad batch must not stop prune-expand
                failed += 1
                log(f"[{TASK}] call failed ({type(exc).__name__}); {len(batch)} class(es) left as they were")
                parsed = None
        got = parse_decisions(parsed.get("decisions") if isinstance(parsed, dict) else None, len(batch))
        for k, it in enumerate(batch):
            if k in got:
                it["decision"] = got[k]

    log(f"[{TASK}] reviewing {len(items)} endpoint class(es) against {len(menu)} root(s) "
        f"({spec.get('provider')}:{spec.get('model')})")
    await asyncio.gather(*(one(items[i:i + size]) for i in range(0, len(items), size)))
    again = [it for it in items if "decision" not in it]
    if again:
        small = max(1, size // 4)
        await asyncio.gather(*(one(again[i:i + small]) for i in range(0, len(again), small)))
    stats = apply_decisions(ontology, items, roots, set(created_props or ()))
    stats.update(items=len(items), retried=len(again), failed_calls=failed,
                 cost_usd=round(router.total_cost_usd - start, 4))
    log(f"[{TASK}] kept {stats['kept']}, dropped {stats['dropped']} "
        f"({stats['relations_removed']} relation(s) removed), re-parented {stats['reparented']}, "
        f"defined {stats['defined']}, unanswered {stats['unanswered']}, "
        f"${stats['cost_usd']:.4f}")
    stats["decisions"] = [{"class": it["class"], "old_parent": it["parent"], **(it.get("decision") or {})}
                          for it in items]
    return stats
