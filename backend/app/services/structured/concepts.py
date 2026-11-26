"""Final catalog -> ontology concepts (runs after measure repair and SQL verification).

Profiling maps columns/metrics/filters to classes from a sample; refinement
then defines measures (some first discovered there, some encoded as values of
an indicator column) and verification fixes the final queries. This module
reconciles the FINAL catalog with the ontology:

1. **Roles from the final queries** (no LLM): for every query whose exact SQL
   passed its test, :func:`metadata.describe_query` says what it measures,
   groups by, filters by and which columns qualify its unit or period. Each
   reusable concept gets ONE key (``col:<column>``, ``measure:<id>``,
   ``indicator:<column>=<value>``) and every item references keys by role
   (``item["concepts"]``).
2. **Resolution**: a key keeps a class it already has (from profiling or an
   earlier reconciliation); otherwise its label is resolved exactly against
   the database class registry; only what is still open goes to the model
   (task ``table_concept_map``, domain-neutral prompt), which reuses a
   candidate class, reports ambiguity, or proposes a new class.
3. **Persistence**: ``meta["concept_map"]`` (key -> roles, columns, class,
   status reused | minted | proposed | ambiguous | unresolved) and
   ``meta["class_proposals"]``. Proposals are NOT minted here (Phase 2 adds no
   classes): :func:`apply_class_proposals` (CLI ``table-ontology-update``)
   adds them to an ontology version folder with the existing
   class-proposal helper (fuzzy dedup, parent resolution, the configured
   base IRI), imports that version, and a later reconciliation resolves the
   keys to the minted IRIs. Repeating it reuses the same keys and IRIs.

Nothing here names a dataset, a column or a value.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from sqlalchemy import text as sql_text

from backend.app.services.structured.metadata import describe_query, items, selectable

ROLES = ("measures", "counts", "groups_by", "filters_by", "identifies", "qualifies")
_LEGACY_ROLES = {"qualifiers": "qualifies"}
STATUSES = ("reused", "minted", "proposed", "ambiguous", "unresolved")
MAX_INDICATOR_CONCEPTS = 40       # indicator values of families mapped per table
MAX_CONCEPTS_PER_CALL = 60


def _norm(label: str | None) -> str:
    return re.sub(r"[^a-z0-9]", "", str(label or "").lower())


def _fixed_population(it: dict) -> list[dict]:
    m = it.get("measure") if isinstance(it.get("measure"), dict) else None
    if m and (m.get("population") or {}):
        pop = m["population"]
        return [*(pop.get("include") or []), *({**w, "excluded": True} for w in pop.get("exclude") or [])]
    return [{k: w[k] for k in ("column", "op", "value") if k in w}
            for w in (it.get("query") or {}).get("where") or []
            if isinstance(w, dict) and w.get("param") in (None, "") and "value" in w]


def catalog_concepts(meta: dict) -> tuple[dict[str, dict], dict[str, dict[str, list[str]]]]:
    """(concepts {key: {roles, columns, label, description, values, evidence}},
    {item id: {role: [keys]}}) for every SELECTABLE item.

    Roles: measures (a quantity), counts (what a count counts: the COUNTED
    SUBJECT, a column's things or the table's rows -- never a 'count' class),
    groups_by, filters_by, identifies (the subject identifier a per-subject
    call takes), qualifies (qualifier / period columns). Every role a concept
    plays carries its SQL evidence per item (the expression or parameter).
    A measure's value column is the measure's concept (one mapping, not
    two), and columns of one family sharing a stem with a period suffix are
    one concept (``family:<stem>``)."""
    from backend.app.services.structured.wide_profile import column_families

    cols = {c["name"]: c for c in meta.get("columns") or []}
    fam_of = {c: stem for stem, members in column_families(list(cols)).items() for c in members}
    concepts: dict[str, dict] = {}
    roles_by_item: dict[str, dict[str, list[str]]] = {}
    measure_cols: dict[str, str] = {}
    for it in meta.get("measures") or []:
        m = it.get("measure") if isinstance(it.get("measure"), dict) else None
        if m and m.get("value_column") in cols and selectable(meta, it, semantic=False)[0]:
            measure_cols.setdefault(m["value_column"], f"measure:{m['id']}")

    def _add(key: str, role: str, item_id: str, evidence: str, **info) -> str:
        c = concepts.setdefault(key, {"roles": [], "used_by": [], "evidence": {}, **info})
        if role not in c["roles"]:
            c["roles"].append(role)
        ev = c["evidence"].setdefault(role, [])
        if len(ev) < 5 and {"item": item_id, "sql": evidence} not in ev:
            ev.append({"item": item_id, "sql": evidence})
        return key

    def _col(name: str | None) -> str | None:
        if not name:
            return None
        base = str(name).split("(")[-1].rstrip(")").split(".")[-1]
        return base if base in cols else None

    def _col_key(c: str, role: str, item_id: str, evidence: str) -> str:
        if role == "measures" and c in measure_cols:
            return _add(measure_cols[c], role, item_id, evidence)
        if c in fam_of:
            stem = fam_of[c]
            members = [x for x, st in fam_of.items() if st == stem]
            return _add(f"family:{stem}", role, item_id, evidence, label=stem,
                        description=f"one quantity over the columns {', '.join(members)}", columns=members)
        return _add(f"col:{c}", role, item_id, evidence, label=c, description=cols[c].get("description"),
                    columns=[c])

    indicators = 0
    for sec, it in items(meta):
        if not selectable(meta, it, semantic=False)[0]:
            continue
        iid = it["id"]
        cap = it.get("capability") or describe_query(meta, sec, it)
        r: dict[str, list[str]] = {k: [] for k in ROLES}
        m = it.get("measure") if isinstance(it.get("measure"), dict) else None
        spec = it.get("query") if isinstance(it.get("query"), dict) else {}
        if m:
            key = _add(f"measure:{m['id']}", "measures", iid, f"value column {m.get('value_column')}",
                       label=m.get("label") or m["id"], description=m.get("description"),
                       columns=[m.get("value_column")], unit=m.get("unit"),
                       population=m.get("population") or None)
            r["measures"].append(key)
            fam = (m.get("item") or {})
            if fam.get("family"):
                for v in fam.get("values") or []:
                    if indicators >= MAX_INDICATOR_CONCEPTS:
                        break
                    indicators += 1
                    r["measures"].append(_add(f"indicator:{fam['column']}={v}", "measures", iid,
                                              f"{fam['column']} = {v!r}", label=str(v),
                                              description=f"value '{v}' of indicator column {fam['column']}",
                                              columns=[fam["column"]], values=[v]))
            for q in m.get("subject_columns") or []:
                if q in cols:
                    r["identifies"].append(_col_key(q, "identifies", iid, f"parameter on {q}"))
            for q in [*(m.get("qualifier_columns") or []), *(m.get("period_columns") or [])]:
                if q in cols:
                    r["qualifies"].append(_col_key(q, "qualifies", iid, f"{q} in the observation key"))
        else:
            for a in [x for x in spec.get("select") or [] if isinstance(x, dict) and x.get("agg")]:
                agg, c = str(a.get("agg")), _col(a.get("column"))
                expr = f"{agg}({c or '*'})"
                if agg in {"count", "count_distinct"}:
                    if c:              # the things in that column
                        key = _col_key(c, "counts", iid, expr)
                    else:              # the table's rows
                        key = _add(f"rows:{meta['table_name']}", "counts", iid, expr,
                                   label=str(meta.get("grain") or f"row of {meta['table_name']}"),
                                   description=f"one row of {meta['table_name']}: {meta.get('grain') or ''}",
                                   columns=[])
                    concepts[key].setdefault("count", {"counted_subject": concepts[key].get("label"),
                                                       "unit": "count", "population": _fixed_population(it),
                                                       "provenance": {"item": iid, "sql_sha": it.get("sql_sha")}})
                    r["counts"].append(key)
                elif c:
                    r["measures"].append(_col_key(c, "measures", iid, expr))
            if not r["measures"] and not r["counts"]:
                for q in cap.get("quantity") or []:
                    c = _col(q)
                    if c:
                        r["measures"].append(_col_key(c, "measures", iid, f"returns {c}"))
        for g in cap.get("grouping_columns") or []:
            c = _col(g)
            if c:
                r["groups_by"].append(_col_key(c, "groups_by", iid, f"GROUP BY {c}"))
        params = {p.get("column"): p.get("name") for p in it.get("params") or [] if p.get("column")}
        for f in cap.get("filter_columns") or []:
            c = _col(f)
            if c and c != (m or {}).get("value_column"):
                ev = f":{params[c]}" if c in params else "fixed condition"
                r["filters_by"].append(_col_key(c, "filters_by", iid, f"{c} filter ({ev})"))
        for k in r:
            r[k] = list(dict.fromkeys(r[k]))
            for key in r[k]:
                if iid not in concepts[key]["used_by"]:
                    concepts[key]["used_by"].append(iid)
        roles_by_item[iid] = r
    return concepts, roles_by_item


async def class_registry(session) -> dict[str, dict[str, Any]]:
    """{iri: {label, parent_iri}} of the database class registry."""
    rows = (await session.execute(sql_text("""
        SELECT c.iri, c.label, (SELECT p.iri FROM graphrag.graph_relationships r
                                  JOIN graphrag.ontology_classes p ON p.id = r.target_node_id
                                 WHERE r.source_node_type = 'ontology_class' AND r.source_node_id = c.id
                                   AND r.relationship_type = 'subClassOf' LIMIT 1)
          FROM graphrag.ontology_classes c"""))).all()
    return {iri: {"label": label, "parent_iri": piri} for iri, label, piri in rows}


def _existing_class(meta: dict, key: str) -> tuple[str | None, str | None]:
    """(iri, label) the concept already carries from profiling."""
    kind, _, rest = key.partition(":")
    if kind == "col":
        c = next((c for c in meta.get("columns") or [] if c["name"] == rest), {})
        return c.get("class_iri"), c.get("class_label")
    if kind == "measure":
        it = next((i for i in meta.get("measures") or [] if (i.get("measure") or {}).get("id") == rest), {})
        return it.get("class_iri"), it.get("class_label")
    return None, None


async def reconcile_concepts(meta: dict, router=None, *, base_iri: str | None = None,
                             log: Callable[[str], None] = print) -> dict:
    """Map the final catalog's concepts to classes (see module doc). ``router``
    None: no LLM -- only kept, profiled and exact-label resolutions.
    Returns a report; updates ``meta`` in place."""
    from backend.app.db.session import session_scope
    from backend.app.services.structured.profiler import ClassIndex

    async with session_scope() as s:
        registry = await class_registry(s)
    by_label: dict[str, str] = {}
    for iri, e in registry.items():
        by_label.setdefault(_norm(e["label"]), iri)
    concepts, roles_by_item = catalog_concepts(meta)
    old = meta.get("concept_map") or {}
    cmap: dict[str, dict] = {}
    todo: list[str] = []
    for key, c in concepts.items():
        prev = old.get(key) or {}
        if not prev:
            # a key merged since the earlier map (a measure's value column, a
            # column family): inherit what one of its member columns had
            prev = next((old[f"col:{col}"] for col in c.get("columns") or []
                         if old.get(f"col:{col}", {}).get("iri") or old.get(f"col:{col}", {}).get("status")
                         in ("proposed", "ambiguous")), {})
        entry = {**{k: c.get(k) for k in ("roles", "used_by", "columns", "label", "values", "unit",
                                          "evidence", "count", "population")
                    if c.get(k) is not None}}
        iri = prev.get("iri") if prev.get("iri") in registry else None
        how = prev.get("status") if iri else None
        if not iri:
            p_iri, p_label = _existing_class(meta, key)
            if p_iri in registry:
                iri, how = p_iri, "reused"
            else:
                for lab in (prev.get("proposed_label"), p_label, c.get("label") if key.startswith(
                        ("measure:", "indicator:")) else None):
                    hit = by_label.get(_norm(lab)) if lab else None
                    if hit:
                        iri = hit
                        # a proposal that now resolves was minted by the ontology update
                        # (the helper may place it in its parent's namespace)
                        how = "minted" if prev.get("status") == "proposed" else "reused"
                        break
        if iri:
            entry.update(iri=iri, class_label=registry[iri]["label"], parent_iri=registry[iri]["parent_iri"],
                         status=how or "reused")
        else:
            entry["status"] = "unresolved"
            if prev.get("status") in ("proposed", "ambiguous"):
                entry.update({k: prev[k] for k in ("status", "proposed_label", "parent_label",
                                                   "ambiguous", "reason") if prev.get(k)})
            else:
                if prev.get("reason"):          # a no-LLM pass keeps the earlier verdict's reason
                    entry["reason"] = prev["reason"]
                todo.append(key)
        cmap[key] = entry
    calls = 0
    if todo and router is not None:
        from backend.app.services.prompts import PROMPTS

        index = ClassIndex({iri: {"labels": [e["label"]]} for iri, e in registry.items() if e["label"]})
        table_info = {"table": meta["table_name"], "description": meta.get("description"),
                      "grain": meta.get("grain")}
        queue = [(todo[i:i + MAX_CONCEPTS_PER_CALL], True) for i in range(0, len(todo), MAX_CONCEPTS_PER_CALL)]
        while queue:
            batch, may_retry = queue.pop(0)
            words: list[str] = []
            from backend.app.services.structured.profiler import _split_tokens

            payload = []
            for key in batch:
                c = concepts[key]
                words += _split_tokens(f"{c.get('label')} {c.get('description') or ''}")
                payload.append({"key": key, "roles": c["roles"], "label": c.get("label"),
                                "description": c.get("description"), "columns": c.get("columns"),
                                **({"values": c["values"]} if c.get("values") else {}),
                                **({"unit": c["unit"]} if c.get("unit") else {}),
                                **({"counted_subject": c["count"]["counted_subject"],
                                    "population": c["count"]["population"]} if c.get("count") else {})})
            system, user = PROMPTS["table_concept_map"](table_info, payload, index.candidates(words))
            res = await router.chat("table_concept_map", system=system, user=user)
            calls += 1
            try:
                out = json.loads(re.sub(r"^```(?:json)?\s*|\s*```$", "", (res.text or "").strip()))
            except json.JSONDecodeError:
                out = {}
            answered: set[str] = set()
            for r in out.get("concepts") or []:
                if not isinstance(r, dict) or r.get("key") not in cmap:
                    continue
                answered.add(r["key"])
                e = cmap[r["key"]]
                iri = r.get("iri") if r.get("iri") in registry else by_label.get(_norm(r.get("label")))
                if iri:
                    e.update(iri=iri, class_label=registry[iri]["label"],
                             parent_iri=registry[iri]["parent_iri"], status="reused")
                elif [a for a in r.get("ambiguous") or [] if a in registry]:
                    e.update(status="ambiguous", ambiguous=[a for a in r["ambiguous"] if a in registry],
                             reason=str(r.get("reason") or "")[:300])
                elif r.get("label"):
                    e.update(status="proposed", proposed_label=str(r["label"]),
                             parent_label=str(r.get("parent_label") or "") or None,
                             reason=str(r.get("reason") or "")[:300])
                else:
                    e.update(status="unresolved", reason=str(r.get("reason") or "no general concept")[:300])
            skipped = [k for k in batch if k not in answered]
            if skipped and may_retry:
                queue.append((skipped, False))      # once more, only what the reply left out
            for key in skipped:
                cmap[key]["reason"] = "the mapping reply did not cover this concept"
    for _sec, it in items(meta):
        if it["id"] in roles_by_item:
            it["concepts"] = roles_by_item[it["id"]]
        else:
            it.pop("concepts", None)
    meta["concept_map"] = cmap
    meta["class_proposals"] = [
        {"key": k, "LABEL": e["proposed_label"], "PARENT_LABEL": e.get("parent_label") or
         ("Metric" if "measures" in e.get("roles", []) else "Dimension"),
         "DESCRIPTION": f"{meta['table_name']}: {e.get('label')}"[:300]}
        for k, e in cmap.items() if e.get("status") == "proposed" and e.get("proposed_label")]
    counts = {st: sum(1 for e in cmap.values() if e.get("status") == st) for st in STATUSES}
    log(f"[structured] {meta['table_name']}: ontology concepts of the final catalog -- "
        + ", ".join(f"{n} {st}" for st, n in counts.items() if n) + f" ({calls} LLM call(s))")
    return {"concepts": len(cmap), "status": counts, "llm_calls": calls}


def concept_class_refs(meta: dict) -> list[dict]:
    """Class links of the reconciled concepts: (kind, name, class_iri)."""
    kind_of = {"measures": "tableMetric", "counts": "tableMetric", "filters_by": "tableFilter",
               "groups_by": "tableColumn", "identifies": "tableColumn", "qualifies": "tableColumn"}
    out = []
    for key, e in (meta.get("concept_map") or {}).items():
        if e.get("iri"):
            have = [_LEGACY_ROLES.get(r, r) for r in e.get("roles") or []]
            role = next((r for r in ROLES if r in have), "groups_by")
            out.append({"kind": kind_of[role], "name": key, "class_iri": e["iri"],
                        "class_label": e.get("class_label")})
    return out


async def apply_class_proposals(input_folder: Path, *, output_root: Path, base_iri: str,
                                parent_iri: str | None, import_db: bool = True,
                                log: Callable[[str], None] = print) -> dict:
    """Mint the persisted class proposals of every loaded table into a NEW
    ontology version (no measure discovery is re-run): the existing helper
    dedups them against existing classes and resolves parents; the version
    folder is written (merged.json + merged.owl + manifest) and, with
    ``import_db``, upserted into the database class registry."""
    from backend.app.db.session import session_scope
    from backend.app.helpers.ontology_pruning import add_new_classes_from_match_not_found
    from backend.app.services import folder_io, ontology_export, versioning
    from backend.app.services.structured.db_load import registered_tables

    async with session_scope() as s:
        reg = await registered_tables(s)
    proposals, seen = [], set()
    for name, r in sorted(reg.items()):
        for p in r["meta"].get("class_proposals") or []:
            k = _norm(p["LABEL"])
            if k and k not in seen:
                seen.add(k)
                proposals.append({"LABEL": p["LABEL"], "PARENT_LABEL": p.get("PARENT_LABEL"),
                                  "DESCRIPTION": p.get("DESCRIPTION") or name})
    ontology = folder_io.load_version_folder(input_folder)
    before = set((ontology.get("classes_dict") or {}).keys())
    created: list[str] = []
    if proposals:
        _o, created = add_new_classes_from_match_not_found(
            loaded_ontology=ontology, match_results={"MATCHES FOUND": [], "MATCH NOT FOUND": proposals},
            new_class_base_iri=base_iri, default_parent_iri=parent_iri, inplace=True)
    version_dir = versioning.new_version_dir(output_root, "table-concepts")
    folder_io.write_merged_json(version_dir, ontology)
    ontology_export.write_owl(ontology, version_dir / folder_io.MERGED_OWL)
    versioning.write_manifest(version_dir, operation="table-concepts", parent_version=input_folder,
                              input_documents=[], model_ids={},
                              extra={"proposals": len(proposals), "created": len(created)})
    versioning.write_stats(version_dir, {"proposals": proposals, "created_classes": created,
                                         "reused_existing": len(proposals) - len(created),
                                         "classes_before": len(before),
                                         "classes_after": len(ontology.get("classes_dict") or {})})
    imported = None
    if import_db:
        from backend.app.services.db_ontology_import import import_ontology_folder

        imported = await import_ontology_folder(version_dir)
    log(f"[structured] ontology update: {len(proposals)} proposal(s), {len(created)} new class(es) "
        f"-> {version_dir}")
    return {"version_dir": str(version_dir), "proposals": len(proposals), "created": created,
            "imported": bool(imported)}


async def check_agreement(owl_path: Path | None = None) -> dict:
    """Every class IRI a catalog references must exist in the database class
    registry, and (given the exported OWL of the current ontology version)
    every minted one in that OWL."""
    from backend.app.db.session import session_scope
    from backend.app.services.structured.db_load import registered_tables

    async with session_scope() as s:
        reg = await registered_tables(s)
        registry = await class_registry(s)
    owl = owl_path.read_text(encoding="utf-8") if owl_path and owl_path.exists() else None
    missing_db, missing_owl, checked = [], [], 0
    for name, r in reg.items():
        meta = r["meta"]
        refs = [(c["name"], c.get("class_iri")) for c in meta.get("columns") or [] if c.get("class_iri")]
        refs += [(k, e.get("iri")) for k, e in (meta.get("concept_map") or {}).items() if e.get("iri")]
        for owner, iri in refs:
            checked += 1
            if iri not in registry:
                missing_db.append({"table": name, "owner": owner, "iri": iri})
            if owl is not None and (meta.get("concept_map") or {}).get(owner, {}).get("status") == "minted" \
                    and iri not in owl:
                missing_owl.append({"table": name, "owner": owner, "iri": iri})
    return {"checked": checked, "missing_in_db": missing_db, "missing_in_owl": missing_owl,
            "ok": not missing_db and not missing_owl}
