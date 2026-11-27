"""Second metadata pass on a LOADED table: measure its structure, define its
measures, then test every stored query against the data.

``profiler`` writes the first metadata from a profile and a sample (no
database). Here the table is in Postgres, so the metadata can be checked
against how the data actually behaves:

1. **Analyse** (:mod:`analysis`, SQL only): grain, functional dependencies,
   repeat statistics (EVIDENCE of report/version columns, never a verdict),
   item columns, total rows.
2. **Define** (task ``table_refine``, one LLM call with a domain-neutral
   prompt): column corrections, MEASURES (value column, item values,
   independent identity columns, display labels, period and qualifier
   columns, and -- only with a reason -- the version column whose latest value
   wins), usage notes, typical questions.
3. **Build** measure queries deterministically from each definition: one row
   per observation = identity + item + period + qualifier columns. Source rows
   collapse only under the definition's version rule; identity columns are
   optional filters.
4. **Test** the EXACT stored SQL of every item (compiled once the metadata it
   depends on exists, lookups with both tables' total-row markers) with
   parameter values read from the table; the outcome is recorded per item as
   ``passed`` / ``failed`` / ``not_exercised`` against the SQL's hash
   (:func:`metadata.record_test`). A non-empty result is not enough: a measure
   must return every source observation exactly once (tied conflicting
   versions are flagged for review), a join must hold its declared
   cardinality and not multiply aggregated rows, an aggregate must not add a
   value restated across confirmed versions. Failed metric/filter/query/lookup
   specs get one repair call (``table_sql_repair``); failed measure
   DEFINITIONS get one bounded repair call (``table_measure_repair``) that
   keeps the original, the failure and the proposal; what still fails is
   dropped (catalog items) or kept for inspection (``failed_measures``).
5. After the optional verification (:mod:`verify`), every item is
   re-finalised with the final metadata and whatever changed is re-tested
   (:func:`ensure_tested`); only ``passed`` items are selectable at retrieval.
   ``catalog_version`` marks the result; older catalogs migrate with no LLM
   (``table-metadata-refine --relink``).

The refined metadata replaces the node's metadata and the cache entry
(``refined.version``), so later runs make no LLM call.
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from collections.abc import Awaitable, Callable
from typing import Any

from sqlalchemy import text as sql_text

from backend.app.services.structured.analysis import analyze_table, summary_for_prompt, version_columns
from backend.app.services.structured.metadata import (
    CATALOG_VERSION,
    COLUMN_ROLES,
    SECTIONS,
    column_types,
    finalize_item,
    item_spec,
    items,
    record_test,
    sql_sha,
    tag_items,
    time_columns,
)
from backend.app.services.structured.query_builder import QuerySpecError, build_sql, compact
from backend.app.services.structured.sql_exec import run_sql

REFINE_VERSION = "tr-1"
MAX_MEASURES = 400               # safety net only: the inventory is batched, never truncated
MAX_FAMILY_VALUES = 200          # indicator values a parameterised measure family may cover
COVERAGE_STATUSES = ("materialized", "supported_not_materialized", "unsupported", "unsupported_by_source",
                     "semantic_ambiguity", "unresolved", "incomplete_due_to_budget",
                     "incomplete_due_to_truncation", "invalid_reply")
MAX_ITEM_VALUES = 8     # variants of ONE quantity, not a list of quantities
SAMPLE_ROWS = 8
EmbedFn = Callable[[list[str]], Awaitable[list[list[float]]]]


def _parse(text: str) -> dict:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", (text or "").strip())
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        try:
            obj = json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:
            obj = {}
    return obj if isinstance(obj, dict) else {}


# ------------------------------------------------------------------ measures


def _determined(col: str, by: list[str], deps: dict[str, list[str]]) -> bool:
    return any(col in (deps.get(i) or []) for i in by)


def split_identity(candidates: list[str], display: list[str], analysis: dict | None
                   ) -> tuple[list[str], list[str], list[str]]:
    """(identity columns, display columns, review notes).

    An identity column is INDEPENDENT: together with the others it identifies
    an observation (a plan and a drug). A display column is a label one of the
    identity columns determines (a plan's name). The split is checked against
    the MEASURED dependencies: a column is display only when an identity
    column exactly determines it. When the data only MOSTLY determines it, the
    distinction is ambiguous -- it stays an identity column (no observation
    is merged away) and the definition is flagged for re-refinement."""
    an = analysis or {}
    deps, near = an.get("dependencies") or {}, an.get("near_dependencies") or {}
    ident: list[str] = []
    disp: list[str] = []
    notes: list[str] = []
    for c in candidates:
        if not ident or not _determined(c, ident, deps):
            if ident and _determined(c, ident, near):
                notes.append(f"{c} is only mostly determined by {', '.join(ident)}: kept as an identity "
                             "column (observations preserved); re-refine to confirm")
            ident.append(c)
        else:
            disp.append(c)
    for c in display:
        if c in ident or c in disp:
            continue
        if _determined(c, ident, deps):
            disp.append(c)
        else:
            why = ("only mostly determined by" if _determined(c, ident, near) else "not determined by")
            notes.append(f"{c} was given as a display label but is {why} {', '.join(ident)}: kept as an "
                         "identity column (observations preserved); re-refine to confirm")
            ident.append(c)
    return ident, disp, notes


def normalize_measure(raw: Any, meta: dict, analysis: dict | None = None) -> tuple[dict | None, str | None]:
    """Validate one measure definition against the table. (measure, error).

    Nothing the model proposed as an identity, period or qualifier column is
    removed because its values repeat (a price can stay the same for years):
    a column leaves the observation key only when the definition names it the
    ``version_column`` with a ``version_reason`` -- then rows that differ only
    in it are the same observation and the highest version wins. A definition
    written before identity/display columns existed (``entity_columns``) is
    split with the measured dependencies (:func:`split_identity`)."""
    if not isinstance(raw, dict):
        return None, "not an object"
    cols = {c["name"]: c for c in meta.get("columns") or []}
    mid = re.sub(r"[^a-z0-9_]+", "_", str(raw.get("id") or raw.get("label") or "").lower()).strip("_")[:40]
    if not mid:
        return None, "missing id"
    vcol = raw.get("value_column")
    if vcol not in cols or cols[vcol].get("type") not in {"integer", "numeric"}:
        return None, f"value_column {vcol!r} is not a numeric column"
    item = raw.get("item")
    if item:
        if not isinstance(item, dict) or item.get("column") not in cols:
            return None, f"item column {item!r} does not exist"
        vals = item.get("values") if isinstance(item.get("values"), list) else [item.get("value")]
        vals = [str(v) for v in vals if v not in (None, "")]
        if not vals:
            return None, "item has no values"
        # a FAMILY (indicators sharing one definition, unit and aggregation
        # rule) keeps its values and takes the indicator as a parameter; more
        # values than naming variants of one quantity, without that claim,
        # make the column a dimension of the measure (a parameter)
        family = bool(item.get("family")) and len(set(vals)) > 1
        if family:
            item = {"column": item["column"], "values": list(dict.fromkeys(vals))[:MAX_FAMILY_VALUES],
                    "family": True}
        else:
            item = ({"column": item["column"], "values": list(dict.fromkeys(vals))}
                    if len(set(vals)) <= MAX_ITEM_VALUES else {"column": item["column"], "as_param": True})

    def _list(key: str) -> list[str]:
        return [c for c in dict.fromkeys(raw.get(key) or []) if isinstance(c, str)]

    legacy = not raw.get("identity_columns")
    cand = _list("entity_columns") if legacy else _list("identity_columns")
    display = [] if legacy else _list("display_columns")
    periods = [c for c in _list("period_columns") if c not in cand + display]
    quals = _list("qualifier_columns")
    bad = [c for c in cand + display + periods + quals if c not in cols]
    if bad:
        return None, f"unknown column(s): {', '.join(bad)}"
    review: list[str] = []
    reason = str(raw.get("version_reason") or "").strip()
    order, order_notes = _version_order(raw, cols, set(cand + display + periods + quals), reason)
    review += order_notes
    version = order[0]["column"] if order else None
    population, pop_notes = _population(raw.get("population"), cols)
    review += pop_notes
    dims = [c for c in _list("dimension_columns") if c in cols and c not in cand + display + periods + quals]
    if item and (cols[item["column"]].get("role") == "time" or item["column"] in periods + quals):
        item = None        # a period/qualifier is not an item: keep the measure, drop the constant
    if item and (item.get("as_param") or item["column"] in cand + display):
        if item["column"] not in cand:
            cand.insert(1 if cand else 0, item["column"])
        display = [d for d in display if d != item["column"]]
        item = None
    if not cand:
        return None, "needs at least one identity column"
    subject = [c for c in _list("subject_columns") if c in cols]
    for c in subject:
        if c not in cand:
            cand.insert(0, c)
    ident, disp, notes = split_identity(cand, display, analysis)
    review += notes
    subject, ident, disp, sub_notes = _subject(subject, ident, disp, cols, analysis,
                                               per_row=not periods and not item)
    review += sub_notes
    # a few-valued column the period determines is the period's GRANULARITY
    # (annual/monthly, quarter/year-to-date...): mixing them in one range
    # mixes different quantities, so it is a qualifier whatever the model said
    domains = meta.get("value_domains") or {}
    deps = (analysis or {}).get("dependencies") or {}
    for p in periods:
        for d in deps.get(p, []):
            if (d not in ident + disp + periods + quals and d != version
                    and len((domains.get(d) or {}).get("values") or []) > 1):
                quals.append(d)
    quals = [q for q in quals if q not in ident + disp + periods]
    return {"id": mid, "label": str(raw.get("label") or mid), "description": str(raw.get("description") or ""),
            "value_column": vcol, "item": item,
            "identity_columns": ident, "display_columns": disp,
            "entity_columns": ident + disp,                      # older readers
            "period_columns": periods, "qualifier_columns": quals,
            "subject_columns": subject,
            "uniqueness_columns": [c for c in ident if c not in subject],
            "dimension_columns": dims,
            "version_column": version, "version_reason": reason if version else "",
            "latest_by": version, "version_order": order,
            "tie_policy": str(raw.get("tie_policy") or "").strip() or None,
            "population": population,
            "additive_over_time": bool(raw.get("additive_over_time"))
            or str(raw.get("aggregation_across_time") or "").lower() == "sum",
            **_semantics(raw, cols),
            "param_columns": [], "review": review}, None


_POP_OPS = {"=", "!=", "in", "not_in", "is_null", "not_null", "<", "<=", ">", ">="}
_NEGATE = {"=": "!=", "!=": "=", "in": "not_in", "not_in": "in", "is_null": "not_null",
           "not_null": "is_null", "<": ">=", "<=": ">", ">": "<=", ">=": "<"}


def _version_order(raw: dict, cols: dict, observation: set[str], reason: str) -> tuple[list[dict], list[str]]:
    """The ordered version rule: [{column, direction, nulls, basis}] where
    ``basis`` is ``precedence`` (the source says a later value supersedes:
    a restatement, a later release) or ``tie_break`` (deterministic only, no
    business meaning). Legacy ``version_column`` = one precedence entry,
    highest first. A precedence entry needs ``version_reason``."""
    notes: list[str] = []
    entries = raw.get("version_order")
    if not isinstance(entries, list) or not entries:
        v = raw.get("version_column") or raw.get("latest_by")
        entries = [{"column": v, "direction": "desc", "nulls": "last", "basis": "precedence"}] if v else []
    out: list[dict] = []
    for e in entries:
        if isinstance(e, str):
            e = {"column": e}
        if not isinstance(e, dict):
            continue
        c = e.get("column")
        if c not in cols:
            if c:
                notes.append(f"version column {c!r} does not exist: not used")
            continue
        if c in observation:
            notes.append(f"{c} is both part of the observation and a version column: kept as part of the "
                         "observation, not used for versions")
            continue
        d = str(e.get("direction") or e.get("dir") or "desc").lower()
        n = str(e.get("nulls") or "last").lower()
        b = str(e.get("basis") or "precedence").lower()
        out.append({"column": c, "direction": d if d in {"asc", "desc"} else "desc",
                    "nulls": n if n in {"first", "last"} else "last",
                    "basis": b if b in {"precedence", "tie_break"} else "precedence"})
    if out and not reason and not raw.get("latest_by"):
        notes.append("version column(s) given without a reason: not used")
        return [], notes
    if out and out[0]["basis"] != "precedence":
        notes.append("a version order must start with a precedence column (a tie-break alone "
                     "says nothing about which value is current): not used")
        return [], notes
    return out, notes


def _population(raw: Any, cols: dict) -> tuple[dict, list[str]]:
    """Fixed inclusion / exclusion predicates of the population a quantity is
    defined over: {"include": [cond], "exclude": [cond]} with cond
    {column, op, value}. Exclusions are stored negated, as where entries."""
    notes: list[str] = []
    out: dict[str, list] = {"include": [], "exclude": []}
    if not isinstance(raw, dict):
        return out, notes
    for side in ("include", "exclude"):
        for c in raw.get(side) or []:
            if not isinstance(c, dict) or c.get("column") not in cols:
                notes.append(f"population {side} {c!r}: unknown column, not used")
                continue
            op = str(c.get("op") or ("in" if isinstance(c.get("value"), list) else "=")).lower()
            if op not in _POP_OPS:
                notes.append(f"population {side} on {c['column']}: op {op!r} not supported, not used")
                continue
            if op not in {"is_null", "not_null"} and "value" not in c:
                notes.append(f"population {side} on {c['column']}: no value, not used")
                continue
            e = {"column": c["column"], "op": op}
            if "value" in c:
                e["value"] = c["value"]
            out[side].append(e)
    return out, notes


def population_where(pop: dict | None) -> list[dict]:
    """Where entries enforcing a population (inclusions as given, exclusions
    negated)."""
    out = []
    for e in (pop or {}).get("include") or []:
        out.append(dict(e))
    for e in (pop or {}).get("exclude") or []:
        if e["op"] in {"=", "in"}:
            # NULL-safe: a row whose column is empty is not an excluded row
            vals = e["value"] if isinstance(e.get("value"), list) else [e.get("value")]
            out.append({"column": e["column"], "op": "not_in", "value": vals})
            continue
        out.append({**e, "op": _NEGATE[e["op"]]})
    for w in out:
        if w["op"] == "!=" and isinstance(w.get("value"), list):
            w["op"] = "not_in"
    return out


def _subject(subject: list[str], ident: list[str], disp: list[str], cols: dict, analysis: dict | None,
             *, per_row: bool) -> tuple[list[str], list[str], list[str], list[str]]:
    """The identifier of the thing measured. Given explicitly, it is kept.
    Otherwise: for a measure with one value per row whose identity is the
    table's grain, a row key with the identifier role (unique and filled on
    every row) is the subject and the old identity columns become display
    labels (a row key determines every column of its row); else the
    identifier-role columns of the identity, else its first column."""
    notes: list[str] = []
    an = analysis or {}
    if subject:
        return subject, ident, disp, notes
    keys = [c for c in an.get("row_keys") or [] if (cols.get(c) or {}).get("role") == "identifier"]
    grain = set(an.get("grain") or [])
    if per_row and keys and an.get("grain_unique") and grain and grain <= set(ident) \
            and not set(keys) & set(ident):
        key = keys[0]
        notes.append(f"subject {key}: unique on every row; {', '.join(ident)} kept as display labels")
        return [key], [key], [*[c for c in ident if c != key], *disp], notes
    ids = [c for c in ident if (cols.get(c) or {}).get("role") == "identifier"]
    return (ids or ident[:1]), ident, disp, notes


_AGG_RULES = {"sum", "average", "weighted_average", "last", "none", "min", "max", "count"}


def _semantics(raw: dict, cols: dict) -> dict:
    """The descriptive part of a measure definition (unit, valid calculations,
    groupings, aggregation across entities / time, weights, denominators,
    period rule): kept as given when it names real columns."""
    def _cols(key: str) -> list[str]:
        return [c for c in dict.fromkeys(raw.get(key) or []) if isinstance(c, str) and c in cols]

    def _rule(key: str) -> str | None:
        v = str(raw.get(key) or "").strip().lower().replace(" ", "_")
        return v if v in _AGG_RULES else (v or None)

    out = {"unit": (str(raw["unit"]) if raw.get("unit") else None),
           "calculations": [str(c) for c in raw.get("calculations") or [] if c][:12],
           "group_by_columns": _cols("group_by_columns"), "filter_columns": _cols("filter_columns"),
           "aggregation_across_entities": _rule("aggregation_across_entities"),
           "aggregation_across_time": _rule("aggregation_across_time"),
           "weight_column": raw.get("weight_column") if raw.get("weight_column") in cols else None,
           "denominator_column": raw.get("denominator_column") if raw.get("denominator_column") in cols else None,
           "period_rule": str(raw["period_rule"]) if raw.get("period_rule") else None,
           "calculation_note": str(raw.get("calculation_note") or "")[:500],
           "missing_evidence": str(raw["missing_evidence"]) if raw.get("missing_evidence") else None}
    if not out["unit"]:
        unit_col = next((c for c in cols.values() if c.get("name") == raw.get("value_column")), {})
        out["unit"] = unit_col.get("unit")
    return out


def observation_key(m: dict) -> list[str]:
    """Columns that identify one observation of the measure: every identity
    column, the item column when the measure covers several item values, the
    period columns and the qualifiers. Deduplication uses exactly this key."""
    ident = m.get("identity_columns") or (m.get("entity_columns") or [])[:1]
    item = m.get("item") or {}
    item_col = [item["column"]] if item.get("column") and len(item.get("values") or []) > 1 else []
    return list(dict.fromkeys([*ident, *(m.get("param_columns") or []), *item_col,
                               *m.get("period_columns", []), *m.get("qualifier_columns", [])]))


def measure_item(m: dict, meta: dict) -> dict:
    """One stored query per measure, built without an LLM: one row per
    observation (:func:`observation_key`). Source rows are collapsed only when
    the definition has a version rule (``version_column`` + reason): then the
    highest version of each observation is kept. Without one, every source row
    stays (a key that does not identify one value then fails its test instead
    of silently picking a row). Each identity column is an optional filter
    parameter, the listed qualifiers and a period range are parameters."""
    domains = meta.get("value_domains") or {}
    ident = m.get("identity_columns") or (m.get("entity_columns") or [])[:1]
    display = m.get("display_columns") or []
    dims = m.get("dimension_columns") or []
    order_rule = m.get("version_order") or (
        [{"column": m.get("version_column") or m.get("latest_by"), "direction": "desc", "nulls": "last",
          "basis": "precedence"}] if (m.get("version_column") or m.get("latest_by")) else [])
    key = observation_key(m)
    sel = list(dict.fromkeys([*ident, *display, *dims, *key, m["value_column"],
                              *[o["column"] for o in order_rule]]))
    where: list[dict] = population_where(m.get("population"))
    if m["item"]:
        vals = m["item"]["values"]
        where.append({"column": m["item"]["column"], "op": "in" if len(vals) > 1 else "=",
                      "value": vals if len(vals) > 1 else vals[0]})
        if m["item"].get("family"):
            # one template for the whole family: pick one indicator, or leave
            # it out to list every indicator of the family
            where.append({"column": m["item"]["column"], "op": "=", "param": m["item"]["column"],
                          "optional": True})
    where.append({"column": m["value_column"], "op": "not_null"})
    # identity parameters are optional: leaving them out lists the measure
    # for EVERY observation (qualifiers and the period stay required, so a
    # listing never mixes units, bases or granularities)
    for c in [*ident, *(m.get("param_columns") or []), *dims]:
        where.append({"column": c, "op": "=", "param": c, "optional": True})
    # columns the definition names as filters stay optional filters even when
    # they are not part of the key (a key reduced to a document number must
    # still answer "for this filer"); columns the population or item fixes,
    # the period and the qualifiers are handled on their own
    taken = {w["column"] for w in where} | set(m.get("period_columns") or []) | set(m["qualifier_columns"])
    for c in m.get("filter_columns") or []:
        if c not in taken:
            taken.add(c)
            where.append({"column": c, "op": "=", "param": c, "optional": True})
            if c not in sel:
                sel.append(c)
    # a qualifier an identity column (nearly) determines -- each item has its
    # own unit, say -- is optional too: fixing it would drop the other items
    # from a listing, and leaving it out mixes nothing (it is per item)
    an = meta.get("analysis") or {}
    described = {d for c in ident for d in [*(an.get("dependencies") or {}).get(c, []),
                                            *(an.get("near_dependencies") or {}).get(c, [])]}
    for q in m["qualifier_columns"]:
        if len((domains.get(q) or {}).get("values") or []) > 1:
            where.append({"column": q, "op": "=", "param": q,
                          **({"optional": True} if q in described else {})})
    if m["period_columns"]:
        p = m["period_columns"][-1]
        where.append({"column": p, "op": "between", "param": [f"{p}_from", f"{p}_to"]})
    spec: dict = {"select": sel, "where": where}
    order = [{"column": c, "dir": "asc"} for c in key]
    if order_rule:
        spec["distinct_on"] = key
        order += [{"column": o["column"], "dir": o["direction"], "nulls": o["nulls"]} for o in order_rule]
    spec["order_by"] = order
    rule = ", then ".join(f"{o['column']} {o['direction']} (nulls {o['nulls']}, {o['basis']})"
                          for o in order_rule)
    desc = (m["description"] + " One value per " + ", ".join(key)
            + (f"; when a value is restated the first row by {rule} wins ({m.get('version_reason')})"
               if order_rule else "") + ".").strip()
    return {"id": f"measure_{m['id']}", "label": m["label"], "description": desc,
            "columns": sel, "class_iri": None, "class_label": None,
            "query": spec, "params": [], "sql": "", "measure": m,
            **({"review": list(m["review"])} if m.get("review") else {})}


def _keep_rejected_definition(meta: dict, definition, stage: str, reason: str) -> None:
    """A measure definition refused before it became a stored query is kept
    (as proposed) so a later repair starts from it."""
    import datetime as _dt

    meta.setdefault("rejected_definitions", []).append({
        "id": definition.get("id") if isinstance(definition, dict) else str(definition),
        "stage": stage, "reason": reason, "definition": definition,
        "at": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds")})


def apply_uniqueness_evidence(m: dict, identity: dict, meta: dict | None = None) -> dict:
    """Identity columns the measured key did not need (``redundant``: no
    observation collides without them) become dimension columns -- still
    returned and filterable, no longer part of the key or the subject.
    With ``meta``, columns that define meaning (:func:`meaning_columns`) stay
    in the key even when older evidence listed them as redundant."""
    ev = identity.get("uniqueness") or {}
    keep = meaning_columns(meta, m) if meta is not None else {}
    drop = [c for c in ev.get("redundant") or [] if c in (m.get("identity_columns") or []) and c not in keep]
    if not drop:
        return m
    ident = [c for c in m["identity_columns"] if c not in drop]
    subj = [c for c in m.get("subject_columns") or [] if c not in drop] or ident[:1]
    m = {**m, "identity_columns": ident, "subject_columns": subj,
         "uniqueness_columns": [c for c in m.get("uniqueness_columns") or [] if c in ident and c not in subj],
         "dimension_columns": list(dict.fromkeys([*(m.get("dimension_columns") or []), *drop]))}
    m["review"] = [*(m.get("review") or []),
                   f"{', '.join(drop)}: no observation collides without it (measured); kept as a "
                   "dimension -- still returned and an optional filter parameter -- not as part of the key"]
    return m


def rebuild_measures(meta: dict, all_types: dict[str, dict[str, str]]) -> list[dict]:
    """Re-derive every measure query from its stored definition (no LLM), so
    a change to how measures are built reaches already-refined tables. A
    definition written before identity/display columns is re-normalised
    against the measured dependencies first. A measure that no longer builds
    keeps its previous query (its test then decides whether it may run)."""
    out = []
    for it in meta.get("measures") or []:
        m = it.get("measure")
        if not isinstance(m, dict):
            out.append(it)
            continue
        m = apply_uniqueness_evidence(m, (it.get("verification") or {}).get("identity") or {}, meta)
        # every stored definition is re-normalised: rules added since it was
        # written (subject identifier, ordered versions, population) apply
        norm, _err = normalize_measure(m, meta, meta.get("analysis"))
        if norm is not None:
            m = norm
        new = measure_item(m, meta)
        for k in ("class_iri", "class_label", "aliases"):
            if it.get(k) is not None:
                new[k] = it.get(k)
        new["id"] = it.get("id") or new["id"]
        if finalize_item(meta, "measures", new, all_types) is not None:
            out.append(it)
            continue
        _carry_review(meta, it, new)
        out.append(new)
    return out


def _carry_review(meta: dict, old: dict, new: dict) -> None:
    """A rebuilt item keeps its semantic review (verdict and reviewed text)
    only when its SQL is unchanged; otherwise the verdict is marked stale and
    the item waits for a new review (the old verdict is kept for the record).
    A catalog verified before verdicts were stored per item takes the verdict
    from the table's verification report."""
    from backend.app.services.structured.metadata import semantic_status

    v = old.get("verify_verdict")
    if not isinstance(v, dict) or not v.get("verdict"):
        legacy = semantic_status(meta, old)
        if legacy in {"not_required", "unreviewed", "pending", "stale"}:
            return
        v = old["verify_verdict"] = {"verdict": legacy, "source": "verification report (legacy catalog)"}
    if old.get("sql") and old.get("sql") == new.get("sql"):
        for k in ("verify_verdict", "verified", "label", "description", "review", "hold"):
            if old.get(k) is not None:
                new[k] = old[k]
        if not v.get("sql_sha"):
            v["sql_sha"] = sql_sha(old["sql"])
        return
    new["verify_verdict"] = {"verdict": "stale", "previous": v,
                             "reason": "the rebuilt SQL differs from the SQL that was reviewed"}


# ------------------------------------------------------------------ tests


def _base_where(spec: dict, table: str, cols: set[str]) -> list[dict]:
    """The spec's constant filters on the base table (no params)."""
    out = []
    for w in spec.get("where") or []:
        c = w.get("column")
        name = c if isinstance(c, str) else (c or {}).get("column")
        if isinstance(name, str) and "." in name:
            t, name = name.split(".", 1)
            if t != table:
                continue
        if name in cols and ("value" in w or w.get("op") in {"is_null", "not_null"}):
            out.append(w)
    return out


async def example_params(meta: dict, section: str, item: dict, built, *,
                         include_optional: bool = False,
                         fixed: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """A WITNESS: parameter values read from ONE real row that satisfies the
    item's constant filters (between params get that row's value at both
    ends). Only REQUIRED parameters are read by default -- an optional filter
    left out must not restrict the witness (a row lacking a value for an
    optional column is a valid witness); optional filters are tested one by
    one afterwards (``include_optional`` with the required values ``fixed``)."""
    table = meta["table_name"]
    types = column_types(meta)
    spec = item_spec(section, item) or {}
    if spec.get("from") and spec["from"] != table:
        return None
    pcols = {}
    for p in built.params:
        if p.get("optional") and not include_optional:
            continue
        col = p.get("column") or ""
        base = re.sub(r"^(year|month|quarter|lower)_", "", col) if col not in types else col
        if base in types:
            pcols[p["name"]] = base
    need = sorted(set(pcols.values()))
    if not need:
        return {p["name"]: 5 for p in built.params if not p.get("optional") or include_optional}
    eq = []
    for p in built.params:          # required values already chosen: the row must share them
        c = pcols.get(p["name"])
        if fixed and p["name"] in fixed and c and fixed[p["name"]] is not None \
                and not isinstance(fixed[p["name"]], list):
            eq.append({"column": c, "op": "=", "value": fixed[p["name"]]})
    # the probe also selects / groups the item's own plain columns, so the
    # builder adds the SAME program conditions (total rows: parts only when a
    # column is selected) as it does to the item
    own = [c for c in sorted(_grouped_and_filtered(spec)) if c in types and c not in need]
    probe = {"select": need + own,
             "where": _base_where(spec, table, set(types)) + eq
             + [{"column": c, "op": "not_null"} for c in need],
             "limit": 1}
    try:
        pb = build_sql(probe, table, {table: types})
    except QuerySpecError:
        return None
    probe_sql = _with_fixed_conditions(pb.sql, item.get("sql") or "") if not spec.get("join") else pb.sql
    res = await run_sql(probe_sql, {}, {table}, max_rows=1, timeout_ms=20000)
    if res.error or not res.rows:
        return None
    row = dict(zip(res.columns, res.rows[0], strict=False))
    out: dict[str, Any] = {}
    for p in built.params:
        if p.get("optional") and p["name"] not in pcols:
            continue                    # left out: every value
        col = pcols.get(p["name"])
        if col is None:
            out[p["name"]] = 5          # a limit / top-n
            continue
        v = row.get(col)
        if (p.get("column") or "").startswith("year_") and v:
            v = int(str(v)[:4])
        out[p["name"]] = [v] if p["type"].endswith("[]") else v
    return out


def _with_fixed_conditions(probe_sql: str, item_sql: str) -> str:
    """The probe plus every parameter-free condition of the item's COMPILED
    SQL (including those the builder added: total rows, empty values), so the
    witness row is one the item can return."""
    import sqlglot
    from sqlglot import exp

    try:
        item_ast = sqlglot.parse_one(item_sql, read="postgres")
        probe_ast = sqlglot.parse_one(probe_sql, read="postgres")
    except Exception:
        return probe_sql
    where = item_ast.args.get("where")
    if where is None:
        return probe_sql
    conds = list(where.this.flatten()) if isinstance(where.this, exp.And) else [where.this]
    for c in conds:
        if c.find(exp.Placeholder) is None and c.find(exp.Parameter) is None and c.find(exp.Subquery) is None:
            probe_ast = probe_ast.where(c.copy(), append=True)
    return probe_ast.sql(dialect="postgres")


def _grouped_and_filtered(spec: dict) -> set[str]:
    out = set()
    for r in (spec.get("group_by") or []) + [s for s in spec.get("select") or [] if not (isinstance(s, dict) and s.get("agg"))]:
        out.add(r if isinstance(r, str) else (r or {}).get("column"))
    for w in spec.get("where") or []:
        if str(w.get("op") or "=") == "=":
            c = w.get("column")
            out.add(c if isinstance(c, str) else (c or {}).get("column"))
    return {c for c in out if c}


def _done(item: dict, status: str, **detail) -> dict:
    v = record_test(item, status, **detail)
    return {"ok": status == "passed", **v}


async def _internal(sql: str, params: dict, declared: list[dict], *, timeout_ms: int = 20000) -> list | None:
    """One INTERNAL diagnostic statement (never an answer): read-only, timed
    out, parameters bound by the driver. None when it fails."""
    from backend.app.db.engine import get_sessionmaker
    from backend.app.services.structured.sql_exec import coerce_params

    bound = coerce_params(params or {}, declared)
    for p in declared or []:
        if p.get("optional") and p.get("name") not in bound:
            bound[p["name"]] = None
    names = set(re.findall(r"(?<![:\w]):([A-Za-z_]\w*)", sql))
    bound = {k: v for k, v in bound.items() if k in names}
    async with get_sessionmaker()() as s:
        try:
            await s.execute(sql_text("SET TRANSACTION READ ONLY"))
            await s.execute(sql_text(f"SET LOCAL statement_timeout = {int(timeout_ms)}"))
            await s.execute(sql_text("SET LOCAL search_path TO graphrag"))
            return [tuple(r) for r in (await s.execute(sql_text(sql), bound)).all()]
        except Exception:
            return None
        finally:
            await s.rollback()


def _qi(c: str) -> str:
    return '"' + c.replace('"', '""') + '"'


async def listing_params(meta: dict, item: dict, params: dict) -> dict:
    """Parameters that LIST the measure: optional filters left out, the period
    range over every period of the table, the rest as in ``params``."""
    m = item.get("measure") or {}
    # every optional filter left out (identities, an indicator family, a
    # qualifier an identity determines): the listing covers all of them
    out = {k: v for k, v in params.items()
           if not any(p.get("name") == k and p.get("optional") for p in item.get("params") or [])}
    periods = m.get("period_columns") or []
    if periods:
        p = periods[-1]
        rows = await _internal(f"SELECT MIN({_qi(p)}), MAX({_qi(p)}) FROM {_qi(meta['table_name'])}", {}, [])
        if rows and rows[0][0] is not None:
            out[f"{p}_from"], out[f"{p}_to"] = rows[0][0], rows[0][1]
    return out


MAX_QUALIFIER_GROUPS = 25        # identity check: qualifier combinations run one by one (else sampled)


def data_version(meta: dict | None) -> str | None:
    return ((meta or {}).get("source") or {}).get("file_sha")


def data_versions(meta: dict, item: dict, partners: dict[str, dict] | None) -> dict[str, str | None]:
    """{table: data version} of every table the item's SQL reads."""
    table = meta["table_name"]
    tables = list(item.get("tables") or []) or [table]
    if item.get("other_table") and item["other_table"] not in tables:
        tables.append(item["other_table"])
    return {t: data_version(meta if t == table else (partners or {}).get(t)) for t in tables}


def _expected_where(meta: dict, m: dict, item: dict, qual: dict[str, Any]) -> tuple[list[str], dict]:
    """WHERE conditions (and binds) of the SOURCE rows a measure listing must
    cover, derived from the definition and the table's total rows -- not from
    the generated query."""
    from backend.app.services.structured.analysis import total_markers

    totals = total_markers(meta)
    conds = [f"{_qi(m['value_column'])} IS NOT NULL"]
    binds: dict[str, Any] = {}
    it = m.get("item") or {}
    if it.get("values"):
        names = []
        for i, v in enumerate(it["values"]):
            binds[f"iv{i}"] = v
            names.append(f":iv{i}")
        conds.append(f"CAST({_qi(it['column'])} AS text) IN ({', '.join(names)})")
    ident = m.get("identity_columns") or (m.get("entity_columns") or [])[:1]
    selected = set(ident) | set(m.get("display_columns") or []) | set(observation_key(m)) | \
        ({m.get("version_column")} - {None})
    for i, (c, v) in enumerate(qual.items()):
        binds[f"q{i}"] = None if v is None else str(v)
        conds.append(f"CAST({_qi(c)} AS text) IS NOT DISTINCT FROM :q{i}")
    for c, marker in totals.items():
        if c in qual:
            continue
        if c in ident:                       # optional identity param left out: the published total
            binds[f"t_{c}"] = str(marker)
            conds.append(f"CAST({_qi(c)} AS text) = :t_{c}")
        elif c in selected:                  # selected: the parts only
            binds[f"t_{c}"] = str(marker)
            conds.append(f"CAST({_qi(c)} AS text) IS DISTINCT FROM :t_{c}")
        else:                                # not used: the total rows only
            binds[f"t_{c}"] = str(marker)
            conds.append(f"CAST({_qi(c)} AS text) = :t_{c}")
    for c in item.get("empty_excluded") or []:
        conds.append(f"NULLIF(CAST({_qi(c)} AS text), '') IS NOT NULL")
    for i, w in enumerate(population_where(m.get("population"))):
        conds.append(_cond_sql(w, binds, f"pop{i}"))
    return conds, binds


def _cond_sql(w: dict, binds: dict, name: str) -> str:
    """One population where entry as SQL over text-cast values (bound)."""
    c, op = f"CAST({_qi(w['column'])} AS text)", w["op"]
    if op == "is_null":
        return f"{_qi(w['column'])} IS NULL"
    if op == "not_null":
        return f"{_qi(w['column'])} IS NOT NULL"
    vals = w["value"] if isinstance(w.get("value"), list) else [w.get("value")]
    if op in {"in", "not_in", "=", "!="} :
        names = []
        for j, v in enumerate(vals):
            binds[f"{name}_{j}"] = None if v is None else str(v)
            names.append(f":{name}_{j}")
        if op in {"in", "="}:
            return f"{c} IN ({', '.join(names)})"
        return f"({_qi(w['column'])} IS NULL OR {c} NOT IN ({', '.join(names)}))"
    binds[name] = vals[0]
    return f"{_qi(w['column'])} {op} :{name}"


async def _bound_internal(sql: str, binds: dict) -> list | None:
    """Read-only diagnostic with plain text binds (no declared parameters)."""
    from sqlalchemy import text as sql_text

    from backend.app.db.session import session_scope

    try:
        async with session_scope() as s:
            await s.execute(sql_text("SET TRANSACTION READ ONLY"))
            await s.execute(sql_text("SET LOCAL search_path TO graphrag"))
            await s.execute(sql_text("SET LOCAL statement_timeout = 20000"))
            return list((await s.execute(sql_text(sql), binds)).all())
    except Exception:  # a diagnostic that cannot run is reported as not exercised
        return None


async def measure_identity_check(meta: dict, item: dict, params: dict,
                                 types_by_table: dict[str, dict[str, str]]) -> dict:
    """Every source observation must come back exactly once, in EVERY
    qualifier group. The expected observations are derived from the
    definition directly on the source table (:func:`_expected_where`), not from
    the generated query; the measure's own stored SQL is run per qualifier
    group with its identity filters left out. Also checks version conflicts
    over the whole table and, when an identity column has total rows, that
    the measure returns the published totals at the same grain.

    Returns {"ok", "ran", "coverage": "complete"|"sampled", "groups", ...};
    ``ran`` False when the diagnostic could not run (-> not_exercised)."""
    m = item.get("measure") or {}
    key = observation_key(m)
    table = _qi(meta["table_name"])
    declared = item.get("params") or []
    lp = await listing_params(meta, item, params)
    qcols = [p.get("column") for p in declared
             if p.get("column") in (m.get("qualifier_columns") or []) and not p.get("optional")]
    qcols = list(dict.fromkeys(c for c in qcols if c))
    combos: list[dict] = [{}]
    total_groups = 1
    if qcols:
        cond, b = _expected_where(meta, m, item, {})
        rows = await _bound_internal(
            f"SELECT DISTINCT {', '.join('CAST(' + _qi(c) + ' AS text)' for c in qcols)} FROM {table} "
            f"WHERE {' AND '.join(cond)} ORDER BY 1", b)
        if rows is None:
            return {"ok": False, "ran": False, "error": "qualifier groups could not be listed"}
        total_groups = len(rows)
        combos = [dict(zip(qcols, r, strict=False)) for r in rows[:MAX_QUALIFIER_GROUPS]]
    keys = ", ".join(_qi(c) for c in key)
    out: dict[str, Any] = {"key": key, "groups": total_groups, "groups_checked": len(combos),
                           "coverage": "complete" if len(combos) >= total_groups else "sampled",
                           "source": 0, "returned_rows": 0, "returned_keys": 0, "ran": True}
    by_name = {p.get("column"): p.get("name") for p in declared}
    for combo in combos:
        cond, b = _expected_where(meta, m, item, combo)
        periods = m.get("period_columns") or []
        if periods and lp.get(f"{periods[-1]}_from") is not None:
            p = periods[-1]
            b["pf"], b["pt"] = lp[f"{p}_from"], lp[f"{p}_to"]
            cond.append(f"{_qi(p)} BETWEEN :pf AND :pt")
        exp = await _bound_internal(f"SELECT COUNT(*) FROM (SELECT DISTINCT {keys} FROM {table} "
                                    f"WHERE {' AND '.join(cond)}) d", b)
        run = {**lp, **{by_name[c]: v for c, v in combo.items() if by_name.get(c)}}
        got = await _internal(f"SELECT COUNT(*), (SELECT COUNT(*) FROM (SELECT DISTINCT {keys} FROM "
                              f"({item['sql']}) m2) d) FROM ({item['sql']}) m", run, declared)
        if not exp or not got:
            return {**out, "ok": False, "ran": False, "error": "identity check did not run"}
        e, r_rows, r_keys = int(exp[0][0]), int(got[0][0]), int(got[0][1])
        out["source"] += e
        out["returned_rows"] += r_rows
        out["returned_keys"] += r_keys
        if r_rows > r_keys:
            return {**out, "ok": False, "qualifier_group": combo,
                    "error": (f"{r_rows - r_keys} observation(s) come back more than once"
                              + (f" in qualifier group {combo}" if combo else "")
                              + f": ({', '.join(key)}) does not identify one value. Add the column that tells "
                              "the rows apart to the identity/qualifier columns, or define why they are the "
                              "same observation (version_column + version_reason).")}
        if r_keys != e:
            return {**out, "ok": False, "qualifier_group": combo,
                    "error": (f"returns {r_keys} of the {e} observations in the source rows"
                              + (f" of qualifier group {combo}" if combo else "")
                              + " under the same filters")}
    conflict = await version_conflicts(meta, m, item)
    if conflict is not None:
        if not conflict.get("ran", True):
            return {**out, "ok": False, "ran": False, "error": "version-conflict check did not run"}
        out["collapsed"], out["tied"] = conflict["collapsed"], conflict["tied"]
        out["tie_examples"] = conflict["examples"]
        if conflict["tied"]:
            if conflict["tie_breaks"] and m.get("tie_policy"):
                out["tie_resolution"] = (f"{conflict['tied']} conflicting observation(s) resolved by the "
                                         f"explicit policy: {m['tie_policy']}")
            else:
                prec = ", ".join(conflict["precedence"])
                return {**out, "ok": False,
                        "error": (f"{conflict['tied']} observation(s) have several rows at the winning "
                                  f"precedence ({prec}) with different values"
                                  + (f"; {', '.join(conflict['tie_breaks'])} only breaks the tie "
                                     "deterministically, it is no evidence of which value is current"
                                     if conflict["tie_breaks"] else "")
                                  + ": the measure stays unavailable until the source or an explicit "
                                    "tie policy resolves them")}
    subj = await subject_check(meta, m, item, lp, declared)
    if subj is not None:
        out["subject"] = subj
        if not subj.get("ok"):
            return {**out, "ok": False, "ran": subj.get("ran", True), "error": subj.get("error")}
    out["uniqueness"] = await uniqueness_evidence(meta, m, item)
    total = await _published_total_check(meta, m, item, lp, declared)
    if total is not None:
        out["published_total"] = total
        if not total.get("ok"):
            return {**out, "ok": False, "ran": total.get("ran", True), "error": total.get("error")}
    out["ok"] = True
    return out


async def version_conflicts(meta: dict, m: dict, item: dict, *, examples: int = 5) -> dict | None:
    """Observations whose rows tie at the winning PRECEDENCE version (the
    leading ``basis: precedence`` entries of the version order) with
    different values. Tie-break entries are deliberately ignored: they pick a
    row deterministically but are no evidence that its value is current.
    None when the measure has no version rule."""
    order = m.get("version_order") or (
        [{"column": m.get("version_column") or m.get("latest_by"), "direction": "desc", "nulls": "last",
          "basis": "precedence"}] if (m.get("version_column") or m.get("latest_by")) else [])
    if not order:
        return None
    prec = [o for o in order if o["basis"] == "precedence"]
    ties = [o["column"] for o in order if o["basis"] == "tie_break"]
    key = observation_key(m)
    keys = ", ".join(_qi(c) for c in key)
    pcols = ", ".join(_qi(o["column"]) for o in prec)
    porder = ", ".join(f"{_qi(o['column'])} {o['direction'].upper()} NULLS {o['nulls'].upper()}" for o in prec)
    val = _qi(m["value_column"])
    cond, b = _expected_where(meta, m, item, {})
    table = _qi(meta["table_name"])
    base = f"""
        WITH g AS (SELECT {keys}, {pcols}, COUNT(DISTINCT {val}) AS nd,
                          array_agg(DISTINCT CAST({val} AS text)) AS vals
                     FROM {table} WHERE {' AND '.join(cond)} GROUP BY {keys}, {pcols}),
             r AS (SELECT g.*, ROW_NUMBER() OVER (PARTITION BY {keys} ORDER BY {porder}) AS rk,
                          COUNT(*) OVER (PARTITION BY {keys}) AS versions FROM g)"""
    r = await _bound_internal(base + """
        SELECT COUNT(*) FILTER (WHERE rk = 1 AND versions > 1), COUNT(*) FILTER (WHERE rk = 1 AND nd > 1)
          FROM r""", b)
    if r is None:
        return {"ran": False}
    out = {"ran": True, "collapsed": int(r[0][0]), "tied": int(r[0][1]),
           "precedence": [o["column"] for o in prec], "tie_breaks": ties, "examples": []}
    if out["tied"]:
        ex = await _bound_internal(base + f"""
            SELECT {keys}, {pcols}, vals FROM r WHERE rk = 1 AND nd > 1 ORDER BY {keys} LIMIT {int(examples)}""", b)
        cols = [*key, *out["precedence"], "values"]
        out["examples"] = [dict(zip(cols, [str(x) if not isinstance(x, list) else x for x in row], strict=False))
                           for row in ex or []]
    return out


async def subject_check(meta: dict, m: dict, item: dict, lp: dict, declared: list) -> dict | None:
    """A per-subject measure must be callable with ONE known subject key and
    return that key next to its value. The key is read from the source rows
    the definition covers; the stored SQL is run with it."""
    subject = m.get("subject_columns") or []
    if not subject:
        return None
    by_col = {p.get("column"): p for p in declared}
    missing = [c for c in subject if c not in by_col]
    if missing:
        return {"ok": False, "error": f"subject column(s) {', '.join(missing)} cannot be passed as a parameter"}
    sel = set(item.get("columns") or [])
    if not set(subject) <= sel:
        return {"ok": False, "error": f"subject column(s) {', '.join(sorted(set(subject) - sel))} are not "
                                      "returned with the value"}
    cols = ", ".join(_qi(c) for c in subject)
    # a subject the measure itself lists with the same (non-subject) parameter
    # values: the call then differs only in the subject parameter
    base = {k: v for k, v in lp.items() if k not in {by_col[c]["name"] for c in subject}}
    row = await _internal(f"SELECT {cols} FROM ({item['sql']}) s "
                          f"WHERE {' AND '.join(_qi(c) + ' IS NOT NULL' for c in subject)} LIMIT 1",
                          base, declared)
    if row is None:
        return {"ok": False, "ran": False, "error": "subject check did not run"}
    if not row:
        return None                    # the listing is empty (the identity check covers it)
    keyv = dict(zip(subject, row[0], strict=False))
    params = {**base, **{by_col[c]["name"]: keyv[c] for c in subject}}
    got = await _internal(f"SELECT {cols}, COUNT(*) OVER () FROM ({item['sql']}) s LIMIT 50", params, declared)
    if got is None:
        return {"ok": False, "ran": False, "error": "subject call did not run"}
    if not got:
        return {"ok": False, "key": {k: str(v) for k, v in keyv.items()},
                "error": f"called with subject {keyv} it returned no rows"}
    wrong = [r for r in got if [str(x) for x in r[:len(subject)]] != [str(keyv[c]) for c in subject]]
    if wrong:
        return {"ok": False, "key": {k: str(v) for k, v in keyv.items()},
                "error": f"called with subject {keyv} it returned other subjects"}
    return {"ok": True, "key": {k: str(v) for k, v in keyv.items()}, "rows": int(got[0][-1])}


async def uniqueness_evidence(meta: dict, m: dict, item: dict) -> dict:
    """Which identity columns are needed to tell observations apart, measured
    on the source rows. Per identity column: ``collisions_without`` (key
    groups that would mix several of its values without it) and
    ``max_rows_sharing_a_value`` (a placeholder or shared code shows up
    here). ``redundant``: columns removed greedily -- the most widely shared
    values first -- while the remaining key still tells every observation
    apart; at least one column stays."""
    out: dict[str, Any] = {}
    ident = list(m.get("identity_columns") or [])
    if len(ident) < 2:
        return out
    key = observation_key(m)
    cond, b = _expected_where(meta, m, item, {})
    where = " AND ".join(cond)
    t = _qi(meta["table_name"])

    async def _collisions(rest: list[str], c: str) -> int | None:
        if not rest:
            return None
        r = await _bound_internal(
            f"SELECT COUNT(*) FROM (SELECT 1 FROM {t} WHERE {where} GROUP BY {', '.join(_qi(x) for x in rest)} "
            f"HAVING COUNT(DISTINCT {_qi(c)}) > 1) d", b)
        return None if r is None else int(r[0][0])

    for c in ident:
        n = await _collisions([x for x in key if x != c], c)
        share = await _bound_internal(f"SELECT MAX(n) FROM (SELECT COUNT(*) AS n FROM {t} WHERE {where} "
                                      f"GROUP BY {_qi(c)}) d", b)
        out[c] = {"collisions_without": n, "max_rows_sharing_a_value": int(share[0][0] or 0) if share else None}
    order = sorted(ident, key=lambda c: -((out.get(c) or {}).get("max_rows_sharing_a_value") or 0))
    left, redundant = list(key), []
    protected = meaning_columns(meta, m)
    for c in order:
        if len([x for x in left if x in ident]) <= 1:
            break
        if c in protected:
            continue        # zero collisions here does not show the column carries no meaning
        if await _collisions([x for x in left if x != c], c) == 0:
            left.remove(c)
            redundant.append(c)
    if redundant:
        out["redundant"] = redundant
    kept = {c: why for c, why in protected.items() if c in ident}
    if kept:
        out["protected"] = kept
    return out


def _related_names(col: dict) -> set[str]:
    out = set()
    for r in col.get("related") or []:
        n = r.get("column") or r.get("name") if isinstance(r, dict) else r
        if isinstance(n, str):
            out.add(n)
    return out


def meaning_columns(meta: dict, m: dict) -> dict[str, str]:
    """Identity columns that define WHAT an observation is, so measured
    redundancy alone never demotes them (a sample without collisions does not
    show they carry no meaning): {column: reason}. Generic signals only:
    published total rows, a ``qualifier`` role in the column metadata, and
    the measure's item column(s) with the columns the profile relates to them."""
    from backend.app.services.structured.analysis import total_markers

    cols = {c["name"]: c for c in meta.get("columns") or []}
    item_cols = {m["item"]["column"]} if isinstance(m.get("item"), dict) and m["item"].get("column") else set()
    related: set[str] = set()
    for ic in item_cols:
        related |= _related_names(cols.get(ic) or {})
    related |= {n for n, c in cols.items() if item_cols & _related_names(c)}
    out: dict[str, str] = {}
    totals = set(total_markers(meta))
    for c in m.get("identity_columns") or []:
        col = cols.get(c) or {}
        roles = {*(col.get("roles") or []), col.get("role")}
        if c in totals:
            out[c] = "has published total rows (decides total vs parts)"
        elif "qualifier" in roles:
            out[c] = "role qualifier: it defines what the value means"
        elif c in item_cols:
            out[c] = "the measure's item column"
        elif c in related:
            out[c] = "related to the item column in the column metadata (identifies the concept with it)"
    return out


async def _published_total_check(meta: dict, m: dict, item: dict, lp: dict, declared: list) -> dict | None:
    """When an identity column has total rows, the measure with that filter
    left out claims the published total: it must return exactly the source
    total rows' values at the same grain."""
    from backend.app.services.structured.analysis import total_markers

    totals = total_markers(meta)
    ident = m.get("identity_columns") or []
    cols = [c for c in ident if c in totals]
    if not cols:
        return None
    key = [c for c in observation_key(m) if c not in cols]
    val = _qi(m["value_column"])
    cond, b = _expected_where(meta, m, item, {})
    periods = m.get("period_columns") or []
    if periods and lp.get(f"{periods[-1]}_from") is not None:
        b["pf"], b["pt"] = lp[f"{periods[-1]}_from"], lp[f"{periods[-1]}_to"]
        cond.append(f"{_qi(periods[-1])} BETWEEN :pf AND :pt")
    kk = ", ".join(_qi(c) for c in key) or "1"
    src = await _bound_internal(f"SELECT COUNT(*), SUM({val}) FROM (SELECT {kk}, {val} FROM "
                                f"{_qi(meta['table_name'])} WHERE {' AND '.join(cond)}) s", b)
    got = await _internal(f"SELECT COUNT(*), SUM({val}) FROM ({item['sql']}) m", lp, declared)
    if not src or not got:
        return {"ok": False, "ran": False, "error": "published-total check did not run"}
    s_n, s_sum, g_n, g_sum = src[0][0], src[0][1], got[0][0], got[0][1]
    same = int(s_n) == int(g_n) and (s_sum == g_sum or (s_sum is not None and g_sum is not None
                                                        and abs(float(s_sum) - float(g_sum)) < 1e-6))
    out = {"ok": same, "ran": True, "columns": cols, "source_rows": int(s_n), "returned_rows": int(g_n)}
    if not same:
        out["error"] = (f"with {', '.join(cols)} left out the measure should return the published total rows "
                        f"({int(s_n)} rows, sum {s_sum}); it returned {int(g_n)} rows, sum {g_sum}")
    return out


async def join_check(meta: dict, item: dict) -> dict | None:
    """A lookup's join on its COMPLETE key: the declared cardinality must hold
    and an aggregate must not multiply the rows it adds up. None without a
    join. {"ok", "base_rows_multiplied", "other_rows_multiplied", "error"?}"""
    spec = item_spec("lookups", item) or {}
    j = spec.get("join") or {}
    other, base = j.get("table"), spec.get("from") or meta["table_name"]
    on = [(p.get("column"), p.get("other_column")) for p in j.get("on") or []]
    if not other or not on:
        return None
    cond = " AND ".join(f"b.{_qi(a)} = o.{_qi(c)}" for a, c in on)
    r = await _internal(f"""
        SELECT (SELECT COUNT(*) FROM {_qi(base)} b JOIN {_qi(other)} o ON {cond}),
               (SELECT COUNT(*) FROM {_qi(base)} b WHERE EXISTS (SELECT 1 FROM {_qi(other)} o WHERE {cond})),
               (SELECT COUNT(*) FROM {_qi(other)} o WHERE EXISTS (SELECT 1 FROM {_qi(base)} b WHERE {cond}))""",
                        {}, [])
    if not r:
        return {"ok": False, "error": "join check did not run"}
    joined, base_hit, other_hit = (int(x) for x in r[0])
    out: dict[str, Any] = {"joined_rows": joined, "base_rows_multiplied": joined > base_hit,
                           "other_rows_multiplied": joined > other_hit, "ok": True}
    keytxt = ", ".join(f"{a} = {other}.{c}" for a, c in on)
    card = str(item.get("cardinality") or "").lower()
    if card in {"many_to_one", "one_to_one"} and out["base_rows_multiplied"]:
        out.update(ok=False, error=f"declared {card} but one {base} row matches several {other} rows on "
                                   f"the complete key ({keytxt})")
    elif card in {"one_to_many", "one_to_one"} and out["other_rows_multiplied"]:
        out.update(ok=False, error=f"declared {card} but one {other} row matches several {base} rows on "
                                   f"the complete key ({keytxt})")
    if not out["ok"]:
        return out
    aggs = [a for a in spec.get("select") or [] if isinstance(a, dict) and a.get("agg")]
    for a in aggs:
        if str(a.get("agg")) in {"min", "max", "count_distinct"}:
            continue                      # repeated rows do not change these
        col = a.get("column")
        name = col.get("column") if isinstance(col, dict) else col
        tbl = (col.get("table") if isinstance(col, dict) else None) or (
            str(name).split(".", 1)[0] if isinstance(name, str) and "." in str(name) else base)
        if name in (None, "", "*"):
            hit = out["base_rows_multiplied"] or out["other_rows_multiplied"]
        else:
            hit = out["base_rows_multiplied"] if tbl == base else out["other_rows_multiplied"]
        if hit:
            out.update(ok=False, error=(f"{str(a.get('agg')).upper()}({name or '*'}) is multiplied by the join: "
                                        f"rows of {tbl if name not in (None, '', '*') else 'the join'} match "
                                        f"several rows of the other table on {keytxt}"))
            return out
    return out


async def test_item(meta: dict, section: str, item: dict, analysis: dict,
                    types_by_table: dict[str, dict[str, str]], *,
                    partners: dict[str, dict] | None = None) -> dict:
    """Test the item and stamp the data version of every table its SQL reads
    (a later change of any of them invalidates the result: see
    :func:`ensure_tested`)."""
    r = await _test_item(meta, section, item, analysis, types_by_table, partners=partners)
    m = item.get("measure") if section == "measures" and isinstance(item.get("measure"), dict) else None
    ident = ((item.get("verification") or {}).get("identity") or {})
    if m is not None and r.get("ok") and (ident.get("uniqueness") or {}).get("redundant"):
        # the measured key is smaller: rebuild on it and test that (once)
        m2 = apply_uniqueness_evidence(m, ident, meta)
        norm, _err = normalize_measure(m2, meta, analysis)
        if norm is not None and list(norm.get("identity_columns") or []) == list(m2["identity_columns"]):
            trial = measure_item(norm, meta)
            for k in ("id", "class_iri", "class_label", "aliases", "tags"):
                if item.get(k) is not None:
                    trial[k] = item[k]
            r2 = await _test_item(meta, section, trial, analysis, types_by_table, partners=partners)
            if r2.get("ok"):
                before = list(m.get("identity_columns") or [])
                _carry_review(meta, item, trial)      # new SQL -> its semantic verdict is stale
                item.clear()
                item.update(trial)
                item["verification"]["key_reduced_from"] = before
                r = r2
    if item.get("verification"):
        item["verification"]["data_versions"] = data_versions(meta, item, partners)
    from backend.app.services.structured.metadata import describe_query

    item["capability"] = describe_query(meta, section, item)
    return r


async def _test_item(meta: dict, section: str, item: dict, analysis: dict,
                     types_by_table: dict[str, dict[str, str]], *,
                     partners: dict[str, dict] | None = None) -> dict:
    """Test the item's EXACT stored SQL (compiled first from its spec with the
    current metadata) and record the outcome in ``item["verification"]``.

    ``status`` is ``passed``, ``failed`` or ``not_exercised`` (no parameter
    values could be read from the table: the item stays out of the retrieval
    catalog until a later run exercises it). A non-empty result is not enough:
    a measure must return every source observation exactly once, a join must
    not multiply the rows it aggregates, an aggregate must not add a value
    restated across versions. Returns {"ok", "status", ...}."""
    from types import SimpleNamespace

    if isinstance(item.get("query"), dict):
        err = finalize_item(meta, section, item, types_by_table, partners=partners)
        if err:
            return _done(item, "failed", error=f"does not compile: {err}")
    sql, declared = item.get("sql") or "", item.get("params") or []
    if not sql:
        return _done(item, "failed", error="no SQL")
    tables = set(item.get("tables") or []) or {meta["table_name"]} | (
        {item["other_table"]} if item.get("other_table") else set())
    params = await example_params(meta, section, item, SimpleNamespace(params=declared))
    if params is None:
        return _done(item, "not_exercised", note="no example row found for its parameters")
    res = await run_sql(sql, params, tables, max_rows=50, timeout_ms=20000, declared_params=declared)
    if res.error:
        return _done(item, "failed", error=res.error, params=params)
    if not res.rows:
        return _done(item, "failed", error="returned no rows with parameter values taken from the table",
                     params=params)
    checks = ["runs", "returns rows for values read from the table"]
    detail: dict[str, Any] = {}
    spec = item_spec(section, item) or {}
    if spec.get("join"):
        jc = await join_check(meta, item)
        if jc is not None:
            detail["join"] = {k: v for k, v in jc.items() if k not in {"ok", "error"}}
            if not jc["ok"]:
                return _done(item, "failed", error=jc["error"], params=params, join=detail["join"])
            checks.append("join cardinality on the complete key; no multiplied aggregate rows")
    if section == "measures" and isinstance(item.get("measure"), dict):
        ic = await measure_identity_check(meta, item, params, types_by_table)
        detail["identity"] = {k: v for k, v in ic.items() if k not in {"ok", "error"}}
        if not ic.get("ran", True):
            return _done(item, "not_exercised", note=ic.get("error"), params=params, identity=detail["identity"])
        if not ic["ok"]:
            return _done(item, "failed", error=ic["error"], params=params, identity=detail["identity"])
        checks.append("returns every source observation exactly once "
                      f"({ic.get('coverage')}: {ic.get('groups_checked')} of {ic.get('groups')} qualifier group(s))")
        if ic.get("published_total"):
            checks.append("returns the published total rows when its identity filter is left out")
    opt = await _optional_filter_tests(meta, section, item, params, tables, declared)
    if opt:
        detail["optional_filters"] = opt
        bad = {k: v for k, v in opt.items() if v.startswith("failed")}
        if bad:
            return _done(item, "failed", error="optional filter " + "; ".join(f"{k}: {v}" for k, v in bad.items()),
                         params=params, **detail)
        if any(v == "passed" for v in opt.values()):
            checks.append("optional filters return rows for a value read from the table")
    aggs = [s for s in spec.get("select") or [] if isinstance(s, dict)
            and str(s.get("agg") or "") in {"sum", "avg"}]
    if aggs and not spec.get("join"):
        err = await _restated_check(meta, spec, params, analysis, types_by_table)
        if err:
            return _done(item, "failed", error=err, params=params)
        checks.append("no value restated across versions added twice")
    return _done(item, "passed", rows=res.row_count, params=params, checks=checks, **detail)


async def _optional_filter_tests(meta: dict, section: str, item: dict, params: dict, tables: set[str],
                                 declared: list[dict]) -> dict[str, str]:
    """Each optional single-value filter, tested on its own: a value read from
    a row that also satisfies the required witness values must return rows,
    and only rows with that value when the column is returned."""
    from types import SimpleNamespace

    out: dict[str, str] = {}
    for p in declared:
        if not p.get("optional") or p.get("type", "").endswith("[]") or not p.get("column"):
            continue
        one = SimpleNamespace(params=[q for q in declared if not q.get("optional") or q["name"] == p["name"]])
        wp = await example_params(meta, section, item, one, include_optional=True, fixed=params)
        if not wp or wp.get(p["name"]) is None:
            out[p["name"]] = "no witness row"
            continue
        trial = {**params, p["name"]: wp[p["name"]]}
        res = await run_sql(item["sql"], trial, tables, max_rows=200, timeout_ms=20000, declared_params=declared)
        if res.error:
            out[p["name"]] = f"failed: {res.error}"
        elif not res.rows:
            out[p["name"]] = f"failed: no rows for {p['name']} = {wp[p['name']]!r}"
        elif p["column"] in (res.columns or []):
            i = list(res.columns).index(p["column"])
            other = [r[i] for r in res.rows if str(r[i]) != str(wp[p["name"]])]
            out[p["name"]] = (f"failed: {p['name']} = {wp[p['name']]!r} also returned {other[:3]!r}"
                              if other else "passed")
        else:
            out[p["name"]] = "passed"
    return out


async def _restated_check(meta: dict, spec: dict, params: dict, analysis: dict,
                          types_by_table: dict[str, dict[str, str]]) -> str | None:
    """An aggregate must not add the same value restated in several versions,
    nor combine periods while grouping by a version column. Only CONFIRMED
    version columns (:func:`version_columns`) count."""
    table = meta["table_name"]
    report = version_columns(meta, analysis)
    if not report:
        return None
    keyed = _grouped_and_filtered(spec)
    groups = [s for s in spec.get("select") or [] if not (isinstance(s, dict) and s.get("agg"))]
    where = spec.get("where") or []
    identity = [c for c in (analysis or {}).get("grain") or [] if c not in report]
    if identity:
        diag = {"select": [*groups, {"agg": "count", "as": "n"},
                           {"agg": "count_distinct", "columns": identity, "as": "k"}],
                "where": where, "group_by": spec.get("group_by") or []}
        r = await _run_diag(diag, table, params, types_by_table)
        if r and any(int(x["n"]) > int(x["k"]) for x in r):
            worst = max(r, key=lambda x: int(x["n"]) - int(x["k"]))
            return (f"aggregates the same reported value more than once: {worst['n']} rows but only "
                    f"{worst['k']} distinct values of {', '.join(identity)}; rows differ only in "
                    f"{', '.join(report)}, which identify the version the value came from. Filter "
                    "those to one version, or use a measure query.")
    periods = [c for c in time_columns(meta) if c not in report]
    if periods and (keyed & set(report)) and not (keyed & set(periods)):
        diag = {"select": [*groups, {"agg": "count_distinct", "columns": periods, "as": "p"}],
                "where": where, "group_by": spec.get("group_by") or []}
        r = await _run_diag(diag, table, params, types_by_table)
        if r and any(int(x["p"]) > 1 for x in r):
            n = max(int(x["p"]) for x in r)
            return (f"groups by {', '.join(sorted(keyed & set(report)))}, which identify the VERSION, not "
                    f"the period: one result row combines {n} different periods "
                    f"({', '.join(periods)}). Group or filter by the period columns instead.")
    return None


async def ensure_tested(meta: dict, *, analysis: dict, all_types: dict[str, dict[str, str]],
                        partners: dict[str, dict] | None = None, force: bool = False) -> dict:
    """Make every stored item's SQL final for the CURRENT metadata and tested:
    items whose SQL changed (or that were never tested at this catalog
    version) are re-tested. Sets ``catalog_version``. Returns counts."""
    counts = {"passed": 0, "failed": 0, "not_exercised": 0, "retested": 0}
    for sec, it in list(items(meta)):
        before = it.get("sql")
        if isinstance(it.get("query"), dict):
            err = finalize_item(meta, sec, it, all_types, partners=partners)
            if err:
                _done(it, "failed", error=f"does not compile: {err}")
        v = it.get("verification") or {}
        dv_now = data_versions(meta, it, partners)
        dv_was = v.get("data_versions") or {}
        data_changed = any(t in dv_was and dv_was[t] != sha for t, sha in dv_now.items() if sha)
        stale = (force or before != it.get("sql") or v.get("sql_sha") != sql_sha(it.get("sql"))
                 or v.get("catalog_version") != CATALOG_VERSION or data_changed
                 or (v and "data_versions" not in v))
        if data_changed:
            it["verification"] = {**v, "status": "not_exercised", "note": "a table it reads changed"}
        if stale and (force or data_changed or v.get("status") != "failed"):
            await test_item(meta, sec, it, analysis, all_types, partners=partners)
            counts["retested"] += 1
        counts[(it.get("verification") or {}).get("status") or "failed"] += 1
    meta["catalog_version"] = CATALOG_VERSION
    review_duplicate_sql(meta)
    return counts


def _declared_population(it: dict) -> Any:
    m = it.get("measure") if isinstance(it.get("measure"), dict) else None
    if m is not None:
        return m.get("population") or {"include": [], "exclude": []}
    return {"fixed": [w for w in (it.get("query") or {}).get("where") or []
                      if isinstance(w, dict) and w.get("param") in (None, "")]}


async def revive_dropped(meta: dict, *, all_types: dict[str, dict[str, str]],
                         partners: dict[str, dict] | None = None) -> list[dict]:
    """Stored queries dropped because they did not compile or failed their
    data test (NOT verifier or duplicate drops) are rebuilt with the current
    builder -- safe parameter names, current rules -- and tested again. One
    that now passes goes back into its section with its text marked for
    review (``verify_verdict`` pending). Returns one row per retried item:
    id, section, earlier error, outcome, parameter renames."""
    out = []
    keep = []
    for d in meta.get("dropped") or []:
        stage = str(d.get("stage") or "")
        reason = str(d.get("reason") or d.get("error") or "")
        q = d.get("query")
        if isinstance(q, str):
            try:
                q = json.loads(q)
            except json.JSONDecodeError:
                q = None
        sec = d.get("section")
        if (not isinstance(q, dict) or sec not in {"metrics", "filters", "queries"}
                or stage in {"verification", "duplicate_sql"} or reason.startswith("verification")):
            keep.append(d)
            continue
        item = {"id": d["id"], "label": d.get("label") or d["id"].replace("_", " "),
                "description": d.get("description") or "", "query": q, "params": d.get("params") or [],
                "sql": "", "class_iri": None, "class_label": None}
        used = {it["id"] for _s, it in items(meta)}
        if item["id"] in used:
            keep.append(d)
            continue
        r = await test_item(meta, sec, item, meta.get("analysis") or {}, all_types, partners=partners)
        row = {"id": item["id"], "section": sec, "before": reason[:300], "outcome": r["status"],
               **({"param_renames": item["param_renames"]} if item.get("param_renames") else {}),
               **({"error": r.get("error")} if r["status"] != "passed" else {})}
        out.append(row)
        if r["status"] == "passed":
            item["review"] = [f"restored after a rebuild (was dropped: {reason[:200]}); text not yet reviewed"]
            item["verify_verdict"] = {"verdict": "pending"}
            meta[sec].append(item)
        else:
            keep.append(d)
    meta["dropped"] = keep
    if out:
        meta["revived"] = out
    return out


def review_duplicate_sql(meta: dict) -> list[dict]:
    """Differently named items whose stored SQL is identical. Same declared
    population: one quantity under two names -- the later one becomes an
    alias of the first (kept). Different declared populations: the SQL
    cannot honour both, so both fail their test until reviewed. Recorded in
    ``meta["sql_duplicates"]`` with each item's declared population."""
    groups: dict[str, list[tuple[str, dict]]] = {}
    for sec, it in items(meta):
        if (it.get("verification") or {}).get("status") == "passed" and it.get("sql"):
            groups.setdefault(sql_sha(it["sql"]), []).append((sec, it))
    found = []
    for sha, grp in groups.items():
        if len(grp) < 2:
            continue
        pops = {it["id"]: _declared_population(it) for _s, it in grp}
        same = len({json.dumps(p, sort_keys=True, default=str) for p in pops.values()}) == 1
        _keep_sec, keep = grp[0]
        entry = {"sql_sha": sha, "ids": [it["id"] for _s, it in grp], "populations": pops,
                 "decision": "aliased" if same else "failed: declared populations differ, SQL is identical"}
        for sec, it in grp[1:]:
            if same:
                keep["aliases"] = list(dict.fromkeys([*(keep.get("aliases") or []), it["id"],
                                                      *(it.get("aliases") or [])]))
                meta[sec].remove(it)
                meta.setdefault("dropped", []).append({"id": it["id"], "section": sec, "stage": "duplicate_sql",
                                                       "reason": f"same SQL and population as {keep['id']}",
                                                       "alias_of": keep["id"]})
        if not same:
            for _s, it in grp:
                _done(it, "failed", error=("identical SQL to " + ", ".join(i for i in entry["ids"] if i != it["id"])
                                           + " although their declared populations differ: the population "
                                             "is not enforced"))
        note = f"identical SQL: {', '.join(entry['ids'])} ({entry['decision']})"
        for _s, it in grp:
            it["review"] = list(dict.fromkeys([*(it.get("review") or []), note]))
        found.append(entry)
    meta["sql_duplicates"] = found
    return found


async def _run_diag(spec: dict, table: str, params: dict, types_by_table) -> list[dict] | None:
    try:
        b = build_sql(spec, table, {table: types_by_table[table]})
    except QuerySpecError:
        return None
    res = await run_sql(b.sql, params, {table}, max_rows=200, timeout_ms=20000, declared_params=b.params)
    if res.error:
        return None
    return [dict(zip(res.columns, r, strict=False)) for r in res.rows]


# ------------------------------------------------------------------ driver


async def _sample_rows(meta: dict) -> list[dict]:
    from backend.app.db.session import session_scope

    cols = [c["name"] for c in meta.get("columns") or []]
    async with session_scope() as s:
        rows = (await s.execute(sql_text(
            f'SELECT {", ".join(chr(34) + c + chr(34) for c in cols)} FROM graphrag."{meta["table_name"]}" '
            f"TABLESAMPLE SYSTEM (5) LIMIT {SAMPLE_ROWS}"))).all()
        if len(rows) < SAMPLE_ROWS:
            rows = (await s.execute(sql_text(
                f'SELECT {", ".join(chr(34) + c + chr(34) for c in cols)} FROM graphrag."{meta["table_name"]}" '
                f"LIMIT {SAMPLE_ROWS}"))).all()
    return [{c: (None if v is None else str(v)[:80]) for c, v in zip(cols, r, strict=True)} for r in rows]


def _prompt_table(meta: dict, analysis: dict, samples: list[dict]) -> dict:
    domains = meta.get("value_domains") or {}
    cols = []
    for c in meta.get("columns") or []:
        d = domains.get(c["name"]) or {}
        e = {"name": c["name"], "type": c.get("type"), "role": c.get("role"),
             "description": c.get("description")}
        if "values" in d:
            e["values"] = d["values"][:40] + (["..."] if len(d["values"]) > 40 else [])
        elif "min" in d:
            e["range"] = [d["min"], d["max"]]
        cols.append(e)
    catalog = [{"id": it["id"], "section": sec, "label": it.get("label"), "query": compact(it["query"])}
               for sec, it in items(meta) if isinstance(it.get("query"), dict) and sec != "measures"]
    for c, mc in zip(cols, meta.get("columns") or [], strict=False):
        if mc.get("roles"):
            c["inventory_roles"] = mc["roles"]
        if mc.get("unit"):
            c["unit"] = mc["unit"]
    inv = meta.get("inventory") or {}
    return {"table": meta["table_name"], "description": meta.get("description"),
            "grain_description": meta.get("grain"), "row_count": analysis.get("row_count"),
            "columns": cols, "structure": summary_for_prompt(analysis),
            "candidate_inventory": {k: inv.get(k) for k in ("candidates", "countables",
                                                            "remaining_candidates") if inv.get(k)},
            "sample_rows": samples, "catalog": catalog}


REFINE_CANDIDATE_BATCH = 15      # candidate ids per table_refine call (grows / shrinks with throughput)
_INVENTORY_STATUSES = {"materialized", "supported_not_materialized", "unsupported", "unresolved",
                       "semantic_ambiguity", "unsupported_by_source"}


def _merge_unique(acc: dict, part: dict) -> None:
    """Merge a reply; measures / templates / columns are unique by id/name
    (a continuation never duplicates what an earlier call defined)."""
    for k, key in (("measures", "id"), ("templates", "id"), ("columns", "name")):
        have = {str((x or {}).get(key)) for x in acc.get(k) or [] if isinstance(x, dict)}
        for x in part.get(k) or []:
            if isinstance(x, dict) and str(x.get(key)) not in have:
                acc.setdefault(k, []).append(x)
                have.add(str(x.get(key)))
    acc.setdefault("common_questions", []).extend(x for x in part.get("common_questions") or [] if x)
    if part.get("usage_notes") and not acc.get("usage_notes"):
        acc["usage_notes"] = part["usage_notes"]


async def _refine_batched(router, ptable: dict, *, log: Callable[[str], None], table: str,
                          stats: list | None = None) -> tuple[dict, int]:
    """table_refine driven by CANDIDATE IDS: each call covers a batch of
    profiled candidates (by id) and must return one inventory entry per id.
    Complete entries of a cut-off reply are kept; missing ids are asked again
    in a smaller batch; budgets and a no-progress stop bound the work, and an
    id not completed is recorded with why (budget / truncation / invalid).
    No fixed number of rounds."""
    from backend.app.services.prompts import PROMPTS
    from backend.app.services.structured.continuation import continuation_settings, parse_reply, run_by_ids
    from backend.app.services.structured.wide_profile import candidate_id

    inv = ptable.get("candidate_inventory") or {}
    cands = [c for c in inv.get("candidates") or [] if isinstance(c, dict)]
    for i, c in enumerate(cands):
        c["candidate_id"] = c.get("candidate_id") or candidate_id(c, i)
    acc: dict = {}
    if not cands:                      # nothing to batch by: one call
        system, user = PROMPTS["table_refine"](ptable, batch=None)
        res = await router.chat("table_refine", system=system, user=user)
        part, _s = parse_reply(res.text)
        _merge_unique(acc, part)
        acc.setdefault("inventory", []).extend(x for x in part.get("inventory") or [] if x)
        return acc, 1
    ids = [c["candidate_id"] for c in cands]
    size, budget = continuation_settings(router, "table_refine", REFINE_CANDIDATE_BATCH)

    async def _call(batch: list[str], n: int):
        defined = sorted({str((m or {}).get("id")) for k in ("measures", "templates")
                          for m in acc.get(k) or [] if isinstance(m, dict)})
        cont = {"candidate_ids": batch, "already_defined": defined, "first_call": n == 1}
        system, user = PROMPTS["table_refine"](ptable, batch=cont)
        res = await router.chat("table_refine", system=system, user=user)
        part, salvaged = parse_reply(res.text)
        if salvaged:
            log(f"[structured] {table}: refine reply cut off -> kept its complete entries")
        _merge_unique(acc, part)
        recs = []
        for e in part.get("inventory") or []:
            if isinstance(e, dict):
                recs.append({**e, "id": str(e.get("candidate_id") or "")})
        return recs, res

    def _valid(_id: str, r: dict) -> str | None:
        return None if r.get("status") in _INVENTORY_STATUSES else "no valid status"

    done, failed, st = await run_by_ids("table_refine", ids, _call, validate=_valid, batch_size=size,
                                        budget=budget, log=log)
    by_id = {c["candidate_id"]: c for c in cands}
    acc["inventory"] = [{**{k: v for k, v in r.items() if k != "id"}, "candidate_id": cid,
                         "candidate": r.get("candidate") or by_id[cid].get("label") or cid}
                        for cid, r in done.items()]
    for cid, why in failed.items():
        c = by_id[cid]
        acc["inventory"].append({"candidate_id": cid, "candidate": c.get("label") or cid,
                                 "source": c.get("source_columns") or c.get("indicator"),
                                 "status": why, "reason": f"refinement stopped ({st.stop_reason}): {why}"})
    if stats is not None:
        stats.append(st.to_dict())
    log(f"[structured] {table}: refine covered {st.ids_done}/{st.ids_total} candidate(s) in "
        f"{st.calls} call(s) ({st.stop_reason})")
    return acc, st.calls


def _templates(meta: dict, raw: list, all_types, partners) -> list[str]:
    """Query templates the refinement proposed for missing calculations and
    breakdowns: added to ``metrics`` (they are tested like every item; a failed
    one is repaired once or dropped)."""
    from backend.app.services.structured.profiler import _norm_params

    used = {it["id"] for _s, it in items(meta)}
    added = []
    for t in raw:
        if not isinstance(t, dict) or not isinstance(t.get("query"), dict):
            continue
        tid = re.sub(r"[^a-z0-9_]+", "_", str(t.get("id") or t.get("label") or "").lower()).strip("_")[:48]
        if not tid:
            continue
        base, i = tid, 2
        while tid in used:
            tid, i = f"{base}_{i}", i + 1
        used.add(tid)
        it = {"id": tid, "label": str(t.get("label") or tid), "description": str(t.get("description") or ""),
              "columns": [], "class_iri": None, "class_label": None, "query": t["query"],
              "params": _norm_params(t.get("params")), "sql": "", "template_for": t.get("measure"),
              "added_by": "refine", **({"claimed_capability": str(t["capability"])[:500]}
                                       if t.get("capability") else {})}
        err = finalize_item(meta, "metrics", it, all_types, partners=partners)
        if err:
            meta.setdefault("dropped", []).append({"id": tid, "section": "metrics",
                                                   "query": compact(t["query"]),
                                                   "error": f"template does not compile: {err}"})
            continue
        meta["metrics"].append(it)
        added.append(tid)
    return added


def _coverage(meta: dict, raw: list, rejected: list[dict]) -> list[dict]:
    """The coverage inventory: one entry per candidate with one of
    :data:`COVERAGE_STATUSES` and the reason for every gap. Candidates of the
    profiling inventory the reply did not mention are kept as unresolved."""
    from backend.app.services.structured.wide_profile import candidate_id

    out, seen, seen_ids = [], set(), set()
    for e in raw:
        if not isinstance(e, dict) or not (e.get("candidate") or e.get("candidate_id")):
            continue
        st = e.get("status") if e.get("status") in COVERAGE_STATUSES else "unresolved"
        key = re.sub(r"[^a-z0-9]", "", str(e.get("candidate") or e.get("candidate_id")).lower())
        cid = e.get("candidate_id")
        if key in seen or (cid and cid in seen_ids):
            continue
        seen.add(key)
        if cid:
            seen_ids.add(cid)
        out.append({"candidate": str(e.get("candidate") or cid), "candidate_id": cid, "source": e.get("source"),
                    "status": st, "measure": e.get("measure"), "reason": str(e.get("reason") or "")[:500]})
    for i, c in enumerate((meta.get("inventory") or {}).get("candidates") or []):
        cid = c.get("candidate_id") or candidate_id(c, i)
        key = re.sub(r"[^a-z0-9]", "", str(c.get("label") or c.get("id") or "").lower())
        if key and key not in seen and cid not in seen_ids:
            seen.add(key)
            out.append({"candidate": c.get("label") or c.get("id"), "candidate_id": cid,
                        "source": c.get("source_columns") or c.get("indicator"), "status": "unresolved",
                        "reason": "profiled candidate not addressed by the refinement"})
    for r in rejected:
        out.append({"candidate": r.get("measure"), "status": "unresolved", "measure": r.get("measure"),
                    "reason": f"definition rejected: {r.get('error')}"})
    return out


LEDGER_ORDER = ("dedicated_measure", "generic_query", "partial", "unexercised", "verifier_drop",
                "unsupported", "unfinished", "gap")
_UNSUPPORTED = {"unsupported", "unsupported_by_source", "semantic_ambiguity"}
_UNFINISHED = {"incomplete_due_to_budget", "incomplete_due_to_truncation", "invalid_reply"}


def _slug(x: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(x or "").lower()).strip("_")[:60]


def _sql_columns(sql: str, names: set[str]) -> set[str]:
    found = set()
    for quoted, bare in re.findall(r'"([^"]+)"|\b([a-z_][a-z0-9_]*)\b', sql or ""):
        w = quoted or bare
        if w in names:
            found.add(w)
    return found


def _source_list(src: Any, names: set[str]) -> list[str]:
    if isinstance(src, list):
        return [c for c in src if c in names]
    found = [w for w in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", str(src or "")) if w in names]
    # 'x_2020 through x_2024': the family members in between
    m = re.findall(r"([A-Za-z_][A-Za-z0-9_]*?_)(\d{4}) through \1(\d{4})", str(src or ""))
    for stem, lo, hi in m:
        found += [f"{stem}{y}" for y in range(int(lo), int(hi) + 1) if f"{stem}{y}" in names]
    return list(dict.fromkeys(found))


def _agg_columns(it: dict) -> tuple[set[str], set[str]]:
    """(columns counted, columns aggregated otherwise) by the item's spec."""
    counted, other = set(), set()
    for x in (it.get("query") or {}).get("select") or []:
        if isinstance(x, dict) and x.get("agg"):
            c = str(x.get("column") or "*").split(".")[-1]
            (counted if str(x["agg"]).startswith("count") else other).add(c)
    return counted, other


def _cap_tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[A-Za-z0-9_\-\.:]+", text or "") if len(t) > 3}


def _capability_texts(meta: dict) -> list[str]:
    out = [f"{c.get('label') or ''} {c.get('source_columns') or ''}"
           for c in (meta.get("inventory") or {}).get("candidates") or []]
    out += [f"{e.get('candidate') or ''} {e.get('source') or ''}" for e in meta.get("coverage") or []]
    return out


async def indicator_values(meta: dict) -> dict[str, list[str]]:
    """Values of the table's text columns that capability texts NAME (exact
    match on the loaded data; one query per candidate column). Used by the
    coverage ledger when a column has too many values to list."""
    toks = sorted(set().union(*(_cap_tokens(t) for t in _capability_texts(meta))) - {
        c["name"] for c in meta.get("columns") or []})
    if not toks:
        return {}
    out: dict[str, list[str]] = {}
    for c in meta.get("columns") or []:
        if c.get("type") != "text" or c.get("role") in {"text", "measure"}:
            continue
        rows = await _bound_internal(
            f"SELECT DISTINCT {_qi(c['name'])} FROM {_qi(meta['table_name'])} "
            f"WHERE {_qi(c['name'])} = ANY(CAST(:toks AS text[])) LIMIT 500", {"toks": toks})
        if rows:
            out[c["name"]] = [r[0] for r in rows]
    return out


def _indicator_in_text(text: str, meta: dict, known: dict[str, list[str]] | None = None) -> dict | None:
    """A capability written as text that names VALUES of a column (an
    indicator family): {column, values}. Values come from the listed value
    domains and from ``known`` (:func:`indicator_values`); the column whose
    values the text names most wins; none when no value appears."""
    toks = _cap_tokens(text)
    best = None
    pools = {c: d.get("values") or [] for c, d in (meta.get("value_domains") or {}).items()}
    for c, vals in (known or {}).items():
        pools[c] = [*pools.get(c, []), *vals]
    for c, vals in pools.items():
        hit = list(dict.fromkeys(v for v in vals if isinstance(v, str) and len(v) > 3 and v in toks))
        if hit and (best is None or len(hit) > len(best["values"])):
            best = {"column": c, "values": hit}
    return best


def coverage_ledger(meta: dict, versions: dict[str, str] | None = None,
                    known_values: dict[str, list[str]] | None = None) -> list[dict]:
    """One row per CAPABILITY (a profiled candidate, or a refinement
    inventory entry), computed from the FINAL catalog: which selectable
    items answer it and how. Stable ids: the candidate id from profiling,
    else ``cap_<slug of the candidate>``. Matching (generic):

    * an INDICATOR capability (named values of a listed column) is answered
      by a measure restricted to those values, or by a query taking that
      column as a parameter whose allowed values include them;
    * a QUANTITY capability (numeric source columns) by the selectable
      items whose SQL reads those columns -- together; a measure covering
      all of them is dedicated;
    * a COUNT capability (non-numeric source columns) by a query counting
      them, or a measure named for it.

    category: dedicated_measure | generic_query | partial | pending_review |
    unexercised | verifier_drop | unsupported | unfinished | gap, with ``stage`` and
    ``reason`` for gaps. Stored in ``meta["coverage_ledger"]``."""
    from backend.app.services.structured.metadata import selectable
    from backend.app.services.structured.wide_profile import candidate_id

    cols = {c["name"]: c for c in meta.get("columns") or []}
    names = set(cols)
    numeric = {n for n, c in cols.items() if c.get("type") in {"integer", "numeric"}}
    live = []
    for sec, it in items(meta):
        live.append({"sec": sec, "it": it, "status": (it.get("verification") or {}).get("status"),
                     "selectable": selectable(meta, it, versions, semantic=False)[0],
                     "reviewed": selectable(meta, it, versions)[0],
                     "reads": _sql_columns(it.get("sql") or "", names), "agg": _agg_columns(it),
                     "m": it.get("measure") if isinstance(it.get("measure"), dict) else None})
    dropped = meta.get("dropped") or []

    caps: list[dict] = []
    for i, c in enumerate((meta.get("inventory") or {}).get("candidates") or []):
        caps.append({"capability_id": c.get("candidate_id") or candidate_id(c, i), "label": c.get("label") or c.get("id"),
                     "source": c.get("source_columns"), "indicator": c.get("indicator"), "origin": "profiling"})
    inv = {}
    for e in meta.get("coverage") or []:
        key = e.get("candidate_id") or ""
        inv.setdefault(key or _slug(e.get("candidate")), e)
        if not key:
            caps.append({"capability_id": "cap_" + _slug(e.get("candidate")), "label": e.get("candidate"),
                         "source": e.get("source"), "indicator": None, "origin": "refinement", "entry": e})
    rows: list[dict] = []
    seen: set[str] = set()
    for cap in caps:
        cid = cap["capability_id"]
        if cid in seen:
            continue
        seen.add(cid)
        e = cap.get("entry") or inv.get(cid) or inv.get(_slug(cap["label"])) or {}
        src = _source_list(cap["source"], names)
        ind = cap.get("indicator") if isinstance(cap.get("indicator"), dict) and cap["indicator"].get("column") \
            else _indicator_in_text(f"{cap['label']} {cap['source']}", meta, known_values)
        qty = [x for x in src if x in numeric]
        named = {str(e.get("measure"))} if e.get("measure") else set()
        sel = [x for x in live if x["selectable"]]
        row: dict[str, Any] = {"capability_id": cid, "label": cap["label"], "origin": cap["origin"],
                               "source_columns": src, **({"indicator": ind} if ind else {})}
        cat, by, detail = None, [], None
        by_id = [x for x in sel if cid in (x["it"].get("capability_ids") or [])
                 or (x["m"] and x["m"].get("id") in named) or x["it"]["id"] in named]
        row["_basis"] = {"id_hits": [x["it"]["id"] for x in by_id],
                         "source_from_text": cap["origin"] == "refinement" and not isinstance(cap["source"], list),
                         "indicator_from_text": bool(ind) and not (isinstance(cap.get("indicator"), dict)
                                                                   and cap["indicator"].get("column"))}
        if ind and ind.get("column") in names:
            want = {str(v) for v in ind.get("values") or []}
            got: set[str] = set()
            for x in sel:
                m = x["m"]
                if m and (m.get("item") or {}).get("column") == ind["column"]:
                    cov = want & {str(v) for v in (m.get("item") or {}).get("values") or []}
                    if cov:
                        got |= cov
                        by.append(x["it"]["id"])
                        cat = cat or "dedicated_measure"
            if not want or got < want:
                for x in sel:
                    ps = [p for p in x["it"].get("params") or [] if p.get("column") == ind["column"]]
                    if ps and (not qty or set(qty) & x["reads"]):
                        allowed = {str(v) for p in ps for v in p.get("allowed_values") or []}
                        cov = (want & allowed) if allowed else want
                        if cov:
                            got |= cov
                            by.append(x["it"]["id"])
                            cat = cat or "generic_query"
            if want and got and got < want:
                cat, detail = "partial", f"{len(got)}/{len(want)} values of {ind['column']} answerable"
            elif cat:
                detail = f"values of {ind['column']} answerable" + (" (by parameter)" if cat == "generic_query" else "")
        elif qty:
            # a measure restricted to some values / a population answers only
            # those, not the unrestricted quantity
            full = [x for x in sel if x["m"] and set(qty) <= x["reads"] and not (x["m"].get("item") or {}).get(
                "values") and not ((x["m"].get("population") or {}).get("include")
                                   or (x["m"].get("population") or {}).get("exclude"))]
            union = set().union(*(x["reads"] for x in sel)) if sel else set()
            if full:
                cat, by, detail = "dedicated_measure", [x["it"]["id"] for x in full], "a measure reads every source column"
            elif set(qty) <= union:
                cat = "generic_query"
                by = [x["it"]["id"] for x in sel if set(qty) & x["reads"]]
                detail = "the source columns are read by these items together"
            elif set(qty) & union:
                cat, by = "partial", [x["it"]["id"] for x in sel if set(qty) & x["reads"]]
                detail = f"{len(set(qty) & union)}/{len(qty)} source column(s) read; missing: " \
                         + ", ".join(sorted(set(qty) - union))[:200]
        elif src:
            counters = [x for x in sel if set(src) & x["agg"][0]]
            if counters:
                cat, by, detail = "generic_query", [x["it"]["id"] for x in counters], "a query counts these columns"
        if not cat and by_id:
            cat, by, detail = ("dedicated_measure" if any(x["m"] for x in by_id) else "generic_query"), \
                [x["it"]["id"] for x in by_id], "named by id"
        drop = next((d for d in dropped if (d.get("stage") == "verification" or str(
            d.get("reason") or d.get("error") or "").startswith("verification")) and (
            d.get("id") in named or str(d.get("id") or "").removeprefix("measure_") in named
            or (ind and any(_slug(v).replace("_", "") in _slug(d.get("id")).replace("_", "")
                            for v in ind.get("values") or [])))), None)
        if cat:
            by = list(dict.fromkeys(by))
            ready = [b for b in by if next((x["reviewed"] for x in live if x["it"]["id"] == b), False)]
            waiting = [b for b in by if b not in ready]
            if not ready:
                # every answering item passed its data test but awaits a semantic verdict
                row.update(category="pending_review", if_reviewed=cat, answered_by=by[:8], detail=detail,
                           stage="verification", reason="answered only by items awaiting semantic review")
                rows.append(row)
                continue
            row.update(category=cat, answered_by=ready[:8], detail=detail)
            if waiting:
                row["pending_review"] = waiting[:8]
            if drop and cat != "dedicated_measure":
                row["dedicated_dropped"] = {"id": drop.get("id"), "stage": "verification",
                                            "reason": str(drop.get("reason") or "")[:300]}
            rows.append(row)
            continue
        def _on_indicator(x: dict, ind: dict | None = ind) -> bool:
            if not ind:
                return False
            return any(p.get("column") == ind.get("column") for p in x["it"].get("params") or []) or bool(
                x["m"] and (x["m"].get("item") or {}).get("column") == ind.get("column")
                and set(map(str, ind.get("values") or [])) & set(map(str, (x["m"].get("item") or {}).get("values") or [])))

        unex = [x for x in live if x["status"] == "not_exercised" and (
            (qty and set(qty) & x["reads"]) or x["it"]["id"] in named or (x["m"] and x["m"].get("id") in named)
            or _on_indicator(x))]
        status = e.get("status")
        if unex:
            row.update(category="unexercised", answered_by=[x["it"]["id"] for x in unex][:4], stage="testing",
                       reason=(unex[0]["it"].get("verification") or {}).get("note") or "no witness row")
        elif drop:
            row.update(category="verifier_drop", stage="verification", dropped_id=drop.get("id"),
                       reason=str(drop.get("reason") or drop.get("error") or "")[:300])
        elif status in _UNSUPPORTED:
            row.update(category="unsupported", stage="refinement", reason=e.get("reason"))
        elif status in _UNFINISHED:
            row.update(category="unfinished", stage="refinement", reason=e.get("reason") or status)
        else:
            row.update(category="gap", stage="refinement", status=status or "unresolved",
                       reason=e.get("reason") or "no selectable stored query answers it")
        rows.append(row)
    by_item = {x["it"]["id"]: x for x in live}
    for row in rows:
        basis = row.pop("_basis", {})
        ans = [b for b in row.get("answered_by") or [] if b in by_item]
        hits = set(basis.get("id_hits") or [])
        if ans and set(ans) <= hits:
            row["match"] = "candidate_id"
        elif ans:
            why = ("source columns parsed from free text" if basis.get("source_from_text")
                   else "indicator values found in the text" if basis.get("indicator_from_text")
                   else "shared source columns")
            row["match"] = "inferred"
            row["match_basis"] = (f"{why}; several capabilities can use the same columns, so this match "
                                  "is inferred until checked")
        row["answers"] = [_answer_detail(meta, cols, by_item[b], row.get("source_columns") or [])
                          for b in (row.get("answered_by") or [])[:8] if b in by_item]
    meta["coverage_ledger"] = rows
    return rows


_COMPONENT_ROLES = {"denominator", "weight"}


def _answer_detail(meta: dict, cols: dict, x: dict, src: list[str]) -> dict:
    """What one answering item does for a capability: operation, period,
    population, required/optional bindings, and whether it returns the
    capability's answer or only its components (a quantity and its
    denominator/weight returned side by side, no ratio computed)."""
    it, m = x["it"], x["m"]
    spec = it.get("query") if isinstance(it.get("query"), dict) else {}
    aggs = [f"{a.get('agg')}({(a.get('column') or {}).get('column') if isinstance(a.get('column'), dict) else a.get('column') or '*'})"
            for a in spec.get("select") or [] if isinstance(a, dict) and a.get("agg")]
    group = [g if isinstance(g, str) else (g or {}).get("column") for g in spec.get("group_by") or []]
    if m:
        op = "one value per observation (" + ", ".join(observation_key(m)) + ")"
    elif aggs:
        op = ", ".join(aggs) + (f" grouped by {', '.join(str(g) for g in group)}" if group else "")
    else:
        op = "row listing"
    params = it.get("params") or []
    period_cols = {c["name"] for c in meta.get("columns") or [] if c.get("role") == "period"
                   or "period" in (c.get("roles") or [])}
    fixed = [w for w in spec.get("where") or [] if isinstance(w, dict) and "value" in w and not w.get("param")]
    read = sorted(set(src) & x["reads"])
    roles = {c: {*(cols.get(c, {}).get("roles") or []), cols.get(c, {}).get("role")} for c in read}
    comp = [c for c in read if roles[c] & _COMPONENT_ROLES]
    qty = [c for c in read if "quantity" in roles[c] and c not in comp]
    return {"query_id": it["id"], "section": x["sec"], "operation": op,
            "required_bindings": [p["name"] for p in params if not p.get("optional")],
            "optional_bindings": [p["name"] for p in params if p.get("optional")],
            "period": sorted({p.get("column") for p in params if p.get("column") in period_cols}
                             | ({*(m.get("period_columns") or [])} if m else set())),
            "population": (m.get("population") if m and m.get("population") else None)
            or ([{k: w.get(k) for k in ("column", "op", "value")} for w in fixed] or None),
            "returns": ("components" if comp and qty else "answer"),
            **({"components": {"quantity": qty, "denominator_or_weight": comp}} if comp and qty else {}),
            "eligible": x["reviewed"], "data_test": x["status"]}


def finish_coverage(meta: dict) -> None:
    """After testing/repair: a 'materialized' entry whose measure is not
    selectable becomes unresolved (with the test outcome as reason); the gaps
    are copied to ``coverage_gaps`` for the selector."""
    from backend.app.services.structured.metadata import selectable

    by_id = {}
    for _s, it in items(meta):
        by_id[it["id"]] = it
        if isinstance(it.get("measure"), dict):
            by_id[it["measure"]["id"]] = it
    failed = {f["id"]: f for f in meta.get("failed_measures") or []}
    for e in meta.get("coverage") or []:
        mid = e.get("measure")
        if e["status"] != "materialized" or not mid:
            continue
        it = by_id.get(mid) or by_id.get(f"measure_{mid}")
        if it is None:
            f = failed.get(mid) or failed.get(f"measure_{mid}")
            e.update(status="unresolved", reason=f"measure failed its tests: {(f or {}).get('failure') or 'not built'}")
            continue
        ok, why = selectable(meta, it)
        if not ok:
            e.update(status="unresolved" if (it.get("verification") or {}).get("status") == "failed"
                     else "supported_not_materialized", reason=f"query {it['id']}: {why}")
    meta["coverage_gaps"] = [{k: e.get(k) for k in ("candidate", "status", "reason")}
                             for e in meta.get("coverage") or [] if e["status"] != "materialized"]


async def refine_table(meta: dict, router, *, all_types: dict[str, dict[str, str]],
                       verify: bool = False, partners: dict[str, dict] | None = None,
                       log: Callable[[str], None] = print) -> dict:
    """Refine ``meta`` in place (tables must be loaded). Returns a report.

    ``partners`` (table -> metadata of the other loaded tables) lets a lookup's
    SQL include the partner's total-row markers before it is tested."""
    from backend.app.db.session import session_scope
    from backend.app.services.prompts import PROMPTS

    t0 = time.monotonic()
    table = meta["table_name"]
    async with session_scope() as s:
        analysis = await analyze_table(s, meta)
    meta["analysis"] = analysis
    log(f"[structured] {table}: grain {analysis.get('grain')} "
        f"(unique={analysis.get('grain_unique')}), repeated across "
        f"{summary_for_prompt(analysis).get('values_repeated_across') or 'nothing'}")

    ptable = _prompt_table(meta, analysis, await _sample_rows(meta))
    refine_stages: list[dict] = []
    reply, calls = await _refine_batched(router, ptable, log=log, table=table, stats=refine_stages)

    # column corrections
    by_name = {c["name"]: c for c in meta["columns"]}
    changed_cols = []
    for c in reply.get("columns") or []:
        if not isinstance(c, dict) or c.get("name") not in by_name:
            continue
        col = by_name[c["name"]]
        if c.get("role") in COLUMN_ROLES and c["role"] != col.get("role"):
            col["role"] = c["role"]
            changed_cols.append(c["name"])
        if c.get("description"):
            col["description"] = str(c["description"]).strip()
    # measures -> deterministic queries
    rejected = []
    meta["measures"] = []
    seen = set()
    raw_measures = reply.get("measures") or []
    if len(raw_measures) > MAX_MEASURES:
        log(f"[structured] {table}: {len(raw_measures)} measures proposed; only {MAX_MEASURES} built -- "
            "the rest are recorded as supported_not_materialized")
        reply.setdefault("inventory", []).extend(
            {"candidate": (r or {}).get("label") or (r or {}).get("id"), "status": "supported_not_materialized",
             "reason": f"beyond the {MAX_MEASURES}-measure safety limit"} for r in raw_measures[MAX_MEASURES:]
            if isinstance(r, dict))
    for raw in raw_measures[:MAX_MEASURES]:
        m, err = normalize_measure(raw, meta, analysis)
        if err or m["id"] in seen:
            rejected.append({"measure": raw.get("id") if isinstance(raw, dict) else raw,
                             "error": err or "duplicate id"})
            _keep_rejected_definition(meta, raw, "definition", err or "duplicate id")
            continue
        seen.add(m["id"])
        it = measure_item(m, meta)
        err = finalize_item(meta, "measures", it, all_types, partners=partners)
        if err:
            rejected.append({"measure": m["id"], "error": err})
            _keep_rejected_definition(meta, m, "compile", err)
            continue
        meta["measures"].append(it)
    meta["usage_notes"] = [str(n) for n in reply.get("usage_notes") or [] if n][:8]
    templates_added = _templates(meta, reply.get("templates") or [], all_types, partners)
    meta["coverage"] = _coverage(meta, reply.get("inventory") or [], rejected)

    # test everything; one repair round for the stored specs that fail
    from backend.app.services.structured.value_domains import attach_value_domains

    async with session_scope() as s:
        await attach_value_domains(s, meta)
    results: dict[str, dict] = {}
    failures = []
    for sec, it in list(items(meta)):
        r = await test_item(meta, sec, it, analysis, all_types, partners=partners)
        results[it["id"]] = r
        if r["status"] == "failed":
            failures.append((sec, it, r))
    repaired, dropped = [], []
    fixable = [(sec, it, r) for sec, it, r in failures if sec != "measures"]
    if fixable:
        system, user = PROMPTS["table_sql_repair"](
            table, _repair_context(meta, analysis),
            [{"id": it["id"], "error": r["error"], "params": it.get("params"), "query": it.get("query")}
             for _s, it, r in fixable])
        calls += 1
        out = _parse((await router.chat("table_refine", system=system, user=user)).text)
        fixed = {str(x.get("id")): x for x in out.get("items") or [] if isinstance(x, dict)}
        for sec, it, r in fixable:
            fx = fixed.get(it["id"])
            if fx and isinstance(fx.get("query"), dict):
                trial = {**it, "query": fx["query"], "params": fx.get("params") or it.get("params")}
                trial.pop("verification", None)
                r2 = await test_item(meta, sec, trial, analysis, all_types, partners=partners)
                if r2["status"] == "passed":
                    it.update(trial)
                    it["repaired"] = True
                    results[it["id"]] = r2
                    repaired.append(it["id"])
                    continue
            dropped.append({"id": it["id"], "section": sec, "error": r["error"]})
            from backend.app.services.structured.metadata import record_rejection

            record_rejection(meta, sec, it, stage="testing", reason=f"failed its data test: {r['error']}")
    mfail = [(it, r) for sec, it, r in failures if sec == "measures"]
    if mfail:
        n, fixed_ids = await repair_measures(meta, router, mfail, analysis=analysis, all_types=all_types,
                                             partners=partners, results=results)
        calls += n
        repaired += fixed_ids
    async with session_scope() as s:
        await attach_value_domains(s, meta)
    verified = None
    if verify:   # independent check of columns and every stored query (task table_verify)
        from backend.app.services.structured.verify import verify_table

        verified = await verify_table(meta, router, analysis=analysis, all_types=all_types,
                                      partners=partners, log=log)
        calls += verified["llm_calls"]
        async with session_scope() as s:
            await attach_value_domains(s, meta)
    # The SQL retrieval will run is the SQL tested last: re-finalize every item
    # with the final metadata (verification may have changed column roles) and
    # re-test whatever changed.
    final = await ensure_tested(meta, analysis=analysis, all_types=all_types, partners=partners)
    if verified is not None:
        final, n = await reverify_changed(meta, router, analysis=analysis, all_types=all_types,
                                          partners=partners, log=log) or (final, 0)
        calls += n
    finish_coverage(meta)
    async with session_scope() as s:
        await _measure_classes(s, meta)
    tag_items(meta)

    # typical questions: keep the ones whose query runs and returns rows
    meta["common_questions"] = await _test_questions(meta, reply.get("common_questions") or [], all_types)

    report = {"version": REFINE_VERSION, "llm_calls": calls,
              "measures": [it["id"] for it in meta["measures"]], "rejected_measures": rejected,
              "columns_changed": changed_cols, "repaired": repaired, "dropped": dropped,
              "test_status": final, "catalog_version": CATALOG_VERSION,
              "templates": templates_added,
              "coverage": {st: sum(1 for e in meta.get("coverage") or [] if e["status"] == st)
                           for st in COVERAGE_STATUSES},
              "not_exercised": [it["id"] for _s, it in items(meta)
                                if (it.get("verification") or {}).get("status") == "not_exercised"],
              "failed_measures": [f["id"] for f in meta.get("failed_measures") or []],
              "review": {it["id"]: it["review"] for it in meta["measures"] if it.get("review")},
              "tested": {k: {kk: vv for kk, vv in v.items() if kk != "params"} for k, v in results.items()},
              "questions": len(meta["common_questions"]), "seconds": round(time.monotonic() - t0, 1)}
    if verified is not None:
        report["verified"] = verified
    meta["refined"] = {"version": REFINE_VERSION, "model": _model(router), **{
        k: report[k] for k in ("measures", "rejected_measures", "columns_changed", "repaired", "dropped",
                               "questions", "not_exercised", "failed_measures", "templates",
                               "coverage")}, "stages": refine_stages}
    log(f"[structured] {table}: {len(meta['measures'])} measure(s), {len(repaired)} repaired, "
        f"{len(dropped)} dropped after testing, {len(report['failed_measures'])} measure(s) unresolved, "
        f"{len(report['not_exercised'])} not exercised, {final['passed']} selectable, "
        f"{len(meta['common_questions'])} typical question(s), {calls} LLM call(s), {report['seconds']}s")
    return report


async def reverify_changed(meta: dict, router, *, analysis: dict, all_types: dict[str, dict[str, str]],
                           partners: dict[str, dict] | None = None,
                           log: Callable[[str], None] = print) -> tuple[dict, int] | None:
    """Safety net after the final recompile: a stored query whose SQL changed
    after its semantic verdict would stay unavailable to retrieval ("semantic
    review stale"). Verify just those items again (one bounded pass, no
    column step) and re-test. Returns (test counts, LLM calls), or None when
    every verdict still matches its SQL."""
    from backend.app.services.structured.metadata import semantic_status
    from backend.app.services.structured.verify import verify_table

    stale = [it["id"] for _s, it in items(meta) if semantic_status(meta, it) == "stale"]
    if not stale:
        return None
    log(f"[structured] {meta.get('table_name')}: {len(stale)} stored quer(ies) changed after their "
        f"verdict -- verifying them again")
    rep = await verify_table(meta, router, analysis=analysis, all_types=all_types, partners=partners,
                             only_pending=True, log=log)
    final = await ensure_tested(meta, analysis=analysis, all_types=all_types, partners=partners)
    return final, int(rep.get("llm_calls") or 0)


MEASURE_REPAIR_ROUNDS = 1      # bounded: one call for all failed measures of a table


async def repair_measures(meta: dict, router, failed: list[tuple[dict, dict]], *, analysis: dict,
                          all_types: dict[str, dict[str, str]], partners: dict[str, dict] | None,
                          results: dict[str, dict]) -> tuple[int, list[str]]:
    """Ingestion-time repair of measure DEFINITIONS that failed their test
    (bounded: :data:`MEASURE_REPAIR_ROUNDS` call). Each failure keeps its
    original definition, the failure reason and the proposed correction in
    ``meta["measure_repairs"]``; a correction is rebuilt and re-tested before
    it is published. What is still unresolved moves to
    ``meta["failed_measures"]`` (inspectable, never selectable).
    Returns (LLM calls, repaired item ids)."""
    from backend.app.services.prompts import PROMPTS

    calls, fixed_ids = 0, []
    pending = list(failed)
    log_rows: list[dict] = []
    for _round in range(MEASURE_REPAIR_ROUNDS):
        if not pending:
            break
        ctx = _prompt_table(meta, analysis, await _sample_rows(meta))
        ctx.pop("catalog", None)
        system, user = PROMPTS["table_measure_repair"](ctx, [
            {"id": it["measure"]["id"], "definition": {k: v for k, v in it["measure"].items()
                                                       if k not in {"entity_columns", "param_columns",
                                                                    "latest_by", "review"}},
             "built_query": compact(it.get("query") or {}), "failure": r.get("error"),
             **({"identity_check": r["identity"]} if r.get("identity") else {})}
            for it, r in pending])
        calls += 1
        out = _parse((await router.chat("table_refine", system=system, user=user)).text)
        proposals = {str(x.get("id")): x for x in out.get("measures") or [] if isinstance(x, dict)}
        still = []
        for it, r in pending:
            orig = it["measure"]
            row: dict[str, Any] = {"id": it["id"], "original": orig, "failure": r.get("error"),
                                   "proposed": proposals.get(orig["id"])}
            prop = proposals.get(orig["id"])
            m, err = normalize_measure({**prop, "id": orig["id"]}, meta, analysis) if prop else (None, None)
            if m is None:
                row["outcome"] = "unresolved"
                row["error"] = err or "no correction proposed"
                still.append((it, r))
                log_rows.append(row)
                continue
            trial = measure_item(m, meta)
            trial["id"] = it["id"]
            for k in ("class_iri", "class_label", "aliases"):
                if it.get(k) is not None:
                    trial[k] = it[k]
            r2 = await test_item(meta, "measures", trial, analysis, all_types, partners=partners)
            if r2["status"] == "passed":
                it.clear()
                it.update(trial)
                it["repaired"] = True
                results[it["id"]] = r2
                fixed_ids.append(it["id"])
                row["outcome"] = "repaired"
            else:
                row["outcome"] = "unresolved"
                row["error"] = r2.get("error") or r2.get("status")
                still.append((it, {**r, "error": row["error"]}))
            log_rows.append(row)
        pending = still
    for it, r in pending:
        if it in meta.get("measures") or []:
            meta["measures"].remove(it)
        meta.setdefault("failed_measures", []).append(
            {"id": it["id"], "definition": it.get("measure"), "query": compact(it.get("query") or {}),
             "failure": r.get("error")})
    meta["measure_repairs"] = [*(meta.get("measure_repairs") or []), *log_rows]
    return calls, fixed_ids


async def _measure_classes(session, meta: dict) -> None:
    """A measure is tagged with the ontology class its label names (exact
    label, case- and space-insensitive), when the ontology has one."""
    for it in meta.get("measures") or []:
        if it.get("class_iri"):
            continue
        keys = {re.sub(r"[^a-z0-9]", "", str(it.get(k) or "").lower())
                for k in ("label",)} | {re.sub(r"[^a-z0-9]", "", it["measure"]["id"])}
        keys.discard("")
        if not keys:
            continue
        row = (await session.execute(sql_text(
            "SELECT iri, label FROM graphrag.ontology_classes "
            "WHERE regexp_replace(lower(label), '[^a-z0-9]', '', 'g') = ANY(CAST(:k AS text[])) LIMIT 1"),
            {"k": sorted(keys)})).first()
        if row:
            it["class_iri"], it["class_label"] = row[0], row[1]


def _model(router) -> str:
    try:
        spec = router.task_spec("table_refine")
        return f"{spec.get('provider')}:{spec.get('model')}"
    except Exception:
        return "unknown"


def _repair_context(meta: dict, analysis: dict) -> str:
    lines = [f"- {c['name']} [{c.get('type')}, {c.get('role')}]: {c.get('description') or ''}"
             for c in meta["columns"]]
    return "\n".join(lines) + "\n\nSTRUCTURE (measured on the data):\n" + json.dumps(
        summary_for_prompt(analysis), ensure_ascii=False, default=str)


async def _test_questions(meta: dict, raw: list, all_types) -> list[dict]:
    from backend.app.services.structured.metadata import find_item, selectable
    from backend.app.services.structured.value_domains import validate_call_params

    out = []
    measure_ids = {it["measure"]["id"]: it["id"] for it in meta.get("measures") or []}
    for q in raw[:12]:
        if not isinstance(q, dict) or not q.get("question"):
            continue
        qid = measure_ids.get(str(q.get("query_id")), str(q.get("query_id")))
        hit = find_item(meta, qid)
        if hit is None:
            continue
        _sec, it = hit
        if not selectable(meta, it, semantic=False)[0]:
            continue                       # examples only point at queries retrieval may run
        params = q.get("params") if isinstance(q.get("params"), dict) else {}
        params, bad, _subs = validate_call_params(params, it.get("params"), it.get("allowed_combinations"))
        if bad:
            continue
        res = await run_sql(it["sql"], params, set(it.get("tables") or [meta["table_name"]]),
                            max_rows=50, timeout_ms=20000, declared_params=it.get("params"))
        if res.error or not res.rows:
            continue
        out.append({"question": str(q["question"]), "query_id": qid, "params": params,
                    "rows": res.row_count})
    return out


async def _verify_only(meta: dict, router, *, all_types: dict[str, dict[str, str]],
                       partners: dict[str, dict] | None = None, only_pending: bool = False,
                       log: Callable[[str], None] = print) -> dict:
    """Verification of an already refined table: re-measure (SQL only), make
    every stored SQL final and tested, run :func:`verify.verify_table` (only
    the items still pending a verdict when ``only_pending``), then re-check
    what depends on the items."""
    from backend.app.db.session import session_scope
    from backend.app.services.structured.value_domains import attach_value_domains
    from backend.app.services.structured.verify import verify_table

    async with session_scope() as s:
        meta["analysis"] = await analyze_table(s, meta)
        await attach_value_domains(s, meta)
    await ensure_tested(meta, analysis=meta["analysis"], all_types=all_types, partners=partners)
    rep = await verify_table(meta, router, analysis=meta["analysis"], all_types=all_types,
                             partners=partners, only_pending=only_pending, log=log)
    await ensure_tested(meta, analysis=meta["analysis"], all_types=all_types, partners=partners)
    again = await reverify_changed(meta, router, analysis=meta["analysis"], all_types=all_types,
                                   partners=partners, log=log)
    if again:
        rep["llm_calls"] += again[1]
    async with session_scope() as s:
        await attach_value_domains(s, meta)
        await _measure_classes(s, meta)
    tag_items(meta)
    meta["common_questions"] = await _test_questions(meta, meta.get("common_questions") or [], all_types)
    return {"verified": rep, "verified_only": True, "llm_calls": rep["llm_calls"]}


def concepts_enabled(router) -> bool:
    try:
        spec = router.task_spec("table_concept_map")
    except Exception:
        return False
    return bool(spec) and spec.get("enabled", True) is not False


def _base_iri() -> str | None:
    try:
        from backend.app.core.config import get_settings

        return ((get_settings().app_config or {}).get("ontology") or {}).get("default_base_iri")
    except Exception:
        return None


async def refine_registered(router, *, tables: list[str] | None = None, refresh: bool = False,
                            relink: bool = False, verify: bool | None = None, reverify: bool = False,
                            cache=None, embed: EmbedFn | None = None, link_entities: bool = True,
                            log: Callable[[str], None] = print) -> list[dict]:
    """Refine loaded tables (all, or ``tables``) whose metadata is not yet
    refined at :data:`REFINE_VERSION`; update node, embedding, edges, cache.

    ``verify`` (default: on when the ``table_verify`` task is configured and
    not ``enabled: false``) adds the independent check of :mod:`verify`; a
    table already refined but not verified at its current version is
    verified only (no re-refine). ``reverify`` verifies again regardless.
    ``relink`` never calls an LLM."""
    from backend.app.db.graph_version import bump_version, current_version
    from backend.app.db.session import session_scope
    from backend.app.services.structured.db_load import rebuild_edges, registered_tables, upsert_node
    from backend.app.services.structured.metadata import metadata_text
    from backend.app.services.structured.verify import VERIFY_VERSION, verify_enabled

    async with session_scope() as s:
        reg = await registered_tables(s)
    names = [n for n in (tables or sorted(reg)) if n in reg]
    all_types = {n: column_types(r["meta"]) for n, r in reg.items()}
    partners = {n: r["meta"] for n, r in reg.items()}
    # A lookup's SQL carries its partner's total-row markers: measure the
    # partner (SQL only) when it was never analysed.
    wanted = {it.get("other_table") for n in names for it in reg[n]["meta"].get("lookups") or []}
    for other in sorted(w for w in wanted if w in partners and (
            (partners[w].get("analysis") or {}).get("file_sha") != data_version(partners[w]))):
        async with session_scope() as s:
            partners[other]["analysis"] = await analyze_table(s, partners[other])
    do_verify = verify_enabled(router) if verify is None else bool(verify)
    reports = []
    for name in names:
        meta = reg[name]["meta"]
        if router is not None:
            # the config decides whether retrieval waits for semantic verdicts
            # (a zero-model relink has no router and keeps the recorded value)
            meta["semantic_review"] = {"required": verify_enabled(router)}
        vinfo = meta.get("verified") or {}
        from backend.app.services.structured.verify import pending_items

        # computed now, not read from the last report: a relink can make
        # verdicts stale (their SQL changed) after that report was written
        pending = vinfo.get("version") == VERIFY_VERSION and bool(pending_items(meta))
        needs_verify = do_verify and (reverify or vinfo.get("version") != VERIFY_VERSION or pending)
        if not refresh and (meta.get("refined") or {}).get("version") == REFINE_VERSION \
                and not relink and needs_verify:
            try:
                rep = await _verify_only(meta, router, all_types=all_types, partners=partners,
                                         only_pending=pending and not reverify, log=log)
            except Exception as exc:  # one table must not stop the rest
                log(f"[structured] FAILED verifying {name}: {type(exc).__name__}: {exc}")
                reports.append({"table": name, "error": str(exc)})
                continue
        elif not refresh and (meta.get("refined") or {}).get("version") == REFINE_VERSION:
            migrate = meta.get("catalog_version") != CATALOG_VERSION
            if not relink and not migrate:
                log(f"[structured] {name}: already refined ({REFINE_VERSION}), no LLM call")
                continue
            # no LLM: re-derive measures (older definitions re-split against the
            # measured dependencies), re-compile and re-test every stored query
            # on its exact SQL, re-derive tags and graph links
            from backend.app.services.structured.metadata import pin_reviewed_sql

            pin_reviewed_sql(meta)          # a verdict stays valid only for the SQL it reviewed
            async with session_scope() as s:
                meta["analysis"] = await analyze_table(s, meta)       # SQL only
            meta["measures"] = rebuild_measures(meta, all_types)
            revived = await revive_dropped(meta, all_types=all_types, partners=partners)
            from backend.app.services.structured.value_domains import attach_value_domains

            async with session_scope() as s:
                await attach_value_domains(s, meta)              # SQL only (domains + stats)
            tested = await ensure_tested(meta, analysis=meta["analysis"], all_types=all_types,
                                         partners=partners, force=True)
            async with session_scope() as s:
                await _measure_classes(s, meta)
            tag_items(meta)
            meta["common_questions"] = await _test_questions(meta, meta.get("common_questions") or [],
                                                             all_types)
            rep = {"relinked": True, "migrated": migrate, "test_status": tested, "llm_calls": 0,
                   "revived": revived}
            log(f"[structured] {name}: {'migrated to' if migrate else 'relinked at'} catalog "
                f"{CATALOG_VERSION} (no LLM): {tested['passed']} passed, {tested['failed']} failed, "
                f"{tested['not_exercised']} not exercised")
        else:
            rep = None
        try:
            rep = rep or await refine_table(meta, router, all_types=all_types, verify=do_verify,
                                            partners=partners, log=log)
        except Exception as exc:  # one table must not stop the rest
            log(f"[structured] FAILED refining {name}: {type(exc).__name__}: {exc}")
            reports.append({"table": name, "error": str(exc)})
            continue
        # the FINAL catalog's concepts -> ontology classes (after repair and
        # verification); no LLM on a relink, and none when the task is absent
        from backend.app.services.structured.concepts import reconcile_concepts

        try:
            concept_router = router if (not rep.get("relinked") and concepts_enabled(router)) else None
            rep["concepts"] = await reconcile_concepts(meta, concept_router, base_iri=_base_iri(), log=log)
        except Exception as exc:  # reconciliation must not lose the refined catalog
            log(f"[structured] {name}: concept reconciliation failed ({type(exc).__name__}: {exc})")
        from backend.app.services.structured.db_load import ensure_value_indexes

        async with session_scope() as s:                          # SQL only
            rep["value_indexes"] = len(await ensure_value_indexes(s, name, meta))
        # coverage ledger from the FINAL catalog (after repair, verification,
        # duplicate review and concept reconciliation)
        rep["coverage_ledger"] = dict(Counter(r["category"] for r in coverage_ledger(
            meta, known_values=await indicator_values(meta))))
        vec = (await embed([metadata_text(meta)]))[0] if embed is not None else None
        async with session_scope() as s:
            gv = await current_version(s)
            node_id = await upsert_node(s, meta, vec, gv, reg[name]["id"])
            await rebuild_edges(s, node_id, meta, gv, link_entities=link_entities, embed=embed, log=log)
        if cache is not None and meta.get("cache_key"):
            base = cache.get(meta["cache_key"], name) or {}
            entry = {**base, **meta}
            entry.setdefault("status", "complete")
            if entry.get("format"):
                cache.put(entry)
        reports.append({"table": name, **rep})
    if reports:
        async with session_scope() as s:
            await bump_version(s)
    return reports


__all__ = [
    "REFINE_VERSION",
    "measure_item",
    "normalize_measure",
    "rebuild_measures",
    "refine_registered",
    "refine_table",
    "test_item",
]
_ = SECTIONS  # measures is one of the item sections
