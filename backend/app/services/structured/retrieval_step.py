"""``query --use-structured-data``: answer from the loaded tables via SQL only.

Runs after the graph walk, before answer synthesis:

1. **Find tables through the graph.** TableMetadata nodes linked to the
   entities the walk reached (``tableEntity`` edges) or to the ontology classes
   the question matched (``tableColumn``/``tableMetric``/``tableFilter``);
   a vector search over the metadata nodes fills the remaining slots.
2. **Decide and pick** (task ``table_query_select``, one call): is table data
   needed, and which STORED metric/filter/query/lookup answers it, with which
   parameter values. The model never writes SQL: when no stored query fits it
   may return a new JSON query spec.
3. **Build and run** deterministically: :mod:`query_builder` compiles the
   stored (or new) spec to SQL, which then runs through the guard
   (``sql_exec``): read-only, timed out, rows capped. A query that errors or
   returns nothing gets one retry call, which is told the error.
4. Each result becomes an evidence item ``kind="sql_result"`` with its own
   citation id. Table data reaches the prompt ONLY through these results.

Two checks keep a call honest:

* **Identifiers are never guessed.** A value compared with an identifier
  column that names individual items (ids, codes, accession numbers) must
  come from the question, from an earlier call's result, or name a row whose
  descriptive values match the question; otherwise the call is refused before
  it runs and the retry is told what that id actually is.
* **Chained calls.** A param may be ``{"from_call": k, "column": c}``: its
  value(s) come from call ``k``'s result (one run per value, up to
  ``MAX_FANOUT``; ``"as_list": true`` passes them all to one list param), so
  "look the item up by name, then use its id in another table" needs no guess.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
from collections.abc import Awaitable, Callable
from datetime import date
from typing import Any

from sqlalchemy import text as sql_text

from backend.app.services.predicates import VIAO_NS
from backend.app.services.structured.analysis import total_markers
from backend.app.services.structured.metadata import (
    catalog_for_prompt,
    column_types,
    compile_item,
    find_item,
    item_spec,
    time_columns,
)
from backend.app.services.structured.query_builder import QuerySpecError, build_sql
from backend.app.services.structured.sql_exec import SqlResult, run_sql
from backend.app.services.structured.sql_guard import SqlGuardError
from backend.app.services.structured.value_domains import check_params, spec_declared

MAX_FANOUT = 5                  # runs of one chained call (one per referenced value)
ENTITY_ID_MIN_DISTINCT = 10     # an identifier column this varied names individual items
DISTINCTIVE_SHARE = 0.2         # a word in at most this share of a column's values identifies a row
MISSING_VALUE_ERROR = ("value(s) not found in their column (see missing_values, which lists the closest "
                       "values that DO exist): checked before the query ran; do not use them again")
EMPTY_EXCLUDED_NOTE = ("Rows with an empty {columns} are not listed or counted as a value "
                       "here (an empty value is not a category).")
UNRANKED_NOTE = ("This query kept only the latest row(s) by {column}, but {rows} other matching "
                 "row(s) have an empty {column} and were never shown; they may hold values the "
                 "returned row(s) lack. Do not take a missing value as absent from the table.")
UNSUPPORTED_ID_ERROR = (
    "identifier value(s) not taken from the question or from an earlier call's result (see "
    "unsupported_identifiers; row_values shows what each id actually is in the table). Never "
    "fill an identifier from memory: filter by the item's name/description column instead, or "
    "find the id with one call and pass it to the next with {\"from_call\": k, \"column\": ...}.")
# Legal-form suffixes plus ordinary English function words: a question's
# "did", "have" or "which" must never vouch for an identifier's row.
_ID_STOPWORDS = {"the", "and", "for", "with", "inc", "ltd", "llc", "plc", "corp", "corporation",
                 "company", "group", "holdings", "limited", "co", "of", "in", "on", "a", "an",
                 "about", "after", "all", "also", "any", "are", "around", "been", "before", "being",
                 "between", "both", "but", "by", "can", "could", "did", "does", "doing", "done",
                 "during", "each", "from", "had", "has", "have", "having", "her", "his", "how",
                 "into", "its", "many", "more", "most", "much", "not", "only", "other", "over",
                 "per", "than", "that", "their", "them", "then", "there", "these", "they", "this",
                 "those", "through", "under", "was", "were", "what", "when", "where", "which",
                 "while", "who", "whom", "whose", "why", "will", "within", "without", "would",
                 "you", "your", "our", "out", "same", "such", "very", "list", "show", "tell",
                 "give", "find", "total", "number", "whether"}

_TABLE_EDGE_KINDS = ("tableEntity", "tablePlace", "tableColumn", "tableMetric", "tableFilter", "tableTime")


def _vec(v: list[float]) -> str:
    return "[" + ",".join(f"{x:.7g}" for x in v) + "]"


async def candidate_tables(
    entity_ids: list[uuid.UUID],
    class_ids: list[uuid.UUID],
    qvec: list[float] | None,
    *,
    max_tables: int = 3,
    vector_max_distance: float | None = None,
    time_ids: list[uuid.UUID] | None = None,
) -> list[dict[str, Any]]:
    """[{id, table, meta, score, via, hits}] best first. ``hits`` records what
    the question reached the table through (entity/place columns, class
    labels, periods) so the table's tagged queries can be ranked."""
    from backend.app.db.session import session_scope

    scored: dict[uuid.UUID, dict[str, Any]] = {}
    async with session_scope() as s:
        if entity_ids or class_ids or time_ids:
            rows = (await s.execute(sql_text("""
                SELECT a.id, gr.target_node_type, gr.relationship_type, gr.extra_metadata, oc.iri, oc.label
                  FROM graphrag.intelligence_artifacts a
                  JOIN graphrag.graph_relationships gr
                    ON gr.source_node_type = 'intelligence_artifact' AND gr.source_node_id = a.id
                   AND gr.relationship_type = ANY(CAST(:kinds AS text[]))
                  LEFT JOIN graphrag.ontology_classes oc
                    ON gr.target_node_type = 'ontology_class' AND oc.id = gr.target_node_id
                 WHERE a.artifact_type = 'TableMetadata' AND a.status = 'ACTIVE'
                   AND ((gr.target_node_type = 'entity' AND gr.target_node_id = ANY(CAST(:ents AS uuid[])))
                     OR (gr.target_node_type = 'ontology_class' AND gr.target_node_id = ANY(CAST(:cls AS uuid[])))
                     OR (gr.target_node_type = 'time_instance' AND gr.target_node_id = ANY(CAST(:tms AS uuid[]))))
            """), {"kinds": list(_TABLE_EDGE_KINDS), "ents": [str(e) for e in entity_ids],
                   "cls": [str(c) for c in class_ids],
                   "tms": [str(t) for t in time_ids or []]})).all()
            counts: dict[tuple[uuid.UUID, str], int] = {}
            for aid, ttype, kind, extra, ciri, clabel in rows:
                d = scored.setdefault(aid, {"score": 0.0, "via": [], "hits": {
                    "columns": set(), "classes": set(), "time": False}})
                extra = extra or {}
                if ttype == "time_instance":
                    if not d["hits"]["time"]:
                        d["score"] += 1.0          # once: a long series covers many periods
                    d["hits"]["time"] = True
                else:
                    d["score"] += 2.0 if ttype == "entity" else 1.0
                if ttype == "entity":
                    d["hits"]["columns"].update(extra.get("columns") or [])
                if ttype == "ontology_class":
                    d["hits"]["classes"].update(x for x in (ciri, clabel) if x)
                key = "place" if kind == "tablePlace" else ttype
                counts[(aid, key)] = counts.get((aid, key), 0) + 1
            for (aid, key), n in counts.items():
                scored[aid]["via"].append(f"{key}:{n}")
        if qvec is not None and len(scored) < max_tables:
            rows = (await s.execute(sql_text("""
                SELECT id, (embedding <-> CAST(:q AS vector)) AS dist
                  FROM graphrag.intelligence_artifacts
                 WHERE artifact_type = 'TableMetadata' AND status = 'ACTIVE' AND embedding IS NOT NULL
                 ORDER BY dist LIMIT :n
            """), {"q": _vec(qvec), "n": max_tables})).all()
            for aid, dist in rows:
                if vector_max_distance is not None and float(dist) > vector_max_distance:
                    continue
                if aid not in scored:
                    scored[aid] = {"score": 0.5 / (1.0 + float(dist)), "via": [f"vector:{float(dist):.3f}"],
                                   "hits": {"columns": set(), "classes": set(), "time": False}}
        if not scored:
            return []
        best = sorted(scored.items(), key=lambda kv: -kv[1]["score"])[:max_tables]
        metas = dict((await s.execute(sql_text("""
            SELECT id, extra_metadata FROM graphrag.intelligence_artifacts
             WHERE id = ANY(CAST(:ids AS uuid[]))
        """), {"ids": [str(a) for a, _ in best]})).all())
    out = []
    for aid, d in best:
        meta = metas.get(aid) or {}
        if meta.get("table_name"):
            out.append({"id": aid, "table": meta["table_name"], "meta": meta, **d})
    return out


def rank_items(meta: dict, hits: dict | None) -> dict[str, tuple[float, list[str]]]:
    """Score each stored query by the graph tags the question matched: its
    classes among the classes the question reached, its entity/place columns
    among the columns whose values the question named, its period columns
    when the question names a period. Measures get a small edge (tested,
    one value per period)."""
    from backend.app.services.structured.metadata import item_tags, items

    hits = hits or {}
    out: dict[str, tuple[float, list[str]]] = {}
    for sec, it in items(meta):
        tags = it.get("tags") or item_tags(meta, sec, it)
        score, why = 0.0, []
        cls = [c.get("label") or c.get("iri") for c in tags.get("classes") or []
               if (c.get("iri") in hits.get("classes", set()) or c.get("label") in hits.get("classes", set()))]
        if cls:
            score += 2.0 * len(set(cls))
            why.append("class " + ", ".join(sorted(set(cls))))
        cols = sorted(set(tags.get("entity_columns") or []) | set(tags.get("place_columns") or []))
        cols = [c for c in cols if c in hits.get("columns", set())]
        if cols:
            score += 1.5 * len(cols)
            why.append("entity in " + ", ".join(cols))
        if hits.get("time") and tags.get("time_columns"):
            score += 1.0
            why.append("period in " + ", ".join(tags["time_columns"]))
        if score and sec == "measures":
            score += 0.5
        out[it["id"]] = (score, why)
    return out


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


async def table_inventory() -> dict[str, dict]:
    """Every ACTIVE table: {table: {id, iri, meta}} (for the ALL TABLES list
    and for describing a table that is not a candidate)."""
    from backend.app.db.session import session_scope

    async with session_scope() as s:
        rows = (await s.execute(sql_text("""
            SELECT id, artifact_identifier, extra_metadata FROM graphrag.intelligence_artifacts
             WHERE artifact_type = 'TableMetadata' AND status = 'ACTIVE'
             ORDER BY extra_metadata->>'table_name'
        """))).all()
    return {m["table_name"]: {"id": aid, "iri": iri, "meta": m}
            for aid, iri, m in rows if (m or {}).get("table_name")}


def inventory_for_prompt(inv: dict[str, dict], limit: int = 60) -> list[dict]:
    out = []
    for name, rec in list(inv.items())[:limit]:
        desc = str(rec["meta"].get("description") or "").strip()
        first = re.split(r"(?<=[.!?])\s", desc, maxsplit=1)[0][:220]
        out.append({"table": name, "rows": rec["meta"].get("row_count"), "about": first})
    return out


# ------------------------------------------------------------ relative periods

_RELATIVE = re.compile(
    r"\b(past|last|previous|recent|latest|trailing|preceding)\b|\b(this|current)\s+(year|quarter|month)\b"
    r"|\byear[- ]to[- ]date\b|\bytd\b|\bso far\b", re.I)
_CURRENT_YEAR = re.compile(r"\b(?:current|present)\s+year\s+is\s+(\d{4})|\b(?:it|today)\s+is\s+(?:\w+\s+)*?(\d{4})", re.I)


def relative_period_question(question: str, today_year: int) -> bool:
    """The question counts periods back from 'now' and names no period of its
    own (a year other than the stated current one fixes the window itself)."""
    if not _RELATIVE.search(question or ""):
        return False
    current = {today_year} | {int(y) for m in _CURRENT_YEAR.finditer(question) for y in m.groups() if y}
    named = {int(y) for y in re.findall(r"\b(1[6-9]\d\d|2[01]\d\d)\b", question)}
    return not (named - current)


def _period_index(value: Any) -> tuple[str, int] | None:
    from backend.app.services.structured.column_stats import granularity
    from backend.app.services.structured.graph_links import period_of

    g, p = granularity(value), period_of(value)
    if p is None or g not in {"year", "quarter", "month"}:
        return None
    y, q, m = p
    return g, (y if g == "year" else y * 4 + (q - 1) if g == "quarter" else y * 12 + (m - 1))


def _period_value(like: Any, gran: str, idx: int) -> Any:
    """Render period index ``idx`` in the spelling of ``like``."""
    s = str(like).strip()
    if gran == "year":
        return idx if isinstance(like, int) else str(idx)
    if gran == "quarter":
        y, q = divmod(idx, 4)
        m = re.match(r"^(\d{4})(\D*)([1-4])$", s)
        return f"{y}{m.group(2)}{q + 1}" if m else f"{y}Q{q + 1}"
    y, mo = divmod(idx, 12)
    m = re.match(r"^\d{4}(\D*)\d{1,2}$", s)
    sep = m.group(1) if m else "-"
    return f"{y}{sep}{mo + 1:02d}"


def between_params(spec: dict | None) -> list[tuple[str, str, str]]:
    """(column, from-param, to-param) of the spec's parameterised ranges."""
    out = []
    for w in (spec or {}).get("where") or []:
        if (isinstance(w, dict) and str(w.get("op") or "").lower() == "between"
                and isinstance(w.get("param"), list) and len(w["param"]) == 2
                and isinstance(w.get("column"), str)):
            out.append((w["column"].split(".")[-1], str(w["param"][0]), str(w["param"][1])))
    return out


def anchor_relative_periods(question: str, call: dict, meta: dict, *, today_year: int) -> dict | None:
    """When the question counts back from 'now' ('past 3 years', 'recent'),
    move a call's period window so it ends at the LATEST period the table
    holds at that granularity, keeping its length. Returns the adjustment
    (and changes ``call['params']``) or None. Deterministic; any table."""
    from backend.app.services.structured.column_stats import latest_period

    if not relative_period_question(question, today_year):
        return None
    if call.get("query_id"):
        found = find_item(meta, call["query_id"])
        spec = item_spec(found[0], found[1]) if found else None
    else:
        spec = call.get("query")
    params = call.get("params") or {}
    for col, p_from, p_to in between_params(spec):
        lo, hi = params.get(p_from), params.get(p_to)
        if lo is None or hi is None:
            continue
        a, b = _period_index(lo), _period_index(hi)
        latest = latest_period(meta, col, hi)
        c = _period_index(latest) if latest is not None else None
        if not a or not b or not c or a[0] != b[0] or b[0] != c[0] or b[1] == c[1] or b[1] < a[1]:
            continue
        span = b[1] - a[1]
        new_lo, new_hi = _period_value(lo, a[0], c[1] - span), _period_value(hi, a[0], c[1])
        call["params"] = {**params, p_from: new_lo, p_to: new_hi}
        return {"column": col, "was": [lo, hi], "now": [new_lo, new_hi], "latest": latest}
    return None


def describe_evidence(names: list[str], inv: dict[str, dict], start_rank: int) -> list[dict]:
    from backend.app.services.structured.column_stats import metadata_card

    out = []
    for i, name in enumerate(n for n in dict.fromkeys(names) if n in inv):
        rec = inv[name]
        iri = rec.get("iri") or f"{VIAO_NS}#TableMetadata_{re.sub(r'[^A-Za-z0-9_]', '_', name)}"
        out.append({"kind": "table_metadata", "node_id": rec["id"], "iri": iri,
                    "rank": start_rank + i, "score": 1.0, "text": metadata_card(rec["meta"]),
                    "char_cap": 12000, "document_title": f"table {name}", "table": name,
                    "source_file": (rec["meta"].get("source") or {}).get("rel_path")})
    return out


def _norm_value(v: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(v).lower())


def _sig_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9][a-z0-9\-]{2,}", str(text).lower())
            if w not in _ID_STOPWORDS}


def _word_regex(word: str) -> str:
    """Postgres ARE matching ``word`` as a whole word (not inside another word)."""
    return r"\m" + re.sub(r"([^a-z0-9])", r"\\\1", word) + r"\M"


def entity_id_columns(meta: dict) -> set[str]:
    """Identifier columns that name individual items (not a handful of
    categories such as a source file): role identifier and >= 10 distinct values."""
    stats = meta.get("column_stats") or {}
    out = set()
    for c in meta.get("columns") or []:
        if c.get("role") != "identifier" or (c.get("type") or "text") != "text":
            continue
        d = (stats.get(c["name"]) or {}).get("distinct")
        if d is None or d >= ENTITY_ID_MIN_DISTINCT:
            out.add(c["name"].lower())
    return out


def name_columns(meta: dict, id_column: str) -> list[str]:
    """Descriptive text columns of a row (what an id stands for): the columns
    the id determines, else every varied text column but the id itself."""
    stats = meta.get("column_stats") or {}
    text_cols = {c["name"] for c in meta.get("columns") or []
                 if (c.get("type") or "text") == "text" and c.get("role") not in {"measure", "time"}}
    deps = ((meta.get("analysis") or {}).get("dependencies") or {}).get(id_column) or []
    cands = [c for c in deps if c in text_cols] or sorted(text_cols - {id_column})
    return [c for c in cands
            if ((stats.get(c) or {}).get("distinct") or ENTITY_ID_MIN_DISTINCT) >= ENTITY_ID_MIN_DISTINCT][:8]


def expand_refs(call: dict, done: dict[int, list[SqlResult]]) -> tuple[list[dict], str | None]:
    """Fill ``{"from_call": k, "column": c}`` params from call k's result rows.
    Returns ([concrete calls], error). Scalar params fan out (one call per
    value, at most MAX_FANOUT runs); ``"as_list": true`` passes all values."""
    params = call.get("params") if isinstance(call.get("params"), dict) else {}
    refs = {n: v for n, v in params.items() if isinstance(v, dict) and "from_call" in v}
    if not refs:
        return [call], None
    lists: dict[str, list] = {}
    scalars: list[tuple[str, list]] = []
    for name, ref in refs.items():
        try:
            k = int(ref.get("from_call"))
        except (TypeError, ValueError):
            return [], f"param {name}: from_call must be the number of an earlier call"
        results = done.get(k)
        if not results:
            return [], (f"param {name}: call {k} has no result to take values from (it failed, "
                        "returned no rows, or is not an earlier call)")
        col = str(ref.get("column") or name)
        vals: list = []
        for res in results:
            idx = next((i for i, c in enumerate(res.columns) if c.lower() == col.lower()), None)
            if idx is None:
                return [], f"param {name}: call {k}'s result has no column {col!r} (columns: {', '.join(res.columns)})"
            for row in res.rows:
                v = row[idx]
                if v is not None and v not in vals:
                    vals.append(v)
        if not vals:
            return [], f"param {name}: column {col!r} of call {k}'s result holds no values"
        if ref.get("as_list"):
            lists[name] = vals
        else:
            scalars.append((name, vals))
    combos: list[dict] = [{}]
    total = 1
    for name, vals in scalars:
        total *= len(vals)
        combos = [{**c, name: v} for c in combos for v in vals[:MAX_FANOUT]]
    combos = combos[:MAX_FANOUT]
    note = {n: refs[n] for n in refs}
    out = []
    for c in combos:
        out.append({**call, "params": {**params, **lists, **c}, "_from_calls": note,
                    **({"_fanout_cut": f"first {MAX_FANOUT} of {total} value combinations"}
                       if total > MAX_FANOUT else {})})
    return out, None


async def identifier_guard(sql: str, params: dict, *, question: str, seen_values: set[str],
                           types: dict[str, dict[str, str]], metas: dict[str, dict],
                           context_norm: str = "") -> list[dict]:
    """Values compared with an item-identifier column that the question does
    not state, no earlier result returned, and whose row's descriptive values
    share no word with the question. Read-only."""
    id_cols = {t: entity_id_columns(m) for t, m in metas.items()}
    if not any(id_cols.values()):
        return []
    lowered = {t: {c.lower(): ty for c, ty in cols.items()} for t, cols in types.items()}
    checks = [c for c in compared_values(sql, params, lowered, id_cols)
              if c["column"] in id_cols.get(c["table"], set())]
    if not checks:
        return []
    q_norm, q_words = _norm_value(question), _sig_words(question)
    out = []
    from backend.app.db.engine import get_sessionmaker
    from backend.app.services.structured.db_load import q

    pending = []
    for c in checks:
        v = _norm_value(c["value"])
        if v and (v in q_norm or v in seen_values or (len(v) >= 4 and v in context_norm)):
            continue
        pending.append(c)
    if not pending:
        return []
    async with get_sessionmaker()() as s:
        try:
            await s.execute(sql_text("SET TRANSACTION READ ONLY"))
            await s.execute(sql_text("SET LOCAL statement_timeout = 3000"))
            for c in pending:
                meta = metas.get(c["table"]) or {}
                names = name_columns(meta, c["column"])
                vals: list[str] = []
                matched = False
                if names:
                    rows = (await s.execute(sql_text(
                        f"SELECT {', '.join(q(n) for n in names)} FROM graphrag.{q(c['table'])} "
                        f"WHERE {q(c['column'])} = :v LIMIT 3"), {"v": c["value"]})).all()
                    vals = list(dict.fromkeys(str(x) for r in rows for x in r if x not in (None, "")))
                    # the id's own row names what the question asks about -- by a
                    # word that is distinctive in its column (not one every row shares)
                    for n in names:
                        own = {w for r in rows for w in _sig_words(r[names.index(n)] or "")}
                        for w in sorted(own & q_words)[:6]:
                            share = (await s.execute(sql_text(
                                f"SELECT COUNT(DISTINCT {q(n)}) FILTER (WHERE {q(n)} ~* :w)::float "
                                f"/ GREATEST(COUNT(DISTINCT {q(n)}), 1) FROM graphrag.{q(c['table'])}"),
                                {"w": _word_regex(w)})).scalar() or 0.0
                            if share <= DISTINCTIVE_SHARE:
                                matched = True
                                break
                        if matched:
                            break
                if matched:
                    continue
                out.append({"table": c["table"], "column": c["column"], "value": c["value"],
                            "row_values": vals[:6] or "no row has this value"})
        except Exception:  # best effort: a failed check never blocks the call
            return []
        finally:
            await s.rollback()
    return out


async def table_metas(names: set[str]) -> dict[str, dict]:
    """ACTIVE metadata of the named tables (lookup partners outside the
    candidate set need their column types to compile)."""
    if not names:
        return {}
    from backend.app.db.session import session_scope

    async with session_scope() as s:
        rows = (await s.execute(sql_text("""
            SELECT extra_metadata FROM graphrag.intelligence_artifacts
             WHERE artifact_type = 'TableMetadata' AND status = 'ACTIVE'
               AND extra_metadata->>'table_name' = ANY(CAST(:t AS text[]))
        """), {"t": sorted(names)})).scalars().all()
    return {m["table_name"]: m for m in rows if isinstance(m, dict) and m.get("table_name")}


async def _execute(call: dict, by_table: dict[str, dict], *, max_rows: int, timeout_ms: int,
                   all_tables: set[str], types: dict[str, dict[str, str]],
                   totals: dict[str, dict[str, str]] | None = None,
                   guard: Callable[[str, dict], Awaitable[list[dict]]] | None = None,
                   prevalidate: Callable[[str, dict], Awaitable[tuple[dict, list[dict], list[dict]]]]
                   | None = None,
                   ) -> tuple[dict, SqlResult | None, str | None]:
    """Build (deterministically) and run one selected call. Returns (record, result, error).
    ``guard(sql, params)`` returns identifier values the call may not use
    (checked before the SQL runs)."""
    table = call.get("table")
    params = call.get("params") if isinstance(call.get("params"), dict) else {}
    if call.get("query_id"):
        meta = (by_table.get(table) or {}).get("meta")
        hit = find_item(meta, call["query_id"]) if meta else None
        if hit is None:
            # The model may name the table wrong; search every candidate.
            for cand in by_table.values():
                hit = find_item(cand["meta"], call["query_id"])
                if hit:
                    meta, table = cand["meta"], cand["table"]
                    break
        if hit is None:
            err = f"unknown query_id {call['query_id']!r}"
            return {"table": table, "query_id": call["query_id"], "error": err}, None, err
        section, item = hit
        rec = {"table": meta["table_name"], "query_id": item["id"], "label": item.get("label"),
               "source": "catalog", "section": section}
        try:
            built = compile_item(meta, section, item, types, totals=totals)
        except (QuerySpecError, SqlGuardError) as exc:
            err = f"stored query does not compile: {exc}"
            return {**rec, "error": err}, None, err
        declared = (item.get("params") if isinstance(item.get("query"), dict) else None) or built.params
        params, bad, subs = check_params(params, declared, item.get("allowed_combinations"))
        if subs:
            rec["substituted"] = subs
        if bad:
            err = "parameter value(s) not in the column's valid values (see invalid_values)"
            return {**rec, "params": params, "invalid_values": bad, "error": err}, None, err
        unsupported = await guard(built.sql, params) if guard else []
        if unsupported:
            return {**rec, "params": params, "unsupported_identifiers": unsupported,
                    "error": UNSUPPORTED_ID_ERROR}, None, UNSUPPORTED_ID_ERROR
        if prevalidate:
            params, missing, vsubs = await prevalidate(built.sql, params)
            if vsubs:
                rec["substituted"] = [*(rec.get("substituted") or []), *vsubs]
            if missing:
                rec["missing_values"] = missing
                if not all(m.get("dropped_from") for m in missing):
                    return {**rec, "params": params, "error": MISSING_VALUE_ERROR}, None, MISSING_VALUE_ERROR
        res = await run_sql(built.sql, params, built.tables & (all_tables | {meta["table_name"]}),
                            max_rows=max_rows, timeout_ms=timeout_ms, declared_params=declared)
    elif isinstance(call.get("query"), dict):
        rec = {"table": table, "query_id": None, "label": call.get("purpose") or "new query spec",
               "source": "spec", "query": call["query"], "why_not_catalog": call.get("why_not_catalog")}
        if table not in by_table:
            err = f"table {table!r} is not a candidate table"
            return {**rec, "error": err}, None, err
        try:
            from backend.app.services.structured.metadata import time_columns

            tmeta = (by_table.get(table) or {}).get("meta") or {}
            built = build_sql(call["query"], table, {t: c for t, c in types.items() if t in all_tables},
                              time_columns={table: time_columns(tmeta)}, totals=totals)
        except QuerySpecError as exc:
            err = f"query spec does not compile: {exc}"
            return {**rec, "error": err}, None, err
        domains = ((by_table.get(table) or {}).get("meta") or {}).get("value_domains") or {}
        declared = spec_declared(built.params, domains)
        params, bad, subs = check_params(params, declared)
        if subs:
            rec["substituted"] = subs
        if bad:
            err = "parameter value(s) not in the column's valid values (see invalid_values)"
            return {**rec, "params": params, "invalid_values": bad, "error": err}, None, err
        unsupported = await guard(built.sql, params) if guard else []
        if unsupported:
            return {**rec, "params": params, "unsupported_identifiers": unsupported,
                    "error": UNSUPPORTED_ID_ERROR}, None, UNSUPPORTED_ID_ERROR
        if prevalidate:
            params, missing, vsubs = await prevalidate(built.sql, params)
            if vsubs:
                rec["substituted"] = [*(rec.get("substituted") or []), *vsubs]
            if missing:
                rec["missing_values"] = missing
                if not all(m.get("dropped_from") for m in missing):
                    return {**rec, "params": params, "error": MISSING_VALUE_ERROR}, None, MISSING_VALUE_ERROR
        res = await run_sql(built.sql, params, built.tables, max_rows=max_rows, timeout_ms=timeout_ms,
                            declared_params=declared)
    else:
        err = "call has neither query_id nor query"
        return {"table": table, "error": err}, None, err
    if built.empty_excluded:
        rec["empty_excluded"] = list(built.empty_excluded)
    rec.update({"sql": res.sql, "params": res.params, "row_count": res.row_count,
                "truncated": res.truncated, "total_rows": res.total_rows, "seconds": res.seconds,
                "error": res.error})
    if res.error:
        return rec, res, res.error
    if res.row_count == 0:
        return rec, res, "returned no rows"
    return rec, res, None


MAX_VALUE_CHECKS = 8
MAX_SUGGESTIONS = 8


def unranked_count_sql(sql: str, period_columns: set[str]) -> tuple[str, str] | None:
    """For a row query that keeps only the latest row(s) -- no aggregate,
    ORDER BY a period column DESC, cut by LIMIT or DISTINCT ON -- return
    (count SQL, column): the same query counting the matching rows whose
    period is EMPTY. Those rows sort last, so the cut never shows them, yet
    they can hold values the returned row lacks. None for any other query."""
    import sqlglot
    from sqlglot import exp

    # sqlglot rewrites ``:name`` placeholders: park them as identifiers
    marked = re.sub(r"(?<![:\w]):([A-Za-z_]\w*)", r"__yegi_p_\1__", sql)
    try:
        tree = sqlglot.parse_one(marked, read="postgres")
    except Exception:
        return None
    if not isinstance(tree, exp.Select) or tree.args.get("group") or tree.find(exp.AggFunc):
        return None
    if not (tree.args.get("limit") or isinstance(tree.args.get("distinct"), exp.Distinct)):
        return None
    order = tree.args.get("order")
    col = None
    for o in (order.expressions if order else []):
        c = o.this
        if o.args.get("desc") and isinstance(c, exp.Column) and c.name.lower() in period_columns:
            col = c
            break
    if col is None:
        return None
    count = tree.copy()
    for key in ("order", "limit", "distinct", "offset"):
        count.set(key, None)
    count = count.select(exp.Count(this=exp.Star()), append=False).where(
        exp.Is(this=col.copy(), expression=exp.Null()), append=True)
    out = re.sub(r'"?__yegi_p_([A-Za-z_]\w*?)__"?', r":\1", count.sql(dialect="postgres"))
    return out, col.name


async def _unranked_rows(rec: dict, res: SqlResult | None, meta: dict | None) -> dict | None:
    """``{"column", "rows"}`` when a latest-row query passed over matching rows
    whose period column is empty (see ``unranked_count_sql``)."""
    if res is None or res.error or not res.rows or not rec.get("sql") or not meta:
        return None
    periods = {c.lower() for c in time_columns(meta)} | {
        str(c.get("name")).lower() for c in meta.get("columns") or [] if c.get("role") == "time"}
    built = unranked_count_sql(rec["sql"], periods)
    if not built:
        return None
    count_sql, col = built
    try:
        out = await run_sql(count_sql, rec.get("params") or {}, {rec["table"]}, max_rows=1, timeout_ms=3000)
        n = int(out.rows[0][0]) if not out.error and out.rows else 0
    except Exception:  # best effort: a failed check never hides the result
        return None
    return {"column": col, "rows": n} if n > 0 else None


def compared_values(sql: str, params: dict, columns: dict[str, dict[str, str]],
                    id_columns: dict[str, set[str]] | None = None) -> list[dict]:
    """``text_column = 'value'`` / ``IN (...)`` comparisons, from literals and
    bound :params: [{table, column, value, context}], where ``context`` holds
    the statement's OTHER single-value equality filters on the same table
    (``cik = '0000811809'``) -- a tag can exist in the table and still not
    exist for that company. ``columns``: table -> {column: type}; only text
    columns of known tables count."""
    from sqlglot import exp

    from backend.app.services.structured.sql_guard import SqlGuardError, parse_one_select

    try:
        root = parse_one_select(sql)
    except SqlGuardError:
        return []
    alias = {}
    for t in root.find_all(exp.Table):
        if t.name and t.name.lower() in columns:
            alias[t.alias_or_name.lower()] = t.name.lower()
    tables = sorted(set(alias.values()))

    def _table_of(col) -> str | None:
        qual = (col.table or "").lower()
        if qual:
            return alias.get(qual)
        owners = [t for t in tables if col.name.lower() in columns[t]]
        return owners[0] if len(owners) == 1 else None

    def _vals(node) -> list[str]:
        # Compiled specs bind as CAST(:p AS text) and lists as = ANY(CAST(:p AS text[])).
        while isinstance(node, (exp.Cast, exp.Any, exp.Paren)):
            node = node.this
        if isinstance(node, exp.Literal) and node.is_string:
            return [node.this]
        if isinstance(node, exp.Placeholder) and node.name:
            v = (params or {}).get(node.name)
            if isinstance(v, str):
                return [v]
            if isinstance(v, list):
                return [x for x in v if isinstance(x, str)]
        return []

    found: list[tuple[str, str, list[str], bool]] = []   # table, column, values, single-EQ
    for node in root.find_all(exp.EQ, exp.In):
        col, rhs = node.this, (node.expressions if isinstance(node, exp.In) else [node.expression])
        if not isinstance(col, exp.Column):
            continue
        t = _table_of(col)
        if t is None or (columns[t].get(col.name.lower()) or "text") != "text":
            continue
        vals = [v for r in rhs for v in _vals(r)]
        if vals:
            found.append((t, col.name.lower(), vals, isinstance(node, exp.EQ) and len(vals) == 1))

    out: list[dict] = []
    for t, c, vals, _single in found:
        # Only identifier filters (cik, lei, id...) scope the check: another
        # compared value may itself be the wrong one, and using it as context
        # would flag every value next to it.
        ids = (id_columns or {}).get(t, set())
        ctx = [(c2, v2[0]) for t2, c2, v2, single in found
               if t2 == t and c2 != c and single and c2 in ids]
        for v in vals:
            item = {"table": t, "column": c, "value": v, "context": ctx}
            if item not in out:
                out.append(item)
    return out[:MAX_VALUE_CHECKS]


async def missing_values(checks: list[dict]) -> list[dict]:
    """Which compared values do not occur in their column (under the
    statement's other equality filters), with the closest values that do. The
    model only ever sees a handful of example values, so it guesses 'Iron ore'
    for 'Iron ore, cfr spot' and US-GAAP tag names for an IFRS filer.
    Read-only, no LLM."""
    if not checks:
        return []
    from backend.app.db.engine import get_sessionmaker
    from backend.app.services.structured.db_load import q

    out = []
    async with get_sessionmaker()() as s:
        try:
            await s.execute(sql_text("SET TRANSACTION READ ONLY"))
            await s.execute(sql_text("SET LOCAL statement_timeout = 3000"))
            for chk in checks:
                table, col, val, ctx = chk["table"], chk["column"], chk["value"], chk.get("context") or []
                ref = f"graphrag.{q(table)}"
                binds = {"v": val, **{f"c{i}": v for i, (_c, v) in enumerate(ctx)}}
                where_ctx = "".join(f" AND {q(c)} = :c{i}" for i, (c, _v) in enumerate(ctx))
                hit = (await s.execute(sql_text(
                    f"SELECT 1 FROM {ref} WHERE {q(col)} = :v{where_ctx} LIMIT 1"), binds)).first()
                if hit:
                    continue
                words = re.findall(r"[A-Za-z0-9]{3,}", val)[:4]
                like = " OR ".join(f"{q(col)} ILIKE :w{i}" for i in range(len(words)))
                binds.update({f"w{i}": f"%{w[:max(3, len(w) - 1)]}%" for i, w in enumerate(words)})
                # `%` (trigram similarity >= 0.3) and ILIKE both use the column's
                # trigram index when the loader made one (ensure_value_indexes)
                cond = f"({like}) OR {q(col)} % :v" if like else f"{q(col)} % :v"
                rows = (await s.execute(sql_text(
                    f"SELECT {q(col)} AS v FROM {ref} WHERE ({cond}){where_ctx} GROUP BY 1 "
                    f"ORDER BY similarity({q(col)}, :v) DESC, length({q(col)}) LIMIT {MAX_SUGGESTIONS}"),
                    binds)).scalars().all()
                out.append({"table": table, "column": col, "value": val,
                            **({"where": {c: v for c, v in ctx}} if ctx else {}),
                            "existing_similar": [str(r) for r in rows]})
        except Exception as exc:  # hints are best effort
            out.append({"error": f"value check failed: {str(exc).splitlines()[0][:200]}"})
        finally:
            await s.rollback()
    return out


def _replace_value(params: dict, old: str, new: str) -> str | None:
    """Swap ``old`` for ``new`` in the bound params; returns the param name."""
    for name, v in params.items():
        if v == old:
            params[name] = new
            return name
        if isinstance(v, list) and old in v:
            params[name] = [new if x == old else x for x in v]
            return name
    return None


async def prevalidate_values(sql: str, params: dict, *, col_types: dict[str, dict[str, str]],
                             id_cols: dict[str, set[str]] | None = None,
                             ) -> tuple[dict, list[dict], list[dict]]:
    """Check every text value a statement compares against the column it is
    compared with, BEFORE the statement runs (an indexed lookup each, ~1 ms).
    Returns (params, missing, substituted): a value that differs from an
    existing one only in case/spacing/punctuation is replaced (and recorded);
    any other absent value is returned in ``missing`` with the closest values
    that do exist, so the call is refused instead of running empty."""
    checks = compared_values(sql, params, col_types, id_cols)
    if not checks:
        return params, [], []
    missing = [m for m in await missing_values(checks) if "value" in m]
    if not missing:
        return params, [], []
    fixed, subs, left = dict(params), [], []
    for m in missing:
        want = _norm_value(m["value"])
        near = next((v for v in m.get("existing_similar") or [] if _norm_value(v) == want), None)
        name = _replace_value(fixed, m["value"], near) if near and want else None
        if name:
            subs.append({"param": name, "from": m["value"], "to": near})
            continue
        # one absent value of a LIST keeps the call alive for the others (it is
        # still reported, so the retry can fix it); a single value refuses it
        lst = next((n for n, v in fixed.items() if isinstance(v, list) and m["value"] in v
                    and len(v) > 1), None)
        if lst:
            fixed[lst] = [x for x in fixed[lst] if x != m["value"]]
            m = {**m, "dropped_from": lst}
        left.append(m)
    return fixed, left, subs


async def combination_hints(checks: list[dict]) -> list[dict]:
    """For an EMPTY result whose values all exist on their own: which
    combinations of the other compared columns actually occur with each value
    ('Iron ore, cfr spot' exists, '($/mt)' exists, but not together). Read-only."""
    by_table: dict[str, list[dict]] = {}
    for c in checks:
        by_table.setdefault(c["table"], []).append(c)
    if not any(len({c["column"] for c in cs}) > 1 for cs in by_table.values()):
        return []
    from backend.app.db.engine import get_sessionmaker
    from backend.app.services.structured.db_load import q

    out = []
    async with get_sessionmaker()() as s:
        try:
            await s.execute(sql_text("SET TRANSACTION READ ONLY"))
            await s.execute(sql_text("SET LOCAL statement_timeout = 3000"))
            for table, cs in by_table.items():
                cols = list(dict.fromkeys(c["column"] for c in cs))
                if len(cols) < 2:
                    continue
                # Anchor on the most selective value (fewest combinations):
                # 'Iron ore, cfr spot' says something, 'nominal_usd' does not.
                best = None
                for anchor in cs[:4]:
                    others = [c for c in cols if c != anchor["column"]]
                    rows = (await s.execute(sql_text(
                        f"SELECT DISTINCT {', '.join(q(c) for c in others)} FROM graphrag.{q(table)} "
                        f"WHERE {q(anchor['column'])} = :v LIMIT {MAX_SUGGESTIONS + 1}"),
                        {"v": anchor["value"]})).all()
                    if rows and (best is None or len(rows) < len(best[2])):
                        best = (anchor, others, rows)
                if best is not None:
                    anchor, others, rows = best
                    out.append({"table": table, "where": {anchor["column"]: anchor["value"]},
                                "existing_combinations": [dict(zip(others, map(str, r), strict=True))
                                                          for r in rows[:MAX_SUGGESTIONS]]})
        except Exception as exc:  # hints are best effort
            out.append({"error": f"combination check failed: {str(exc).splitlines()[0][:200]}"})
        finally:
            await s.rollback()
    return out


def _listing_text(head: str, res: SqlResult, char_cap: int, summary: str = "") -> str:
    """Compact layout for a LISTING result: the header once, then one line of
    values per row (about half the size of name=value pairs). ``summary`` (the
    exact change statistics) goes first, so the row cap never cuts it."""
    if summary:
        head += summary + " "
    def cell(v) -> str:
        return "" if v is None else str(v).replace("|", "/")
    lines = [" | ".join(res.columns)]
    used = len(head) + len(lines[0])
    shown = 0
    for r in res.rows:
        row = " | ".join(cell(v) for v in r)
        if used + len(row) + 3 > char_cap - 160:
            break
        lines.append(row)
        used += len(row) + 3
        shown += 1
    note = f"{res.row_count} row(s)" + (f" of {res.total_rows if res.total_rows is not None else 'more'}"
                                         " (capped)" if res.truncated else "")
    if shown < len(res.rows):
        note += f"; {shown} shown here, {len(res.rows) - shown} more in the returned data"
    return head + f"LISTING, {note}. Rows: " + " ;; ".join(lines)


def _listing_stats(rec: dict, res: SqlResult, meta: dict) -> dict | None:
    if not rec.get("listing"):
        return None
    from backend.app.services.structured.listing_stats import change_stats, roles_for

    period_col, value_col = roles_for(meta or {}, rec.get("query_id"), res.columns)
    return change_stats(res.columns, res.rows, period_col, value_col) if period_col else None


MAX_CHECK_VALUES = 6


def value_check_evidence(calls: list[dict], by_table: dict[str, dict], start_rank: int, *,
                         char_cap: int = 5000) -> list[dict]:
    """Values a call asked for that the TABLE DOES NOT HOLD (found by an exact
    lookup, or not among the column's listed values), as evidence for the
    answer model -- one item per table. Without it the answer model only sees
    the table's description ('a spending column per drug') and may assume an
    absent item is in the table. A value some call used successfully -- or one
    whose suggested replacement a retry used -- is left out."""
    used = {_norm_value(v) for c in calls if not c.get("error") and c.get("row_count")
            for v in _flat(c.get("params") or {})}
    by_tab: dict[str, dict[tuple, dict]] = {}
    for c in calls:
        table = c.get("table")
        for m in c.get("missing_values") or []:
            if "value" not in m or _norm_value(m["value"]) in used:
                continue
            if any(_norm_value(x) in used for x in m.get("existing_similar") or []):
                continue                         # a retry used the corrected value
            key = (m.get("column"), _norm_value(m["value"]), json.dumps(m.get("where") or {}, sort_keys=True))
            by_tab.setdefault(m.get("table") or table, {}).setdefault(key, {
                "column": m.get("column"), "value": m["value"], "where": m.get("where") or {},
                "near": m.get("existing_similar") or []})
        for m in c.get("invalid_values") or []:
            if "value" not in m or not m.get("column") or _norm_value(m["value"]) in used:
                continue
            key = (m["column"], _norm_value(m["value"]), "{}")
            by_tab.setdefault(table, {}).setdefault(key, {
                "column": m["column"], "value": m["value"], "where": {},
                "listed": m.get("allowed_values") or [], "more": m.get("more_values") or 0})
    out = []
    for table, checks in by_tab.items():
        if not table:
            continue
        lines = []
        for chk in list(checks.values())[:MAX_CHECK_VALUES]:
            ctx = (" together with " + ", ".join(f"{k} = '{v}'" for k, v in chk["where"].items())
                   if chk["where"] else "")
            if "listed" in chk:
                shown = " | ".join(map(str, chk["listed"][:20]))
                more = f" (+{chk['more']} more)" if chk["more"] else ""
                lines.append(f"'{chk['value']}' is not one of the values of column {chk['column']}"
                             f" (its values: {shown}{more}).")
            else:
                near = "; values that do occur and look similar: " + " | ".join(chk["near"][:8]) \
                    if chk["near"] else "; no similar value occurs"
                lines.append(f"'{chk['value']}' does not occur in column {chk['column']}{ctx}{near}.")
        meta = (by_table.get(table) or {}).get("meta") or {}
        text = (f"Value check on table {table}"
                + (f" [{(meta.get('source') or {}).get('rel_path')}]" if meta else "")
                + " (exact lookup in the loaded table, run before any query): "
                + " ".join(lines) + " The table holds no rows for these values; do not assume "
                "it covers them.")[:char_cap]
        digest = hashlib.sha256(text.encode()).hexdigest()[:8]
        frag = f"SqlResult_{table}_value_check_{digest}"
        out.append({
            "kind": "sql_result", "node_id": (by_table.get(table) or {}).get("id"),
            "iri": f"{VIAO_NS}#{re.sub(r'[^A-Za-z0-9_]', '_', frag)}",
            "rank": start_rank + len(out), "score": 1.0, "text": text, "char_cap": char_cap,
            "listing": False, "document_title": f"table {table}", "table": table,
            "query_id": "value_check", "sql": None, "params": {}, "columns": [], "rows": [],
            "row_count": 0, "truncated": False, "total_rows": None,
            "source_file": (meta.get("source") or {}).get("rel_path"), "sql_source": "value_check",
        })
    return out


def relevance_hints(cands: list[dict], rankings: dict[str, dict], top: int = 8) -> list[dict]:
    """Per candidate table (most relevant first): its graph score and the
    stored items whose tags the question matched, with why."""
    out = []
    for c in sorted(cands, key=lambda c: -(c.get("score") or 0)):
        r = rankings.get(c["table"]) or {}
        matched = [{"id": i, "matched": w} for i, (sc, w) in sorted(r.items(), key=lambda kv: -kv[1][0])
                   if sc > 0][:top]
        out.append({"table": c["table"], "reached_via": c.get("via") or [], "matched_items": matched})
    return out


def _flat(params: dict) -> list:
    out = []
    for v in params.values():
        out.extend(v if isinstance(v, list) else [v])
    return [v for v in out if isinstance(v, str)]


def _evidence_text(rec: dict, res: SqlResult, meta: dict, char_cap: int, stats: dict | None = None) -> str:
    head = (f"SQL result from table {rec['table']}"
            + (f" ({rec['label']})" if rec.get("label") else "")
            + (f" [{(meta.get('source') or {}).get('rel_path')}]" if meta else "") + ". ")
    if rec.get("params"):
        head += "Parameters: " + ", ".join(f"{k}={v}" for k, v in rec["params"].items()) + ". "
    if rec.get("listing"):
        from backend.app.services.structured.listing_stats import stats_text

        if rec.get("empty_excluded"):
            head += EMPTY_EXCLUDED_NOTE.format(columns=", ".join(rec["empty_excluded"])) + " "
        return _listing_text(head, res, char_cap, stats_text(stats) if stats else "")
    cols = res.columns
    body = []
    used = len(head)
    for r in res.rows:
        row = "; ".join(f"{c}={'' if v is None else v}" for c, v in zip(cols, r, strict=False))
        if used + len(row) + 4 > char_cap - 120:
            body.append(f"... ({len(res.rows) - len(body)} more fetched row(s) not shown)")
            break
        body.append("(" + row + ")")
        used += len(row) + 4
    tail = ""
    if rec.get("unranked_rows"):
        tail += " " + UNRANKED_NOTE.format(**rec["unranked_rows"])
    if rec.get("empty_excluded"):
        tail += " " + EMPTY_EXCLUDED_NOTE.format(columns=", ".join(rec["empty_excluded"]))
    if res.truncated:
        tail += (f" Result capped at {res.row_count} of "
                f"{res.total_rows if res.total_rows is not None else 'more'} rows.")
    return head + f"{res.row_count} row(s): " + " ".join(body) + tail


async def structured_evidence(
    question: str,
    *,
    router,
    entity_ids: list[uuid.UUID],
    class_ids: list[uuid.UUID],
    qvec: list[float] | None,
    max_tables: int = 3,
    max_rows: int = 50,
    timeout_ms: int = 5000,
    max_calls: int = 4,
    char_cap: int = 5000,
    vector_max_distance: float | None = None,
    verbose: bool = False,
    time_ids: list[uuid.UUID] | None = None,
    listing_max_rows: int = 2000,
    listing_char_cap: int = 60000,
    doc_context: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Evidence items (``kind='sql_result'``) + a plan record for persistence."""
    from backend.app.services.prompts import PROMPTS

    t0 = time.monotonic()
    plan: dict[str, Any] = {"candidates": [], "needed": False, "calls": [], "retried": False}
    cands = await candidate_tables(entity_ids, class_ids, qvec, max_tables=max_tables,
                                   vector_max_distance=vector_max_distance, time_ids=time_ids)
    plan["candidates"] = [{"table": c["table"], "score": round(c["score"], 3), "via": c["via"]}
                          for c in cands]
    rankings = {c["table"]: rank_items(c["meta"], c.get("hits")) for c in cands}
    plan["ranked_queries"] = {t: [i for i, (sc, _w) in sorted(r.items(), key=lambda kv: -kv[1][0]) if sc > 0][:5]
                              for t, r in rankings.items()}
    if not cands:
        plan["reason"] = "no table reachable from the graph walk"
        return [], plan
    by_table = {c["table"]: c for c in cands}
    all_tables = set(by_table)
    for c in cands:  # lookups may join a table outside the candidate set
        for it in c["meta"].get("lookups") or []:
            if it.get("other_table"):
                all_tables.add(it["other_table"])
    partners = await table_metas(all_tables - set(by_table)) if all_tables - set(by_table) else {}
    all_tables = set(by_table) | set(partners)   # a partner that is not loaded cannot be read
    types = {c["table"]: column_types(c["meta"]) for c in cands}
    types.update({n: column_types(m) for n, m in partners.items()})
    # Catalogs are the same text whatever the question (sorted by table, stored
    # order) so repeated calls share a cacheable prompt prefix; what the
    # question matched goes in a short separate block.
    catalogs = [catalog_for_prompt(c["meta"], ranking=rankings.get(c["table"]), stable=True)
                for c in sorted(cands, key=lambda c: c["table"])]
    relevance = relevance_hints(cands, rankings)
    inv = await table_inventory()
    inventory = inventory_for_prompt(inv)
    col_types = {c["table"]: {col["name"].lower(): (col.get("type") or "text")
                              for col in c["meta"].get("columns") or []} for c in cands}
    id_cols = {c["table"]: {col["name"].lower() for col in c["meta"].get("columns") or []
                            if col.get("role") == "identifier"} for c in cands}

    async def _value_failures(rec: dict, res: SqlResult | None) -> list[dict]:
        if res is None or res.error:
            return []
        checks = compared_values(res.sql, res.params, col_types, id_cols)
        missing = [m for m in await missing_values(checks) if "value" in m]
        if not missing and res.row_count == 0:
            combos = [h for h in await combination_hints(checks) if "where" in h]
            if combos:
                rec["value_combinations"] = combos
        return missing

    system, user = PROMPTS["table_query_select"](question, catalogs, max_calls=max_calls, doc_context=doc_context, relevance=relevance,
                                                  inventory=inventory)
    out = await router.chat("table_query_select", system=system, user=user)
    choice = _parse(out.text)
    if "needed" not in choice:
        # cut off at max_tokens or not JSON: an error, never a silent "not needed"
        plan["select_error"] = (f"selection reply unreadable (finish_reason={getattr(out, "finish_reason", None)}, "
                                f"{len(out.text or '')} chars)")
        plan["retried"] = True
        system, user = PROMPTS["table_query_select"](question, catalogs, [{
            "error": "your previous reply was cut off or was not valid JSON. Reply again with FEWER "
                     "calls: one call per part of the question; a question that lists many items is "
                     "ONE listing query, never one call per item."}], max_calls=max_calls,
            inventory=inventory, doc_context=doc_context, relevance=relevance)
        out = await router.chat("table_query_select", system=system, user=user)
        choice = _parse(out.text)
        if "needed" not in choice:
            plan["select_error"] += f"; retry also unreadable (finish_reason={getattr(out, "finish_reason", None)})"
    describe = [t for t in (choice.get("describe") or []) if isinstance(t, str) and t in inv]
    plan["describe"] = describe
    plan["needed"] = bool(choice.get("needed")) or bool(describe)
    plan["reason"] = choice.get("reason")
    if not plan["needed"]:
        return [], plan

    calls = [c for c in (choice.get("calls") or []) if isinstance(c, dict)][:max_calls]
    today_year = date.today().year
    for call in calls:   # 'past 3 years' ends at the table's latest period, not the model's guess
        meta_c = (by_table.get(call.get("table")) or {}).get("meta")
        adj = anchor_relative_periods(question, call, meta_c, today_year=today_year) if meta_c else None
        if adj:
            plan.setdefault("period_adjustments", []).append({"table": call.get("table"), **adj})
    results: list[tuple[dict, SqlResult]] = []
    failures: list[dict] = []
    metas_all = {**{c["table"]: c["meta"] for c in cands}, **partners}
    totals = {t: total_markers(m) for t, m in metas_all.items()}
    seen_values: set[str] = set()       # every value an earlier call returned

    doc_norm = _norm_value(" ".join(str(d.get("text") or "") for d in doc_context or []))

    async def _guard(sql: str, params: dict) -> list[dict]:
        return await identifier_guard(sql, params, question=question, seen_values=seen_values,
                                      types=types, metas=metas_all, context_norm=doc_norm)

    async def _prevalidate(sql: str, params: dict) -> tuple[dict, list[dict], list[dict]]:
        return await prevalidate_values(sql, params, col_types=col_types, id_cols=id_cols)

    def _rows_for(call: dict) -> int:
        return listing_max_rows if call.get("listing") else max_rows

    async def _run_calls(batch: list[dict], *, retry: bool) -> None:
        done: dict[int, list[SqlResult]] = {}       # call number -> its results (chaining)
        for n, call in enumerate(batch, start=1):
            concrete, ref_err = expand_refs(call, done)
            if ref_err:
                rec = {"table": call.get("table"), "query_id": call.get("query_id"), "error": ref_err,
                       **({"retry": True} if retry else {})}
                plan["calls"].append(rec)
                failures.append({"call": call, "error": ref_err})
                continue
            for sub in concrete:
                sub_call = {k: v for k, v in sub.items() if not k.startswith("_")}
                rec, res, err = await _execute(sub_call, by_table, max_rows=_rows_for(sub),
                                               timeout_ms=timeout_ms, all_tables=all_tables,
                                               types=types, totals=totals, guard=_guard,
                                               prevalidate=_prevalidate)
                if retry:
                    rec["retry"] = True
                if sub.get("_from_calls"):
                    rec["from_calls"] = sub["_from_calls"]
                    if sub.get("_fanout_cut"):
                        rec["fanout_cut"] = sub["_fanout_cut"]
                rec["listing"] = bool(sub.get("listing"))
                missing = await _value_failures(rec, res) or rec.get("missing_values") or []
                if missing:
                    rec["missing_values"] = missing
                unranked = None if err else await _unranked_rows(
                    rec, res, (by_table.get(rec.get("table")) or {}).get("meta"))
                if unranked:
                    rec["unranked_rows"] = unranked
                plan["calls"].append(rec)
                if (err or missing or unranked) and not retry:
                    failures.append({"call": call, "sql": rec.get("sql"),
                                     "error": err or ("value(s) not found in their column (see missing_values)"
                                                      if missing else UNRANKED_NOTE.format(**unranked)),
                                     **{k: rec[k] for k in ("invalid_values", "unsupported_identifiers",
                                                            "value_combinations", "missing_values")
                                        if rec.get(k)},
                                     **({"missing_values": missing} if missing else {})})
                if not err and res is not None:
                    # A partial hit (one IN value matched, another did not) is
                    # still evidence; the retry may add the missing part.
                    results.append((rec, res))
                    if res.rows:
                        done.setdefault(n, []).append(res)
                        seen_values.update(_norm_value(v) for r in res.rows for v in r if v is not None)

    await _run_calls(calls, retry=False)

    if failures:
        plan["retried"] = True
        system, user = PROMPTS["table_query_select"](question, catalogs, failures, max_calls=max_calls,
                                                     doc_context=doc_context, relevance=relevance,
                                                     inventory=inventory)
        out = await router.chat("table_query_select", system=system, user=user)
        retry = _parse(out.text)
        retry_calls = [c for c in (retry.get("calls") or []) if isinstance(c, dict)][:max_calls]
        for call in retry_calls:
            meta_c = (by_table.get(call.get("table")) or {}).get("meta")
            adj = anchor_relative_periods(question, call, meta_c, today_year=today_year) if meta_c else None
            if adj:
                plan.setdefault("period_adjustments", []).append({"table": call.get("table"), **adj})
        await _run_calls(retry_calls, retry=True)

    # The retry may repeat a statement that already succeeded; keep one copy.
    seen_runs: set[str] = set()
    unique = []
    for rec, res in results:
        k = json.dumps([" ".join(res.sql.split()), res.params], sort_keys=True, default=str)
        if k not in seen_runs:
            seen_runs.add(k)
            unique.append((rec, res))
    results = unique

    evidence = []
    for i, (rec, res) in enumerate(results, start=1):
        meta = (by_table.get(rec["table"]) or {}).get("meta") or {}
        node = (by_table.get(rec["table"]) or {}).get("id")
        # Same statement + params -> same id (deep_research rounds can then
        # share it); a different run of the same query gets a different id.
        digest = hashlib.sha256(json.dumps([res.sql, res.params], sort_keys=True,
                                           default=str).encode()).hexdigest()[:8]
        frag = f"SqlResult_{rec['table']}_{rec.get('query_id') or 'q'}_{digest}"
        stats = _listing_stats(rec, res, meta)
        evidence.append({
            "kind": "sql_result",
            "node_id": node,
            "iri": f"{VIAO_NS}#{re.sub(r'[^A-Za-z0-9_]', '_', frag)}",
            "rank": i, "score": 1.0,
            "text": _evidence_text(rec, res, meta, listing_char_cap if rec.get("listing") else char_cap, stats),
            "change_stats": stats,
            "char_cap": listing_char_cap if rec.get("listing") else char_cap,
            "listing": bool(rec.get("listing")),
            "document_title": f"table {rec['table']}",
            "table": rec["table"], "query_id": rec.get("query_id"), "sql": res.sql,
            "params": res.params, "columns": res.columns, "rows": res.rows,
            "row_count": res.row_count, "truncated": res.truncated, "total_rows": res.total_rows,
            "source_file": (meta.get("source") or {}).get("rel_path"),
            "sql_source": rec.get("source"),
        })
    evidence += value_check_evidence(plan["calls"], by_table, len(evidence) + 1, char_cap=char_cap)
    evidence += describe_evidence(describe, inv, len(evidence) + 1)
    plan["seconds"] = round(time.monotonic() - t0, 3)
    if verbose:
        print(f"[query] structured data: {len(cands)} candidate table(s) "
              f"{[c['table'] for c in cands]}, needed={plan['needed']}, "
              f"{len(evidence)} result(s), retried={plan['retried']}")
    return evidence, plan
