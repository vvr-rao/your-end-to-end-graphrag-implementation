"""``query --use-structured-data``: answer from the loaded tables via SQL only.

Runs after the graph walk, before answer synthesis:

1. **Find tables through the graph.** TableMetadata nodes linked to the
   entities the walk reached (``tableEntity`` edges) or to the ontology classes
   the question matched (``tableColumn``/``tableMetric``/``tableFilter``);
   a vector search over the metadata nodes fills the remaining slots.
2. **Decide and pick** (task ``table_query_select``, one call): is table data
   needed, and which STORED metric/filter/query/lookup/measure answers each
   part, with which parameter values: every answer-producing call is
   ``{table, query_id, params}``. A part no stored query answers is reported
   as ``no_match``. The model never writes SQL or a query spec; a call that
   carries either is refused, even next to a valid query_id.
3. **Run** the exact SQL ingestion verified (``item["sql"]``, see
   :func:`metadata.selectable`) after checking the parameter values against the
   stored declaration (names, required values, types, valid values and
   combinations); the driver binds them, and the guard (``sql_exec``) keeps it
   read-only, timed out and row-capped. A call shown to be wrong (selection
   error, invalid or absent value) gets one retry, which must keep every value
   the question asked for; a query that returns no rows is a valid answer and
   is never retried or broadened.
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
import math
import re
import time
import unicodedata
import uuid
from collections.abc import Awaitable, Callable
from datetime import date
from typing import Any

from sqlalchemy import text as sql_text

from backend.app.services.predicates import VIAO_NS
from backend.app.services.structured.metadata import (
    catalog_for_prompt,
    column_types,
    find_item,
    item_spec,
    time_columns,
)
from backend.app.services.structured.sql_exec import SqlResult, run_sql

MAX_LISTING_CALLS = 2           # listing calls (rows the answer is read from) per batch
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
def partial_note(shown: int, fetched: int, truncated: bool, total_rows: int | None) -> str:
    """Say plainly when the rows in the prompt are not all the rows the query
    matched: how many are shown, how many it returned, and the full count when
    known. Empty when every matching row is shown."""
    if shown >= fetched and not truncated:
        return ""
    if truncated:
        full = (f"the query matched {total_rows} rows in total" if total_rows is not None
                else f"the query matched more than the {fetched} rows fetched (full count unknown)")
    else:
        full = f"the query returned {fetched} rows"
    return (f" PARTIAL RESULT: {shown} row(s) shown here; {full}. A sum, count, average, share or "
            "ranking computed from the rows shown is NOT the complete figure, and rows not shown are "
            "not missing from the data.")


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


_RRF_K = 60           # reciprocal-rank-fusion constant (the usual value; measured insensitive)
_RANK_STOPWORDS = _ID_STOPWORDS | {"data", "value", "values", "table", "tables"}


def _rank_terms(text: str | None) -> set[str]:
    """Content words for the lexical signal: lower-cased, function words and
    words under three letters dropped, a plain plural reduced to its singular."""
    out = set()
    for w in re.findall(r"[a-z0-9][a-z0-9\-]*", (text or "").lower()):
        w = w.strip("-")
        if len(w) < 3 or w in _RANK_STOPWORDS:
            continue
        if w.endswith("ies") and len(w) > 4:
            w = w[:-3] + "y"
        elif w.endswith("s") and not w.endswith("ss") and len(w) > 3:
            w = w[:-1]
        out.add(w)
    return out


def _idf(n_tables: int, df: int) -> float:
    return math.log((n_tables + 1) / (df + 0.5))


def _fuse(signals: list[dict[uuid.UUID, float]], k: int = _RRF_K) -> dict[uuid.UUID, float]:
    """Reciprocal-rank fusion of higher-is-better signals; a table scores
    only on the signals it has (a positive value)."""
    fused: dict[uuid.UUID, float] = {}
    for sig in signals:
        ranked = sorted((a for a, v in sig.items() if v > 0), key=lambda a: -sig[a])
        for rank, aid in enumerate(ranked, start=1):
            fused[aid] = fused.get(aid, 0.0) + 1.0 / (k + rank)
    return fused


async def candidate_tables(
    entity_ids: list[uuid.UUID],
    class_ids: list[uuid.UUID],
    qvec: list[float] | None,
    *,
    max_tables: int = 3,
    vector_max_distance: float | None = None,
    time_ids: list[uuid.UUID] | None = None,
    entity_scores: dict[uuid.UUID, float] | None = None,
    question: str | None = None,
) -> list[dict[str, Any]]:
    """[{id, table, meta, score, via, hits}] best first. ``hits`` records what
    the question reached the table through (entity/place columns, class
    labels, periods) so the table's tagged queries can be ranked.

    Three signals, each ranking the tables, fused by reciprocal rank:

    - graph: the reached entities, matched classes and periods linked to the
      table, each weighted by how few tables it links to (a country linked to
      every table with a country column says little about which one is meant)
      and an entity by how strongly the walk reached it (``entity_scores``:
      1.0 for a seed, decaying per hop);
    - vector: the question's distance to the table description;
    - lexical: the question's content words found in the table description,
      a word found in few tables counting more.

    Raw link counts used to decide alone, so a question naming many places
    ranked every table with a place column above the one it was about."""
    from backend.app.db.session import session_scope

    hits_of: dict[uuid.UUID, dict[str, Any]] = {}
    via: dict[uuid.UUID, list[str]] = {}
    graph: dict[uuid.UUID, float] = {}
    vector: dict[uuid.UUID, float] = {}
    lexical: dict[uuid.UUID, float] = {}
    dist_of: dict[uuid.UUID, float] = {}

    def _hits(aid):
        return hits_of.setdefault(aid, {"columns": set(), "classes": set(), "time": False})

    async with session_scope() as s:
        tables = (await s.execute(sql_text(
            "SELECT id, text FROM graphrag.intelligence_artifacts "
            "WHERE artifact_type = 'TableMetadata' AND status = 'ACTIVE'"))).all()
        n_tables = len(tables)
        if not n_tables:
            return []
        if entity_ids or class_ids or time_ids:
            rows = (await s.execute(sql_text("""
                SELECT a.id, gr.target_node_type, gr.target_node_id, gr.relationship_type,
                       gr.extra_metadata, oc.iri, oc.label
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
            tables_of: dict[tuple[str, uuid.UUID], set[uuid.UUID]] = {}
            counts: dict[tuple[uuid.UUID, str], set[uuid.UUID]] = {}
            for aid, ttype, tid, kind, extra, ciri, clabel in rows:
                tables_of.setdefault((ttype, tid), set()).add(aid)
                h = _hits(aid)
                if ttype == "time_instance":
                    h["time"] = True
                elif ttype == "entity":
                    h["columns"].update((extra or {}).get("columns") or [])
                elif ttype == "ontology_class":
                    h["classes"].update(x for x in (ciri, clabel) if x)
                key = "place" if kind == "tablePlace" else ttype
                counts.setdefault((aid, key), set()).add(tid)
            time_best: dict[uuid.UUID, float] = {}
            for (ttype, tid), aids in tables_of.items():
                w = _idf(n_tables, len(aids))
                if ttype == "entity":
                    w *= float((entity_scores or {}).get(tid, 1.0))
                for aid in aids:
                    if ttype == "time_instance":   # once: a long series covers many periods
                        time_best[aid] = max(time_best.get(aid, 0.0), w)
                    else:
                        graph[aid] = graph.get(aid, 0.0) + w
            for aid, w in time_best.items():
                graph[aid] = graph.get(aid, 0.0) + w
            for (aid, key), tids in counts.items():
                via.setdefault(aid, []).append(f"{key}:{len(tids)}")
        if qvec is not None:
            for aid, dist in (await s.execute(sql_text("""
                SELECT id, (embedding <-> CAST(:q AS vector)) AS dist
                  FROM graphrag.intelligence_artifacts
                 WHERE artifact_type = 'TableMetadata' AND status = 'ACTIVE' AND embedding IS NOT NULL
            """), {"q": _vec(qvec)})).all():
                if vector_max_distance is not None and float(dist) > vector_max_distance:
                    continue
                dist_of[aid] = float(dist)
                vector[aid] = 1.0 / (1.0 + float(dist))
        q_terms = _rank_terms(question)
        if q_terms:
            terms = {aid: _rank_terms(text) for aid, text in tables}
            df: dict[str, int] = {}
            for ts in terms.values():
                for w in ts & q_terms:
                    df[w] = df.get(w, 0) + 1
            for aid, ts in terms.items():
                sc = sum(_idf(n_tables, df[w]) for w in ts & q_terms)
                if sc > 0:
                    lexical[aid] = sc
        fused = _fuse([graph, vector, lexical])
        if not fused:
            return []
        best = sorted(fused, key=lambda a: (-fused[a], -graph.get(a, 0.0),
                                            dist_of.get(a, float("inf"))))[:max_tables]
        metas = dict((await s.execute(sql_text("""
            SELECT id, extra_metadata FROM graphrag.intelligence_artifacts
             WHERE id = ANY(CAST(:ids AS uuid[]))
        """), {"ids": [str(a) for a in best]})).all())
    out = []
    for aid in best:
        meta = metas.get(aid) or {}
        if not meta.get("table_name"):
            continue
        v = list(via.get(aid, []))
        if aid in graph:
            v.append(f"graph:{graph[aid]:.2f}")
        if aid in dist_of:
            v.append(f"vector:{dist_of[aid]:.3f}")
        if aid in lexical:
            v.append(f"lexical:{lexical[aid]:.2f}")
        out.append({"id": aid, "table": meta["table_name"], "meta": meta, "score": fused[aid],
                    "via": v, "hits": _hits(aid)})
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
    found = find_item(meta, call["query_id"]) if isinstance(call.get("query_id"), str) else None
    spec = item_spec(found[0], found[1]) if found else None
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


# Typographic look-alikes of ASCII punctuation: text copied from documents
# (curly apostrophes, en dashes, no-break spaces) rarely match a table's spelling
# byte for byte.
_TYPO_FOLD = str.maketrans({
    "\u2019": "'", "\u2018": "'", "\u02bc": "'", "\u201b": "'", "\u2032": "'", "\u00b4": "'",
    "`": "'", "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u2033": '"',
    "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-", "\u2212": "-",
    "\u00a0": " ", "\u2007": " ", "\u202f": " ", "\u2009": " "})


def fold_text(text: str) -> str:
    """Case-, accent- and typography-folded copy of ``text`` with EXACTLY one
    character per input character, so a match found in the folded copy has
    the same position in the original."""
    out = []
    for ch in str(text).translate(_TYPO_FOLD):
        base = unicodedata.normalize("NFKD", ch)[:1] or ch
        low = base.lower()
        out.append(low if len(low) == 1 else ch)
    return "".join(out)


def _norm_value(v: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", fold_text(str(v)))


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
        ks = ref.get("from_call")
        try:
            ks = [int(x) for x in ks] if isinstance(ks, list) else [int(ks)]
        except (TypeError, ValueError):
            return [], f"param {name}: from_call must be the number (or a list of numbers) of earlier calls"
        results = []
        for k in ks:   # several earlier calls: their values are pooled
            if not done.get(k):
                return [], (f"param {name}: call {k} has no result to take values from (it failed, "
                            "returned no rows, or is not an earlier call)")
            results.extend(done[k])
        k = ks[0] if len(ks) == 1 else ks
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
    checks = [c for c in compared_values(sql, params, lowered, id_cols, bound_only=True)
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
        # a value that is a NAME ("Crude oil, Brent") is stated by the question
        # when every significant word of it is ("Brent crude oil"), in any order;
        # a code ("CHEMBL435", "ID004") is one word and still needs itself
        v_words = _sig_words(c["value"])
        if len(v_words) >= 2 and v_words <= q_words:
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
                # the identifier's own value is read too: a NAME-like id
                # ("Crude oil, Brent") can carry the distinctive word itself
                own_col = lowered.get(c["table"], {}).get(c["column"], "text") == "text"
                cols = ([c["column"]] if own_col else []) + [n for n in names if n != c["column"]]
                vals: list[str] = []
                matched = False
                if cols:
                    rows = (await s.execute(sql_text(
                        f"SELECT {', '.join(q(n) for n in cols)} FROM graphrag.{q(c['table'])} "
                        f"WHERE {q(c['column'])} = :v LIMIT 3"), {"v": c["value"]})).all()
                    vals = list(dict.fromkeys(str(x) for r in rows for x, n in zip(r, cols, strict=True)
                                              if x not in (None, "") and n in names))
                    # the id's own row names what the question asks about -- by a
                    # word that is distinctive in its column (not one every row shares)
                    async def _share(col: str, word: str, table: str = c["table"]) -> float:
                        return (await s.execute(sql_text(
                            f"SELECT COUNT(DISTINCT {q(col)}) FILTER (WHERE {q(col)} ~* :w)::float "
                            f"/ GREATEST(COUNT(DISTINCT {q(col)}), 1) FROM graphrag.{q(table)}"),
                            {"w": _word_regex(word)})).scalar() or 0.0

                    for n in cols:
                        own = {w for r in rows for w in _sig_words(r[cols.index(n)] or "")}
                        if n == c["column"]:
                            # the id itself: its MOST distinctive word is the one
                            # that singles the row out ('brent' in 'Crude oil,
                            # Brent'), so that word must be the one stated
                            shares = {w: await _share(n, w) for w in sorted(own)[:8]}
                            best = min(shares.values(), default=1.0)
                            matched = best <= DISTINCTIVE_SHARE and any(
                                w in q_words for w, sh in shares.items() if sh == best)
                        else:
                            for w in sorted(own & q_words)[:6]:
                                if await _share(n, w) <= DISTINCTIVE_SHARE:
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


# Keys that would carry SQL or a query spec in a call. Retrieval only runs
# stored, tested queries: a call carrying any of them is refused, even when it
# also names a valid query_id.
INLINE_KEYS = ("query", "sql", "spec", "query_spec", "new_sql", "definition")
INLINE_ERROR = ("a call may only name a stored query: {{\"table\", \"query_id\", \"params\"}}. "
                "Inline SQL or a query spec is refused (found: {keys}).")


def cap_listings(calls: list[dict], plan: dict) -> list[dict]:
    """At most MAX_LISTING_CALLS listing calls per batch (rows the answer is
    read from); further listings are dropped and recorded in the plan."""
    out, n = [], 0
    for c in calls:
        if c.get("listing"):
            n += 1
            if n > MAX_LISTING_CALLS:
                plan.setdefault("listing_cut", []).append(
                    {k: c.get(k) for k in ("table", "query_id", "params")})
                continue
        out.append(c)
    return out


TOTAL_SCOPE_ERROR = ("the call is marked scope \"total\", but stored query {qid!r} returns the PARTS of a "
                     "total (one row per {grain}), not the total; verified stored queries of the same "
                     "table compute it: {alts}. Select one of them (same values for the item, period "
                     "and qualifiers).")
TOTAL_PARTS_NOTE = ("The question asks for a total, but no stored query computes it: these rows are its "
                    "parts, not a total.")


def _row_level(desc: dict) -> bool:
    op = desc.get("operation")
    return isinstance(op, str) and (op.startswith("one value per observation") or op in ("rows", "distinct rows"))


def _family(meta: dict, cols) -> set[str]:
    """Columns plus those they determine or are determined by (a code and its
    name are one dimension), from the measured dependencies."""
    deps = ((meta or {}).get("analysis") or {}).get("dependencies") or {}
    out: set[str] = set()
    for c in cols:
        c = str(c).split(".")[-1]
        out |= {c, *deps.get(c, [])} | {d for d, fixed in deps.items() if c in (fixed or [])}
    return out


def open_dimensions(meta: dict, desc: dict, params: dict | None) -> set[str]:
    """Row dimensions of a row-level query that the call leaves OPEN (rows vary
    over them, so the rows are parts to be summed). A dimension is pinned by
    an equality parameter the call fills (one value, or a list of values the
    call asks for -- one row per listed value is what was asked), by a fixed
    equality condition, or by a range whose two bounds are the same value
    (year 2024..2024). Column families count as one dimension (a code and its
    name). Empty set = every dimension pinned."""
    given = {k: v for k, v in (params or {}).items() if v not in (None, "", [])}
    # only an equality fixes a dimension (an exclusion such as <> 'total row' does not)
    eq = ("=", "in", None)
    fixed = [e.get("column") for e in desc.get("parameters") or []
             if e.get("name") in given and e.get("column") and e.get("op") in eq]
    fixed += [f.get("column") for f in desc.get("fixed_conditions") or []
              if f.get("column") and f.get("op") in ("=", "in")]
    bounds: dict[str, list] = {}
    for e in desc.get("parameters") or []:
        if e.get("op") == "between" and e.get("column") and e.get("name") in given:
            bounds.setdefault(e["column"], []).append(given[e["name"]])
    fixed += [c for c, vals in bounds.items()
              if len(vals) == 2 and not isinstance(vals[0], list) and str(vals[0]) == str(vals[1])]
    fixed_fam = _family(meta, [c for c in fixed if c])
    return _family(meta, [g for g in desc.get("output_grain") or []
                          if isinstance(g, str) and g.split(".")[-1] not in fixed_fam])


def total_alternatives(meta: dict, section: str, item: dict,
                       versions: dict[str, str | None] | None = None,
                       params: dict | None = None) -> list[str]:
    """Verified, retrieval-eligible stored queries of the same table that SUM
    the quantity a row-level item returns, without breaking the total down by
    a dimension the item's rows vary over and the call leaves open (that
    would list the parts again). Empty when the item is itself an aggregate.
    Read from each item's machine description, no LLM."""
    from backend.app.services.structured.metadata import describe_query, items, selectable

    desc = describe_query(meta, section, item)
    if not _row_level(desc):
        return []
    wanted = {str(q).split(".")[-1] for q in desc.get("quantity") or [] if q}
    open_dims = open_dimensions(meta, desc, params)
    if not open_dims:      # every row dimension pinned: the rows ARE the totals asked for
        return []
    out = []
    for sec, it in items(meta):
        if it is item or not selectable(meta, it, versions)[0]:
            continue
        d = describe_query(meta, sec, it)
        ops = d.get("operation") if isinstance(d.get("operation"), list) else []
        if not any(o.startswith("sum(") and o[4:-1].split(".")[-1] in wanted for o in ops):
            continue
        groups = {str(g).split(".")[-1] for g in d.get("grouping_columns") or [] if g}
        if groups & open_dims:
            continue
        out.append(it["id"])
    return out


def selection_error(call: dict, err: str, **extra) -> tuple[dict, None, str]:
    rec = {"table": call.get("table"), "query_id": call.get("query_id"), "source": "catalog",
           "selection_error": True, "error": err, **extra}
    return rec, None, err


async def _execute(call: dict, by_table: dict[str, dict], *, max_rows: int, timeout_ms: int,
                   versions: dict[str, str | None] | None = None,
                   all_tables: set[str],
                   guard: Callable[[str, dict], Awaitable[list[dict]]] | None = None,
                   prevalidate: Callable[[str, dict], Awaitable[tuple[dict, list[dict], list[dict]]]]
                   | None = None,
                   ) -> tuple[dict, SqlResult | None, str | None]:
    """Run one selected STORED query: the exact SQL ingestion verified, with
    the call's parameter values bound by the driver. Returns (record, result,
    error). ``error`` is set only when the call itself is wrong (selection
    error, invalid parameter, refused identifier, failed execution); a
    query that runs and returns no rows is a valid, empty result."""
    from backend.app.services.structured.metadata import rebuild_action, selectable
    from backend.app.services.structured.value_domains import validate_call_params

    table, qid = call.get("table"), call.get("query_id")
    if call.get("params") is not None and not isinstance(call.get("params"), dict):
        return selection_error(call, "params must be a JSON object {name: value}; got "
                               f"{type(call.get('params')).__name__}")
    inline = [k for k in INLINE_KEYS if call.get(k) not in (None, "", {}, [])]
    if inline:
        return selection_error(call, INLINE_ERROR.format(keys=", ".join(inline)))
    if not isinstance(qid, str) or not qid.strip():
        return selection_error(call, "call names no query_id")
    cand = by_table.get(table) if isinstance(table, str) else None
    if cand is None:
        return selection_error(call, f"table {table!r} is not one of the candidate tables "
                                     f"({', '.join(sorted(by_table))})")
    meta = cand["meta"]
    hit = find_item(meta, qid)           # within the NAMED table only
    if hit is None:
        return selection_error(call, f"query_id {qid!r} is not a stored query of table {table!r}")
    section, item = hit
    rec = {"table": meta["table_name"], "query_id": item["id"], "label": item.get("label"),
           "source": "catalog", "section": section}
    ok, why = selectable(meta, item, versions)
    if not ok:
        err = f"stored query {item['id']!r} is unavailable: {why}"
        return {**rec, "selection_error": True, "unavailable": True,
                "action": rebuild_action(meta["table_name"], why), "error": err}, None, err
    shape_in = call.get("reshape")
    from backend.app.services.structured.metadata import describe_query

    # only a ROW-LEVEL query returns the parts of a total; a stored aggregate is
    # one, and so is a row-level query whose every row dimension the call pins
    _desc = describe_query(meta, section, item)
    if str(call.get("scope") or "").lower() == "total" and not (
            isinstance(shape_in, dict) and shape_in.get("aggregate")) \
            and _row_level(_desc) and open_dimensions(meta, _desc, call.get("params") or {}):
        alts = total_alternatives(meta, section, item, versions, call.get("params") or {})
        if alts:
            grain = ", ".join(map(str, describe_query(meta, section, item).get("output_grain") or [])) \
                or "source row"
            err = TOTAL_SCOPE_ERROR.format(qid=item["id"], grain=grain, alts=", ".join(alts))
            return {**rec, "params": call.get("params"), "selection_error": True,
                    "total_alternatives": alts, "error": err}, None, err
        rec["total_without_aggregate"] = True
    from backend.app.services.structured.reshape import ReshapeError, listify, output_columns, reshape_sql

    sql, declared = item["sql"], item["params"]
    raw = dict(call.get("params")) if isinstance(call.get("params"), dict) else {}
    try:   # a list of values for an equality filter: col = ANY(...)
        sql, declared, listed = listify(sql, raw, declared)
    except ReshapeError as exc:
        return {**rec, "params": raw, "error": str(exc)}, None, str(exc)
    if listed:
        rec["listed_params"] = listed
    params, bad, subs = validate_call_params(raw, declared, item.get("allowed_combinations"))
    if subs:
        rec["substituted"] = subs
    if bad:
        err = "parameter value(s) refused before the query ran (see invalid_values)"
        return {**rec, "params": params, "invalid_values": bad, "error": err}, None, err
    unsupported = await guard(sql, params) if guard else []
    if unsupported:
        return {**rec, "params": params, "unsupported_identifiers": unsupported,
                "error": UNSUPPORTED_ID_ERROR}, None, UNSUPPORTED_ID_ERROR
    if prevalidate:
        params, missing, vsubs = await prevalidate(sql, params)
        if vsubs:
            rec["substituted"] = [*(rec.get("substituted") or []), *vsubs]
        if missing:
            rec["missing_values"] = missing
            if not all(m.get("dropped_from") for m in missing):
                return {**rec, "params": params, "error": MISSING_VALUE_ERROR}, None, MISSING_VALUE_ERROR
    tables = set(item.get("tables") or []) or {meta["table_name"]} | (
        {item["other_table"]} if item.get("other_table") else set())
    shape = call.get("reshape")
    if shape not in (None, {}, []):
        # aggregate / group / order / limit over the stored query's own rows
        types_m = {c["name"]: str(c.get("type") or "text") for c in meta.get("columns") or []}
        roles_m = {c["name"]: str(c.get("role") or "") for c in meta.get("columns") or []}
        try:
            kinds = output_columns(sql, types_m, roles_m)
            sql, how = reshape_sql(sql, shape, kinds, max_rows=max_rows)
        except Exception as exc:   # ReshapeError, or a stored query that does not parse
            err = f"reshape refused: {exc}"
            return {**rec, "params": params, "reshape": shape, "error": err}, None, err
        rec["reshape"] = shape
        rec["reshape_note"] = how
    res = await run_sql(sql, params, tables & (all_tables | {meta["table_name"]}),
                        max_rows=max_rows, timeout_ms=timeout_ms, declared_params=declared)
    if item.get("empty_excluded"):
        rec["empty_excluded"] = list(item["empty_excluded"])
    rec.update({"sql": res.sql, "params": res.params, "row_count": res.row_count,
                "truncated": res.truncated, "total_rows": res.total_rows, "seconds": res.seconds,
                "error": res.error})
    if res.error:
        return rec, res, res.error
    return rec, res, None


def requested_values(rec: dict, item: dict | None) -> dict[str, Any]:
    """{column: value} of a failed call's parameters that were NOT shown to be
    invalid: what the user asked for (entity, period, qualifier). A retry must
    keep them."""
    if not item:
        return {}
    bad = set()
    for b in rec.get("invalid_values") or []:
        if b.get("param"):
            bad.add(str(b["param"]))
        for k in (b.get("combination") or {}):
            bad.add(str(k))
    wrong = {_norm_value(m.get("value")) for k in ("missing_values", "unsupported_identifiers")
             for m in rec.get(k) or [] if m.get("value") is not None}
    out: dict[str, Any] = {}
    params = rec.get("params") or {}
    for p in item.get("params") or []:
        n, col = p.get("name"), p.get("column")
        if not col or n in bad or n not in params or params[n] is None:
            continue
        if isinstance(params[n], list):
            # list members: the valid ones are constraints, the wrong ones may change
            good = [v for v in params[n] if _norm_value(v) not in wrong]
            if good:
                out[f"{col}|{n.endswith('_from')}|{n.endswith('_to')}"] = {"members": good}
            continue
        if _norm_value(params[n]) in wrong:
            continue
        out[f"{col}|{n.endswith('_from')}|{n.endswith('_to')}"] = params[n]
    return out


def _norm_any(x: Any) -> Any:
    return sorted(_norm_value(v) for v in x) if isinstance(x, list) else _norm_value(x)


def _grouped_columns(item: dict) -> set[str]:
    """Columns an aggregate item returns one row per value of (none for a
    row-level item)."""
    spec = item.get("query") if isinstance(item.get("query"), dict) else {}
    sel = spec.get("select") or []
    if not any(isinstance(x, dict) and x.get("agg") for x in sel):
        return set()
    grp = spec.get("group_by") or [x for x in sel if not (isinstance(x, dict) and x.get("agg"))]
    return {str(g.get("column") if isinstance(g, dict) else g).split(".")[-1] for g in grp if g}


def lookup_calls(batch: list[dict]) -> set[int]:
    """Numbers (1-based) of the calls in ``batch`` whose results a LATER call
    of the same batch takes a parameter from (``{"from_call": k, ...}``)."""
    out: set[int] = set()
    for n, call in enumerate(batch, start=1):
        params = call.get("params") if isinstance(call.get("params"), dict) else {}
        for v in params.values():
            if isinstance(v, dict) and "from_call" in v:
                ks = v["from_call"] if isinstance(v["from_call"], list) else [v["from_call"]]
                out |= {k for k in ks if isinstance(k, int) and 1 <= k < n}
    return out


def lookup_coverage(call: dict, concrete: list[dict], item: dict | None,
                    keep: dict[str, Any]) -> str | None:
    """Why the values a call TAKES from earlier calls ({"from_call": k}) fail to
    cover what the question asked for on that column (None when they cover it).
    The values are pooled over every run the call fans out to, so a requested
    value missing from the lookup -- or cut by the fan-out cap -- is caught."""
    if not keep or not item:
        return None
    params = call.get("params") if isinstance(call.get("params"), dict) else {}
    for p in item.get("params") or []:
        n, col = p.get("name"), p.get("column")
        ref = params.get(n)
        if not col or not (isinstance(ref, dict) and "from_call" in ref):
            continue
        key = f"{col}|{str(n).endswith('_from')}|{str(n).endswith('_to')}"
        if key not in keep:
            continue
        want = keep[key]
        members = want["members"] if isinstance(want, dict) and "members" in want else [want]
        have: set[str] = set()
        for sub in concrete:
            v = (sub.get("params") or {}).get(n)
            have |= {_norm_value(x) for x in (v if isinstance(v, list) else [v]) if x is not None}
        lost = [v for v in members if _norm_value(v) not in have]
        if lost:
            cut = next((c.get("_fanout_cut") for c in concrete if c.get("_fanout_cut")), None)
            return (f"takes {col} values from an earlier call that do not include {lost!r}, which the "
                    f"question asked for{f' ({cut} were run)' if cut else ''}; look up every requested "
                    "value, or filter on the requested values directly")
    return None


def broadens(call: dict, item: dict | None, keep: dict[str, Any], *,
             grouping_keeps: bool = False) -> str | None:
    """Why a retry call would change or drop a value the user asked for
    (None when it keeps them all). Only parameters on the same column are
    compared: a different stored query may filter differently.
    ``grouping_keeps``: the retry uses an aggregate the program itself named
    for a refused total (same quantity, summed); one that GROUPS BY a
    requested column then keeps the requested values -- each as its own
    result row -- so it does not broaden that column."""
    if not keep:
        return None
    if not item:
        return None                      # an unknown id fails on its own
    params = call.get("params") if isinstance(call.get("params"), dict) else {}
    covered = {p.get("column") for p in item.get("params") or [] if p.get("column")}
    grouped = _grouped_columns(item) if grouping_keeps else set()
    for key, want in keep.items():
        col = key.split("|", 1)[0]
        if col not in covered and col not in grouped:
            return (f"uses a stored query that cannot filter {col} = {want!r}, which the question asked "
                    "for (never broaden the question to get rows)")
    for p in item.get("params") or []:
        n, col = p.get("name"), p.get("column")
        key = f"{col}|{str(n).endswith('_from')}|{str(n).endswith('_to')}"
        if not col or key not in keep:
            continue
        want = keep[key]
        got = params.get(n)
        if isinstance(got, dict) and "from_call" in got:
            continue
        if got is None:
            if p.get("optional"):
                return (f"leaves out {col} = {want!r}, which the question asked for (an empty result "
                        "is an answer; never broaden the question to get rows)")
            continue
        if isinstance(want, dict) and "members" in want:
            have = {_norm_value(v) for v in (got if isinstance(got, list) else [got])}
            lost = [v for v in want["members"] if _norm_value(v) not in have]
            if lost:
                return (f"drops {lost!r} from the {col} list; those values were asked for and not shown "
                        "to be invalid")
            continue
        if _norm_any(got) != _norm_any(want):
            return (f"changes {col} from {want!r} to {got!r}, a value the question asked for that was "
                    "not shown to be invalid")
    return None


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
                    id_columns: dict[str, set[str]] | None = None, *,
                    bound_only: bool = False) -> list[dict]:
    """``text_column = 'value'`` / ``IN (...)`` comparisons, from literals and
    bound :params: [{table, column, value, context}], where ``context`` holds
    the statement's OTHER single-value equality filters on the same table
    (``cik = '0000811809'``) -- a tag can exist in the table and still not
    exist for that company. ``columns``: table -> {column: type}; only text
    columns of known tables count. ``bound_only``: report only values bound
    through :params -- literals of a stored query are part of its tested
    definition (``unit IN ('a', 'b')`` is set membership: no single listed
    value has to exist for the bound filters), so they are not checked; they
    still scope the check as context."""
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

    def _vals(node) -> list[tuple[str, bool]]:
        # (value, bound). Compiled specs bind as CAST(:p AS text) and lists as
        # = ANY(CAST(:p AS text[])).
        while isinstance(node, (exp.Cast, exp.Any, exp.Paren)):
            node = node.this
        if isinstance(node, exp.Literal) and node.is_string:
            return [(node.this, False)]
        if isinstance(node, exp.Placeholder) and node.name:
            v = (params or {}).get(node.name)
            if isinstance(v, str):
                return [(v, True)]
            if isinstance(v, list):
                return [(x, True) for x in v if isinstance(x, str)]
        return []

    # table, column, [(value, bound)], single-EQ
    found: list[tuple[str, str, list[tuple[str, bool]], bool]] = []
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
        ctx = [(c2, v2[0][0]) for t2, c2, v2, single in found
               if t2 == t and c2 != c and single and c2 in ids]
        for v, bound in vals:
            if bound_only and not bound:
                continue
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


def pattern_values(sql: str, params: dict, columns: dict[str, dict[str, str]]) -> list[dict]:
    """Bound values used as a LIKE / ILIKE pattern on a text column (``name
    ILIKE '%' || :name || '%'``): [{table, column, param, value}]. Values with
    their own wildcards are left alone (the model wrote a pattern)."""
    from sqlglot import exp

    from backend.app.services.structured.sql_guard import SqlGuardError, parse_one_select

    try:
        root = parse_one_select(sql)
    except SqlGuardError:
        return []
    alias = {t.alias_or_name.lower(): t.name.lower() for t in root.find_all(exp.Table)
             if t.name and t.name.lower() in columns}
    tables = sorted(set(alias.values()))
    out: list[dict] = []
    for node in root.find_all(exp.Like, exp.ILike):
        col = node.this
        while isinstance(col, (exp.Cast, exp.Paren, exp.Lower, exp.Upper)):
            col = col.this
        if not isinstance(col, exp.Column):
            continue
        qual = (col.table or "").lower()
        owners = [alias[qual]] if qual in alias else [t for t in tables if col.name.lower() in columns[t]]
        if len(owners) != 1 or (columns[owners[0]].get(col.name.lower()) or "text") != "text":
            continue
        for ph in node.expression.find_all(exp.Placeholder):
            v = (params or {}).get(ph.name)
            if isinstance(v, str) and v.strip() and not re.search(r"[%_]", v):
                item = {"table": owners[0], "column": col.name.lower(), "param": ph.name, "value": v}
                if item not in out:
                    out.append(item)
    return out[:MAX_VALUE_CHECKS]


async def fold_pattern_values(checks: list[dict]) -> list[dict]:
    """For each pattern value that matches no row as written, find the
    column's own spelling of it: the same text up to case, accents, curly vs
    straight quotes, dash kinds and space kinds (a document's curly apostrophe
    in a name the table spells with a straight one). Returns [{param, from, to}] for the
    values to replace. Read-only, no LLM; a value with no such match is left
    alone (zero rows is a valid answer)."""
    if not checks:
        return []
    from backend.app.db.engine import get_sessionmaker
    from backend.app.services.structured.db_load import q

    subs = []
    async with get_sessionmaker()() as s:
        try:
            await s.execute(sql_text("SET TRANSACTION READ ONLY"))
            await s.execute(sql_text("SET LOCAL statement_timeout = 3000"))
            for chk in checks:
                ref, col, val = f"graphrag.{q(chk['table'])}", q(chk["column"]), chk["value"]
                hit = (await s.execute(sql_text(
                    f"SELECT 1 FROM {ref} WHERE {col} ILIKE '%' || :v || '%' LIMIT 1"), {"v": val})).first()
                if hit:
                    continue
                words = sorted(set(re.findall(r"\w{3,}", val)), key=len, reverse=True)[:4]
                if not words:
                    continue
                like = " OR ".join(f"{col} ILIKE :w{i}" for i in range(len(words)))
                binds = {f"w{i}": f"%{w}%" for i, w in enumerate(words)}
                # candidates share a whole word with the value; the folded
                # comparison below then ignores case, accents and typography
                cands = (await s.execute(sql_text(
                    f"SELECT DISTINCT {col} FROM {ref} WHERE {like} LIMIT 200"), binds)).scalars().all()
                want = fold_text(val)
                for cand in cands:
                    pos = fold_text(str(cand)).find(want)
                    if pos >= 0:
                        new = str(cand)[pos:pos + len(val)]
                        if new != val:
                            subs.append({"param": chk["param"], "from": val, "to": new,
                                         "reason": "same text in the table's spelling (quotes, dashes, accents)"})
                        break
        except Exception:  # best effort: never blocks the call
            return []
        finally:
            await s.rollback()
    return subs


async def prevalidate_values(sql: str, params: dict, *, col_types: dict[str, dict[str, str]],
                             id_cols: dict[str, set[str]] | None = None,
                             ) -> tuple[dict, list[dict], list[dict]]:
    """Check every text value a statement compares against the column it is
    compared with, BEFORE the statement runs (an indexed lookup each, ~1 ms).
    Returns (params, missing, substituted): a value that differs from an
    existing one only in case/spacing/punctuation is replaced (and recorded);
    any other absent value is returned in ``missing`` with the closest values
    that do exist, so the call is refused instead of running empty."""
    pattern_subs = await fold_pattern_values(pattern_values(sql, params, col_types))
    if pattern_subs:
        params = {**params, **{p["param"]: p["to"] for p in pattern_subs}}
    checks = compared_values(sql, params, col_types, id_cols, bound_only=True)
    if not checks:
        return params, [], pattern_subs
    missing = [m for m in await missing_values(checks) if "value" in m]
    if not missing:
        return params, [], pattern_subs
    fixed, subs, left = dict(params), list(pattern_subs), []
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


def _listing_text(head: str, res: SqlResult, char_cap: int, summary: str = "", *,
                  cols: list[str] | None = None, rows: list[list] | None = None) -> str:
    """Compact layout for a LISTING result: the header once, then one line of
    values per row (about half the size of name=value pairs). ``summary`` (the
    exact change statistics) goes first, so the row cap never cuts it."""
    if summary:
        head += summary + " "
    def cell(v) -> str:
        return "" if v is None else str(v).replace("|", "/")
    cols = cols if cols is not None else list(res.columns)
    rows = rows if rows is not None else res.rows
    lines = [" | ".join(cols)]
    used = len(head) + len(lines[0])
    shown = 0
    for r in rows:
        row = " | ".join(cell(v) for v in r)
        if used + len(row) + 3 > char_cap - 400:
            break
        lines.append(row)
        used += len(row) + 3
        shown += 1
    note = f"{res.row_count} row(s)"
    return (head + f"LISTING, {note}. Rows: " + " ;; ".join(lines)
            + partial_note(shown, len(rows), res.truncated, res.total_rows))


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
            if ("value" not in m or not m.get("column") or "allowed_values" not in m
                    or _norm_value(m["value"]) in used):
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


def no_match_evidence(no_match: list[dict], start_rank: int) -> list[dict]:
    """One evidence item saying which parts of the question NO stored query
    answers, so the answer says the tables do not cover them instead of
    guessing. Empty when every part was matched."""
    parts = [m for m in no_match or [] if m.get("part") or m.get("reason")]
    if not parts:
        return []
    text = ("Table coverage: no stored, tested query of the candidate tables answers "
            + "; ".join(f"'{m['part']}'" + (f" ({m['reason']})" if m.get("reason") else "") for m in parts)
            + ". The tables were not queried for these parts; do not state table values for them.")
    digest = hashlib.sha256(text.encode()).hexdigest()[:8]
    return [{"kind": "sql_result", "node_id": None,
             "iri": f"{VIAO_NS}#SqlResult_no_match_{digest}", "rank": start_rank, "score": 1.0,
             "text": text, "char_cap": 2000, "listing": False, "document_title": "table coverage",
             "table": None, "query_id": "no_match", "sql": None, "params": {}, "columns": [],
             "rows": [], "row_count": 0, "truncated": False, "total_rows": None,
             "source_file": None, "sql_source": "no_match"}]


def relevance_hints(cands: list[dict], rankings: dict[str, dict], top: int = 8,
                    eligible: dict[str, set[str]] | None = None) -> list[dict]:
    """Per candidate table (most relevant first): its graph score and the
    stored items whose tags the question matched, with why -- only items the
    selector is offered (``eligible``: table -> ids in its catalog)."""
    out = []
    for c in sorted(cands, key=lambda c: -(c.get("score") or 0)):
        r = rankings.get(c["table"]) or {}
        ok = (eligible or {}).get(c["table"])
        matched = [{"id": i, "matched": w} for i, (sc, w) in sorted(r.items(), key=lambda kv: -kv[1][0])
                   if sc > 0 and (ok is None or i in ok)][:top]
        out.append({"table": c["table"], "reached_via": c.get("via") or [], "matched_items": matched})
    return out


def _flat(params: dict) -> list:
    out = []
    for v in params.values():
        out.extend(v if isinstance(v, list) else [v])
    return [v for v in out if isinstance(v, str)]


def _evidence_text(rec: dict, res: SqlResult, meta: dict, char_cap: int, stats: dict | None = None,
                   units: dict | None = None) -> str:
    head = (f"SQL result from table {rec['table']}"
            + (f" ({rec['label']})" if rec.get("label") else "")
            + (f" [{(meta.get('source') or {}).get('rel_path')}]" if meta else "") + ". ")
    if rec.get("params"):
        head += "Parameters: " + ", ".join(f"{k}={v}" for k, v in rec["params"].items()) + ". "
    if rec.get("reshape_note"):
        head += f"Reshaped from the stored query's rows: {rec['reshape_note']}. "
    from backend.app.services.structured.units import units_line

    head += units_line(units)
    cols, rows = _with_row_units(res, units)
    if rec.get("listing"):
        from backend.app.services.structured.listing_stats import stats_text

        if rec.get("empty_excluded"):
            head += EMPTY_EXCLUDED_NOTE.format(columns=", ".join(rec["empty_excluded"])) + " "
        if rec.get("total_without_aggregate"):
            head += TOTAL_PARTS_NOTE + " "
        return _listing_text(head, res, char_cap, stats_text(stats) if stats else "", cols=cols, rows=rows)
    body = []
    used = len(head)
    for r in rows:
        row = "; ".join(f"{c}={'' if v is None else v}" for c, v in zip(cols, r, strict=False))
        if used + len(row) + 4 > char_cap - 400:
            break
        body.append("(" + row + ")")
        used += len(row) + 4
    tail = partial_note(len(body), len(rows), res.truncated, res.total_rows)
    if rec.get("total_without_aggregate"):
        tail += " " + TOTAL_PARTS_NOTE
    if rec.get("unranked_rows"):
        tail += " " + UNRANKED_NOTE.format(**rec["unranked_rows"])
    if rec.get("empty_excluded"):
        tail += " " + EMPTY_EXCLUDED_NOTE.format(columns=", ".join(rec["empty_excluded"]))
    return head + f"{res.row_count} row(s): " + " ".join(body) + tail


def _with_row_units(res: SqlResult, units: dict | None) -> tuple[list[str], list[list]]:
    """The result's columns and rows, plus a '<column>_unit' column for each
    output whose unit the program looked up row by row."""
    per = (units or {}).get("row_units") or {}
    cols, rows = list(res.columns), [list(r) for r in res.rows]
    for name, vals in per.items():
        if len(vals) != len(rows):
            continue
        cols.append(f"{name}_unit")
        for r, u in zip(rows, vals, strict=True):
            r.append(u)
    return cols, rows


async def _result_units(rec: dict, res: SqlResult, by_table: dict[str, dict], *,
                        timeout_ms: int) -> dict | None:
    """Units of each output column of a result (see :mod:`structured.units`).
    Never fails the retrieval: an unexpected statement gives no units."""
    from backend.app.services.structured.units import result_units

    metas = {t: c["meta"] for t, c in by_table.items() if c.get("meta")}
    meta = metas.get(rec.get("table")) or {}
    hit = find_item(meta, rec["query_id"]) if rec.get("query_id") else None
    measure = (hit[1].get("measure") if hit and isinstance(hit[1].get("measure"), dict) else None)

    async def run(sql: str, params: dict, tables: set[str]):
        r = await run_sql(sql, params, tables, max_rows=600, timeout_ms=timeout_ms)
        return None if r.error else r.rows

    try:
        return await result_units(res.sql, res.params, res.columns, res.rows, metas,
                                  measure=measure, run=run)
    except Exception:  # units are an aid; the result stands without them
        return None


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
    entity_scores: dict[uuid.UUID, float] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Evidence items (``kind='sql_result'``) + a plan record for persistence."""
    from backend.app.services.prompts import PROMPTS

    t0 = time.monotonic()
    plan: dict[str, Any] = {"candidates": [], "needed": False, "calls": [], "retried": False}
    cands = await candidate_tables(entity_ids, class_ids, qvec, max_tables=max_tables,
                                   vector_max_distance=vector_max_distance, time_ids=time_ids,
                                   entity_scores=entity_scores, question=question)
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
    inv = await table_inventory()
    versions = {n: ((r["meta"].get("source") or {}).get("file_sha")) for n, r in inv.items()}
    catalogs = [catalog_for_prompt(c["meta"], ranking=rankings.get(c["table"]), stable=True,
                                   only_selectable=True, versions=versions)
                for c in sorted(cands, key=lambda c: c["table"])]
    eligible = {cat["table"]: {q["id"] for q in cat["queries"]} for cat in catalogs}
    # Stored queries retrieval may not run (catalog not verified at the current
    # version, failed or unexercised data test): reported with the rebuild action.
    from backend.app.services.structured.metadata import rebuild_action, selectable_items

    unavailable = [u for c in cands for u in selectable_items(c["meta"], versions)[1]]
    if unavailable:
        plan["unavailable"] = unavailable[:50]
        plan["unavailable_action"] = sorted({rebuild_action(u["table"], u.get("reason")) for u in unavailable})
        import logging

        logging.getLogger(__name__).warning(
            "structured data: %d stored quer(ies) unavailable in %s -- rebuild: %s", len(unavailable),
            sorted({u["table"] for u in unavailable}), "; ".join(plan["unavailable_action"]))
    relevance = relevance_hints(cands, rankings, eligible=eligible)
    inventory = inventory_for_prompt(inv)
    col_types = {c["table"]: {col["name"].lower(): (col.get("type") or "text")
                              for col in c["meta"].get("columns") or []} for c in cands}
    id_cols = {c["table"]: {col["name"].lower() for col in c["meta"].get("columns") or []
                            if col.get("role") == "identifier"} for c in cands}

    async def _value_failures(rec: dict, res: SqlResult | None) -> list[dict]:
        if res is None or res.error:
            return []
        checks = compared_values(res.sql, res.params, col_types, id_cols, bound_only=True)
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
    no_match = [{"part": str(m.get("part") or m.get("question") or ""), "reason": str(m.get("reason") or "")}
                for m in (choice.get("no_match") or []) if isinstance(m, dict)]
    if no_match:
        plan["no_match"] = no_match
    plan["needed"] = bool(choice.get("needed")) or bool(describe)
    plan["reason"] = choice.get("reason")
    if not plan["needed"]:
        return no_match_evidence(no_match, 1), plan

    calls = cap_listings([c for c in (choice.get("calls") or []) if isinstance(c, dict)][:max_calls], plan)
    today_year = date.today().year
    for call in calls:   # 'past 3 years' ends at the table's latest period, not the model's guess
        meta_c = (by_table.get(call.get("table")) or {}).get("meta")
        adj = anchor_relative_periods(question, call, meta_c, today_year=today_year) if meta_c else None
        if adj:
            plan.setdefault("period_adjustments", []).append({"table": call.get("table"), **adj})
    results: list[tuple[dict, SqlResult]] = []
    failures: list[dict] = []
    metas_all = {**{c["table"]: c["meta"] for c in cands}, **partners}
    seen_values: set[str] = set()       # every value an earlier call returned

    doc_norm = _norm_value(" ".join(str(d.get("text") or "") for d in doc_context or []))

    async def _guard(sql: str, params: dict) -> list[dict]:
        return await identifier_guard(sql, params, question=question, seen_values=seen_values,
                                      types=types, metas=metas_all, context_norm=doc_norm)

    async def _prevalidate(sql: str, params: dict) -> tuple[dict, list[dict], list[dict]]:
        return await prevalidate_values(sql, params, col_types=col_types, id_cols=id_cols)

    def _rows_for(call: dict) -> int:
        return listing_max_rows if call.get("listing") else max_rows


    def _item_of(call: dict) -> dict | None:
        meta_c = (by_table.get(call.get("table")) or {}).get("meta") if isinstance(call.get("table"), str) else None
        hit = find_item(meta_c, call["query_id"]) if meta_c and isinstance(call.get("query_id"), str) else None
        return hit[1] if hit else None

    def _origin(call: dict) -> tuple[int | None, str | None]:
        """The failed call a retry replaces: its ``retry_of`` (1-based index in
        the failures list), else the only failure, else the one failure with
        the same table and query_id. Its constraints alone are checked."""
        ro = call.get("retry_of")
        if isinstance(ro, int) and 1 <= ro <= len(failures):
            return ro, None
        if len(failures) == 1:
            return 1, None
        same = [i for i, f in enumerate(failures, start=1)
                if (f.get("call") or {}).get("table") == call.get("table")
                and (f.get("call") or {}).get("query_id") == call.get("query_id")]
        if len(same) == 1:
            return same[0], None
        return None, ("names no failed call it replaces (set retry_of to the number of the failed call "
                      "in the failures list)")

    async def _run_calls(batch: list[dict], *, retry: bool) -> None:
        done: dict[int, list[SqlResult]] = {}       # call number -> its results (chaining)
        feeders = lookup_calls(batch)
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
                why = None
                rec_origin = None
                if retry and n in feeders:
                    # a LOOKUP: its rows only supply a value (an id) to a later
                    # call, which is checked against the question's values
                    # itself. It need not filter the question's other values
                    # nor replace a failed call -- refusing it blocked the
                    # very id lookup the identifier guard asks for.
                    rec_origin, _why = _origin(call)
                elif retry:
                    origin, why = _origin(call)
                    if origin is not None:
                        rec_origin = origin
                        failed = failures[origin - 1]
                        # judge parameters taken from an earlier call as written
                        # ({"from_call": k}): after fan-out each sub-call holds ONE
                        # of the looked-up values, never the whole requested list
                        as_written = {**sub_call, "params": {
                            **(sub_call.get("params") or {}),
                            **{k: v for k, v in (call.get("params") or {}).items()
                               if isinstance(v, dict) and "from_call" in v}}}
                        why = broadens(as_written, _item_of(sub_call), failed.get("keep") or {},
                                       grouping_keeps=sub_call.get("query_id") in (
                                           failed.get("total_alternatives") or [])) \
                            or lookup_coverage(call, concrete, _item_of(sub_call), failed.get("keep") or {})
                if why:
                    rec, res, err = selection_error(sub_call, f"retry refused: it {why}", broadens=True)
                else:
                    rec, res, err = await _execute(sub_call, by_table, max_rows=_rows_for(sub),
                                                   timeout_ms=timeout_ms, all_tables=all_tables,
                                                   guard=_guard, prevalidate=_prevalidate, versions=versions)
                if retry:
                    rec["retry"] = True
                    if why is None and rec_origin is not None:
                        rec["retry_of"] = rec_origin
                    if n in feeders:
                        rec["lookup"] = True
                if sub.get("_from_calls"):
                    rec["from_calls"] = sub["_from_calls"]
                    if sub.get("_fanout_cut"):
                        rec["fanout_cut"] = sub["_fanout_cut"]
                rec["listing"] = bool(sub.get("listing"))
                if sub.get("scope"):
                    rec["scope"] = str(sub["scope"])
                missing = await _value_failures(rec, res) or rec.get("missing_values") or []
                if missing:
                    rec["missing_values"] = missing
                unranked = None if err else await _unranked_rows(
                    rec, res, (by_table.get(rec.get("table")) or {}).get("meta"))
                if unranked:
                    rec["unranked_rows"] = unranked
                plan["calls"].append(rec)
                # Retry only a call shown to be wrong (selection error, invalid or
                # absent parameter value). A valid query returning no rows is an
                # answer, and a note about the data is evidence, not an error.
                if (err or missing) and not retry:
                    failures.append({"call": {k: v for k, v in call.items() if k in
                                              ("table", "query_id", "params", "listing", "reshape")},
                                     "error": err or "value(s) not found in their column (see missing_values)",
                                     **{k: rec[k] for k in ("invalid_values", "unsupported_identifiers",
                                                            "value_combinations", "missing_values", "action",
                                                            "total_alternatives")
                                        if rec.get(k)},
                                     **({"missing_values": missing} if missing else {}),
                                     "keep": requested_values(rec, _item_of(sub_call))})
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
        for m in retry.get("no_match") or []:
            if isinstance(m, dict):
                no_match.append({"part": str(m.get("part") or ""), "reason": str(m.get("reason") or "")})
                plan["no_match"] = no_match
        retry_calls = cap_listings([c for c in (retry.get("calls") or []) if isinstance(c, dict)][:max_calls],
                                   plan)
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
        units = await _result_units(rec, res, by_table, timeout_ms=timeout_ms)
        evidence.append({
            "kind": "sql_result",
            "node_id": node,
            "iri": f"{VIAO_NS}#{re.sub(r'[^A-Za-z0-9_]', '_', frag)}",
            "rank": i, "score": 1.0,
            "text": _evidence_text(rec, res, meta, listing_char_cap if rec.get("listing") else char_cap, stats,
                                   units),
            "units": units,
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
    evidence += no_match_evidence(no_match, len(evidence) + 1)
    evidence += describe_evidence(describe, inv, len(evidence) + 1)
    plan["seconds"] = round(time.monotonic() - t0, 3)
    if verbose:
        print(f"[query] structured data: {len(cands)} candidate table(s) "
              f"{[c['table'] for c in cands]}, needed={plan['needed']}, "
              f"{len(evidence)} result(s), retried={plan['retried']}")
    return evidence, plan
