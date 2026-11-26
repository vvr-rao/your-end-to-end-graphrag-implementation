"""Tie a loaded table to the graph's TIME and PLACE nodes, so a question
about a period or a place reaches the table through the graph walk.

**Time.** The periods a table's values cover (its period columns -- never a
column that only dates the report) are read with SQL, turned into the same
canonical identifiers enrich-time uses (``YEAR_2024``, ``Q3_2024``,
``MONTH_2024_07``), minted in ``time_instances`` when missing (with their
parents and ``intervalDuring`` edges, exactly as enrich-time does), and linked
``table -[hasTime]-> time_instance`` (relationship_type ``tableTime``).
Retrieval's temporal expansion (down at full weight, up at half) then lets a
question about "October 2024" reach a table linked to 2024.

**Places.** Columns that hold places (from their entity kind, class or name)
are read; ISO 3166 country and subdivision codes become names (``AU`` ->
Australia, ``CN-ZJ`` -> Zhejiang); values are matched to EXISTING place
entities only. Structured data never mints a place entity -- on the first
load or on relink: a place the graph does not know stays unresolved (it is
reported, and the table still holds it). Containment is written where the data
states it between places that exist: a subdivision code names its country,
and a row that holds a city, a region and a country (columns sharing a
prefix) says the city is in that region and country. Edges
``table -[assertsAbout]-> place`` (``tablePlace``). Time instances follow
their own policy above (minted like enrich-time does).
"""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable
from datetime import date
from itertools import pairwise
from typing import Any

from sqlalchemy import text as sql_text

SCHEMA = "graphrag"
TIME_LEVELS = ("year", "quarter", "month")
MAX_PERIOD_VALUES = 20000
MAX_PLACE_VALUES = 500          # per column
MAX_PLACE_COMBOS = 5000
VALUE_PLACE_MIN_SHARE = 0.5     # columns found by their values: share of distinct values that are places
VALUE_PLACE_MIN_HITS = 3
PLACE_WORDS = {
    # word in a column's entity kind / class / name -> level (1 city .. 3 country)
    "city": 1, "town": 1, "municipality": 1, "locality": 1,
    "region": 2, "state": 2, "province": 2, "subdivision": 2, "county": 2,
    "administrativearea": 2,
    "country": 3, "nation": 3, "jurisdiction": 3,
}
CLASS_FOR_LEVEL = {
    3: ("Country",),
    2: ("AdministrativeArea", "State", "Province", "Region"),
    1: ("City", "Municipality", "Town"),
}
EmbedFn = Callable[[list[str]], Awaitable[list[list[float]]]]


def _q(ident: str) -> str:
    return '"' + ident.replace('"', '""') + '"'


# ------------------------------------------------------------------ time


_YEAR = r"(1[6-9]\d\d|20\d\d|21\d\d)"
_PATTERNS = [
    (re.compile(rf"^{_YEAR}$"), "year"),
    (re.compile(rf"^{_YEAR}\s*[-/ ]?\s*[Qq]([1-4])$"), "quarter"),
    (re.compile(rf"^[Qq]([1-4])\s*[-/ ]?\s*{_YEAR}$"), "quarter_rev"),
    (re.compile(rf"^{_YEAR}\s*[-/ ]?\s*M?(0?[1-9]|1[0-2])$"), "month"),
    (re.compile(rf"^{_YEAR}-(0[1-9]|1[0-2])-(\d\d)(?:[T ].*)?$"), "day"),
    (re.compile(rf"^{_YEAR}(0[1-9]|1[0-2])(\d\d)$"), "day"),
]


def period_of(value: Any) -> tuple[int, int | None, int | None] | None:
    """(year, quarter, month) of a period value, or None. Accepts dates,
    years (int or text) and common period spellings: 2024, 2024Q3, Q3 2024,
    2024M07, 2024-07, 2024-07-15."""
    if value is None:
        return None
    if isinstance(value, date):
        return value.year, (value.month - 1) // 3 + 1, value.month
    if isinstance(value, int | float) and not isinstance(value, bool):
        y = int(value)
        return (y, None, None) if 1600 <= y <= 2199 and y == value else None
    s = str(value).strip()
    for rx, kind in _PATTERNS:
        m = rx.match(s)
        if not m:
            continue
        if kind == "year":
            return int(m.group(1)), None, None
        if kind == "quarter":
            return int(m.group(1)), int(m.group(2)), None
        if kind == "quarter_rev":
            return int(m.group(2)), int(m.group(1)), None
        y, mo = int(m.group(1)), int(m.group(2))
        return y, (mo - 1) // 3 + 1, mo
    return None


def identifiers_for(periods: set[tuple[int, int | None, int | None]], level: str) -> set[str]:
    """Canonical time identifiers at ``level`` (or coarser when a value is
    coarser: a yearly value never becomes twelve months)."""
    out = set()
    for y, q, mo in periods:
        if level == "month" and mo:
            out.add(f"MONTH_{y}_{mo:02d}")
        elif level in {"month", "quarter"} and q:
            out.add(f"Q{q}_{y}")
        else:
            out.add(f"YEAR_{y}")
    return out


async def ensure_time_instances(session, identifiers: set[str], gv: int) -> dict[str, Any]:
    """Mint missing time instances (and their parents up to the year), set
    parent_time_id and the ``intervalDuring`` edges. Returns identifier -> id."""
    from sqlalchemy import select
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    from backend.app.db.models.entities import TimeInstance
    from backend.app.services.db_temporal_enrich import _parent_of, _record_for
    from backend.app.services.predicates import TIME_INTERVAL_DURING

    records: dict[str, dict] = {}
    for ident in identifiers:
        rec = _record_for(ident)
        while rec is not None and rec["time_identifier"] not in records:
            records[rec["time_identifier"]] = rec
            rec = _parent_of(rec)
    if not records:
        return {}
    have = dict((await session.execute(
        select(TimeInstance.time_identifier, TimeInstance.id)
        .where(TimeInstance.time_identifier.in_(list(records))))).all())
    new = [r for k, r in records.items() if k not in have]
    for i in range(0, len(new), 200):
        await session.execute(pg_insert(TimeInstance).values(new[i:i + 200])
                              .on_conflict_do_nothing(index_elements=["time_identifier"]))
    ids = dict((await session.execute(
        select(TimeInstance.time_identifier, TimeInstance.id)
        .where(TimeInstance.time_identifier.in_(list(records))))).all())
    for r in new:
        parent = _parent_of(r)
        if parent is None or parent["time_identifier"] not in ids:
            continue
        cid, pid = ids[r["time_identifier"]], ids[parent["time_identifier"]]
        await session.execute(sql_text(
            f"UPDATE {SCHEMA}.time_instances SET parent_time_id = :p WHERE id = :c AND parent_time_id IS NULL"),
            {"p": pid, "c": cid})
        await session.execute(sql_text(f"""
            INSERT INTO {SCHEMA}.graph_relationships
                (source_node_type, source_node_id, target_node_type, target_node_id,
                 predicate_iri, predicate_label, relationship_type, relationship_source,
                 is_authoritative, graph_version, extra_metadata)
            SELECT 'time_instance', :c, 'time_instance', :p, :pred, 'time:intervalDuring',
                   'intervalDuring', 'TIME_ENRICHMENT', true, :gv, '{{}}'::jsonb
             WHERE NOT EXISTS (SELECT 1 FROM {SCHEMA}.graph_relationships
                                WHERE source_node_id = :c AND target_node_id = :p AND predicate_iri = :pred)
        """), {"c": cid, "p": pid, "pred": TIME_INTERVAL_DURING, "gv": gv})
    return ids


def period_columns(meta: dict) -> list[str]:
    """Columns holding the period a VALUE covers: the measures' period
    columns, else time columns in the measured grain; never a column a
    measure definition confirmed as the version of a value. Empty before the
    table is analysed."""
    from backend.app.services.structured.analysis import version_columns

    analysis = meta.get("analysis") or {}
    report = set(version_columns(meta, analysis))
    cols = []
    for it in meta.get("measures") or []:
        for c in (it.get("measure") or {}).get("period_columns") or []:
            if c not in cols and c not in report:
                cols.append(c)
    if not cols:
        timeish = {c["name"] for c in meta.get("columns") or [] if c.get("role") == "time"
                   or c.get("type") in {"date", "timestamp"}}
        cols = [c for c in analysis.get("grain") or [] if c in timeish and c not in report]
    return cols


async def table_periods(session, meta: dict) -> set[tuple[int, int | None, int | None]]:
    table = meta["table_name"]
    types = {c["name"]: c.get("type") for c in meta.get("columns") or []}
    out: set = set()
    for col in period_columns(meta):
        if types.get(col) in {"date", "timestamp"}:
            rows = (await session.execute(sql_text(
                f"SELECT DISTINCT CAST(date_trunc('month', {_q(col)}) AS date) FROM {SCHEMA}.{_q(table)} "
                f"WHERE {_q(col)} IS NOT NULL LIMIT {MAX_PERIOD_VALUES}"))).scalars().all()
        else:
            rows = (await session.execute(sql_text(
                f"SELECT DISTINCT {_q(col)} FROM {SCHEMA}.{_q(table)} "
                f"WHERE {_q(col)} IS NOT NULL LIMIT {MAX_PERIOD_VALUES}"))).scalars().all()
        for v in rows:
            p = period_of(v)
            if p:
                out.add(p)
    return out


# ------------------------------------------------------------------ places


def _words(s: str) -> list[str]:
    return [w for w in re.split(r"[^a-z]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", s or "").lower()) if w]


def place_level(column: dict) -> int | None:
    """1 city / 2 region / 3 country, from the column's entity kind, class
    label or name; None for a column that does not hold places."""
    if column.get("type", "text") != "text" or column.get("role") in {"measure", "text"}:
        return None
    for source in (column.get("entity_kind"), column.get("class_label"), column.get("name")):
        joined = "".join(_words(source or ""))
        for w in [*_words(source or ""), joined]:
            if w in PLACE_WORDS:
                return PLACE_WORDS[w]
    return None


def _pretty(s: str) -> str:
    s = s.strip()
    return s.title() if s.isupper() and len(s) > 3 else s


def resolve_place(value: str, level: int) -> dict | None:
    """{"name", "level", "code"?, "country"?} -- ISO codes become names; a
    subdivision code also names its country."""
    import pycountry

    v = (value or "").strip()
    if not v:
        return None
    if re.fullmatch(r"[A-Z]{2}-[A-Z0-9]{1,3}", v):
        sub = pycountry.subdivisions.get(code=v)
        if sub is not None:
            country = pycountry.countries.get(alpha_2=v[:2])
            return {"name": re.sub(r"\s*\[[^\]]*\]\s*$", "", sub.name), "level": 2, "code": v,
                    "country": _country_name(country) if country else None}
    if re.fullmatch(r"[A-Z]{2}|[A-Z]{3}", v) and level >= 2:
        c = pycountry.countries.get(**({"alpha_2": v} if len(v) == 2 else {"alpha_3": v}))
        if c is not None:
            return {"name": _country_name(c), "level": 3, "code": c.alpha_2}
    if level == 3:
        try:
            c = pycountry.countries.lookup(v)
            return {"name": _country_name(c), "level": 3, "code": c.alpha_2}
        except LookupError:
            pass
    return {"name": _pretty(v), "level": level}


def _country_name(c) -> str:
    return getattr(c, "common_name", None) or c.name


def _level_of_class(label: str | None) -> int | None:
    low = (label or "").lower()
    for lvl, labels in CLASS_FOR_LEVEL.items():
        if low in {x.lower() for x in labels}:
            return lvl
    return None


def value_place_level(values: list[Any], known_places: dict[str, int | None]) -> int | None:
    """Level of a column judged from its VALUES, whatever the column is called:
    a value counts when it names a place the graph already holds
    (``known_places``: normalised name -> level, None when the class does not
    say) or a country (ISO 3166 name or 3-letter code). Two-letter values are
    not evidence -- state, language and unit codes collide with ISO alpha-2.
    A place column when at least VALUE_PLACE_MIN_HITS values and
    VALUE_PLACE_MIN_SHARE of the distinct values count; the level is the most
    common level among them (country when only countries / unknown levels)."""
    import pycountry

    from backend.app.services.db_entity_extract import _normalize_name

    vals = [str(v).strip() for v in values if v is not None and str(v).strip()]
    if len(vals) < VALUE_PLACE_MIN_HITS:
        return None
    votes: dict[int, int] = {}
    hits = 0
    for v in vals:
        n = _normalize_name(v)
        hit, lvl = False, None
        if n in known_places:
            hit, lvl = True, known_places[n]
        if not re.fullmatch(r"[A-Za-z]{2}", v) and not re.fullmatch(r"[\d\W_]+", v):
            try:
                pycountry.countries.lookup(v)
                hit, lvl = True, lvl or 3
            except LookupError:
                pass
        if hit:
            hits += 1
            if lvl:
                votes[lvl] = votes.get(lvl, 0) + 1
    if hits < VALUE_PLACE_MIN_HITS or hits / len(vals) < VALUE_PLACE_MIN_SHARE:
        return None
    return max(votes, key=lambda k: (votes[k], k)) if votes else 3


def place_groups(columns: list[dict]) -> list[list[tuple[str, int]]]:
    """Place columns that describe ONE place (same name prefix, e.g. an
    address block), each [(column, level)] ordered city -> country."""
    groups: dict[str, list[tuple[str, int]]] = {}
    for c in columns:
        lvl = place_level(c)
        if lvl is None:
            continue
        words = c["name"].lower().split("_")
        prefix = "_".join(w for w in words if not any(p in w for p in PLACE_WORDS)) or "_"
        groups.setdefault(prefix, []).append((c["name"], lvl))
    return [sorted(g, key=lambda x: x[1]) for g in groups.values()]


async def _place_class_ids(session) -> dict[int, Any]:
    out = {}
    for lvl, labels in CLASS_FOR_LEVEL.items():
        for lab in labels:
            cid = (await session.execute(sql_text(
                f"SELECT id FROM {SCHEMA}.ontology_classes WHERE lower(label) = lower(:l) LIMIT 1"),
                {"l": lab})).scalar()
            if cid:
                out[lvl] = cid
                break
    return out


async def link_places(session, meta: dict, gv: int, *, embed: EmbedFn | None = None) -> dict[str, Any]:
    """Resolve the table's place values to EXISTING place entities and write
    the containment the data states between them. Never creates an entity.
    Returns {"entities": {entity_id: {columns, values}}, "minted": 0,
    "unresolved": [place names], "containment": n}."""
    from backend.app.services.db_entity_extract import _normalize_name
    from backend.app.services.db_geo_enrich import DEFAULT_GEO_CLASS_LABELS, _place_entities

    table = meta["table_name"]
    columns = meta.get("columns") or []
    groups = place_groups(columns)
    out: dict[str, Any] = {"entities": {}, "minted": 0, "unresolved": [], "containment": 0,
                           "found_by_values": []}
    # existing places by normalised name
    existing: dict[str, Any] = {}
    known_levels: dict[str, int | None] = {}
    for eid, name, label, _cid in await _place_entities(session, DEFAULT_GEO_CLASS_LABELS):
        existing.setdefault(_normalize_name(name), eid)
        known_levels.setdefault(_normalize_name(name), _level_of_class(label))

    # columns the words did not mark: judge them by their values. Each one is
    # its own group -- nothing says two such columns describe the same place
    # (a reporter and a partner country), so no containment is read from them.
    named = {c for g in groups for c, _ in g}
    for c in columns:
        if (c["name"] in named or c.get("type", "text") != "text"
                or c.get("role") in {"measure", "text"}):
            continue
        vals = (await session.execute(sql_text(
            f"SELECT {_q(c['name'])} FROM {SCHEMA}.{_q(table)} WHERE {_q(c['name'])} IS NOT NULL "
            f"GROUP BY 1 ORDER BY COUNT(*) DESC LIMIT {MAX_PLACE_VALUES}"))).scalars().all()
        lvl = value_place_level(list(vals), known_levels)
        if lvl is not None:
            groups.append([(c["name"], lvl)])
            out["found_by_values"].append(c["name"])
    if not groups:
        return out

    resolved: dict[tuple[str, str], dict] = {}       # (column, raw) -> place
    for group in groups:
        for col, lvl in group:
            vals = (await session.execute(sql_text(
                f"SELECT {_q(col)}, COUNT(*) FROM {SCHEMA}.{_q(table)} WHERE {_q(col)} IS NOT NULL "
                f"GROUP BY 1 ORDER BY 2 DESC LIMIT {MAX_PLACE_VALUES}"))).all()
            for raw, _n in vals:
                p = resolve_place(str(raw), lvl)
                if p:
                    resolved[(col, str(raw))] = p

    # one entity per (name, level); country of a subdivision code too
    wanted: dict[tuple[str, int], dict] = {}
    for p in resolved.values():
        wanted.setdefault((p["name"], p["level"]), p)
        if p.get("country"):
            wanted.setdefault((p["country"], 3), {"name": p["country"], "level": 3})
    ids: dict[tuple[str, int], Any] = {}
    for key, p in wanted.items():
        eid = existing.get(_normalize_name(p["name"]))
        if eid is not None:
            ids[key] = eid
        elif p["name"] not in out["unresolved"]:
            out["unresolved"].append(p["name"])

    # table -> place, per entity
    for (col, raw), p in resolved.items():
        eid = ids.get((p["name"], p["level"]))
        if eid is None:
            continue
        h = out["entities"].setdefault(eid, {"columns": set(), "values": []})
        h["columns"].add(col)
        if len(h["values"]) < 5 and raw not in h["values"]:
            h["values"].append(raw)

    # containment stated by the data
    pairs: set[tuple[Any, Any, str]] = set()
    for p in resolved.values():
        if p.get("country") and (p["name"], 2) in ids and (p["country"], 3) in ids:
            pairs.add((ids[(p["name"], 2)], ids[(p["country"], 3)], f"ISO 3166-2 code {p['code']}"))
    for group in groups:
        if len(group) < 2:
            continue
        cols = [c for c, _ in group]
        rows = (await session.execute(sql_text(
            f"SELECT DISTINCT {', '.join(_q(c) for c in cols)} FROM {SCHEMA}.{_q(table)} "
            f"LIMIT {MAX_PLACE_COMBOS}"))).all()
        for row in rows:
            chain = []
            for (col, _lvl), raw in zip(group, row, strict=True):
                p = resolved.get((col, str(raw))) if raw is not None else None
                if p and (p["name"], p["level"]) in ids:
                    chain.append(p)
            for child, parent in pairwise(chain):
                if child["level"] < parent["level"]:
                    pairs.add((ids[(child["name"], child["level"])], ids[(parent["name"], parent["level"])],
                               f"same row of table {table}"))
    out["containment"] = await _write_containment(session, pairs, gv, table)
    return out


async def _write_containment(session, pairs, gv: int, table: str) -> int:
    from backend.app.services.predicates import GRAPHRAG_LOCATED_IN

    n = 0
    for child, parent, why in pairs:
        if child == parent:
            continue
        res = await session.execute(sql_text(f"""
            INSERT INTO {SCHEMA}.graph_relationships
                (source_node_type, source_node_id, target_node_type, target_node_id,
                 predicate_iri, predicate_label, relationship_type, relationship_source,
                 is_authoritative, graph_version, extra_metadata)
            SELECT 'entity', :c, 'entity', :p, :pred, 'locatedInPlace', 'locatedInPlace',
                   'DOCUMENT_EXTRACTION', true, :gv, CAST(:x AS jsonb)
             WHERE NOT EXISTS (SELECT 1 FROM {SCHEMA}.graph_relationships
                                WHERE source_node_id = :c AND target_node_id = :p AND predicate_iri = :pred)
        """), {"c": child, "p": parent, "pred": GRAPHRAG_LOCATED_IN, "gv": gv,
               "x": json.dumps({"evidence_kind": "table_data", "found_by": "structured-data",
                                "table": table, "evidence": why})})
        n += res.rowcount or 0
    return n


# ------------------------------------------------------------------ driver


async def time_and_place_edges(session, node_id, meta: dict, gv: int, *, embed: EmbedFn | None = None,
                               time_level: str = "year") -> tuple[list[dict], dict]:
    """Edges table -> time instances and table -> places, plus a summary."""
    from backend.app.services.predicates import TIME_HAS_TIME, VIAO_ASSERTS_ABOUT

    level = time_level if time_level in TIME_LEVELS else "year"
    edges: list[dict] = []
    periods = await table_periods(session, meta)
    idents = identifiers_for(periods, level)
    ids = await ensure_time_instances(session, idents, gv) if idents else {}
    cols = period_columns(meta)
    for ident in sorted(idents):
        if ident in ids:
            edges.append({
                "source_node_type": "intelligence_artifact", "source_node_id": node_id,
                "target_node_type": "time_instance", "target_node_id": ids[ident],
                "predicate_iri": TIME_HAS_TIME, "predicate_label": "time:hasTime",
                "relationship_type": "tableTime", "relationship_source": "TIME_ENRICHMENT",
                "is_authoritative": True, "source_artifact_id": node_id, "graph_version": gv,
                "extra_metadata": json.dumps({"table": meta["table_name"], "columns": cols})})
    places = await link_places(session, meta, gv, embed=embed)
    for eid, h in places["entities"].items():
        edges.append({
            "source_node_type": "intelligence_artifact", "source_node_id": node_id,
            "target_node_type": "entity", "target_node_id": eid,
            "predicate_iri": VIAO_ASSERTS_ABOUT, "predicate_label": "viao:assertsAbout",
            "relationship_type": "tablePlace", "relationship_source": "DOCUMENT_EXTRACTION",
            "is_authoritative": True, "source_artifact_id": node_id, "graph_version": gv,
            "extra_metadata": json.dumps({"table": meta["table_name"], "columns": sorted(h["columns"]),
                                          "values": h["values"]})})
    summary = {"time_level": level, "periods": len(idents), "time_columns": cols,
               "places": len(places["entities"]), "places_minted": 0,
               "places_unresolved": places["unresolved"][:50],
               "containment": places["containment"],
               "place_columns_by_values": places.get("found_by_values") or []}
    return edges, summary
