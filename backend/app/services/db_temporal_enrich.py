"""Milestone D: temporal enrichment over ingested chunks.

Per chunk: regex-extract years / quarters / months / days from
`chunks.text`. Normalize to canonical identifiers:
  - YEAR_2024
  - Q3_2024
  - MONTH_2024_07
  - DAY_2024_07_15

Side effects:
  1. Upsert `time_instances` rows for every distinct identifier seen.
  2. Walk up to `highest_level` (default 'year'): mint missing parents
     + set parent_time_id.
  3. Gap-fill at `lowest_level` (default 'month'): between the
     minimum and maximum minted month, mint every intermediate month
     even if no chunk referenced it. Same for days when lowest=day.
  4. Insert `graph_relationships` rows:
       chunk -> time:hasTime -> time_instance         (source=TIME_ENRICHMENT)
       child_time -> time:intervalDuring -> parent    (source=TIME_ENRICHMENT)

No LLM calls. Pure regex + calendar math + SQL.
Generic: works on any corpus that has been ingested via Milestone B.
"""
from __future__ import annotations

import calendar
import re
import time
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from sqlalchemy import and_, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from backend.app.db.graph_version import bump_version, current_version
from backend.app.db.models.documents import Chunk
from backend.app.db.models.entities import TimeInstance
from backend.app.db.models.graph import GraphRelationship
from backend.app.db.session import session_scope
from backend.app.services.predicates import TIME_HAS_TIME, TIME_INTERVAL_DURING

_MONTHS_LONG = (
    "january february march april may june july august "
    "september october november december"
).split()
_MONTHS_SHORT = (
    "jan feb mar apr may jun jul aug sep oct nov dec"
).split()
_MONTH_TO_INT = {
    name: i + 1 for i, name in enumerate(_MONTHS_LONG)
}
_MONTH_TO_INT.update({name: i + 1 for i, name in enumerate(_MONTHS_SHORT)})

# Order matters: most specific first so a YYYY-MM-DD doesn't also fire
# as a YYYY-MM.
_RE_DAY_ISO = re.compile(r"\b(19[7-9]\d|20[0-3]\d)-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])\b")
_RE_DAY_NAMED = re.compile(
    r"\b(" + "|".join(_MONTHS_LONG + _MONTHS_SHORT) + r")\.?\s+"
    r"(\d{1,2})(?:st|nd|rd|th)?,?\s+"
    r"(19[7-9]\d|20[0-3]\d)\b",
    re.IGNORECASE,
)
_RE_MONTH_ISO = re.compile(r"\b(19[7-9]\d|20[0-3]\d)-(0[1-9]|1[0-2])\b")
_RE_MONTH_NAMED = re.compile(
    r"\b(" + "|".join(_MONTHS_LONG + _MONTHS_SHORT) + r")\.?\s+"
    r"(19[7-9]\d|20[0-3]\d)\b",
    re.IGNORECASE,
)
_RE_QUARTER = re.compile(
    r"\b(?:(Q[1-4])[\s-]?(?:FY)?(19[7-9]\d|20[0-3]\d)|"
    r"(19[7-9]\d|20[0-3]\d)[\s-]?(Q[1-4]))\b",
    re.IGNORECASE,
)
_RE_YEAR = re.compile(r"\b(19[7-9]\d|20[0-3]\d)\b")

_HIGHEST: dict[str, int] = {"year": 4, "quarter": 3, "month": 2, "day": 1}
_LOWEST: dict[str, int] = {"day": 1, "month": 2, "quarter": 3, "year": 4}


@dataclass
class EnrichSummary:
    chunks_processed: int = 0
    chunks_skipped: int = 0
    instances_minted: int = 0
    parents_filled: int = 0
    gaps_filled: int = 0
    chunk_time_edges: int = 0
    parent_edges: int = 0
    wall_seconds: float = 0.0
    new_graph_version: int = 0
    sample_identifiers: list[str] = field(default_factory=list)


def _year_record(year: int) -> dict[str, Any]:
    return {
        "time_identifier": f"YEAR_{year}",
        "time_level": "year",
        "start_date": date(year, 1, 1),
        "end_date": date(year, 12, 31),
        "display_label": str(year),
        "extra_metadata": {},
    }


def _quarter_record(year: int, q: int) -> dict[str, Any]:
    start_month = (q - 1) * 3 + 1
    end_month = start_month + 2
    return {
        "time_identifier": f"Q{q}_{year}",
        "time_level": "quarter",
        "start_date": date(year, start_month, 1),
        "end_date": date(year, end_month, calendar.monthrange(year, end_month)[1]),
        "display_label": f"Q{q} {year}",
        "extra_metadata": {"quarter": q, "year": year},
    }


def _month_record(year: int, month: int) -> dict[str, Any]:
    return {
        "time_identifier": f"MONTH_{year}_{month:02d}",
        "time_level": "month",
        "start_date": date(year, month, 1),
        "end_date": date(year, month, calendar.monthrange(year, month)[1]),
        "display_label": f"{calendar.month_name[month]} {year}",
        "extra_metadata": {"year": year, "month": month},
    }


def _day_record(year: int, month: int, day: int) -> dict[str, Any]:
    d = date(year, month, day)
    return {
        "time_identifier": f"DAY_{year}_{month:02d}_{day:02d}",
        "time_level": "day",
        "start_date": d,
        "end_date": d,
        "display_label": d.isoformat(),
        "extra_metadata": {"year": year, "month": month, "day": day},
    }


def _parent_of(rec: dict[str, Any]) -> dict[str, Any] | None:
    """Return the parent time-instance record (one level up) or None
    if the record is already at the top (year)."""
    level = rec["time_level"]
    if level == "year":
        return None
    meta = rec.get("extra_metadata") or {}
    year = meta.get("year")
    if year is None:
        return None
    if level == "quarter":
        return _year_record(year)
    if level == "month":
        q = (meta["month"] - 1) // 3 + 1
        return _quarter_record(year, q)
    if level == "day":
        return _month_record(year, meta["month"])
    return None


def _extract_dates(text: str) -> set[str]:
    """Return the set of canonical time_identifiers found in text."""
    found: set[str] = set()
    consumed_spans: list[tuple[int, int]] = []

    def _claim(span: tuple[int, int]) -> bool:
        for s, e in consumed_spans:
            if not (span[1] <= s or span[0] >= e):
                return False
        consumed_spans.append(span)
        return True

    for m in _RE_DAY_ISO.finditer(text):
        if not _claim(m.span()):
            continue
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            date(y, mo, d)
            found.add(f"DAY_{y}_{mo:02d}_{d:02d}")
        except ValueError:
            pass

    for m in _RE_DAY_NAMED.finditer(text):
        if not _claim(m.span()):
            continue
        mo_name = m.group(1).lower()
        d = int(m.group(2))
        y = int(m.group(3))
        mo = _MONTH_TO_INT.get(mo_name)
        if not mo:
            continue
        try:
            date(y, mo, d)
            found.add(f"DAY_{y}_{mo:02d}_{d:02d}")
        except ValueError:
            pass

    for m in _RE_MONTH_ISO.finditer(text):
        if not _claim(m.span()):
            continue
        y, mo = int(m.group(1)), int(m.group(2))
        found.add(f"MONTH_{y}_{mo:02d}")

    for m in _RE_MONTH_NAMED.finditer(text):
        if not _claim(m.span()):
            continue
        mo_name = m.group(1).lower()
        y = int(m.group(2))
        mo = _MONTH_TO_INT.get(mo_name)
        if mo:
            found.add(f"MONTH_{y}_{mo:02d}")

    for m in _RE_QUARTER.finditer(text):
        if not _claim(m.span()):
            continue
        q_str = m.group(1) or m.group(4)
        y_str = m.group(2) or m.group(3)
        if not q_str or not y_str:
            continue
        q = int(q_str[1])
        y = int(y_str)
        found.add(f"Q{q}_{y}")

    for m in _RE_YEAR.finditer(text):
        if not _claim(m.span()):
            continue
        y = int(m.group(1))
        found.add(f"YEAR_{y}")

    return found


def _record_for(identifier: str) -> dict[str, Any] | None:
    if identifier.startswith("YEAR_"):
        return _year_record(int(identifier[5:]))
    if identifier.startswith("Q"):
        # Q3_2024
        q = int(identifier[1])
        y = int(identifier.split("_", 1)[1])
        return _quarter_record(y, q)
    if identifier.startswith("MONTH_"):
        parts = identifier.split("_")
        return _month_record(int(parts[1]), int(parts[2]))
    if identifier.startswith("DAY_"):
        parts = identifier.split("_")
        return _day_record(int(parts[1]), int(parts[2]), int(parts[3]))
    return None


async def enrich_temporal(
    *,
    limit: int | None = None,
    highest_level: str = "year",
    lowest_level: str = "month",
) -> EnrichSummary:
    """Drive: scan chunks not yet enriched, mint time_instances + edges."""
    t0 = time.time()
    summary = EnrichSummary()

    # Pull chunks that don't yet have a time:hasTime edge.
    async with session_scope() as session:
        already_enriched_subq = select(GraphRelationship.source_chunk_id).where(
            GraphRelationship.relationship_source == "TIME_ENRICHMENT",
            GraphRelationship.predicate_iri == TIME_HAS_TIME,
        )
        stmt = (
            select(Chunk.id, Chunk.text)
            .where(
                Chunk.status == "ACTIVE",
                Chunk.id.notin_(already_enriched_subq),
            )
            .order_by(Chunk.created_at)
        )
        if limit is not None:
            stmt = stmt.limit(limit)
        result = await session.execute(stmt)
        chunks = result.all()

    if not chunks:
        print("[enrich-time] no chunks left to enrich")
        return summary

    print(f"[enrich-time] {len(chunks)} chunk(s) to scan")

    # Per-chunk extraction -> mapping chunk_id -> set of identifiers
    chunk_to_identifiers: list[tuple[Any, set[str]]] = []
    all_identifiers: set[str] = set()
    for chunk_id, text in chunks:
        ids = _extract_dates(text)
        if not ids:
            summary.chunks_skipped += 1
            continue
        chunk_to_identifiers.append((chunk_id, ids))
        all_identifiers.update(ids)
        summary.chunks_processed += 1

    if not all_identifiers:
        print("[enrich-time] no dates extracted; nothing to insert")
        summary.wall_seconds = time.time() - t0
        return summary

    print(
        f"[enrich-time] {len(all_identifiers)} distinct time identifier(s) "
        f"from {summary.chunks_processed} chunk(s); "
        f"{summary.chunks_skipped} chunk(s) had no dates"
    )

    # Walk up parents to fill the hierarchy (respect highest_level).
    highest_rank = _HIGHEST.get(highest_level, 4)
    all_records: dict[str, dict[str, Any]] = {}
    for ident in all_identifiers:
        rec = _record_for(ident)
        if rec is None:
            continue
        all_records[ident] = rec
        cur = rec
        while True:
            parent = _parent_of(cur)
            if parent is None:
                break
            if _HIGHEST[parent["time_level"]] > highest_rank:
                break
            if parent["time_identifier"] in all_records:
                cur = all_records[parent["time_identifier"]]
                break
            all_records[parent["time_identifier"]] = parent
            summary.parents_filled += 1
            cur = parent

    # Gap-fill at lowest_level. Find min..max bound from observed records.
    lowest_rank = _LOWEST.get(lowest_level, 2)
    if lowest_rank in (1, 2):  # day or month gap-fill
        # We just gap-fill MONTHS for simplicity in v0 (day-level
        # gap-fill across long ranges would create tens of thousands
        # of rows).
        months = sorted(
            (rec for rec in all_records.values() if rec["time_level"] == "month"),
            key=lambda r: r["start_date"],
        )
        if len(months) >= 2:
            start = months[0]["start_date"]
            end = months[-1]["start_date"]
            y, mo = start.year, start.month
            while date(y, mo, 1) <= end:
                ident = f"MONTH_{y}_{mo:02d}"
                if ident not in all_records:
                    all_records[ident] = _month_record(y, mo)
                    summary.gaps_filled += 1
                    # also ensure its parent chain
                    cur = all_records[ident]
                    while True:
                        parent = _parent_of(cur)
                        if parent is None or _HIGHEST[parent["time_level"]] > highest_rank:
                            break
                        if parent["time_identifier"] in all_records:
                            break
                        all_records[parent["time_identifier"]] = parent
                        summary.parents_filled += 1
                        cur = parent
                mo += 1
                if mo == 13:
                    mo = 1
                    y += 1

    # Upsert all time_instances. Idempotent on (time_identifier).
    payloads = list(all_records.values())
    print(f"[enrich-time] upserting {len(payloads)} time_instance row(s)")

    UPSERT_BATCH = 200
    async with session_scope() as session:
        for i in range(0, len(payloads), UPSERT_BATCH):
            chunk = payloads[i : i + UPSERT_BATCH]
            stmt = pg_insert(TimeInstance).values(chunk)
            stmt = stmt.on_conflict_do_update(
                index_elements=["time_identifier"],
                set_={
                    "time_level": stmt.excluded.time_level,
                    "start_date": stmt.excluded.start_date,
                    "end_date": stmt.excluded.end_date,
                    "display_label": stmt.excluded.display_label,
                    "extra_metadata": stmt.excluded.extra_metadata,
                },
            )
            await session.execute(stmt)

        # Fetch ID + parent_time_id back.
        result = await session.execute(
            select(TimeInstance.id, TimeInstance.time_identifier).where(
                TimeInstance.time_identifier.in_(all_records.keys())
            )
        )
        id_by_ident = {ident: tid for tid, ident in result.all()}
    summary.instances_minted = len(payloads)

    # Second pass: set parent_time_id on each child.
    parent_updates = []
    for ident, rec in all_records.items():
        parent_rec = _parent_of(rec)
        if parent_rec is None:
            continue
        if parent_rec["time_identifier"] not in id_by_ident:
            continue
        parent_updates.append(
            {
                "id": id_by_ident[ident],
                "parent_time_id": id_by_ident[parent_rec["time_identifier"]],
            }
        )

    if parent_updates:
        async with session_scope() as session:
            for upd in parent_updates:
                await session.execute(
                    select(TimeInstance.id).where(TimeInstance.id == upd["id"])
                )
                # Use direct UPDATE since we're doing many small fixes.
                from sqlalchemy import update as sql_update
                await session.execute(
                    sql_update(TimeInstance).where(
                        TimeInstance.id == upd["id"]
                    ).values(parent_time_id=upd["parent_time_id"])
                )

    # Build edges.
    async with session_scope() as session:
        gv = await current_version(session)

    edge_payloads: list[dict[str, Any]] = []
    # Chunk -> time:hasTime -> time_instance
    for chunk_id, idents in chunk_to_identifiers:
        for ident in idents:
            tid = id_by_ident.get(ident)
            if not tid:
                continue
            edge_payloads.append({
                "source_node_type": "chunk",
                "source_node_id": chunk_id,
                "target_node_type": "time_instance",
                "target_node_id": tid,
                "predicate_iri": TIME_HAS_TIME,
                "predicate_label": "time:hasTime",
                "relationship_type": "hasTime",
                "relationship_source": "TIME_ENRICHMENT",
                "is_authoritative": True,
                "source_chunk_id": chunk_id,
                "source_document_id": None,
                "source_artifact_id": None,
                "graph_version": gv,
                "extra_metadata": {},
            })

    # child_time -> time:intervalDuring -> parent_time
    for ident, rec in all_records.items():
        parent_rec = _parent_of(rec)
        if parent_rec is None:
            continue
        if parent_rec["time_identifier"] not in id_by_ident:
            continue
        edge_payloads.append({
            "source_node_type": "time_instance",
            "source_node_id": id_by_ident[ident],
            "target_node_type": "time_instance",
            "target_node_id": id_by_ident[parent_rec["time_identifier"]],
            "predicate_iri": TIME_INTERVAL_DURING,
            "predicate_label": "time:intervalDuring",
            "relationship_type": "intervalDuring",
            "relationship_source": "TIME_ENRICHMENT",
            "is_authoritative": True,
            "source_chunk_id": None,
            "source_document_id": None,
            "source_artifact_id": None,
            "graph_version": gv,
            "extra_metadata": {},
        })

    # Insert edges. Dedup against existing rows (chunk->time edges
    # may exist if a chunk was re-enriched after partial failure).
    EDGE_BATCH = 500
    if edge_payloads:
        async with session_scope() as session:
            # Pull existing (source_id, predicate_iri, target_id) keys
            # for chunks we're processing to avoid duplicates.
            chunk_ids = list({p["source_node_id"] for p in edge_payloads
                              if p["source_node_type"] == "chunk"})
            existing_keys = set()
            if chunk_ids:
                result = await session.execute(
                    select(
                        GraphRelationship.source_node_id,
                        GraphRelationship.target_node_id,
                        GraphRelationship.predicate_iri,
                    ).where(
                        and_(
                            GraphRelationship.source_node_id.in_(chunk_ids),
                            GraphRelationship.relationship_source == "TIME_ENRICHMENT",
                        )
                    )
                )
                existing_keys = {(s, t, p) for s, t, p in result.all()}

            # Also dedup parent edges
            time_ids = list({
                p["source_node_id"] for p in edge_payloads
                if p["source_node_type"] == "time_instance"
            })
            if time_ids:
                result = await session.execute(
                    select(
                        GraphRelationship.source_node_id,
                        GraphRelationship.target_node_id,
                        GraphRelationship.predicate_iri,
                    ).where(
                        and_(
                            GraphRelationship.source_node_id.in_(time_ids),
                            GraphRelationship.predicate_iri == TIME_INTERVAL_DURING,
                        )
                    )
                )
                existing_keys.update((s, t, p) for s, t, p in result.all())

            filtered = [
                p for p in edge_payloads
                if (p["source_node_id"], p["target_node_id"], p["predicate_iri"])
                not in existing_keys
            ]

            for i in range(0, len(filtered), EDGE_BATCH):
                await session.execute(
                    pg_insert(GraphRelationship).values(
                        filtered[i : i + EDGE_BATCH]
                    )
                )

            chunk_edges = sum(1 for p in filtered if p["source_node_type"] == "chunk")
            parent_edges = sum(1 for p in filtered if p["source_node_type"] == "time_instance")
            summary.chunk_time_edges = chunk_edges
            summary.parent_edges = parent_edges

    async with session_scope() as session:
        summary.new_graph_version = await bump_version(session)

    summary.wall_seconds = time.time() - t0
    summary.sample_identifiers = sorted(all_records.keys())[:10]

    print(
        f"[enrich-time] DONE: instances={summary.instances_minted} "
        f"(parents+={summary.parents_filled}, gaps+={summary.gaps_filled}), "
        f"chunk_edges={summary.chunk_time_edges}, "
        f"parent_edges={summary.parent_edges}, "
        f"wall={summary.wall_seconds:.1f}s, "
        f"graph_version -> {summary.new_graph_version}"
    )

    return summary
