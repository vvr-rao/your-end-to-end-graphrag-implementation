"""Concept areas: from the words people ask with, down to the chunks about them.

"Financial information for Samsung SDI" names nothing the graph holds: `revenue` is
a Quantity, with no link to anything financial, and concept seeding found 0 entities
for that term (measured 2026-11-04). `domain_concepts.owl` now groups kinds of
concept under areas (Financials -> FinancialPerformance -> Revenue), with
`conceptTerm` annotations listing the phrases that mean each class.

Two things use the terms:

- A question containing one expands down the class tree to every entity under that
  class, and from them to the chunks asserting about them. This is the only route
  from a class to a chunk (chunk -> assertsAbout -> entity -> class), and it does
  not depend on the entity walk or on a capped list of seeds, so a broad area keeps
  all its children.
- An entity named by one is typed with that class. Extraction types generically
  (`revenue` was a `Metric` although a `Revenue` class existed), and an entity left
  there is not under the area. Costs nothing, so it always runs.

An ontology with no terms makes both a no-op.
"""
from __future__ import annotations

import json
import re
import uuid
from collections.abc import Iterable
from typing import Any

from sqlalchemy import text as sql_text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db.session import session_scope
from backend.app.services.retrieval_sql import _CONCEPT_CLASS_IDS_SQL

TERM_KEY = "https://veerla-ramrao.ai/ontology/domain-concepts#conceptTerm"
_WORD = r"[^\W_]"

_TERM_CLASSES_SQL = sql_text("""
    SELECT id, label, extra_metadata -> 'annotations' -> :key
      FROM graphrag.ontology_classes
     WHERE extra_metadata -> 'annotations' ? :key
""")


async def term_classes(session: AsyncSession) -> list[tuple[uuid.UUID, str, list[str]]]:
    """`(class id, label, terms)` for every class with concept terms."""
    rows = await session.execute(_TERM_CLASSES_SQL, {"key": TERM_KEY})
    out: list[tuple[uuid.UUID, str, list[str]]] = []
    for cid, label, terms in rows.all():
        values = [terms] if isinstance(terms, str) else list(terms or [])
        out.append((cid, label, [t for t in values if isinstance(t, str) and t.strip()]))
    return out


def classes_in_question(
    question: str, classes: Iterable[tuple[Any, str, Iterable[str]]]
) -> list[Any]:
    """Ids of the classes whose terms the question contains, longest term first:
    "manufacturing costs" is not also a question about "costs"."""
    terms = sorted(
        ((t, cid) for cid, _label, ts in classes for t in ts),
        key=lambda item: -len(item[0]),
    )
    found: list[Any] = []
    taken: list[tuple[int, int]] = []
    for term, cid in terms:
        pattern = rf"(?<!{_WORD}){re.escape(term)}(?!{_WORD})"
        for match in re.finditer(pattern, question, re.IGNORECASE):
            start, end = match.span()
            if any(start < t_end and t_start < end for t_start, t_end in taken):
                continue
            taken.append((start, end))
            if cid not in found:
                found.append(cid)
    return found


_AREA_CHUNKS_SQL = sql_text("""
WITH RECURSIVE down(id) AS (
    SELECT CAST(x AS uuid) FROM unnest(CAST(:classes AS uuid[])) AS x
  UNION
    SELECT gr.source_node_id
      FROM down JOIN graphrag.graph_relationships gr
        ON gr.target_node_id = down.id
       AND gr.source_node_type = 'ontology_class'
       AND gr.target_node_type = 'ontology_class'
       AND gr.predicate_label = 'rdfs:subClassOf'
), members AS (
    SELECT e.id, e.class_id
      FROM graphrag.entities e JOIN down ON down.id = e.class_id
     WHERE e.status = 'ACTIVE'
       AND coalesce(e.extra_metadata ->> 'concept_kind', '') <> 'named'
), named AS (
    SELECT DISTINCT gr.source_node_id AS chunk_id
      FROM graphrag.graph_relationships gr
     WHERE gr.predicate_label = 'viao:assertsAbout'
       AND gr.source_node_type = 'chunk'
       AND gr.target_node_id = ANY(CAST(:named AS uuid[]))
)
SELECT gr.source_node_id,
       count(DISTINCT m.class_id)::float
         + CASE WHEN n.chunk_id IS NOT NULL THEN 1000 ELSE 0 END AS score
  FROM graphrag.graph_relationships gr
  JOIN members m ON m.id = gr.target_node_id
  JOIN graphrag.chunks c ON c.id = gr.source_node_id AND c.status = 'ACTIVE'
  LEFT JOIN named n ON n.chunk_id = gr.source_node_id
 WHERE gr.predicate_label = 'viao:assertsAbout'
   AND gr.source_node_type = 'chunk'
   AND gr.target_node_type = 'entity'
 GROUP BY gr.source_node_id, n.chunk_id
 ORDER BY score DESC, gr.source_node_id
 LIMIT :limit
""")


async def area_chunks(
    session: AsyncSession,
    class_ids: list[Any],
    named_ids: list[uuid.UUID],
    *,
    limit: int = 100,
) -> list[tuple[uuid.UUID, float]]:
    """Chunks about anything under these classes, best first: those also mentioning
    an entity the question names, then by how many different kinds they cover."""
    if not class_ids:
        return []
    rows = await session.execute(_AREA_CHUNKS_SQL, {
        "classes": [str(c) for c in class_ids],
        "named": [str(n) for n in named_ids],
        "limit": limit,
    })
    return [(cid, float(score)) for cid, score in rows.all()]


def plan_retype(
    entities: Iterable[tuple[Any, str, Any]],
    classes: Iterable[tuple[Any, str, Iterable[str]]],
    normalize: Any,
) -> list[tuple[Any, Any]]:
    """`(entity id, class id)` for entities named exactly by a class's term or label
    and not already typed with it."""
    by_term: dict[str, Any] = {}
    for cid, label, terms in classes:
        for term in (*terms, label):
            by_term.setdefault(normalize(term), cid)
    moves: list[tuple[Any, Any]] = []
    for eid, name, class_id in entities:
        target = by_term.get(normalize(name))
        if target is not None and target != class_id:
            moves.append((eid, target))
    return moves


async def retype_entities_by_term(
    roots: list[str], *, dry_run: bool = False, log=print
) -> dict[str, int]:
    """Type each concept named by a concept term with that term's class. Keeps a
    more specific class the entity already has: `SegmentRevenue` stays."""
    from backend.app.services.db_entity_extract import _normalize_name

    stats = {"concepts": 0, "retyped": 0}
    labels = [r.strip().lower() for r in roots if r and r.strip()]
    if not labels:
        return stats
    async with session_scope() as session:
        classes = await term_classes(session)
        if not classes:
            return stats
        rows = await session.execute(sql_text("""
            SELECT e.id, e.name, e.class_id FROM graphrag.entities e
             WHERE e.status = 'ACTIVE'
               AND coalesce(e.extra_metadata ->> 'concept_kind', '') <> 'named'
               AND e.class_id IN (SELECT id FROM (""" + _CONCEPT_CLASS_IDS_SQL.text + """) c)
        """), {"labels": labels})
        entities = [tuple(r) for r in rows.all()]
        stats["concepts"] = len(entities)
        moves = plan_retype(entities, classes, _normalize_name)
        below = await session.execute(sql_text("""
            WITH RECURSIVE down(root, id) AS (
                SELECT CAST(x AS uuid), CAST(x AS uuid)
                  FROM unnest(CAST(:targets AS uuid[])) AS x
              UNION
                SELECT down.root, gr.source_node_id
                  FROM down JOIN graphrag.graph_relationships gr
                    ON gr.target_node_id = down.id
                   AND gr.source_node_type = 'ontology_class'
                   AND gr.target_node_type = 'ontology_class'
                   AND gr.predicate_label = 'rdfs:subClassOf'
            )
            SELECT root, id FROM down
        """), {"targets": [str(t) for t in {m[1] for m in moves}]})
        under = {tuple(r) for r in below.all()}
        current = {e[0]: e[2] for e in entities}
        moves = [(eid, cid) for eid, cid in moves if (cid, current[eid]) not in under]
        stats["retyped"] = len(moves)
        if dry_run:
            return stats
        for eid, cid in moves:
            await session.execute(sql_text("""
                UPDATE graphrag.entities
                   SET extra_metadata = extra_metadata
                         || CAST(:meta AS jsonb),
                       class_id = :cid
                 WHERE id = :eid
            """), {"eid": eid, "cid": cid,
                   "meta": json.dumps({"retyped_from": str(current[eid])})})
    if stats["retyped"]:
        log(f"[concept-areas] {stats['retyped']} of {stats['concepts']} concepts typed "
            f"from their concept term")
    return stats
