"""Concept areas: from the words people ask with, down to the chunks about them.

"Financial information for Samsung SDI" names nothing the graph holds: `revenue` is
a Quantity, with no link to anything financial, and concept seeding found 0 entities
for that term (measured 2026-11-04). `domain_concepts.owl` keeps three layers apart:

- TYPES (`rdfs:subClassOf`): Revenue is a Measure, Maintenance is a Process.
- TOPICS: concept areas (Financials, Transportation...), nested with `skos:broader`.
- MEMBERSHIP: `dc:inConceptArea` on a concept's class IRI names its areas
  (Revenue inConceptArea FinancialPerformance; Port in Transportation AND Logistics).

`conceptTerm` lists the phrases that route a QUESTION to a concept or area. A
question containing one expands: the matched class -> narrower areas (inverse
`skos:broader`) -> their member concepts (`inConceptArea`) -> every subclass of
those -> the entities typed with them -> the chunks asserting about them. This is
the only route from a class to a chunk (chunk -> assertsAbout -> entity -> class),
and it does not depend on the entity walk or a capped list of seeds, so a broad area
keeps all its children. Databases built before v2 nest areas with subClassOf; the
subclass step still covers them.

Membership is never stored as an edge and never links two entities: two companies
reporting Revenue share an area, and that is not a fact about them.

Entity retyping is separate and stricter: an entity named exactly by a concept's
label, `skos:prefLabel` or `skos:altLabel` (never a `conceptTerm`, never an area)
is typed with that concept. Ambiguous names (margin, equity, yield, turnover,
earnings) are never used to retype.

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

DC_NS = "https://veerla-ramrao.ai/ontology/domain-concepts#"
TERM_KEY = DC_NS + "conceptTerm"
IN_AREA_KEY = DC_NS + "inConceptArea"
_SKOS = "http://www.w3.org/2004/02/skos/core#"
BROADER_KEY = _SKOS + "broader"
PREF_LABEL_KEY = _SKOS + "prefLabel"
ALT_LABEL_KEY = _SKOS + "altLabel"
# Names that mean different things in different contexts ("margin" of profit,
# of collateral, of a price). A question may route through them; an entity named
# by one is never retyped from it.
AMBIGUOUS_NAMES = frozenset({"margin", "margins", "equity", "yield", "yields",
                             "turnover", "earnings"})
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
        out.append((cid, label, _values(terms)))
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


def _values(raw: Any) -> list[str]:
    """Annotation values as strings: the merge stores plain strings, the raw
    parser `{"value": ...}` / `{"iri": ...}` dicts; both are read."""
    items = raw if isinstance(raw, list) else ([] if raw is None else [raw])
    out: list[str] = []
    for v in items:
        if isinstance(v, dict):
            v = v.get("iri") or v.get("value")
        if isinstance(v, str) and v.strip():
            out.append(v.strip())
    return out


_TOPIC_LINKS_SQL = sql_text("""
    SELECT id, iri,
           extra_metadata -> 'annotations' -> :broader,
           extra_metadata -> 'annotations' -> :in_area
      FROM graphrag.ontology_classes
     WHERE extra_metadata -> 'annotations' ? :broader
        OR extra_metadata -> 'annotations' ? :in_area
""")


async def topic_links(session: AsyncSession) -> list[tuple[Any, str, list[str], list[str]]]:
    """`(class id, iri, broader IRIs, area IRIs)` for every class that names a
    broader concept or a concept area."""
    rows = await session.execute(
        _TOPIC_LINKS_SQL, {"broader": BROADER_KEY, "in_area": IN_AREA_KEY})
    return [(cid, iri, _values(b), _values(a)) for cid, iri, b, a in rows.all()]


def expand_topics(
    class_ids: Iterable[Any],
    links: Iterable[tuple[Any, str, Iterable[str], Iterable[str]]],
    iri_of: dict[Any, str] | None = None,
) -> list[Any]:
    """The matched classes plus everything narrower in the TOPIC layer: concepts
    whose `skos:broader` is one of them and concepts `inConceptArea` one of them,
    transitively. Downward only; subclasses are added later by the SQL.

    `iri_of` maps a matched class id to its IRI when the class has no link row
    of its own (an area at the top names no broader area)."""
    links = list(links)
    iri_by_id = dict(iri_of or {})
    narrower: dict[str, list[tuple[Any, str]]] = {}
    for cid, iri, broader, areas in links:
        iri_by_id.setdefault(cid, iri)
        for parent in {*broader, *areas}:
            narrower.setdefault(parent, []).append((cid, iri))
    out: list[Any] = []
    seen: set[Any] = set()
    queue = [c for c in class_ids]
    while queue:
        cid = queue.pop(0)
        if cid in seen:
            continue
        seen.add(cid)
        out.append(cid)
        for child, _iri in narrower.get(iri_by_id.get(cid, ""), ()):
            if child not in seen:
                queue.append(child)
    return out


async def area_chunks(
    session: AsyncSession,
    class_ids: list[Any],
    named_ids: list[uuid.UUID],
    *,
    limit: int = 100,
) -> list[tuple[uuid.UUID, float]]:
    """Chunks about anything the matched classes cover, best first: those also
    mentioning an entity the question names, then by how many different kinds
    they cover. Coverage = the topic layer (narrower areas and their member
    concepts) and then every subclass."""
    if not class_ids:
        return []
    links = await topic_links(session)
    iris = await session.execute(sql_text(
        "SELECT id, iri FROM graphrag.ontology_classes WHERE id = ANY(CAST(:ids AS uuid[]))"
    ), {"ids": [str(c) for c in class_ids]})
    expanded = expand_topics(class_ids, links, dict(iris.all()))
    rows = await session.execute(_AREA_CHUNKS_SQL, {
        "classes": [str(c) for c in expanded],
        "named": [str(n) for n in named_ids],
        "limit": limit,
    })
    return [(cid, float(score)) for cid, score in rows.all()]


def _label_words(label: str) -> str:
    """'CostOfGoodsSold' -> 'Cost Of Goods Sold' (a class label as a name)."""
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", label or "")
    return re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", text)


def plan_retype(
    entities: Iterable[tuple[Any, str, Any]],
    classes: Iterable[tuple[Any, str, Iterable[str]]],
    normalize: Any,
) -> list[tuple[Any, Any]]:
    """`(entity id, class id)` for entities named exactly by a class's label or
    one of its `names` (prefLabel / altLabel) and not already typed with it.
    Ambiguous names (`AMBIGUOUS_NAMES`) never retype; a name two classes share
    is dropped rather than guessed."""
    by_name: dict[str, Any] = {}
    clash: set[str] = set()
    for cid, label, names in classes:
        for name in {normalize(n) for n in (*names, label, _label_words(label)) if n}:
            if not name or name in AMBIGUOUS_NAMES:
                continue
            if name in by_name and by_name[name] != cid:
                clash.add(name)
            by_name.setdefault(name, cid)
    moves: list[tuple[Any, Any]] = []
    for eid, name, class_id in entities:
        key = normalize(name)
        if key in clash:
            continue
        target = by_name.get(key)
        if target is not None and target != class_id:
            moves.append((eid, target))
    return moves


_NAME_CLASSES_SQL = """
    SELECT oc.id, oc.label,
           oc.extra_metadata -> 'annotations' -> :pref,
           oc.extra_metadata -> 'annotations' -> :alt
      FROM graphrag.ontology_classes oc
     WHERE oc.iri LIKE :ns
       -- curated concepts only: classes the pipeline minted into the namespace
       -- carry none of these
       AND (oc.extra_metadata -> 'annotations' ?| ARRAY[:pref, :alt, :term])
       AND NOT coalesce(oc.extra_metadata -> 'annotations' ? :deprecated, false)
       AND oc.id IN (SELECT id FROM (""" + _CONCEPT_CLASS_IDS_SQL.text + """) c)
"""


async def name_classes(
    session: AsyncSession, root_labels: list[str]
) -> list[tuple[uuid.UUID, str, list[str]]]:
    """`(class id, label, names)` for the curated domain-concepts classes under the
    concept roots: a concept area is not under them, so it is never a retype
    target, and classes the pipeline minted into the namespace are not curated."""
    rows = await session.execute(sql_text(_NAME_CLASSES_SQL), {
        "pref": PREF_LABEL_KEY, "alt": ALT_LABEL_KEY, "term": TERM_KEY, "ns": DC_NS + "%",
        "deprecated": "http://www.w3.org/2002/07/owl#deprecated", "labels": root_labels,
    })
    return [(cid, label, _values(p) + _values(a)) for cid, label, p, a in rows.all()]


async def retype_entities_by_term(
    roots: list[str], *, dry_run: bool = False, log=print
) -> dict[str, int]:
    """Type each concept entity named exactly by a domain concept (its label,
    prefLabel or altLabel) with that concept. Keeps a more specific class the
    entity already has: `SegmentRevenue` stays."""
    from backend.app.services.db_entity_extract import _normalize_name

    stats = {"concepts": 0, "retyped": 0}
    labels = [r.strip().lower() for r in roots if r and r.strip()]
    if not labels:
        return stats
    async with session_scope() as session:
        classes = await name_classes(session, labels)
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
            f"from a concept's name")
    return stats
