"""Geographic containment between places the corpus already names.

WHY THIS EXISTS. Temporal enrichment builds a Year -> Quarter -> Month -> Day
tree from calendar arithmetic, so a question about 2023 reaches a chunk dated
October 2023. Geography had no analogue, and extraction only mints a
containment edge when a chunk states one outright -- which prose never does,
because "Frankfurt" assumes the reader knows it is in Germany. Measured on the
40-document news corpus: ONE `locatedin` edge in the entire graph.

That single gap is what makes "which company in Germany" unanswerable by the
graph. The chain needs two rungs:

    Company X -[headquartered in]-> Frankfurt      from the text (extraction)
    Frankfurt -[locatedInPlace]--> Germany         from here

TWO DELIBERATE LIMITS, both about not inventing a graph:

1. **Only places the corpus already names.** A container that is not already
   an entity is not minted. The pass adds edges, never nodes, so the graph
   stays bounded by the corpus and a run cannot quietly import a gazetteer.
2. **These edges are world knowledge, not evidence.** They carry
   `relationship_source='LLM_INFERENCE'` and
   `extra_metadata.evidence_kind='world_knowledge'`, and retrieval keeps them
   OUT of the evidence packet. They exist to be walked, never to be cited --
   no answer should ever rest on a sentence no document wrote.

Opt-in (`enrich-geo`), like every other pass that spends money.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import text as sql_text

from backend.app.db.graph_version import current_version
from backend.app.db.session import session_scope
from backend.app.services.db_artifact_gen import _extract_json
from backend.app.services.llm_router import LLMRouter
from backend.app.services.predicates import GRAPHRAG_LOCATED_IN
from backend.app.services.prompts import PROMPTS

# Seed labels for "this class denotes a place". The subclass closure of
# whatever these match is taken too, so an ontology that calls them
# `PopulatedPlace` / `SovereignState` is picked up through its parent. Override
# via `geo_enrichment.class_labels` in config.yaml for an ontology that uses
# none of these words.
#
# `SpatialThing` is deliberately ABSENT despite being the obvious W3C root:
# `foaf:Person` is declared a subclass of it, so seeding from there drags every
# person in the corpus into the place list (measured: 262 of 461 "places" were
# people). Anything genuinely geographic reaches this list through one of the
# concrete labels below.
DEFAULT_GEO_CLASS_LABELS: tuple[str, ...] = (
    "Place", "Location", "GeographicFeature",
    "Continent", "Country", "Region", "State", "Province",
    "AdministrativeArea", "City", "Municipality", "Town",
)

# Classes whose descendants are never places, whatever they inherit from.
# Subtracted from the closure as a second guard against the SpatialThing
# problem in an ontology that spells its abstract root differently.
NON_PLACE_CLASS_LABELS: tuple[str, ...] = (
    "Agent", "Person", "Organization", "Organisation", "Event", "Document",
)

# Labels that denote a CONTAINER rather than a contained place. Containers are
# few (19 countries in a 40-document corpus) and children many, so every batch
# carries the full container list as context instead of hoping alphabetical
# batching puts Frankfurt and Germany in the same call -- which it does not.
CONTAINER_CLASS_LABELS: frozenset[str] = frozenset({
    "continent", "country", "region", "state", "province",
    "administrativearea", "place", "location",
})

# Containment LEVELS, used as a deterministic guard on what the model returns.
# A container must sit strictly ABOVE the place it contains, which is what
# rejects the two failure modes measured on gpt-4o-mini: "Canada is in United
# States" (two countries) and "United States is in United States of America"
# (the same country twice). No model judgement can be trusted to be transitive
# and acyclic; arithmetic can.
_GEO_LEVEL: dict[str, int] = {
    "municipality": 1, "town": 1, "city": 1,
    "administrativearea": 2, "state": 2, "province": 2,
    "country": 3,
    "region": 4,
    "continent": 5,
}


def _level(class_label: str | None) -> int:
    """Containment level of a class label, 0 when it says nothing useful."""
    return _GEO_LEVEL.get((class_label or "").strip().lower(), 0)


def levels_are_plausible(child_level: int, parent_level: int) -> bool:
    """True if a place at `parent_level` can contain one at `child_level`.

    Unknown levels (0) are the interesting case. A generic label ("Place",
    "Location", or a class the extractor invented) tells us nothing, so the
    pair is admitted only when the OTHER end is specific enough to carry the
    claim: an unknown child may sit in a country or above, and an unknown
    parent is rejected outright -- it is exactly where a mistyped entity would
    smuggle a bogus container in.
    """
    if child_level and parent_level:
        return parent_level > child_level
    return parent_level >= 3


def containment_is_plausible(child_label: str | None,
                             parent_label: str | None) -> bool:
    """`levels_are_plausible` on two class labels, with no ontology lookup."""
    return levels_are_plausible(_level(child_label), _level(parent_label))


# Contained places per call, on top of the container list.
_BATCH = 40
# Ceiling on containers carried into every batch, so a corpus with hundreds of
# countries cannot blow the prompt up.
_MAX_CONTAINERS = 80


@dataclass
class GeoEnrichSummary:
    place_entities: int = 0
    batches: int = 0
    edges_created: int = 0
    edges_existing: int = 0
    unresolved: int = 0
    rejected_by_level: int = 0
    cost_usd: float = 0.0
    samples: list[str] = field(default_factory=list)


_CLOSURE_SQL = sql_text("""
    WITH RECURSIVE roots AS (
        SELECT id FROM graphrag.ontology_classes
         WHERE lower(label) = ANY(CAST(:labels AS text[]))
    ), down(id, depth) AS (
        SELECT id, 0 FROM roots
      UNION
        SELECT gr.source_node_id, down.depth + 1
          FROM down JOIN graphrag.graph_relationships gr
            ON gr.target_node_id = down.id
           AND gr.source_node_type = 'ontology_class'
           AND gr.target_node_type = 'ontology_class'
           AND gr.predicate_label = 'rdfs:subClassOf'
         WHERE down.depth < 4
    )
    SELECT id FROM down
""")


async def _class_closure(session: Any, labels: tuple[str, ...]) -> set[Any]:
    """Class ids whose label matches, plus their subclasses (4 deep)."""
    if not labels:
        return set()
    rows = await session.execute(
        _CLOSURE_SQL, {"labels": [lbl.lower() for lbl in labels]}
    )
    return {r[0] for r in rows.all()}


_ANCESTOR_LEVEL_SQL = sql_text("""
WITH RECURSIVE up(start_id, id, depth) AS (
    SELECT CAST(x AS uuid), CAST(x AS uuid), 0
      FROM unnest(CAST(:ids AS uuid[])) AS x
  UNION ALL
    SELECT u.start_id, gr.target_node_id, u.depth + 1
      FROM up u
      JOIN graphrag.graph_relationships gr
        ON gr.source_node_id = u.id
       AND gr.source_node_type = 'ontology_class'
       AND gr.target_node_type = 'ontology_class'
       AND gr.predicate_label = 'rdfs:subClassOf'
     WHERE u.depth < 4
) CYCLE id SET is_cycle USING path
SELECT up.start_id, c.label, min(up.depth)
  FROM up JOIN graphrag.ontology_classes c ON c.id = up.id
 WHERE NOT up.is_cycle
 GROUP BY up.start_id, c.label
""")


async def _effective_levels(
    session: Any, class_ids: list[Any]
) -> dict[Any, int]:
    """Containment level per class, falling back to its NEAREST ancestor that
    has one.

    Needed because prune-expand mints individuals as classes: this corpus has
    classes literally named `Uzbekistan`, `Saudi Arabia` and `Asia`, so the
    label alone says nothing and every correct containment through them was
    being rejected (29 of 72 on the 30-document run). Their parents are
    `Country` and `Continent` -- the ontology already knows, the guard just
    was not asking.

    Nearest wins, so a class under both `City` and some abstract root takes
    the city level rather than whichever sorts first.
    """
    if not class_ids:
        return {}
    rows = await session.execute(
        _ANCESTOR_LEVEL_SQL, {"ids": [str(c) for c in class_ids]}
    )
    best: dict[Any, tuple[int, int]] = {}          # class_id -> (depth, level)
    for start_id, label, depth in rows.all():
        lvl = _level(label)
        if not lvl:
            continue
        cur = best.get(start_id)
        if cur is None or int(depth) < cur[0]:
            best[start_id] = (int(depth), lvl)
    return {cid: lvl for cid, (_d, lvl) in best.items()}


async def _place_entities(
    session: Any,
    class_labels: tuple[str, ...],
    non_place_labels: tuple[str, ...] = NON_PLACE_CLASS_LABELS,
) -> list[tuple[Any, str, str, Any]]:
    """(entity_id, name, class_label, class_id) for every ACTIVE entity typed with a
    geographic class or a subclass of one, minus anything that also descends
    from a class of agents, organizations or events.

    The subtraction is not paranoia: `foaf:Person` is declared a subclass of
    `geo:SpatialThing`, so a closure over abstract place roots reaches every
    person in the corpus.
    """
    place_ids = await _class_closure(session, class_labels)
    if not place_ids:
        return []
    place_ids -= await _class_closure(session, non_place_labels)
    if not place_ids:
        return []
    rows = await session.execute(sql_text("""
        SELECT e.id, e.name, c.label, e.class_id
          FROM graphrag.entities e
          JOIN graphrag.ontology_classes c ON c.id = e.class_id
         WHERE e.status = 'ACTIVE'
           AND e.class_id = ANY(CAST(:ids AS uuid[]))
         ORDER BY e.name
    """), {"ids": [str(i) for i in place_ids]})
    return [(r[0], r[1], r[2], r[3]) for r in rows.all()]


async def enrich_geography(
    *,
    class_labels: tuple[str, ...] = DEFAULT_GEO_CLASS_LABELS,
    dry_run: bool = False,
    limit: int | None = None,
    verbose: bool = False,
    router: LLMRouter | None = None,
) -> GeoEnrichSummary:
    """Mint `locatedInPlace` edges among the corpus's place entities."""
    summary = GeoEnrichSummary()
    router = router or LLMRouter()

    async with session_scope() as session:
        places = await _place_entities(session, class_labels)
        if limit:
            places = places[:limit]
        # Level per class, resolved through the subClassOf chain -- see
        # `_effective_levels` for why the label alone is not enough.
        levels = await _effective_levels(
            session, list({cid for _e, _n, _l, cid in places})
        )
    summary.place_entities = len(places)
    if len(places) < 2:
        if verbose:
            print(
                f"[enrich-geo] {len(places)} place entity(ies) typed under "
                f"{list(class_labels)[:4]}... -- nothing to connect. If the "
                f"corpus is full of places, the ontology's place classes are "
                f"named something else; set geo_enrichment.class_labels."
            )
        return summary

    by_name = {name: (eid, label, levels.get(cid, _level(label)))
               for eid, name, label, cid in places}
    # A place is a CONTAINER by its resolved level (>= country), not by its
    # literal class label -- the label may be "Uzbekistan".
    containers = [
        {"name": n, "type": lbl} for _e, n, lbl, cid in places
        if levels.get(cid, _level(lbl)) >= 3
    ][:_MAX_CONTAINERS]
    contained = [
        {"name": n, "type": lbl} for _e, n, lbl, cid in places
        if levels.get(cid, _level(lbl)) < 3
    ]
    # (container names are already in `by_name`; no separate index needed)
    if verbose:
        print(f"[enrich-geo] {len(containers)} container place(s), "
              f"{len(contained)} contained place(s)")
    # Containers ride along in every batch AND get one batch to themselves, so
    # both Frankfurt -> Germany and Hesse -> Germany can be stated.
    # Each batch is sorted by name, which interleaves containers with the
    # places they contain. Grouping all containers first and all contained
    # places after measurably breaks the cheap model: the identical 46-place
    # list returned 5 containments alphabetically and ZERO grouped, because an
    # empty answer is always available to it and a list that opens with 24
    # countries reads like one.
    def _sorted(items: list[dict[str, str]]) -> list[dict[str, str]]:
        return sorted(items, key=lambda d: d["name"])

    batches = [_sorted(containers)] if len(containers) > 1 else []
    batches += [
        _sorted(containers + contained[i : i + _BATCH])
        for i in range(0, len(contained), _BATCH)
    ]

    pairs: list[tuple[Any, Any, str]] = []
    for batch in batches:
        sys_p, user_p = PROMPTS["geographic_containment"](batch)
        summary.batches += 1
        if dry_run:
            continue
        try:
            out = await router.chat(
                "geographic_containment", system=sys_p, user=user_p
            )
            data = _extract_json(out.text) or {}
        except Exception as exc:                            # pragma: no cover
            print(f"[enrich-geo] WARNING batch {summary.batches} failed: {exc}")
            continue
        for item in (data.get("containments") or [])[: len(batch) * 3]:
            if not isinstance(item, dict):
                continue
            child = (item.get("place") or "").strip()
            parent = (item.get("contained_in") or "").strip()
            if not child or not parent or child == parent:
                continue
            # Both ends must already be entities: this pass adds edges, never
            # nodes. A container the corpus never names is simply skipped.
            if child not in by_name or parent not in by_name:
                summary.unresolved += 1
                continue
            if not levels_are_plausible(by_name[child][2],
                                        by_name[parent][2]):
                summary.rejected_by_level += 1
                if verbose:
                    print(f"[enrich-geo] rejected: {child} "
                          f"({by_name[child][1]}) in {parent} "
                          f"({by_name[parent][1]})")
                continue
            pairs.append((by_name[child][0], by_name[parent][0],
                          f"{child} is in {parent}"))
    summary.cost_usd = router.total_cost_usd
    summary.samples = [p[2] for p in pairs[:8]]
    if dry_run or not pairs:
        return summary

    async with session_scope() as session:
        gv = await current_version(session)
        existing = await session.execute(sql_text("""
            SELECT source_node_id, target_node_id
              FROM graphrag.graph_relationships
             WHERE predicate_iri = :p
        """), {"p": GRAPHRAG_LOCATED_IN})
        have = {(a, b) for a, b in existing.all()}
        rows = []
        for child_id, parent_id, text_ in pairs:
            if (child_id, parent_id) in have:
                summary.edges_existing += 1
                continue
            have.add((child_id, parent_id))
            rows.append({
                "source_node_type": "entity",
                "source_node_id": child_id,
                "target_node_type": "entity",
                "target_node_id": parent_id,
                "predicate_iri": GRAPHRAG_LOCATED_IN,
                "predicate_label": "locatedInPlace",
                "relationship_type": "locatedInPlace",
                "relationship_source": "LLM_INFERENCE",
                "is_authoritative": False,
                "source_chunk_id": None,
                "source_document_id": None,
                "source_artifact_id": None,
                "graph_version": gv,
                "extra_metadata": json.dumps({
                    # Marks the edge as NOT sourced from any document, which
                    # is what keeps it out of the evidence packet.
                    "evidence_kind": "world_knowledge",
                    "found_by": "enrich-geo",
                    "relation": "is located in",
                    "evidence": text_,
                }),
            })
        for r in rows:
            await session.execute(sql_text("""
                INSERT INTO graphrag.graph_relationships
                  (source_node_type, source_node_id, target_node_type,
                   target_node_id, predicate_iri, predicate_label,
                   relationship_type, relationship_source, is_authoritative,
                   graph_version, extra_metadata)
                VALUES
                  (:source_node_type, :source_node_id, :target_node_type,
                   :target_node_id, :predicate_iri, :predicate_label,
                   :relationship_type, :relationship_source, :is_authoritative,
                   :graph_version, CAST(:extra_metadata AS jsonb))
            """), r)
        summary.edges_created = len(rows)
    return summary
