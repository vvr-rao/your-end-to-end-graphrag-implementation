"""Geographic containment: the place hierarchy a region question walks down.

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

TWO STEPS, the first free of any model:

1. **From the ontology.** A place matching a geography class follows that
   class's `partOfClass` chain, and the containers on the chain become entities
   if they are not already, as enrich-time mints the parents of a period.
   Japan -> East Asia -> Asia exists even if no chunk produced an `Asia` entity,
   which is what a question about Asia needs to start from. Only containers
   above a place the corpus names are minted, so the ontology's 283 geography
   classes do not all become nodes.
2. **From the model.** Cities, and names the ontology does not spell the same
   way (`Republic of Korea`), are linked by `geographic_containment`, between
   entities only: a container the model names that is not an entity is skipped,
   so a run cannot quietly import a gazetteer.

These edges are world knowledge, not evidence. They carry
   `relationship_source='LLM_INFERENCE'` and
`extra_metadata.evidence_kind='world_knowledge'`, and retrieval labels them
rather than presenting them as something a document said. The ontology step's
edges carry LLM_INFERENCE too: `ONTOLOGY` edges survive a corpus wipe
(`db_ontology_wipe`), and these point at entities that do not.

Retrieval walks them with `retrieval_sql.expand_place_entities`, down only.

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
#
# Small, for the reason measured twice on 2026-09-20. A single call covering
# many items loses recall badly -- the model labels or omits rather than
# searching, because that is the cheap path once attention is divided. The
# orphan check showed it first (31 orphans in one call -> 2 rescues; the same
# chunk, model and prompt found the edge when asked about ONE), and enrich-geo
# had the identical shape: 5 batches over 299 places, ~109 per call, left 85 of
# 146 cities and states (58%) with no containment edge at all -- including
# Bangalore and Mumbai, which the model resolves correctly to
# "Republic of India" when shown three places.
#
# That missing edge is not cosmetic: it is why "Which cities are in India?"
# fell back to a broad walk and then failed to rank its answer.
_BATCH = 15
# Ceiling on containers carried into every batch, so a corpus with hundreds of
# countries cannot blow the prompt up. Containers ride along in EVERY batch, so
# this is the dominant term in prompt size -- and in how much the model has to
# hold in mind at once.
_MAX_CONTAINERS = 60


@dataclass
class GeoEnrichSummary:
    place_entities: int = 0
    batches: int = 0
    edges_created: int = 0
    edges_existing: int = 0
    unresolved: int = 0
    rejected_by_level: int = 0
    ontology_matched: int = 0
    ontology_nodes: int = 0
    ontology_edges: int = 0
    embedding_cost_usd: float = 0.0
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


_RELEVANT_CONTAINERS_SQL = sql_text("""
SELECT c.id, c.name, min(c.embedding <=> p.embedding) AS d
  FROM graphrag.entities c, graphrag.entities p
 WHERE c.id = ANY(CAST(:containers AS uuid[]))
   AND p.id = ANY(CAST(:batch AS uuid[]))
   AND c.embedding IS NOT NULL
   AND p.embedding IS NOT NULL
 GROUP BY c.id, c.name
 ORDER BY d ASC
 LIMIT :k
""")


async def _relevant_containers(
    session: Any,
    batch_ids: list[Any],
    container_ids: list[Any],
    k: int,
) -> set[Any]:
    """The `k` containers closest in embedding space to any place in `batch`.

    Containers ride along in every batch, so they cannot all be included once a
    corpus has more than a few dozen. The previous rule took a fixed
    ALPHABETICAL prefix, which silently withheld everything after the cut from
    every batch: with 153 container-level places and a cap of 80, `Republic of
    India` sat at position 109 and was never shown to the model at all. Indian
    cities were therefore linked to `Asia` -- the best container they were
    offered -- and "Which cities are in India?" had no edge to walk.

    Selecting per batch instead means a batch of Indian cities is offered
    India, and a batch of European ones is offered European containers. It uses
    the entity embeddings already stored by extraction, so it costs no API
    call.
    """
    if not batch_ids or not container_ids:
        return set()
    rows = await session.execute(_RELEVANT_CONTAINERS_SQL, {
        "containers": [str(c) for c in container_ids],
        "batch": [str(b) for b in batch_ids],
        "k": int(k),
    })
    return {r[0] for r in rows.all()}


@dataclass
class OntologyPlan:
    """What the ontology says about the corpus's places, by class IRI."""
    entities_by_class: dict[str, list[Any]] = field(default_factory=dict)
    to_mint: list[str] = field(default_factory=list)
    links: list[tuple[str, str]] = field(default_factory=list)


def plan_ontology_containment(
    places: list[tuple[Any, list[str], int]],
    classes: dict[str, dict[str, Any]],
) -> OntologyPlan:
    """Match `(entity id, names, level)` to classes and walk `part_of` upward.

    `classes` maps IRI -> {label, part_of: [IRI]}. A chain stops at a class that
    is absent, so a pruned ontology gives a shorter chain rather than a bad edge.

    A city or state never matches, because a name alone cannot say which place
    it is: the US state `Georgia` would otherwise land in Western Asia.
    """
    by_label: dict[str, str] = {}
    for iri, cls in classes.items():
        if cls.get("part_of"):
            by_label.setdefault(_norm(cls.get("label")), iri)
    for iri, cls in classes.items():
        by_label.setdefault(_norm(cls.get("label")), iri)

    plan = OntologyPlan()
    for eid, names, level in places:
        if level in (1, 2):
            continue
        for name in names:
            iri = by_label.get(_norm(name))
            if iri and iri in classes:
                got = plan.entities_by_class.setdefault(iri, [])
                if eid not in got:
                    got.append(eid)
                break

    seen: set[str] = set()
    stack = [i for i, c in classes.items()
             if i in plan.entities_by_class and c.get("part_of")]
    while stack:
        child = stack.pop()
        if child in seen:
            continue
        seen.add(child)
        for parent in classes[child].get("part_of") or []:
            if parent == child or parent not in classes:
                continue
            plan.links.append((child, parent))
            if parent not in plan.entities_by_class and parent not in plan.to_mint:
                plan.to_mint.append(parent)
            stack.append(parent)
    return plan


def _norm(name: str | None) -> str:
    return " ".join("".join(ch if ch.isalnum() else " " for ch in (name or "").lower()).split())


def _container_type(cls: dict[str, Any], classes: dict[str, dict[str, Any]], iri: str) -> str:
    """The class a minted container is typed with: `Continent` for Asia, as
    extraction types countries `Country` rather than `Japan`."""
    for sup in cls.get("superclasses") or []:
        if sup in classes and _level(classes[sup].get("label")):
            return sup
    return iri


_GEO_CLASSES_SQL = sql_text("""
    SELECT id, iri, label,
           extra_metadata -> 'annotations' AS annotations,
           extra_metadata -> 'superclasses' AS superclasses
      FROM graphrag.ontology_classes
     WHERE id = ANY(CAST(:ids AS uuid[]))
""")


async def _geo_classes(session: Any, class_labels: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    ids = await _class_closure(session, class_labels)
    ids -= await _class_closure(session, NON_PLACE_CLASS_LABELS)
    if not ids:
        return {}
    rows = await session.execute(_GEO_CLASSES_SQL, {"ids": [str(i) for i in ids]})
    out: dict[str, dict[str, Any]] = {}
    for cid, iri, label, notes, superclasses in rows.all():
        part_of: list[str] = []
        for key, value in (notes or {}).items():
            if key.endswith("#partOfClass"):
                part_of += [value] if isinstance(value, str) else list(value or [])
        out[iri] = {
            "id": cid, "label": label, "part_of": part_of,
            "superclasses": [s.get("iri") for s in (superclasses or []) if isinstance(s, dict)],
        }
    return out


async def _enrich_from_ontology(
    places: list[tuple[Any, str, str, Any]],
    levels: dict[Any, int],
    class_labels: tuple[str, ...],
    summary: GeoEnrichSummary,
    *,
    dry_run: bool,
    verbose: bool,
    embedder: Any = None,
) -> None:
    from sqlalchemy import select
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    from backend.app.db.embedding_guard import ensure_embedding_model
    from backend.app.db.models.entities import Entity
    from backend.app.services.db_entity_extract import _entity_iri, _normalize_name
    from backend.app.services.embeddings import Embedder

    async with session_scope() as session:
        classes = await _geo_classes(session, class_labels)
        rows = await session.execute(sql_text("""
            SELECT id, extra_metadata -> 'aliases' FROM graphrag.entities
             WHERE id = ANY(CAST(:ids AS uuid[]))
        """), {"ids": [str(p[0]) for p in places]})
        aliases = {r[0]: [a for a in (r[1] or []) if isinstance(a, str)] for r in rows.all()}
    plan = plan_ontology_containment(
        [(eid, [name, *aliases.get(eid, [])], levels.get(cid, _level(label)))
         for eid, name, label, cid in places],
        classes,
    )
    summary.ontology_matched = sum(len(v) for v in plan.entities_by_class.values())
    summary.ontology_nodes = len(plan.to_mint)
    if verbose:
        print(f"[enrich-geo] ontology: {summary.ontology_matched} place(s) matched a "
              f"class, {len(plan.to_mint)} container(s) to add: "
              f"{sorted(classes[i]['label'] for i in plan.to_mint)}")
    if dry_run or not plan.links:
        summary.ontology_edges = 0
        return

    if plan.to_mint:
        embedder = embedder or Embedder()
        await ensure_embedding_model(embedder)
        mints: list[dict[str, Any]] = []
        texts: list[str] = []
        by_identifier: dict[str, str] = {}
        for iri in plan.to_mint:
            cls = classes[iri]
            type_iri = _container_type(cls, classes, iri)
            identifier = _entity_iri(cls["label"], type_iri)
            by_identifier[identifier] = iri
            mints.append({
                "entity_identifier": identifier,
                "name": cls["label"],
                "normalized_name": _normalize_name(cls["label"]),
                "class_id": classes[type_iri]["id"],
                "iri": identifier,
                "status": "ACTIVE",
                "extra_metadata": {"minted_by": "enrich-geo", "ontology_class": iri},
            })
            texts.append(f"{cls['label']} -- {classes[type_iri]['label']}")
        for mint, vec in zip(mints, await embedder.embed(texts), strict=True):
            mint["embedding"] = vec
        summary.embedding_cost_usd = float(getattr(embedder, "total_cost_usd", 0.0) or 0.0)
        async with session_scope() as session:
            await session.execute(
                pg_insert(Entity).values(mints).on_conflict_do_nothing(
                    index_elements=["entity_identifier"]))
            got = await session.execute(
                select(Entity.id, Entity.entity_identifier).where(
                    Entity.entity_identifier.in_(list(by_identifier))))
            for eid, identifier in got.all():
                plan.entities_by_class[by_identifier[identifier]] = [eid]

    pairs: list[tuple[Any, Any, str]] = []
    for child, parent in plan.links:
        targets = plan.entities_by_class.get(parent) or []
        if not targets:
            continue
        for eid in plan.entities_by_class.get(child) or []:
            pairs.append((eid, targets[0],
                          f"{classes[child]['label']} is in {classes[parent]['label']}"))
    summary.samples = [p[2] for p in pairs[:8]]
    async with session_scope() as session:
        summary.ontology_edges, _existing = await _write_containment(
            session, pairs, "enrich-geo-ontology")
        summary.edges_existing += _existing


async def _write_containment(
    session: Any, pairs: list[tuple[Any, Any, str]], found_by: str
) -> tuple[int, int]:
    """Insert the edges not already present. Returns (created, already present)."""
    gv = await current_version(session)
    existing = await session.execute(sql_text("""
        SELECT source_node_id, target_node_id
          FROM graphrag.graph_relationships
         WHERE predicate_iri = :p
    """), {"p": GRAPHRAG_LOCATED_IN})
    have = {(a, b) for a, b in existing.all()}
    created = present = 0
    for child_id, parent_id, text_ in pairs:
        if (child_id, parent_id) in have:
            present += 1
            continue
        have.add((child_id, parent_id))
        await session.execute(sql_text("""
            INSERT INTO graphrag.graph_relationships
              (source_node_type, source_node_id, target_node_type,
               target_node_id, predicate_iri, predicate_label,
               relationship_type, relationship_source, is_authoritative,
               graph_version, extra_metadata)
            VALUES
              ('entity', :source_node_id, 'entity',
               :target_node_id, :predicate_iri, 'locatedInPlace',
               'locatedInPlace', 'LLM_INFERENCE', false,
               :graph_version, CAST(:extra_metadata AS jsonb))
        """), {
            "source_node_id": child_id,
            "target_node_id": parent_id,
            "predicate_iri": GRAPHRAG_LOCATED_IN,
            "graph_version": gv,
            "extra_metadata": json.dumps({
                # Marks the edge as NOT sourced from any document.
                "evidence_kind": "world_knowledge",
                "found_by": found_by,
                "relation": "is located in",
                "evidence": text_,
            }),
        })
        created += 1
    return created, present


async def enrich_geography(
    *,
    class_labels: tuple[str, ...] = DEFAULT_GEO_CLASS_LABELS,
    batch_size: int = _BATCH,
    max_containers: int = _MAX_CONTAINERS,
    dry_run: bool = False,
    limit: int | None = None,
    verbose: bool = False,
    router: LLMRouter | None = None,
    ontology_only: bool = False,
    embedder: Any = None,
) -> GeoEnrichSummary:
    """Add the ontology's containers and containment, then link the rest with
    the model. `ontology_only` stops before the model, for an existing graph."""
    summary = GeoEnrichSummary()

    async def _load() -> tuple[list[tuple[Any, str, str, Any]], dict[Any, int]]:
        async with session_scope() as session:
            found = await _place_entities(session, class_labels)
            if limit:
                found = found[:limit]
            # Level per class, resolved through the subClassOf chain -- see
            # `_effective_levels` for why the label alone is not enough.
            return found, await _effective_levels(
                session, list({cid for _e, _n, _l, cid in found})
            )

    places, levels = await _load()
    if places:
        await _enrich_from_ontology(places, levels, class_labels, summary,
                                    dry_run=dry_run, verbose=verbose, embedder=embedder)
    if summary.ontology_nodes and not dry_run:
        # The model pass may link other places to the new containers.
        places, levels = await _load()
    summary.place_entities = len(places)
    if ontology_only:
        return summary
    router = router or LLMRouter()
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
    all_containers = [
        (eid, {"name": n, "type": lbl}) for eid, n, lbl, cid in places
        if levels.get(cid, _level(lbl)) >= 3
    ]
    contained_full = [
        (eid, {"name": n, "type": lbl}) for eid, n, lbl, cid in places
        if levels.get(cid, _level(lbl)) < 3
    ]
    contained = [d for _e, d in contained_full]
    # (container names are already in `by_name`; no separate index needed)
    if verbose:
        print(f"[enrich-geo] {len(all_containers)} container place(s), "
              f"{len(contained)} contained place(s); "
              f"<= {max_containers} offered per batch, by relevance")
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

    # Containers are chosen PER BATCH by embedding proximity to that batch's
    # places, not by an alphabetical prefix of the whole list. See
    # `_relevant_containers`: the fixed prefix silently withheld every
    # container after the cut, so Indian cities were only ever offered `Asia`.
    cont_ids = [eid for eid, _d in all_containers]
    cont_by_id = {eid: d for eid, d in all_containers}
    batches: list[list[dict[str, str]]] = []
    if len(all_containers) > 1:
        # Container-to-container containment (Hesse -> Germany) still needs a
        # pass of its own, chunked so it is never one huge list either.
        conts_sorted = _sorted([d for _e, d in all_containers])
        batches += [conts_sorted[i : i + max_containers]
                    for i in range(0, len(conts_sorted), max_containers)]
    async with session_scope() as session:
        for i in range(0, len(contained_full), batch_size):
            slice_ = contained_full[i : i + batch_size]
            picked = await _relevant_containers(
                session, [eid for eid, _d in slice_], cont_ids, max_containers,
            )
            batches.append(_sorted(
                [cont_by_id[c] for c in picked if c in cont_by_id]
                + [d for _e, d in slice_]
            ))

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
            # Both ends must already be entities: the model never adds a node.
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
    summary.samples = [p[2] for p in pairs[:8]] or summary.samples
    if dry_run or not pairs:
        return summary

    async with session_scope() as session:
        summary.edges_created, _existing = await _write_containment(
            session, pairs, "enrich-geo")
        summary.edges_existing += _existing
    return summary
