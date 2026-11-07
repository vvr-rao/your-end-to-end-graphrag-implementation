"""Corpus evaluation: how well the built graph covers the corpus.

Run after every graph build (`extract-entities` runs it at the end) and on
demand with `eval-corpus`. Free: database queries only.

Entities are split three ways (see `common.entity_kinds`): concepts (typed under
`extraction.concept_class_roots`, minus those judged named things), places
(typed under a geographic class) and other named entities. "Entities" in the
report means named entities INCLUDING places; places are also shown alone.

Metrics, per chunk kind (summary / fulltext):
  * entities and concepts per chunk (mean, median), from `viao:assertsAbout`
    mention edges;
  * chunks with no entities and no concepts (and with none of each);
  * chunks by how many they mention: 0, 1, 2, 3, 4, 5 and more than 5, counted
    for entities and concepts together, entities alone and concepts alone;
Graph-wide:
  * entity-to-entity relationships, split entity-entity / entity-concept /
    concept-concept, with the place-containment edges counted on their own;
  * average entities per class (classes with at least one entity) and the
    classes with more than 50 entities;
  * entities and concepts with 0, 1, 2, 3 and more than 3 relationships.
"""
from __future__ import annotations

import uuid
from collections import Counter, defaultdict
from typing import Any

from sqlalchemy import text

from backend.app.evals.common import buckets, count_buckets, entity_kinds, mean, median
from backend.app.services.predicates import GRAPHRAG_LOCATED_IN

BIG_CLASS = 50

_CHUNKS_SQL = text("SELECT id, kind FROM graphrag.chunks WHERE status = 'ACTIVE'")
_MENTIONS_SQL = text("""
    SELECT DISTINCT source_node_id, target_node_id
      FROM graphrag.graph_relationships
     WHERE source_node_type = 'chunk' AND target_node_type = 'entity'
       AND predicate_label = 'viao:assertsAbout'
""")
_EDGES_SQL = text("""
    SELECT DISTINCT source_node_id, target_node_id, predicate_iri
      FROM graphrag.graph_relationships
     WHERE source_node_type = 'entity' AND target_node_type = 'entity'
       AND source_node_id <> target_node_id
""")
_CLASS_LABELS_SQL = text("SELECT id, label FROM graphrag.ontology_classes")


def _side(kind: str) -> str:
    return "concept" if kind == "concept" else "entity"


async def corpus_eval(session: Any) -> dict[str, Any]:
    kinds = await entity_kinds(session)
    chunks = {cid: kind for cid, kind in (await session.execute(_CHUNKS_SQL)).all()}
    labels = {cid: lbl for cid, lbl in (await session.execute(_CLASS_LABELS_SQL)).all()}

    per_chunk: dict[uuid.UUID, Counter] = defaultdict(Counter)
    for chunk_id, entity_id in (await session.execute(_MENTIONS_SQL)).all():
        k = kinds.get(entity_id)
        if chunk_id in chunks and k:
            per_chunk[chunk_id][k[0]] += 1

    chunk_report: dict[str, Any] = {}
    for ckind in sorted(set(chunks.values())):
        ids = [c for c, k in chunks.items() if k == ckind]
        ents = [per_chunk[c]["named"] + per_chunk[c]["place"] for c in ids]
        places = [per_chunk[c]["place"] for c in ids]
        cons = [per_chunk[c]["concept"] for c in ids]
        chunk_report[ckind] = {
            "chunks": len(ids),
            "entities_per_chunk_mean": mean(ents),
            "entities_per_chunk_median": median(ents),
            "places_per_chunk_mean": mean(places),
            "concepts_per_chunk_mean": mean(cons),
            "concepts_per_chunk_median": median(cons),
            "chunks_without_entities_or_concepts": sum(
                1 for e, c in zip(ents, cons, strict=True) if e == 0 and c == 0),
            "chunks_without_entities": sum(1 for e in ents if e == 0),
            "chunks_without_concepts": sum(1 for c in cons if c == 0),
            "chunks_by_count": {
                "entities_and_concepts": count_buckets(
                    [e + c for e, c in zip(ents, cons, strict=True)]),
                "entities": count_buckets(ents),
                "concepts": count_buckets(cons),
            },
        }

    rels = Counter()
    degree: Counter = Counter()
    for src, dst, predicate in (await session.execute(_EDGES_SQL)).all():
        a, b = kinds.get(src), kinds.get(dst)
        if not a or not b:
            continue
        pair = "-".join(sorted((_side(a[0]), _side(b[0]))))
        rels[pair] += 1
        if predicate == GRAPHRAG_LOCATED_IN:
            rels["place_containment"] += 1
        degree[src] += 1
        degree[dst] += 1

    per_class: dict[str, Counter] = {"entity": Counter(), "concept": Counter()}
    for _eid, (k, cid) in kinds.items():
        per_class[_side(k)][cid] += 1
    class_report: dict[str, Any] = {}
    for side, counts in per_class.items():
        big = sorted(((labels.get(c, str(c)), n) for c, n in counts.items() if n > BIG_CLASS),
                     key=lambda t: -t[1])
        class_report[side] = {
            "classes_with_entities": len(counts),
            "entities_per_class_mean": mean(list(counts.values())),
            "entities_per_class_median": median(list(counts.values())),
            f"classes_over_{BIG_CLASS}": len(big),
            f"classes_over_{BIG_CLASS}_list": big[:25],
        }
    all_counts = per_class["entity"] + per_class["concept"]
    class_report["all"] = {
        "classes_with_entities": len(all_counts),
        "entities_per_class_mean": mean(list(all_counts.values())),
        f"classes_over_{BIG_CLASS}": sum(1 for n in all_counts.values() if n > BIG_CLASS),
    }

    totals = Counter(k for k, _ in kinds.values())
    return {
        "entities": totals["named"] + totals["place"],
        "places": totals["place"],
        "concepts": totals["concept"],
        "chunks": chunk_report,
        "relationships": {
            "total": rels["entity-entity"] + rels["concept-entity"] + rels["concept-concept"],
            "entity_entity": rels["entity-entity"],
            "entity_concept": rels["concept-entity"],
            "concept_concept": rels["concept-concept"],
            "of_which_place_containment": rels["place_containment"],
        },
        "classes": class_report,
        "relationships_per_node": {
            "entities": buckets([degree[e] for e, (k, _) in kinds.items() if k != "concept"]),
            "concepts": buckets([degree[e] for e, (k, _) in kinds.items() if k == "concept"]),
        },
    }


def format_corpus_eval(r: dict[str, Any]) -> str:
    lines = [f"entities {r['entities']} (of which places {r['places']}), concepts {r['concepts']}"]
    for ckind, c in r["chunks"].items():
        lines.append(
            f"{ckind} chunks {c['chunks']}: entities/chunk mean {c['entities_per_chunk_mean']} "
            f"median {c['entities_per_chunk_median']} (places {c['places_per_chunk_mean']}); "
            f"concepts/chunk mean {c['concepts_per_chunk_mean']} "
            f"median {c['concepts_per_chunk_median']}")
        lines.append(
            f"  without entities or concepts {c['chunks_without_entities_or_concepts']}; "
            f"without entities {c['chunks_without_entities']}; "
            f"without concepts {c['chunks_without_concepts']}")
        for what, b in (c.get("chunks_by_count") or {}).items():
            lines.append(f"  chunks by {what.replace('_', ' ')}: "
                         + ", ".join(f"{k}: {v}" for k, v in b.items()))
    rel = r["relationships"]
    lines.append(
        f"relationships {rel['total']}: entity-entity {rel['entity_entity']}, "
        f"entity-concept {rel['entity_concept']}, concept-concept {rel['concept_concept']} "
        f"(place containment {rel['of_which_place_containment']})")
    for side in ("entity", "concept", "all"):
        c = r["classes"][side]
        lines.append(
            f"{side} classes {c['classes_with_entities']}: mean "
            f"{c['entities_per_class_mean']} per class, {c[f'classes_over_{BIG_CLASS}']} "
            f"with >{BIG_CLASS}"
            + (": " + ", ".join(f"{lbl} {n}" for lbl, n in c[f"classes_over_{BIG_CLASS}_list"])
               if c.get(f"classes_over_{BIG_CLASS}_list") else ""))
    for side, b in r["relationships_per_node"].items():
        lines.append(f"{side} by relationship count: "
                     + ", ".join(f"{k}: {v}" for k, v in b.items()))
    return "\n".join(lines)
