#!/usr/bin/env python3
"""Per-chunk entity + concept coverage, and how broad the attached concepts are.
Read-only; no LLM calls.

    uv run python scripts/bench/concept_coverage.py --dsn postgresql://... [--verdicts verdict.csv] [--json out.json]

For each chunk: named entities and concept entities tied to it (viao:assertsAbout),
and chunks with NEITHER. Concept breadth is reported three ways:
  depth   IS-A steps from the concept root to the entity's class (0 = typed directly to a root)
  spread  share of summary chunks that concept is tied to
  class   how many entities share its class

`--verdicts` (name,class,kind CSV) scores a graph as if those labels were saved.
"""
from __future__ import annotations

import asyncio
import csv
import json
import statistics as st
import sys
from collections import Counter, defaultdict

import asyncpg

ROOTS = ["quantity", "condition", "process", "method", "intervention", "outcome", "policyconcept",
         "economicconcept", "technologyconcept", "industry", "infrastructure", "supplychainconcept",
         "financialobservation"]


def _arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def _pct(a, b):
    return round(100 * a / b, 1) if b else 0.0


async def main() -> None:
    conn = await asyncpg.connect(_arg("--dsn"))
    await conn.execute("SET search_path TO graphrag")
    root_ids = {r["id"] for r in await conn.fetch(
        "SELECT id FROM ontology_classes WHERE lower(label) = ANY($1)", ROOTS)}
    children = defaultdict(list)
    for r in await conn.fetch("""SELECT source_node_id c, target_node_id p FROM graph_relationships
        WHERE predicate_label = 'rdfs:subClassOf' AND source_node_type = 'ontology_class'
          AND target_node_type = 'ontology_class'"""):
        children[r["p"]].append(r["c"])
    depth = {r: 0 for r in root_ids}
    frontier = list(root_ids)
    while frontier:
        nxt = []
        for p in frontier:
            for c in children[p]:
                if c not in depth:
                    depth[c] = depth[p] + 1
                    nxt.append(c)
        frontier = nxt
    ents = {r["id"]: r for r in await conn.fetch("""SELECT e.id, e.name, e.class_id, oc.label AS cls,
        e.extra_metadata->>'concept_kind' AS kind FROM entities e JOIN ontology_classes oc ON oc.id = e.class_id
        WHERE e.status = 'ACTIVE'""")}
    verdict = {}
    if _arg("--verdicts"):
        verdict = {(r[0], r[1]): r[2] for r in csv.reader(open(_arg("--verdicts"), encoding="utf-8"))}

    def is_concept(e) -> bool:
        kind = verdict.get((e["name"], e["cls"])) or e["kind"]
        return e["class_id"] in depth and kind != "named"

    chunks = await conn.fetch("SELECT c.id, c.kind, c.document_id, left(c.text, 140) AS head FROM chunks c")
    per = defaultdict(set)
    for r in await conn.fetch("""SELECT source_node_id c, target_node_id e FROM graph_relationships
        WHERE predicate_label = 'viao:assertsAbout' AND source_node_type = 'chunk' AND target_node_type = 'entity'"""):
        if r["e"] in ents:
            per[r["c"]].add(r["e"])
    out = {}
    for kind in sorted({c["kind"] for c in chunks}):
        rows = [c for c in chunks if c["kind"] == kind]
        n = len(rows)
        both = named_only = concept_only = neither = 0
        none_rows, c_counts, n_counts = [], [], []
        for c in rows:
            es = [ents[e] for e in per[c["id"]]]
            nc = sum(is_concept(e) for e in es)
            nn = len(es) - nc
            c_counts.append(nc)
            n_counts.append(nn)
            if nc and nn:
                both += 1
            elif nn:
                named_only += 1
            elif nc:
                concept_only += 1
            else:
                neither += 1
                none_rows.append(c["head"].replace("\n", " "))
        out[kind] = {"chunks": n, "both": both, "both_pct": _pct(both, n), "named_only": named_only,
                     "concept_only": concept_only, "neither": neither, "neither_pct": _pct(neither, n),
                     "median_named": st.median(n_counts), "median_concepts": st.median(c_counts),
                     "neither_sample": none_rows[:12]}
    # breadth, over summary chunks: each chunk->concept link counts once
    summ = {c["id"] for c in chunks if c["kind"] == "summary"}
    ent_chunks = Counter()
    for c in summ:
        for e in per[c]:
            ent_chunks[e] += 1
    cls_size = Counter(e["class_id"] for e in ents.values() if is_concept(e))
    links = [(e, ents[e]) for c in summ for e in per[c] if is_concept(ents[e])]
    dist = Counter(min(depth[e["class_id"]], 5) for _, e in links)
    concepts = [e for e in ents.values() if is_concept(e)]
    spread = sorted(((ent_chunks[e["id"]], e["name"], e["cls"], depth[e["class_id"]]) for e in concepts), reverse=True)
    out["breadth"] = {
        "concept_entities": len(concepts),
        "links": len(links),
        "links_by_depth_pct": {("5+" if k == 5 else str(k)): _pct(v, len(links)) for k, v in sorted(dist.items())},
        "typed_to_root_entities": sum(depth[e["class_id"]] == 0 for e in concepts),
        "median_chunks_per_concept": st.median(s[0] for s in spread) if spread else 0,
        "concepts_in_1_chunk_pct": _pct(sum(s[0] == 1 for s in spread), len(spread)),
        "concepts_over_5pct_of_chunks": sum(s[0] > 0.05 * len(summ) for s in spread),
        "median_class_size": st.median(cls_size[e["class_id"]] for e in concepts) if concepts else 0,
        "broadest": [f"{s[1]} <{s[2]}> {s[0]} chunks, depth {s[3]}" for s in spread[:10]],
        "root_typed_sample": [e["name"] for e in concepts if depth[e["class_id"]] == 0][:15],
    }
    print(json.dumps(out, indent=1, ensure_ascii=False))
    if _arg("--json"):
        with open(_arg("--json"), "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1, ensure_ascii=False)
    await conn.close()


asyncio.run(main())
