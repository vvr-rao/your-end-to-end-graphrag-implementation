"""Reciprocal Rank Fusion (RRF) for the Milestone F retrieval pipeline.

RRF combines multiple ranked candidate lists into a single ranking
without needing comparable scores. For each candidate, its RRF score
across N lists is:

    RRF(c) = sum over each list_i of  1 / (k + rank_i(c))

where `rank_i(c)` is 1-indexed (best = 1). `k` is a small constant
(60 by default per the original RRF paper) that dampens the influence
of high ranks.

We feed this multiple kinds of rankings:
  - One ranking per vector-probe (step 9c)
  - The seed-distance ranking (BFS hop -> rank)
  - The entity-coverage ranking (how many query entities the chunk
    asserts about)

Returns a sorted list of (uuid, rrf_score) tuples.
"""
from __future__ import annotations

import uuid
from collections import defaultdict


def rrf_fuse(
    ranked_lists: list[list[uuid.UUID]],
    *,
    k: int = 60,
) -> list[tuple[uuid.UUID, float]]:
    """Reciprocal Rank Fusion. Each `ranked_lists[i]` is an ordered
    list of UUIDs (best first). Returns a single sorted (uuid, score)
    list combining them.

    Candidates appearing in more lists, or higher in any one list, get
    more weight. Empty lists are ignored.
    """
    scores: dict[uuid.UUID, float] = defaultdict(float)
    for lst in ranked_lists:
        for rank, cid in enumerate(lst, start=1):
            scores[cid] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: -kv[1])


def cap_per_source(
    fused: list[tuple[uuid.UUID, float]],
    source_of: dict[uuid.UUID, uuid.UUID | None],
    max_per_source: int,
) -> list[tuple[uuid.UUID, float]]:
    """Limit how many items any single source (document) may contribute.

    RRF's usual defence is consensus across independent lists, but the
    lists it receives here are three views of ONE candidate pool -- probe
    reranks, the graph ranking and the document arm all run over the same
    ids. When that pool is already concentrated they agree rather than
    cross-check, and a single document can supply every evidence slot.
    Observed twice on a 30-document corpus: all 30 evidence chunks from
    one document, while 20 documents carried the relevant section.

    This runs AFTER fusion, so relevance still decides the ordering; the
    cap only stops one source monopolising the tail. Items with an unknown
    source are never capped -- guessing would drop real evidence.

    `max_per_source <= 0` disables it, returning the input unchanged.
    """
    if max_per_source <= 0:
        return fused
    seen: defaultdict[uuid.UUID, int] = defaultdict(int)
    kept: list[tuple[uuid.UUID, float]] = []
    for cid, score in fused:
        src = source_of.get(cid)
        if src is None:
            kept.append((cid, score))
            continue
        if seen[src] >= max_per_source:
            continue
        seen[src] += 1
        kept.append((cid, score))
    return kept
