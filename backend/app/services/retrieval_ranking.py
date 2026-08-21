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
    *,
    relevance_ratio: float = 0.9,
) -> list[tuple[uuid.UUID, float]]:
    """Limit how many items one source (document) contributes -- unless
    nothing better is available.

    Why a cap at all: RRF's usual defence is consensus across independent
    lists, but the lists it receives here are three views of ONE candidate
    pool -- probe reranks, the graph ranking and the document arm all run
    over the same ids. When that pool is concentrated they agree rather
    than cross-check. Measured: with the cap off, "How do GLP-1 receptor
    medications work?" draws 30 of 30 evidence chunks from a single
    document while 20 documents carry a MECHANISM OF ACTION section.

    Why not a plain cap: it diversifies unconditionally, so it will
    displace an excellent chunk with a much weaker one purely to hit a
    quota. That is wrong for corpora of large documents -- "the dosage and
    administration instructions for ZEPBOUND" legitimately lives in one
    label, and a 200-page 10-K legitimately answers most of a question
    about that filing. Capping such a query spread it over 7 documents
    when 2 would do.

    So the cap yields when there is no real alternative. A competing
    source counts as a real alternative only if its best unselected item
    scores at least `relevance_ratio` x this item's score. Past the limit
    an item is kept when NO such alternative exists -- i.e. everything
    else on offer is materially worse.

    `relevance_ratio` therefore reads as "how good must a rival be to
    count": 0.9 means within 90%. HIGHER is stricter about what counts as
    competition, so one source keeps more slots; LOWER diversifies more
    eagerly; 0.0 makes the cap absolute.

    Runs after fusion, so relevance still decides ordering. Items with an
    unknown source are never capped -- guessing would drop real evidence.
    `max_per_source <= 0` disables the whole thing.
    """
    if max_per_source <= 0:
        return fused

    # Best remaining score per source, so "is there a real alternative?"
    # is answerable without rescanning the tail for every candidate.
    remaining: defaultdict[uuid.UUID, list[float]] = defaultdict(list)
    for cid, score in fused:
        src = source_of.get(cid)
        if src is not None:
            remaining[src].append(score)

    seen: defaultdict[uuid.UUID, int] = defaultdict(int)
    consumed: defaultdict[uuid.UUID, int] = defaultdict(int)
    kept: list[tuple[uuid.UUID, float]] = []

    for cid, score in fused:
        src = source_of.get(cid)
        if src is None:
            kept.append((cid, score))
            continue
        consumed[src] += 1
        if seen[src] < max_per_source:
            seen[src] += 1
            kept.append((cid, score))
            continue
        # Over the limit: keep only if no other source offers a genuine
        # alternative -- one scoring within `relevance_ratio` of this item.
        best_other = 0.0
        for other, scores in remaining.items():
            if other == src:
                continue
            idx = seen[other]
            if idx < len(scores):
                best_other = max(best_other, scores[idx])
        if best_other < relevance_ratio * score:
            seen[src] += 1
            kept.append((cid, score))
    return kept
