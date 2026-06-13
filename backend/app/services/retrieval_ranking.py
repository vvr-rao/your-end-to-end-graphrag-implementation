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
