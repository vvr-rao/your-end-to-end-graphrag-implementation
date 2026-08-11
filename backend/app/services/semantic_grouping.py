"""Group semantically-similar items so batching keeps them together.

Batching an LLM dedup pass creates a failure mode the single-call version did
not have: two proposals for the same concept can land in different requests, so
no single call ever sees both and the duplicate survives. A deterministic
post-merge catches only exact string matches -- 'Washington DC' and
'Washington, D.C.' slip through, and so do 'USA' / 'United States'.

Sorting by label fixes the first pair and not the second. Embedding the labels
and clustering by cosine similarity fixes both, because it groups by meaning
rather than spelling. Ordering the items by cluster before batching then makes
cross-batch duplicates rare BY CONSTRUCTION rather than something to clean up
afterwards.

Cost is negligible: ~25 tokens per proposal at text-embedding-3-small
($0.02/1M) is ~$0.002 for a 3,700-proposal run, against ~$5 of dedup calls.
Embeddings run on OpenAI in every provider mode, so this works in
anthropic-only mode too (which already requires OPENAI_API_KEY).
"""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import TypeVar

from backend.app.services.embeddings import Embedder

T = TypeVar("T")

# Cosine similarity above which two items are treated as the same concept for
# GROUPING purposes only. This threshold does not decide dedup -- the LLM still
# makes that call. It only decides co-location in a batch, so erring loose is
# cheap (a few unrelated neighbours share a request) while erring tight is not
# (a genuine duplicate pair gets split across requests).
_DEFAULT_THRESHOLD = 0.82

# Above this many items, skip clustering. The greedy pass below is O(n * k) in
# clusters formed; on very large sets the embedding call and the comparisons
# stop being free. Falls back to label sort, which still beats arbitrary order.
_MAX_CLUSTER_ITEMS = 20_000


def _norm(vec: Sequence[float]) -> float:
    return math.sqrt(sum(v * v for v in vec)) or 1.0


def _cosine(a: Sequence[float], b: Sequence[float], na: float, nb: float) -> float:
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


async def order_by_semantic_cluster(
    items: Sequence[T],
    render: Callable[[T], str],
    *,
    threshold: float = _DEFAULT_THRESHOLD,
    embedder: Embedder | None = None,
) -> list[T]:
    """Return `items` reordered so similar items are adjacent.

    Groups by embedding cosine similarity using a single greedy pass: each item
    joins the first cluster whose centroid it is close enough to, else starts
    its own. Greedy rather than agglomerative because the goal is only
    co-location in a batch, not a defensible taxonomy -- and greedy is one pass
    and deterministic given a fixed input order.

    Never raises: a failure here degrades batch quality, it does not make the
    result wrong, so it falls back to a label sort (which still groups
    prefix-variants) rather than taking down a paid run.
    """
    if len(items) < 2:
        return list(items)

    texts = [render(it) for it in items]

    if len(items) > _MAX_CLUSTER_ITEMS:
        return _sorted_fallback(items, texts)

    try:
        vectors = await (embedder or Embedder()).embed(list(texts))
    except Exception as exc:  # noqa: BLE001 -- degrade, never fail the run
        print(f"[dedup-cluster] embedding failed ({exc}); falling back to label sort")
        return _sorted_fallback(items, texts)

    if len(vectors) != len(items):
        return _sorted_fallback(items, texts)

    norms = [_norm(v) for v in vectors]
    centroids: list[tuple[list[float], float]] = []   # (vector, norm)
    members: list[list[int]] = []

    for i, vec in enumerate(vectors):
        best, best_sim = -1, threshold
        for c, (cvec, cnorm) in enumerate(centroids):
            sim = _cosine(vec, cvec, norms[i], cnorm)
            if sim >= best_sim:
                best, best_sim = c, sim
        if best < 0:
            centroids.append((list(vec), norms[i]))
            members.append([i])
        else:
            members[best].append(i)
            # Nudge the centroid toward the new member so a drifting cluster
            # keeps matching later arrivals.
            cvec, _ = centroids[best]
            k = len(members[best])
            for d in range(len(cvec)):
                cvec[d] += (vec[d] - cvec[d]) / k
            centroids[best] = (cvec, _norm(cvec))

    # Largest clusters first: they are the ones whose splitting would cost the
    # most, so they get placed while batches are still empty.
    order = sorted(range(len(members)), key=lambda c: -len(members[c]))
    out: list[T] = []
    for c in order:
        for i in members[c]:
            out.append(items[i])
    n_multi = sum(1 for m in members if len(m) > 1)
    if n_multi:
        print(
            f"[dedup-cluster] {len(items)} item(s) -> {len(members)} cluster(s), "
            f"{n_multi} with >1 member (similar items will share a batch)"
        )
    return out


def _sorted_fallback(items: Sequence[T], texts: Sequence[str]) -> list[T]:
    """Label sort: weaker than clustering but still groups prefix-variants."""
    return [it for _, it in sorted(zip(texts, items), key=lambda p: p[0].lower())]
