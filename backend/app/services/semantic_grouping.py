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
    """Reference implementation. Kept because the vectorised path below is
    tested against it -- it is the definition of what that path must reproduce.
    """
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def _greedy_cluster(vectors: Sequence[Sequence[float]], threshold: float) -> list[list[int]]:
    """Greedy single-pass clustering; returns member indices per cluster.

    Vectorised because the obvious Python version is a performance bug at real
    scale, and it shipped as one. Each comparison is a 1024-dimension dot
    product, and the greedy pass does one per (item, cluster) pair: at 11,310
    proposals that is millions of them. Measured at 76 microseconds each in
    pure Python, dedup spent 25+ minutes pinned at ~80% of a core doing
    arithmetic that numpy does in seconds -- and it looked like a network stall,
    because the stage has no progress output.

    numpy is depended on explicitly for this (it was already present via openai
    and pgvector, but relying on a transitive dependency for correctness is
    fragile).

    Semantics are preserved exactly, including the tie-break: the original used
    `sim >= best_sim` while scanning ascending, so the LAST maximal centroid
    wins. That is not hypothetical -- duplicate proposal texts embed identically
    and tie constantly. np.argmax returns the FIRST maximum, so ties are
    resolved explicitly here.
    """
    import numpy as np

    mat = np.asarray(vectors, dtype=np.float64)
    n_items, dim = mat.shape

    norms = np.linalg.norm(mat, axis=1)
    norms[norms == 0] = 1.0          # mirrors `_norm`'s `or 1.0`

    # Centroids preallocated and grown geometrically; appending row-by-row to a
    # numpy array reallocates the whole block each time.
    cap = max(16, n_items // 8)
    cents = np.empty((cap, dim), dtype=np.float64)
    cnorms = np.empty(cap, dtype=np.float64)
    n_cents = 0
    members: list[list[int]] = []

    for i in range(n_items):
        vec = mat[i]
        best = -1
        if n_cents:
            # One matrix-vector product replaces the inner Python loop.
            sims = (cents[:n_cents] @ vec) / (cnorms[:n_cents] * norms[i])
            top = float(sims.max())
            if top >= threshold:
                # Last index attaining the max -- matches the `>=` scan.
                best = int(np.flatnonzero(sims == top)[-1])

        if best < 0:
            if n_cents == cap:
                cap *= 2
                cents = np.resize(cents, (cap, dim))
                cnorms = np.resize(cnorms, cap)
            cents[n_cents] = vec
            cnorms[n_cents] = norms[i]
            members.append([i])
            n_cents += 1
        else:
            members[best].append(i)
            # Nudge the centroid toward the new member so a drifting cluster
            # keeps matching later arrivals.
            k = len(members[best])
            cents[best] += (vec - cents[best]) / k
            cn = float(np.linalg.norm(cents[best]))
            cnorms[best] = cn or 1.0

    return members


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

    members = _greedy_cluster(vectors, threshold)

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
