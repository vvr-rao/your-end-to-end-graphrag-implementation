"""The vectorised clusterer must match the reference one, and be fast.

Background: the first version compared every (item, centroid) pair with a
pure-Python generator expression over 1024 floats. Measured at 76 microseconds
per comparison, an 11,310-proposal dedup spent 25+ minutes pinned at ~80% of a
core -- and because that stage prints no progress, it was indistinguishable
from a network stall.

Correctness is the thing to protect here: this reorders items for batching, and
a subtle divergence would silently change which near-duplicates share a request
without ever failing anything.
"""

from __future__ import annotations

import math
import random
import time

from backend.app.services.semantic_grouping import _cosine, _greedy_cluster, _norm


def _reference(vectors, threshold):
    """The original pure-Python loop, verbatim, as the oracle."""
    norms = [_norm(v) for v in vectors]
    centroids: list[tuple[list[float], float]] = []
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
            cvec, _ = centroids[best]
            k = len(members[best])
            for d in range(len(cvec)):
                cvec[d] += (vec[d] - cvec[d]) / k
            centroids[best] = (cvec, _norm(cvec))
    return members


def _rand(n, dim=64, seed=0):
    rng = random.Random(seed)
    return [[rng.gauss(0, 1) for _ in range(dim)] for _ in range(n)]


def test_matches_the_reference_on_random_vectors() -> None:
    v = _rand(150, dim=64, seed=1)
    assert _greedy_cluster(v, 0.3) == _reference(v, 0.3)


def test_matches_the_reference_when_everything_clusters() -> None:
    """A low threshold collapses almost everything -- exercises the centroid
    drift update, which is where a running-mean bug would hide."""
    v = _rand(120, dim=32, seed=2)
    assert _greedy_cluster(v, -1.0) == _reference(v, -1.0)


def test_matches_the_reference_when_nothing_clusters() -> None:
    v = _rand(80, dim=32, seed=3)
    assert _greedy_cluster(v, 0.999) == _reference(v, 0.999)


def test_identical_vectors_tie_break_the_same_way() -> None:
    """The real reason tie-breaking matters: duplicate proposal texts embed
    identically, so exact ties are common, not theoretical. The reference scans
    ascending with `>=`, so the LAST maximal centroid wins; np.argmax would
    pick the first."""
    base = _rand(4, dim=16, seed=4)
    v = base + base + base          # every vector has exact duplicates
    assert _greedy_cluster(v, 0.5) == _reference(v, 0.5)


def test_zero_vector_does_not_divide_by_zero() -> None:
    """`_norm` returns 1.0 for a zero vector; the vectorised path must too."""
    v = [[0.0] * 16, [1.0] + [0.0] * 15, [0.0] * 16]
    assert _greedy_cluster(v, 0.5) == _reference(v, 0.5)


def test_every_item_lands_in_exactly_one_cluster() -> None:
    v = _rand(200, dim=32, seed=5)
    flat = [i for cluster in _greedy_cluster(v, 0.25) for i in cluster]
    assert sorted(flat) == list(range(200))


def test_capacity_growth_survives_more_clusters_than_initial_capacity() -> None:
    """Centroids are preallocated and doubled; a set where nearly every item
    starts its own cluster forces several resizes."""
    v = _rand(300, dim=16, seed=6)
    m = _greedy_cluster(v, 0.999)
    assert len(m) > 16, "expected many singleton clusters to exercise growth"
    assert sorted(i for c in m for i in c) == list(range(300))


def test_is_fast_enough_at_realistic_scale() -> None:
    """11,310 proposals at 1024 dimensions is the shape that took 25+ minutes.
    Scaled down to keep CI quick, but at a size where the pure-Python version
    would already be visibly slow."""
    v = _rand(1200, dim=1024, seed=7)
    t = time.perf_counter()
    _greedy_cluster(v, 0.30)
    elapsed = time.perf_counter() - t
    assert elapsed < 20.0, f"clustering 1200x1024 took {elapsed:.1f}s"
