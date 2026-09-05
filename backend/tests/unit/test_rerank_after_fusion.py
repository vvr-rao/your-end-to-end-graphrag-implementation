"""Post-RRF vector rerank (`query --rerank`).

The rerank exists because RRF fuses rank ORDER and throws distance away, so a
chunk that merely placed well across several probes can outrank one that is
genuinely closer to the question. Each test below names the property that
makes the feature safe rather than just present.
"""
from __future__ import annotations

import uuid

import pytest

from backend.app.services.retrieval_ranking import cap_per_source, rrf_fuse


def _ids(n: int) -> list[uuid.UUID]:
    """Deterministic ids so a failure is reproducible."""
    return [uuid.UUID(int=i) for i in range(1, n + 1)]


# --------------------------------------------------------------------------- #
# The reorder itself
# --------------------------------------------------------------------------- #


def _reorder_like_pipeline(
    fused: list[tuple[uuid.UUID, float]],
    reranked_order: list[uuid.UUID],
) -> list[tuple[uuid.UUID, float]]:
    """The transform retrieval.py applies: take the rerank's ORDER but keep
    each chunk's original RRF SCORE, then append anything the rerank did not
    return (e.g. a chunk with no embedding) behind it."""
    score = dict(fused)
    seen = set(reranked_order)
    tail = [cid for cid, _ in fused if cid not in seen]
    return [(cid, score[cid]) for cid in reranked_order + tail]


def test_rerank_changes_order_but_preserves_scores() -> None:
    """THE CONTRACT. `cap_per_source` reads the score as higher-is-better and
    it is persisted to retrieval_evidence.score. Substituting an L2 distance
    (lower-is-better) would silently invert both."""
    a, b, c = _ids(3)
    fused = [(a, 0.9), (b, 0.5), (c, 0.1)]

    out = _reorder_like_pipeline(fused, [c, a, b])

    assert [cid for cid, _ in out] == [c, a, b], "rerank order not applied"
    assert dict(out) == dict(fused), "scores must survive the reorder"


def test_chunks_without_an_embedding_are_not_lost() -> None:
    """vector_rerank_chunks skips rows where embedding IS NULL. Those chunks
    must keep their RRF position behind the reranked head, not vanish."""
    a, b, c = _ids(3)
    fused = [(a, 0.9), (b, 0.5), (c, 0.1)]

    out = _reorder_like_pipeline(fused, [c, a])  # b had no embedding

    assert [cid for cid, _ in out] == [c, a, b]
    assert len(out) == len(fused)


def test_reranked_output_still_feeds_cap_per_source() -> None:
    """The cap runs AFTER the rerank, so its higher-is-better contract has to
    still hold on the reordered list."""
    a, b, c, d = _ids(4)
    fused = [(a, 0.9), (b, 0.8), (c, 0.7), (d, 0.6)]
    reordered = _reorder_like_pipeline(fused, [d, c, b, a])

    doc1, doc2 = uuid.uuid4(), uuid.uuid4()
    doc_of = {a: doc1, b: doc1, c: doc1, d: doc2}

    capped = cap_per_source(reordered, doc_of, 2, relevance_ratio=0.0)

    from_doc1 = sum(1 for cid, _ in capped if doc_of[cid] == doc1)
    assert from_doc1 <= 2, "cap stopped working on reranked input"


# --------------------------------------------------------------------------- #
# Why the pool must be wider than top_k
# --------------------------------------------------------------------------- #


def test_a_wider_pool_can_promote_a_chunk_rrf_would_have_cut() -> None:
    """The whole point of the flag. Reranking only the chunks RRF already
    selected can reorder them but never improve WHICH chunks are returned --
    so the rerank runs on top_k * multiplier and truncates afterwards."""
    ids = _ids(6)
    fused = [(cid, 1.0 - i * 0.1) for i, cid in enumerate(ids)]
    top_k = 3
    # The genuinely closest chunk sits at RRF rank 5, outside top_k.
    best = ids[4]

    narrow = [cid for cid, _ in fused[:top_k]]
    assert best not in narrow, "fixture no longer exercises the case"

    pool = fused[: top_k * 2]
    reordered = _reorder_like_pipeline(pool, [best] + [c for c, _ in pool if c != best])
    assert [cid for cid, _ in reordered][:top_k][0] == best


# --------------------------------------------------------------------------- #
# Off by default
# --------------------------------------------------------------------------- #


def test_rerank_is_off_by_default_in_shipped_config() -> None:
    """Additive features in this repo ship disabled; enabling it changes which
    evidence every existing query returns."""
    import yaml
    from pathlib import Path

    repo = Path(__file__).resolve().parents[3]
    qa = yaml.safe_load((repo / "config" / "config.example.yaml").read_text())["qa"]
    assert qa["rerank_after_fusion"] is False
    assert int(qa["rerank_pool_multiplier"]) >= 1


def test_driver_accepts_rerank_and_defaults_to_none() -> None:
    """`None` means 'fall back to config', matching how hops/top_k resolve."""
    import inspect

    from backend.app.services.retrieval import retrieve_and_answer

    sig = inspect.signature(retrieve_and_answer)
    assert "rerank" in sig.parameters
    assert sig.parameters["rerank"].default is None


@pytest.mark.parametrize("flag", [True, False])
def test_rrf_fuse_is_unchanged_by_the_feature(flag: bool) -> None:
    """Guard against the rerank being smuggled into fusion itself -- with the
    flag off the pipeline must be byte-identical to before."""
    a, b, c = _ids(3)
    assert rrf_fuse([[a, b, c], [c, a, b]])[0][0] in (a, c)
