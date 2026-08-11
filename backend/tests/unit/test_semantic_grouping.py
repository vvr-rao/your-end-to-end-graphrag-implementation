"""Similar proposals share a batch, so batching does not reintroduce duplicates.

Batching stage-3 dedup created a failure mode the single-call version lacked:
the LLM can only collapse what one request contains, and the deterministic
post-merge catches exact string matches only. So 'USA' in batch 1 and
'United States' in batch 7 would both survive.

Ordering by embedding similarity before batching makes that rare by
construction. A label sort would fix 'Washington DC' / 'Washington, D.C.' but
not 'USA' / 'United States' -- that difference is the point of this module.
"""

from __future__ import annotations

import asyncio

from backend.app.services.semantic_grouping import (
    _sorted_fallback,
    order_by_semantic_cluster,
)


class _FakeEmbedder:
    """Deterministic stand-in: items sharing a tag get near-identical vectors."""

    def __init__(self, tags: dict[str, int], dim: int = 8, fail: bool = False) -> None:
        self._tags = tags
        self._dim = dim
        self._fail = fail
        self.calls = 0

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        if self._fail:
            raise RuntimeError("embeddings unavailable")
        out = []
        for t in texts:
            tag = next((v for k, v in self._tags.items() if k in t), -1)
            vec = [0.0] * self._dim
            if tag >= 0:
                vec[tag % self._dim] = 1.0
            else:
                vec[0] = 0.01
                vec[self._dim - 1] = 1.0
            out.append(vec)
        return out


def _run(items, tags, **kw):
    emb = _FakeEmbedder(tags)
    ordered = asyncio.run(
        order_by_semantic_cluster(items, render=lambda s: s, embedder=emb, **kw)
    )
    return ordered, emb


def test_semantically_similar_items_become_adjacent() -> None:
    items = ["USA", "Aspirin", "United States", "Ibuprofen", "America"]
    tags = {"USA": 1, "United States": 1, "America": 1, "Aspirin": 2, "Ibuprofen": 2}
    ordered, _ = _run(items, tags)

    pos = {s: i for i, s in enumerate(ordered)}
    country = sorted(pos[s] for s in ("USA", "United States", "America"))
    drug = sorted(pos[s] for s in ("Aspirin", "Ibuprofen"))
    # each group occupies a contiguous run
    assert country == list(range(country[0], country[0] + 3))
    assert drug == list(range(drug[0], drug[0] + 2))


def test_catches_what_a_label_sort_would_miss() -> None:
    """The motivating case: synonyms that are nowhere near each other
    alphabetically. A label sort puts 'Malaria' between them; clustering
    does not."""
    items = ["Myocardial Infarction", "Malaria", "Nigeria", "Heart Attack", "Tuberculosis"]
    tags = {"Myocardial Infarction": 1, "Heart Attack": 1, "Malaria": 2, "Tuberculosis": 2}

    sorted_order = _sorted_fallback(items, items)
    si = {s: i for i, s in enumerate(sorted_order)}
    assert abs(si["Heart Attack"] - si["Myocardial Infarction"]) > 1, (
        "sort leaves the synonym pair separated"
    )

    ordered, _ = _run(items, tags)
    ci = {s: i for i, s in enumerate(ordered)}
    assert abs(ci["Heart Attack"] - ci["Myocardial Infarction"]) == 1, (
        "clustering puts them in the same batch"
    )


def test_every_item_is_preserved_exactly_once() -> None:
    items = [f"item-{i}" for i in range(30)]
    ordered, _ = _run(items, {})
    assert sorted(ordered) == sorted(items)
    assert len(ordered) == len(items)


def test_embedding_failure_degrades_to_sort_rather_than_raising() -> None:
    """A grouping failure must never take down a paid run -- it only affects
    batch quality, not correctness."""
    items = ["Zebra", "Apple", "Mango"]
    emb = _FakeEmbedder({}, fail=True)
    ordered = asyncio.run(
        order_by_semantic_cluster(items, render=lambda s: s, embedder=emb)
    )
    assert ordered == ["Apple", "Mango", "Zebra"]


def test_trivial_inputs_skip_the_embedding_call() -> None:
    for items in ([], ["only-one"]):
        emb = _FakeEmbedder({})
        out = asyncio.run(
            order_by_semantic_cluster(items, render=lambda s: s, embedder=emb)
        )
        assert list(out) == list(items)
        assert emb.calls == 0, "no spend on inputs that cannot be grouped"


def test_larger_clusters_are_placed_first() -> None:
    # Splitting a big cluster costs more, so it should be laid down first.
    items = ["a1", "a2", "a3", "b1", "solo"]
    tags = {"a": 1, "b": 2}
    ordered, _ = _run(items, tags)
    assert ordered[0].startswith("a")


def test_threshold_controls_grouping_strictness() -> None:
    items = ["USA", "United States"]
    tags = {"USA": 1, "United States": 1}
    # An impossible threshold must not crash; it just yields singletons.
    ordered = asyncio.run(
        order_by_semantic_cluster(
            items, render=lambda s: s, embedder=_FakeEmbedder(tags), threshold=1.01
        )
    )
    assert sorted(ordered) == sorted(items)
