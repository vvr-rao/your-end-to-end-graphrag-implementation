"""Token-budgeted batching."""

from __future__ import annotations

import pytest

from backend.app.services.token_budget import (
    batch_by_token_budget,
    count_tokens,
    count_tokens_each,
    oversized_indices,
    plan_batches,
)


def test_empty_input_yields_no_batches() -> None:
    assert plan_batches([], max_tokens=100) == []
    assert batch_by_token_budget([], render=str, max_tokens=100) == []


def test_everything_fits_in_one_batch() -> None:
    assert plan_batches([10, 20, 30], max_tokens=100) == [(0, 3)]


def test_no_batch_exceeds_the_budget() -> None:
    counts = [40, 40, 40, 40, 40]
    ranges = plan_batches(counts, max_tokens=100)
    for start, end in ranges:
        assert sum(counts[start:end]) <= 100
    # 40+40=80 fits, +40 would be 120 -> close at 2 per batch
    assert ranges == [(0, 2), (2, 4), (4, 5)]


def test_ranges_are_contiguous_and_cover_everything() -> None:
    counts = [7, 3, 9, 1, 5, 8, 2]
    ranges = plan_batches(counts, max_tokens=10)
    assert ranges[0][0] == 0
    assert ranges[-1][1] == len(counts)
    for (_, prev_end), (next_start, _) in zip(ranges, ranges[1:]):
        assert prev_end == next_start


def test_max_items_caps_batch_length_independently_of_tokens() -> None:
    counts = [1] * 10  # tokens are irrelevant here
    assert plan_batches(counts, max_tokens=1000, max_items=3) == [
        (0, 3),
        (3, 6),
        (6, 9),
        (9, 10),
    ]


def test_oversized_single_item_is_isolated_not_merged() -> None:
    # 500 exceeds the whole budget: it cannot be made to fit, but it must not
    # drag its neighbours into a doomed request either.
    counts = [10, 500, 10]
    assert plan_batches(counts, max_tokens=100) == [(0, 1), (1, 2), (2, 3)]


def test_oversized_indices_reports_the_offenders() -> None:
    assert oversized_indices([10, 500, 10, 900], max_tokens=100) == [1, 3]
    assert oversized_indices([10, 20], max_tokens=100) == []


def test_order_is_preserved_across_batch_boundaries() -> None:
    items = [f"item-{i}" for i in range(20)]
    batches = batch_by_token_budget(items, render=str, max_tokens=12)
    assert [it for b in batches for it in b] == items
    assert len(batches) > 1, "expected the budget to force a split"


def test_render_measures_what_is_actually_sent() -> None:
    # Measuring the raw dict would undercount vs the JSON that goes on the wire.
    import json

    items = [{"label": "x" * 200} for _ in range(10)]
    batches = batch_by_token_budget(
        items, render=lambda d: json.dumps(d), max_tokens=60
    )
    for batch in batches:
        rendered = "".join(json.dumps(d) for d in batch)
        assert count_tokens(rendered) <= 60 or len(batch) == 1


def test_count_tokens_handles_special_token_text() -> None:
    # Corpus text legitimately contains this; encoding must not raise.
    assert count_tokens("<|endoftext|> and more") > 0


def test_count_tokens_each_matches_count_tokens() -> None:
    texts = ["alpha", "beta gamma", ""]
    assert count_tokens_each(texts) == [count_tokens(t) for t in texts]


def test_unknown_encoding_falls_back_rather_than_raising() -> None:
    assert count_tokens("hello", encoding_name="not-a-real-encoding") > 0


@pytest.mark.parametrize("bad", [0, -1])
def test_rejects_nonsense_budgets(bad: int) -> None:
    with pytest.raises(ValueError):
        plan_batches([1, 2], max_tokens=bad)
    with pytest.raises(ValueError):
        plan_batches([1, 2], max_tokens=10, max_items=bad)
