"""Rank fusion + evidence diversification + time-span rendering.

Pure-unit; no DB, no LLM, no cost.

`rrf_fuse` had no tests at all despite being the final arbiter of what
reaches the answer. Two defects motivated these:

1. **Temporal (2026-08-21).** `fetch_chunk_time_labels` listed a chunk's
   first N periods in ascending date order, so a chunk spanning
   2011-2024 rendered as "2011, 2013, 2014, 2019" -- the recent end
   truncated away. Forward-bounded questions ("after 2023") then returned
   confident false denials while their backward twins passed.

2. **Concentration.** One document supplied all 30 evidence items while
   20 documents carried the relevant section. RRF's defence is agreement
   between independent lists, but all three chunk voters run over the same
   candidate pool -- when it is concentrated they agree rather than
   cross-check.
"""
from __future__ import annotations

import uuid

import pytest

from backend.app.services.retrieval_ranking import cap_per_source, rrf_fuse
from backend.app.services.retrieval_sql import _render_time_span


def _ids(n: int) -> list[uuid.UUID]:
    return [uuid.uuid4() for _ in range(n)]


# --------------------------------------------------------------------------
# rrf_fuse -- first tests this function has ever had
# --------------------------------------------------------------------------


def test_rrf_rewards_appearing_in_more_lists():
    a, b, c = _ids(3)
    fused = rrf_fuse([[a, b], [a, c], [a]])
    assert fused[0][0] == a


def test_rrf_rewards_a_higher_rank():
    a, b = _ids(2)
    fused = rrf_fuse([[a, b], [a, b]])
    assert [cid for cid, _ in fused] == [a, b]


def test_rrf_ignores_empty_lists():
    a, b = _ids(2)
    assert rrf_fuse([[a, b], [], []]) == rrf_fuse([[a, b]])


def test_rrf_empty_input():
    assert rrf_fuse([]) == []


# --------------------------------------------------------------------------
# cap_per_source
# --------------------------------------------------------------------------


def test_one_source_cannot_monopolise():
    """The reported failure: 30 of 30 evidence items from one document."""
    hog = uuid.uuid4()
    chunks = _ids(30)
    fused = [(c, 1.0 / (i + 1)) for i, c in enumerate(chunks)]
    capped = cap_per_source(fused, {c: hog for c in chunks}, 8)
    assert len(capped) == 8


def test_cap_preserves_relevance_order():
    """Capping must trim the monopoly, not reorder what survives."""
    d1, d2 = uuid.uuid4(), uuid.uuid4()
    chunks = _ids(6)
    source_of = {
        chunks[0]: d1, chunks[1]: d1, chunks[2]: d2,
        chunks[3]: d1, chunks[4]: d2, chunks[5]: d1,
    }
    fused = [(c, 1.0 / (i + 1)) for i, c in enumerate(chunks)]
    capped = [c for c, _ in cap_per_source(fused, source_of, 2)]
    # d1 keeps its two best (0, 1) and loses 3 and 5; d2 keeps both.
    assert capped == [chunks[0], chunks[1], chunks[2], chunks[4]]


def test_items_with_unknown_source_are_never_capped():
    """Guessing a source would drop real evidence."""
    chunks = _ids(5)
    fused = [(c, 1.0) for c in chunks]
    assert len(cap_per_source(fused, {}, 1)) == 5
    assert len(cap_per_source(fused, {c: None for c in chunks}, 1)) == 5


@pytest.mark.parametrize("cap", [0, -1])
def test_cap_of_zero_disables(cap: int):
    """REGRESSION GUARD: the pre-cap behaviour must remain reachable."""
    src = uuid.uuid4()
    chunks = _ids(10)
    fused = [(c, 1.0) for c in chunks]
    assert cap_per_source(fused, {c: src for c in chunks}, cap) == fused


def test_cap_is_a_noop_when_already_diverse():
    chunks = _ids(5)
    fused = [(c, 1.0) for c in chunks]
    source_of = {c: uuid.uuid4() for c in chunks}
    assert cap_per_source(fused, source_of, 8) == fused


# --------------------------------------------------------------------------
# Time-span rendering
# --------------------------------------------------------------------------


def _rows(*labels: str):
    return [(lab, i) for i, lab in enumerate(labels)]


def test_single_period_renders_verbatim():
    assert _render_time_span(_rows("May 2023"), 4) == "May 2023"


def test_short_list_renders_in_full():
    assert _render_time_span(_rows("2019", "2022", "2023", "2024"), 4) == (
        "2019, 2022, 2023, 2024"
    )


def test_long_list_keeps_both_endpoints():
    """THE regression. Ascending-truncation dropped the recent end, which
    made every chunk look out-of-window for a forward bound and
    in-window for a backward one -- the exact asymmetry reported."""
    rendered = _render_time_span(
        _rows("2011", "2013", "2014", "2019", "2020", "2022", "2023", "2024"), 4
    )
    assert rendered == "2011 ... 2024 (8 periods)"
    assert "2024" in rendered, "the recent end must survive truncation"
    assert "2011" in rendered, "the early end must survive too"


def test_span_is_directionally_neutral():
    """Neither bound direction may be starved of its relevant endpoint."""
    rendered = _render_time_span(_rows(*[str(y) for y in range(2010, 2025)]), 4)
    assert rendered.startswith("2010") and "2024" in rendered


def test_no_periods_renders_empty():
    assert _render_time_span([], 4) == ""
