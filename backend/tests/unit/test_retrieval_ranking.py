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


def test_one_source_keeps_everything_when_it_is_the_only_source():
    """A single-document question must NOT be truncated. Measured: with a
    hard cap, "dosage instructions for ZEPBOUND" was spread over 7
    documents when 2 would do -- the failure mode on a corpus of large
    documents such as 10-K filings."""
    only = uuid.uuid4()
    chunks = _ids(30)
    fused = [(c, 1.0 / (i + 1)) for i, c in enumerate(chunks)]
    capped = cap_per_source(fused, {c: only for c in chunks}, 8)
    assert len(capped) == 30, "no alternative exists; the cap must yield"


def test_monopoly_is_broken_when_alternatives_are_competitive():
    """The reported failure: 30 of 30 from one document while 20 other
    documents carried the relevant section."""
    hog, other = uuid.uuid4(), uuid.uuid4()
    hog_chunks, other_chunks = _ids(20), _ids(20)
    # Interleaved and near-equal in score: real competition.
    fused = [(c, 1.0 - i * 0.001) for i, c in enumerate(hog_chunks)]
    fused += [(c, 0.99 - i * 0.001) for i, c in enumerate(other_chunks)]
    fused.sort(key=lambda t: -t[1])
    src = {**{c: hog for c in hog_chunks}, **{c: other for c in other_chunks}}
    kept = cap_per_source(fused, src, 8)
    from collections import Counter
    counts = Counter(src[c] for c, _ in kept)
    assert counts[hog] <= 8, "a competitive alternative must break the monopoly"


def test_strong_chunk_is_not_displaced_by_a_weak_one():
    """The whole point of the escape hatch: never trade an excellent chunk
    for a much worse one purely to hit a quota."""
    strong, weak = uuid.uuid4(), uuid.uuid4()
    strong_chunks, weak_chunks = _ids(12), _ids(12)
    fused = [(c, 1.0) for c in strong_chunks] + [(c, 0.05) for c in weak_chunks]
    src = {**{c: strong for c in strong_chunks}, **{c: weak for c in weak_chunks}}
    kept = [c for c, _ in cap_per_source(fused, src, 4)]
    assert all(c in kept for c in strong_chunks), (
        "the weak source is not a real alternative; the strong one keeps its chunks"
    )


def test_ratio_of_zero_makes_the_cap_absolute():
    """Nothing counts as competition, so the cap never yields."""
    a, b = uuid.uuid4(), uuid.uuid4()
    ac, bc = _ids(10), _ids(10)
    fused = [(c, 1.0) for c in ac] + [(c, 0.01) for c in bc]
    src = {**{c: a for c in ac}, **{c: b for c in bc}}
    kept = cap_per_source(fused, src, 3, relevance_ratio=0.0)
    from collections import Counter
    assert Counter(src[c] for c, _ in kept)[a] == 3


def test_higher_ratio_keeps_one_source_longer():
    """The knob reads as 'how good must a rival be to count'. Stricter
    (higher) means fewer things qualify as alternatives, so the dominant
    source retains more slots."""
    a, b = uuid.uuid4(), uuid.uuid4()
    ac, bc = _ids(10), _ids(10)
    fused = [(c, 1.0) for c in ac] + [(c, 0.8) for c in bc]
    src = {**{c: a for c in ac}, **{c: b for c in bc}}
    from collections import Counter
    lenient = Counter(src[c] for c, _ in cap_per_source(fused, src, 2, relevance_ratio=0.5))
    strict = Counter(src[c] for c, _ in cap_per_source(fused, src, 2, relevance_ratio=0.9))
    assert strict[a] > lenient[a]


def test_cap_never_reorders_what_survives():
    """Capping may drop items but must not reorder the survivors --
    relevance still decides the sequence."""
    d1, d2 = uuid.uuid4(), uuid.uuid4()
    chunks = _ids(6)
    source_of = {
        chunks[0]: d1, chunks[1]: d1, chunks[2]: d2,
        chunks[3]: d1, chunks[4]: d2, chunks[5]: d1,
    }
    fused = [(c, 1.0 / (i + 1)) for i, c in enumerate(chunks)]
    capped = [c for c, _ in cap_per_source(fused, source_of, 2)]
    assert capped == [c for c in chunks if c in capped]


def test_cap_binds_when_the_rival_is_close():
    """d1 is over its cap and d2's next chunk is comparable, so d1's
    surplus is dropped."""
    d1, d2 = uuid.uuid4(), uuid.uuid4()
    d1c, d2c = _ids(5), _ids(5)
    fused = [(c, 1.00 - i * 0.001) for i, c in enumerate(d1c)]
    fused += [(c, 0.999 - i * 0.001) for i, c in enumerate(d2c)]
    fused.sort(key=lambda t: -t[1])
    src = {**{c: d1 for c in d1c}, **{c: d2 for c in d2c}}
    kept = cap_per_source(fused, src, 2)
    from collections import Counter
    assert Counter(src[c] for c, _ in kept)[d1] == 2


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
