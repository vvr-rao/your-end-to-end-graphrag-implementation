"""Guards for batch-streamed, resumable `generate-artifacts` (per-chunk stage).

The single-shot path makes every chunk's LLM call first and writes once at the
end, so a kill or a `--max-cost-usd` trip discards the whole run.
`generate_per_chunk_artifacts_streamed` commits per batch instead.
"""
from __future__ import annotations

import asyncio
import contextlib
import inspect

import pytest

from backend.app.services import db_artifact_gen as dag

# --------------------------------------------------------------------------
# The drift hazard: selection and count must share ONE predicate.
# --------------------------------------------------------------------------

def test_selection_and_count_share_one_predicate():
    sel = inspect.getsource(dag.generate_per_chunk_artifacts)
    cnt = inspect.getsource(dag._unprocessed_artifact_chunk_count)
    assert "_unprocessed_chunk_filter(" in sel
    assert "_unprocessed_chunk_filter(" in cnt


def test_selection_has_a_total_order():
    """`chunk_offset` is meaningless unless the order is total: created_at
    ties within a document, so the id tiebreak is what keeps an offset naming
    the same chunks from one batch to the next."""
    sel = inspect.getsource(dag.generate_per_chunk_artifacts)
    assert ".order_by(Chunk.created_at, Chunk.id)" in sel


# --------------------------------------------------------------------------
# Loop behaviour
# --------------------------------------------------------------------------

def _install(monkeypatch, *, counts, batch_results=None, record=None):
    """Stub the DB touchpoints.

    `counts` is the sequence `_unprocessed_artifact_chunk_count` returns (one
    before and one after each batch). `batch_results` is a list of per-batch
    field dicts; the last one repeats.
    """
    seq = list(counts)
    results = list(batch_results or [{}])
    calls = record if record is not None else {}
    calls.setdefault("gen", [])
    calls.setdefault("bump", 0)

    async def _count(**_kw):
        return seq.pop(0) if seq else 0

    async def _gen(**kw):
        calls["gen"].append(kw)
        r = results[min(len(calls["gen"]) - 1, len(results) - 1)]
        s = dag.ArtifactGenSummary()
        s.by_type = {t: 0 for t in kw["types"]}
        for k, v in r.items():
            setattr(s, k, v)
        return s

    async def _bump(_session):
        calls["bump"] += 1
        return 42

    @contextlib.asynccontextmanager
    async def _scope():
        yield object()

    monkeypatch.setattr(dag, "_unprocessed_artifact_chunk_count", _count)
    monkeypatch.setattr(dag, "generate_per_chunk_artifacts", _gen)
    monkeypatch.setattr(dag, "bump_version", _bump)
    monkeypatch.setattr(dag, "session_scope", _scope)
    return calls


def _run(**kw):
    kw.setdefault("max_cost_usd", 100)
    return asyncio.run(dag.generate_per_chunk_artifacts_streamed(**kw))


def test_batch_size_must_be_positive():
    with pytest.raises(ValueError):
        asyncio.run(dag.generate_per_chunk_artifacts_streamed(batch_size=0))


def test_no_chunks_means_no_llm_work(monkeypatch):
    calls = _install(monkeypatch, counts=[0])
    _run(batch_size=10)
    assert calls["gen"] == []
    assert calls["bump"] == 1


def test_runs_until_the_backlog_drains(monkeypatch):
    calls = _install(
        monkeypatch,
        counts=[25, 15, 15, 5, 5, 0],
        batch_results=[{"chunks_scanned": 10}, {"chunks_scanned": 10},
                       {"chunks_scanned": 5}],
    )
    _run(batch_size=10)
    assert len(calls["gen"]) == 3
    assert [kw["chunk_offset"] for kw in calls["gen"]] == [0, 0, 0]


def test_each_batch_defers_versioning_to_the_end(monkeypatch):
    calls = _install(
        monkeypatch, counts=[20, 10, 10, 0],
        batch_results=[{"chunks_scanned": 10}],
    )
    summary = _run(batch_size=10)
    assert all(kw["bump_graph_version"] is False for kw in calls["gen"])
    assert calls["bump"] == 1
    assert summary.new_graph_version == 42


def test_a_fully_stalled_backlog_terminates(monkeypatch):
    """Chunks the model returns nothing for never get an artifact, so they
    never leave the unprocessed set. Without the offset the loop re-selects
    and re-pays for them forever."""
    calls = _install(
        monkeypatch, counts=[7] * 20, batch_results=[{"chunks_scanned": 7}],
    )
    _run(batch_size=10)
    assert len(calls["gen"]) == 1


def test_stalled_chunks_do_not_starve_the_chunks_behind_them(monkeypatch):
    # 6 chunks: 4 empty ones sort first, 2 good ones behind them. batch_size 2
    # must step past the empties and still reach the good pair, then stop with
    # only the 4 empties left (count 4 <= offset 4).
    calls = _install(
        monkeypatch,
        counts=[6, 6, 6, 6, 6, 4, 4],
        batch_results=[{"chunks_scanned": 2}],
    )
    _run(batch_size=2)
    assert [kw["chunk_offset"] for kw in calls["gen"]] == [0, 2, 4]


def test_a_partially_stalled_batch_advances_by_exactly_its_leftovers(monkeypatch):
    """10 attempted, 7 got artifacts (count 20 -> 13), 3 left behind: the next
    batch must start past those 3, not re-select and re-pay for them."""
    calls = _install(
        monkeypatch,
        counts=[20, 13, 13, 3, 3, 3],
        batch_results=[{"chunks_scanned": 9, "chunks_failed": 1},
                       {"chunks_scanned": 10}],
    )
    _run(batch_size=10)
    assert [kw["chunk_offset"] for kw in calls["gen"]] == [0, 3]
    assert len(calls["gen"]) == 2


def test_cost_budget_shrinks_across_batches_and_stops_the_run(monkeypatch):
    calls = _install(
        monkeypatch,
        counts=[30, 20, 20, 10, 10, 0],
        batch_results=[{"chunks_scanned": 10, "llm_cost_usd": 4.0}],
    )
    _run(batch_size=10, max_cost_usd=10.0)
    assert [kw["max_cost_usd"] for kw in calls["gen"]] == [10.0, 6.0, 2.0]


def test_a_batch_that_attempts_nothing_stops_the_loop(monkeypatch):
    """E.g. the cost cap tripped before any call returned. Spinning on the
    same offset would be an infinite loop."""
    calls = _install(monkeypatch, counts=[10] * 10, batch_results=[{}])
    _run(batch_size=5)
    assert len(calls["gen"]) == 1


def test_limit_is_spread_across_batches(monkeypatch):
    calls = _install(
        monkeypatch,
        counts=[100, 90, 90, 80, 80, 75],
        batch_results=[{"chunks_scanned": 10}, {"chunks_scanned": 10},
                       {"chunks_scanned": 5}],
    )
    _run(batch_size=10, limit=25)
    assert [kw["limit"] for kw in calls["gen"]] == [10, 10, 5]


def test_passthrough_kwargs_reach_every_batch(monkeypatch):
    calls = _install(
        monkeypatch, counts=[10, 0], batch_results=[{"chunks_scanned": 10}],
    )
    _run(batch_size=10, concurrency=64, use_entities=False,
         chunk_kind="fulltext", types=("Claim",))
    kw = calls["gen"][0]
    assert kw["concurrency"] == 64
    assert kw["use_entities"] is False
    assert kw["chunk_kind"] == "fulltext"
    assert kw["types"] == ("Claim",)


def test_merge_accumulates_counters_and_types():
    total, batch = dag.ArtifactGenSummary(), dag.ArtifactGenSummary()
    total.by_type = {"Claim": 2}
    batch.by_type = {"Claim": 3, "Event": 1}
    total.llm_cost_usd, batch.llm_cost_usd = 1.5, 0.25
    batch.artifacts_inserted = 4
    dag._merge_artifact_summaries(total, batch)
    assert total.by_type == {"Claim": 5, "Event": 1}
    assert total.llm_cost_usd == 1.75
    assert total.artifacts_inserted == 4
