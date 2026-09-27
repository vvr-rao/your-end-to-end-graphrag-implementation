"""Guards for batch-streamed, resumable `extract-entities`.

The single-shot path makes every LLM call first and writes once at the end, so
a kill or a `--max-cost-usd` trip at 90% discards the whole run. Measured
2026-09-26: 282 chunks, $8.75, 17 minutes, nothing durable until the final
second. `extract_entities_streamed` commits per batch instead.
"""
from __future__ import annotations

import asyncio
import contextlib
import inspect

import pytest

from backend.app.services import db_entity_extract as dee


# --------------------------------------------------------------------------
# The drift hazard: if the counter and the selection disagree, the progress
# guard stops firing and a zero-entity batch becomes an infinite PAID loop.
# --------------------------------------------------------------------------

def test_count_mirrors_the_selection_predicate():
    sel = inspect.getsource(dee.extract_entities)
    cnt = inspect.getsource(dee._unprocessed_chunk_count)
    for clause in (
        "GraphRelationship.predicate_iri == VIAO_ASSERTS_ABOUT",
        'GraphRelationship.relationship_source == "DOCUMENT_EXTRACTION"',
        "GraphRelationship.source_chunk_id.isnot(None)",
        'Chunk.status == "ACTIVE"',
        "Chunk.id.notin_(already_subq)",
    ):
        assert clause in sel, f"selection lost: {clause}"
        assert clause in cnt, (
            f"_unprocessed_chunk_count must mirror the selection; missing: "
            f"{clause}. If these drift, the streaming progress guard cannot "
            f"detect a stalled batch and the loop re-pays for the same chunks."
        )


# --------------------------------------------------------------------------
# Loop behaviour
# --------------------------------------------------------------------------

def _install(monkeypatch, *, counts, batch_result=None, record=None):
    """Stub the DB touchpoints; `counts` is the sequence _unprocessed returns."""
    seq = list(counts)
    calls = record if record is not None else {}
    calls.setdefault("extract", [])
    calls.setdefault("link", 0)
    calls.setdefault("bump", 0)

    async def _count(**_kw):
        return seq.pop(0) if seq else 0

    async def _extract(**kw):
        calls["extract"].append(kw)
        s = dee.EntityExtractSummary()
        if batch_result:
            for k, v in batch_result.items():
                setattr(s, k, v)
        return s

    async def _link(_summary):
        calls["link"] += 1

    async def _bump(_session):
        calls["bump"] += 1
        return 42

    @contextlib.asynccontextmanager
    async def _scope():
        yield object()

    monkeypatch.setattr(dee, "_unprocessed_chunk_count", _count)
    monkeypatch.setattr(dee, "extract_entities", _extract)
    monkeypatch.setattr(dee, "_link_tables_to_entities", _link)
    monkeypatch.setattr(dee, "bump_version", _bump)
    monkeypatch.setattr(dee, "session_scope", _scope)
    return calls


def test_batch_size_must_be_positive():
    with pytest.raises(ValueError):
        asyncio.run(dee.extract_entities_streamed(batch_size=0))


def test_no_chunks_means_no_llm_work(monkeypatch):
    calls = _install(monkeypatch, counts=[0])
    asyncio.run(dee.extract_entities_streamed(batch_size=10))
    assert calls["extract"] == []
    # Still exactly one version bump: the run happened, it just found nothing.
    assert calls["bump"] == 1


def test_runs_until_the_backlog_drains(monkeypatch):
    # 25 -> 15 -> 5 -> 0 : three batches, then stop.
    calls = _install(
        monkeypatch,
        counts=[25, 15, 15, 5, 5, 0],
        batch_result={"chunks_scanned": 10},
    )
    asyncio.run(dee.extract_entities_streamed(batch_size=10, max_cost_usd=100))
    assert len(calls["extract"]) == 3


def test_a_stalled_batch_terminates_rather_than_looping(monkeypatch):
    """A chunk yielding NO entities gets no assertsAbout edge -> never 'done'.

    Without progression the next batch re-selects the same chunks and re-pays
    for them, forever.
    """
    calls = _install(monkeypatch, counts=[7] * 20)
    asyncio.run(dee.extract_entities_streamed(batch_size=10, max_cost_usd=100))
    assert len(calls["extract"]) == 1, (
        "offset must pass the whole stalled set, ending the loop in one step"
    )


def test_stalled_chunks_do_not_starve_the_chunks_behind_them(monkeypatch):
    """The failure the first guard had: a hard stop on no-progress.

    50 untypeable chunks sort first by created_at. With batch_size=50 the first
    batch is exactly those, makes no progress, and a naive guard would stop --
    never reaching the NEW chunks behind them. Offset progression must keep
    going instead.
    """
    calls = _install(monkeypatch, counts=[7] * 20)
    asyncio.run(dee.extract_entities_streamed(batch_size=2, max_cost_usd=100))
    offsets = [kw["chunk_offset"] for kw in calls["extract"]]
    assert offsets == [0, 2, 4, 6], offsets
    assert len(calls["extract"]) == 4, (
        "must keep stepping forward through the backlog, not stop at batch 1"
    )


def test_offset_stays_put_while_progress_is_being_made(monkeypatch):
    """Processed chunks leave the unprocessed set, so the offset must NOT move.

    Advancing it on a successful batch would skip real work.
    """
    calls = _install(
        monkeypatch, counts=[30, 20, 20, 10, 10, 0],
        batch_result={"chunks_scanned": 10},
    )
    asyncio.run(dee.extract_entities_streamed(batch_size=10, max_cost_usd=100))
    assert [kw["chunk_offset"] for kw in calls["extract"]] == [0, 0, 0]


def test_each_batch_defers_linking_and_versioning(monkeypatch):
    calls = _install(
        monkeypatch, counts=[20, 10, 10, 0], batch_result={"chunks_scanned": 10}
    )
    asyncio.run(dee.extract_entities_streamed(batch_size=10, max_cost_usd=100))
    assert len(calls["extract"]) == 2
    for kw in calls["extract"]:
        assert kw["link_tables"] is False
        assert kw["bump_graph_version"] is False
    # ...and both happen exactly once, for the whole run.
    assert calls["link"] == 1
    assert calls["bump"] == 1


def test_cost_budget_shrinks_across_batches_and_stops_the_run(monkeypatch):
    calls = _install(
        monkeypatch,
        counts=[30, 20, 20, 10, 10, 0],
        batch_result={"chunks_scanned": 10, "llm_cost_usd": 4.0},
    )
    asyncio.run(dee.extract_entities_streamed(batch_size=10, max_cost_usd=10.0))
    budgets = [kw["max_cost_usd"] for kw in calls["extract"]]
    assert budgets == [10.0, 6.0, 2.0], budgets
    # A 4th batch would have had budget <= 0, so the run stops with work left.
    assert len(calls["extract"]) == 3


def test_limit_is_spread_across_batches(monkeypatch):
    calls = _install(
        monkeypatch,
        counts=[100, 90, 90, 80, 80, 70],
        batch_result={"chunks_scanned": 10},
    )
    asyncio.run(dee.extract_entities_streamed(
        batch_size=10, limit=25, max_cost_usd=100))
    sizes = [kw["limit"] for kw in calls["extract"]]
    assert sizes == [10, 10, 5], sizes


# --------------------------------------------------------------------------
# Summary accumulation
# --------------------------------------------------------------------------

def test_merge_summaries_accumulates_counters_and_dicts():
    total, batch = dee.EntityExtractSummary(), dee.EntityExtractSummary()
    total.entities_minted, batch.entities_minted = 5, 7
    total.llm_cost_usd, batch.llm_cost_usd = 1.5, 2.25
    total.entity_drops = {"abstained": 3}
    batch.entity_drops = {"abstained": 4, "off_menu_iri": 1}
    batch.new_graph_version = 9
    dee._merge_summaries(total, batch)
    assert total.entities_minted == 12
    assert total.llm_cost_usd == pytest.approx(3.75)
    assert total.entity_drops == {"abstained": 7, "off_menu_iri": 1}
    assert total.new_graph_version == 9


def test_merge_summaries_caps_unbounded_lists():
    """A 10M-token run must not accumulate every sample from every batch."""
    total = dee.EntityExtractSummary()
    for _ in range(50):
        batch = dee.EntityExtractSummary()
        batch.samples = [{"n": i} for i in range(10)]
        batch.abstained_samples = [{"n": i} for i in range(20)]
        dee._merge_summaries(total, batch)
    assert len(total.samples) == 8
    assert len(total.abstained_samples) == 200


# --------------------------------------------------------------------------
# CLI routing
# --------------------------------------------------------------------------

def test_cli_routes_to_the_streamed_driver_only_when_batching():
    from backend.app.cli import main as cli

    src = inspect.getsource(cli._cmd_extract_entities)
    assert "extract_entities_streamed if _batch > 0 else extract_entities" in src
    assert '"batch_size": _batch' in src


def test_cli_batch_size_resolution():
    """flag > extraction.batch_size > 0 (single-shot)."""
    from backend.app.cli import main as cli

    p = cli.build_parser()
    assert cli._resolve_extraction_opt(
        p.parse_args(["extract-entities", "--batch-size", "64"]),
        "batch_size", "batch_size", 0) == 64
    # Unset falls through to config; shipped default is 0 = single-shot.
    assert int(cli._resolve_extraction_opt(
        p.parse_args(["extract-entities"]), "batch_size", "batch_size", 0) or 0) == 0
