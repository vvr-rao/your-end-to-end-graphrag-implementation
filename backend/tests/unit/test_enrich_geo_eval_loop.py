"""enrich-geo runs its standing corpus evaluation in a SECOND asyncio.run.
The engine cached by the enrichment's (now dead) loop must be dropped first,
or the evaluation fails with "attached to a different loop" and prints
"[eval-corpus] skipped" (sc80 build, 2026-11-26)."""

from __future__ import annotations

import argparse
from types import SimpleNamespace

from backend.app.cli import main as cli


def test_enrich_geo_resets_the_engine_cache_before_its_evaluation(monkeypatch):
    import backend.app.db.engine as engine
    import backend.app.services.db_geo_enrich as geo

    order: list[str] = []
    summary = SimpleNamespace(ontology_matched=0, ontology_nodes=0, ontology_edges=0, embedding_cost_usd=0.0,
                              place_entities=0, batches=0, edges_created=0, edges_existing=0, unresolved=0,
                              rejected_by_level=0, cost_usd=0.0, samples=[])

    async def fake_enrich(**_kw):
        order.append("enrich")
        return summary

    async def fake_eval():
        order.append("eval")
        return {}

    monkeypatch.setattr(geo, "enrich_geography", fake_enrich)
    monkeypatch.setattr(engine, "reset_engine_cache", lambda: order.append("reset"))
    monkeypatch.setattr(cli, "_run_corpus_eval", fake_eval)
    monkeypatch.setattr(cli, "_print_corpus_eval", lambda *_a, **_k: None)
    args = argparse.Namespace(class_labels=None, dry_run=False, limit=None, ontology_only=False)
    assert cli._cmd_enrich_geo(args) == 0
    assert order == ["enrich", "reset", "eval"]


def test_enrich_geo_dry_run_skips_the_evaluation(monkeypatch):
    import backend.app.services.db_geo_enrich as geo

    summary = SimpleNamespace(ontology_matched=0, ontology_nodes=0, ontology_edges=0, embedding_cost_usd=0.0,
                              place_entities=0, batches=0, edges_created=0, edges_existing=0, unresolved=0,
                              rejected_by_level=0, cost_usd=0.0, samples=[])

    async def fake_enrich(**_kw):
        return summary

    async def no_eval():
        raise AssertionError("dry run must not evaluate")

    monkeypatch.setattr(geo, "enrich_geography", fake_enrich)
    monkeypatch.setattr(cli, "_run_corpus_eval", no_eval)
    args = argparse.Namespace(class_labels=None, dry_run=True, limit=None, ontology_only=False)
    assert cli._cmd_enrich_geo(args) == 0
