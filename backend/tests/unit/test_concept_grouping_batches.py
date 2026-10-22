"""Concept-grouping batches run concurrently but merge as if sequential."""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import backend.app.services.pipeline_llm as pl


def test_concurrent_batches_merge_in_batch_order(monkeypatch) -> None:
    monkeypatch.setattr(pl, "_CONCEPT_GROUPING_BATCH_SIZE", 2)
    orphans = [{"label": f"C{i}", "iri": f"x#C{i}"} for i in range(6)]       # -> 3 batches
    in_flight, peak = 0, 0

    class Router:
        async def chat(self, task, system, user):
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            batch = 0 if "C0" in user else 1 if "C2" in user else 2
            await asyncio.sleep(0.03 * (3 - batch))                          # later batches finish first
            in_flight -= 1
            return SimpleNamespace(text=json.dumps({
                "TOP_LEVEL_CONCEPTS": [{"LABEL": "Commodity", "DESCRIPTION": f"from batch {batch}"}],
                "ASSIGNMENTS": [{"CLASS_LABEL": f"C{2 * batch}", "CONCEPT_LABEL": "Commodity"}]}))

    out = asyncio.run(pl._propose_concept_grouping(Router(), orphans, concurrency=3))
    assert peak == 3                                                         # really concurrent
    assert out["TOP_LEVEL_CONCEPTS"] == [{"LABEL": "Commodity", "DESCRIPTION": "from batch 0"}]
    assert [a["CLASS_LABEL"] for a in out["ASSIGNMENTS"]] == ["C0", "C2", "C4"]
