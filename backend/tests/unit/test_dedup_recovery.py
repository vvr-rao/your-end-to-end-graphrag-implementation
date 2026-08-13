"""One bad dedup batch must not destroy the run.

The fail-loud guard was right to refuse an un-deduplicated ontology -- silently
returning the input is the original P1 bug. But it was absolute: a single
unparseable reply out of 32 discarded the other 31 batches and a 65-minute,
~$25 run. Live failure:

    DedupFailedError: 1/32 dedup batch(es) failed ... batch 19/32: response
    was not parseable JSON

Recovery is matched to the two causes: retry once (a transient malformed reply
is just a reroll), then split the batch in half (if the reply was truncated at
max_tokens, halving the input halves the echo -- the only thing that fixes it).
Still raises if a half fails on its own.
"""

from __future__ import annotations

import pytest

from backend.app.services import pipeline_llm as P
from backend.app.services.llm_router import ChatResult


# Single-key, matching production: _plan_dedup_batches emits one batch per key.
_BATCH = {"MATCH NOT FOUND": [{"LABEL": f"C{i}"} for i in range(8)]}
# Multi-key, for the defensive splitting path only.
_MIXED = {
    "MATCH NOT FOUND": [{"LABEL": f"C{i}"} for i in range(8)],
    "MATCH NOT FOUND RELATIONS": [{"LABEL": f"r{i}"} for i in range(4)],
}
_GOOD = '{"MATCH NOT FOUND": [], "MATCH NOT FOUND RELATIONS": []}'


class _Router:
    """Router stub whose reply sequence the test dictates."""

    def __init__(self, replies):
        self._replies = list(replies)
        self.calls = 0
        self.total_cost_usd = 0.0

    def task_spec(self, task):
        # Large enough that the planner produces ONE batch, so call counts
        # below measure retry behaviour and not batch count.
        return {"max_tokens": 32768}

    async def chat(self, task, *, system, user, cache_prefix=None):
        self.calls += 1
        text, finish = self._replies.pop(0)
        return ChatResult(text=text, model="m", provider="openai",
                          finish_reason=finish)


@pytest.fixture(autouse=True)
def _no_embedding_calls(monkeypatch):
    """Clustering only reorders items for batching, and it reaches for a real
    Embedder -- which would make live API calls from a unit test. Stub it to
    identity: none of these tests are about grouping quality."""
    async def _identity(items, render, **kw):
        return list(items)

    monkeypatch.setattr(P, "order_by_semantic_cluster", _identity)


async def _dedup(router):
    return await P._dedup(router, dict(_BATCH), concurrency=1)


# ----------------------------- splitting ---------------------------------- #

def test_split_halves_a_single_key_batch() -> None:
    """The production shape: one batch per key, so splitting halves that list."""
    left, right = P._split_dedup_batch(_BATCH)
    assert len(left["MATCH NOT FOUND"]) == 4
    assert len(right["MATCH NOT FOUND"]) == 4


def test_split_keeps_the_same_mix_if_a_multi_key_batch_ever_appears() -> None:
    """Defensive: moving whole lists would hand one half only relations,
    changing what the prompt is being asked to do rather than just how much."""
    left, right = P._split_dedup_batch(_MIXED)
    for half in (left, right):
        assert half["MATCH NOT FOUND"] and half["MATCH NOT FOUND RELATIONS"]


def test_split_loses_no_items() -> None:
    left, right = P._split_dedup_batch(_MIXED)
    for key in ("MATCH NOT FOUND", "MATCH NOT FOUND RELATIONS"):
        assert left[key] + right[key] == _MIXED[key]


def test_unsplittable_batch_returns_itself_rather_than_looping() -> None:
    assert len(P._split_dedup_batch({"MATCH NOT FOUND": [{"LABEL": "x"}]})) == 1
    assert len(P._split_dedup_batch({"MATCH NOT FOUND": []})) == 1


# ----------------------------- recovery ----------------------------------- #

@pytest.mark.asyncio
async def test_transient_bad_reply_recovers_on_plain_retry() -> None:
    r = _Router([("not json", "stop"), (_GOOD, "stop")])
    await _dedup(r)
    assert r.calls == 2, "should retry once before splitting"


@pytest.mark.asyncio
async def test_truncated_reply_recovers_by_splitting() -> None:
    """finish_reason='length' twice -> split into two halves that both parse."""
    r = _Router([("{trunc", "length"), ("{trunc", "length"),
                 (_GOOD, "stop"), (_GOOD, "stop")])
    await _dedup(r)
    assert r.calls == 4, "attempt, retry, then two halves"


@pytest.mark.asyncio
async def test_still_raises_when_a_half_also_fails() -> None:
    """Recovery must not become the silent pass-through it replaced."""
    r = _Router([("{trunc", "length"), ("{trunc", "length"),
                 (_GOOD, "stop"), ("still bad", "stop")])
    with pytest.raises(P.DedupFailedError):
        await _dedup(r)


@pytest.mark.asyncio
async def test_healthy_batch_costs_exactly_one_call() -> None:
    """Recovery must be free on the common path."""
    r = _Router([(_GOOD, "stop")])
    await _dedup(r)
    assert r.calls == 1


@pytest.mark.asyncio
async def test_truncation_is_named_in_the_failure_message() -> None:
    """'not parseable JSON' and 'truncated at max_tokens' have opposite
    remedies; the message must say which one happened."""
    r = _Router([("{trunc", "length")] * 2 + [("{trunc", "length")] * 2)
    with pytest.raises(P.DedupFailedError, match="truncated at max_tokens"):
        await _dedup(r)


@pytest.mark.asyncio
async def test_api_exception_is_also_retried_not_fatal() -> None:
    class _Flaky(_Router):
        async def chat(self, task, *, system, user, cache_prefix=None):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("connection reset")
            return ChatResult(text=_GOOD, model="m", provider="openai",
                              finish_reason="stop")

    r = _Flaky([])
    await _dedup(r)
    assert r.calls == 2
