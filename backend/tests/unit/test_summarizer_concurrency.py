"""The summarizer's per-call limiter must be reusable and must actually limit.

The evaluate/revise chain enters its limiter once per LLM call. An
@asynccontextmanager generator can only be entered ONCE -- the second `async
with` raises, and the chain's except-blocks swallow it, silently returning a
truncated summary with no error. That failure mode is invisible without
counting calls, hence these tests.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from backend.app.services import evaluated_summarizer as ES

# 1 summarize + 1 question-gen + 4 evaluate + 3 revise, when nothing ever passes
_WORST_CASE_CALLS = 9


class _Result:
    def __init__(self, text: str) -> None:
        self.text = text
        self.cost_usd = 0.0


class _FakeRouter:
    """Never reports 'passed', so the chain runs its full revision budget."""

    total_cost_usd = 0.0

    def __init__(self, delay: float = 0.0) -> None:
        self.calls = 0
        self.concurrent = 0
        self.peak_concurrent = 0
        self._delay = delay

    def task_spec(self, task: str) -> dict:
        return {}

    async def chat(self, task: str, **kw):
        self.calls += 1
        self.concurrent += 1
        self.peak_concurrent = max(self.peak_concurrent, self.concurrent)
        try:
            if self._delay:
                await asyncio.sleep(self._delay)
            if task == "summary_question_gen":
                return _Result(json.dumps({"questions": ["q1", "q2"]}))
            if task == "summary_evaluate":
                return _Result(json.dumps(
                    {"passed": False, "missing_items": ["x"], "section_issues": []}
                ))
            return _Result("## EVIDENCE\nsummary body")
        finally:
            self.concurrent -= 1


def test_chain_completes_without_a_semaphore() -> None:
    """sem=None must run the WHOLE chain.

    Regression: the null limiter was a single @asynccontextmanager object, so
    the second `async with` raised and the chain bailed after one call --
    returning a summary that had never been evaluated or revised.
    """
    r = _FakeRouter()
    summary, audit = asyncio.run(
        ES._summarize_chunk_with_eval(r, "source", num_questions=2, eval_rounds=3, sem=None)
    )
    assert r.calls == _WORST_CASE_CALLS, (
        f"expected the full {_WORST_CASE_CALLS}-call chain, got {r.calls} -- "
        "the limiter is not reusable"
    )
    assert summary.strip()
    assert audit["rounds"] > 1, "the evaluate/revise loop did not run"


def test_chain_completes_with_a_semaphore() -> None:
    r = _FakeRouter()
    asyncio.run(
        ES._summarize_chunk_with_eval(
            r, "source", num_questions=2, eval_rounds=3, sem=asyncio.Semaphore(2)
        )
    )
    assert r.calls == _WORST_CASE_CALLS


def test_no_limit_is_reentrant() -> None:
    """The exact property whose absence caused the bug."""

    async def _enter_twice() -> None:
        limiter = ES._NoLimit()
        async with limiter:
            pass
        async with limiter:  # must not raise
            pass

    asyncio.run(_enter_twice())


def test_semaphore_caps_calls_in_flight_not_chains() -> None:
    """The limiter now bounds concurrent API CALLS, which is what provider rate
    limits measure -- so the configured number maps directly onto TPM."""
    r = _FakeRouter(delay=0.01)
    limit = 3

    async def _run() -> None:
        sem = asyncio.Semaphore(limit)
        await asyncio.gather(*[
            ES._summarize_chunk_with_eval(
                r, "source", num_questions=2, eval_rounds=3, sem=sem
            )
            for _ in range(6)
        ])

    asyncio.run(_run())
    assert r.peak_concurrent <= limit, (
        f"peak {r.peak_concurrent} exceeded the semaphore limit {limit}"
    )
    assert r.calls == 6 * _WORST_CASE_CALLS


@pytest.mark.parametrize("rounds", [0, 1, 3])
def test_eval_rounds_controls_chain_length(rounds: int) -> None:
    r = _FakeRouter()
    asyncio.run(
        ES._summarize_chunk_with_eval(
            r, "source", num_questions=2, eval_rounds=rounds, sem=None
        )
    )
    # 1 summarize + 1 question-gen + (rounds+1) evaluate + rounds revise
    assert r.calls == 2 + (rounds + 1) + rounds
