"""The slow-call warning must be input-aware.

Background: a task configured `max_tokens: 8192, timeout: 180` needs >=45 tok/s
sustained for a maximum-length reply to arrive in time. Below that the request
cannot succeed AT ALL -- it times out, then retries. That is the 45-minute
document.

The first warning compared `elapsed > timeout * 0.5`, which is blind to how much
output was produced. That is wrong in both directions:

  * It cannot tell a slow decode from a large prefill. A call spending 100s
    reading a 300k-token prompt and 6s writing the reply is healthy; the flat
    rule reports it the same as one genuinely crawling toward its deadline.
  * It stays silent on the case that matters. A task returning 200 tokens in
    25% of a 60s budget looks fine and is not -- a full 8,192-token reply at
    that rate needs ~20 minutes.

Worth recording, because the initial reading of `summary_evaluate` was wrong:
it hit 59% of its budget and looked like the first bullet, but at max_tokens
4096 / timeout 150 it REQUIRED 27.3 tok/s and 27 tok/s was measured. It was a
true positive sitting on its own boundary, which is what prompted rebasing
every preset's timeouts to leave real headroom (test_timeout_budgets.py).
"""

from __future__ import annotations

import logging

import pytest

from backend.app.services.llm_router import ChatResult, LLMRouter


def _result(prompt: int, completion: int) -> ChatResult:
    return ChatResult(
        text="x", model="gpt-4.1-mini", provider="openai",
        prompt_tokens=prompt, completion_tokens=completion,
    )


def _warn(router: LLMRouter, caplog, **kw) -> str:
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="backend.app.services.llm_router"):
        router._warn_if_near_deadline(
            task=kw.get("task", "t"), provider_name="openai",
            model="gpt-4.1-mini", elapsed=kw["elapsed"], timeout=kw["timeout"],
            max_tokens=kw["max_tokens"], result=kw["result"],
        )
    return caplog.text


@pytest.fixture
def router() -> LLMRouter:
    return LLMRouter.__new__(LLMRouter)  # no config needed; method is pure


def test_prefill_dominated_call_is_not_flagged(router, caplog) -> None:
    """The false-alarm class the flat rule could not distinguish.

    A 300k-token prompt is ~100s of prefill on its own. The reply then streams
    out at ~170 tok/s in the remaining 6s -- fast. At 71% of a 150s budget the
    old `elapsed > timeout * 0.5` rule warned; the projection correctly does
    not, because a full 4096-token reply still fits inside the deadline.
    """
    text = _warn(
        router, caplog, elapsed=106, timeout=150, max_tokens=4096,
        result=_result(prompt=300_000, completion=1_000),
    )
    assert text == "", "prefill time must not be charged to the decode rate"


def test_summary_evaluate_at_its_configured_boundary_is_flagged(router, caplog) -> None:
    """This is the case that motivated the work, and the finding was the
    opposite of the initial reading: `summary_evaluate` at max_tokens 4096 /
    timeout 150 requires 27.3 tok/s, and 27 tok/s was MEASURED. It was not a
    false alarm -- the task genuinely could not emit a full-length reply in
    time. The config now carries margin; the warning must still catch the
    shape if anyone reintroduces it.
    """
    text = _warn(
        router, caplog, elapsed=88, timeout=150, max_tokens=4096,
        result=_result(prompt=8_000, completion=2_300),
    )
    assert "under-budgeted" in text


def test_slow_decode_is_flagged_even_when_well_inside_the_timeout(router, caplog) -> None:
    """The case the flat rule MISSED. 200 tokens in 15s of a 60s budget is 25%
    -- looks healthy -- but that is ~13 tok/s, so a full 8,192-token reply
    needs ~10 minutes against a 60s deadline."""
    text = _warn(
        router, caplog, elapsed=15, timeout=60, max_tokens=8192,
        result=_result(prompt=500, completion=200),
    )
    assert "under-budgeted" in text
    assert "8192-token reply" in text


def test_warning_names_a_timeout_that_would_actually_work(router, caplog) -> None:
    """A warning that only says 'too slow' leaves the reader to do the
    arithmetic that caused the bug in the first place."""
    text = _warn(
        router, caplog, elapsed=15, timeout=60, max_tokens=8192,
        result=_result(prompt=500, completion=200),
    )
    suggested = int(text.split("raise timeout to >=")[1].split()[0])
    assert suggested > 60, "a suggestion at or below the current timeout is useless"


def test_healthy_fast_call_is_silent(router, caplog) -> None:
    assert _warn(
        router, caplog, elapsed=3, timeout=180, max_tokens=8192,
        result=_result(prompt=1_000, completion=900),
    ) == ""


def test_the_45_minute_document_shape_is_flagged(router, caplog) -> None:
    """Regression for the original incident: 8192 max_tokens / 180s timeout
    demands 45 tok/s. At ~27 tok/s observed, this must warn."""
    text = _warn(
        router, caplog, elapsed=100, timeout=180, max_tokens=8192,
        result=_result(prompt=2_000, completion=2_700),
    )
    assert "under-budgeted" in text


def test_missing_usage_falls_back_to_the_flat_rule(router, caplog) -> None:
    """Some providers/paths report no usage. Losing the check entirely there
    would be worse than the imprecise version of it."""
    text = _warn(
        router, caplog, elapsed=150, timeout=180, max_tokens=8192,
        result=_result(prompt=0, completion=0),
    )
    assert "no usage reported" in text


def test_missing_usage_and_fast_call_stays_silent(router, caplog) -> None:
    assert _warn(
        router, caplog, elapsed=5, timeout=180, max_tokens=8192,
        result=_result(prompt=0, completion=0),
    ) == ""


def test_prefill_estimate_cannot_consume_the_whole_elapsed_time(router, caplog) -> None:
    """A huge prompt with a fast reply must not drive decode_s to ~0 and
    produce a nonsense rate (division-by-near-zero → no warning ever)."""
    text = _warn(
        router, caplog, elapsed=1, timeout=60, max_tokens=8192,
        result=_result(prompt=900_000, completion=10),
    )
    assert "under-budgeted" in text, "must still evaluate, not silently divide out"


def test_zero_timeout_is_not_treated_as_a_deadline(router, caplog) -> None:
    assert _warn(
        router, caplog, elapsed=999, timeout=0, max_tokens=8192,
        result=_result(prompt=100, completion=100),
    ) == ""
