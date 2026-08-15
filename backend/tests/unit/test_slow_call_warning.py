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

    A 150k-token prompt is ~50s of prefill on its own. The reply then streams
    out at ~230 tok/s over the remaining 13s -- fast. At 72% of a 90s budget the
    old `elapsed > timeout * 0.5` rule warned; the projection correctly does
    not, because a full 4096-token reply still fits inside the deadline.

    Sized so the decode span clears _MIN_DECODE_S -- otherwise this would be
    SKIPPED as an unmeasurable sample and pass for the wrong reason.
    """
    text = _warn(
        router, caplog, elapsed=65, timeout=90, max_tokens=4096,
        result=_result(prompt=150_000, completion=3_000),
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


def test_huge_prompt_with_a_fast_reply_neither_crashes_nor_invents_a_rate(
    router, caplog
) -> None:
    """A 900k-token prompt against a 1s elapsed time is degenerate: prefill
    alone exceeds it. The clamps must keep decode_s positive (no division by
    zero) AND the sample must be rejected -- 10 tokens in 1s supports no rate.

    An earlier version of this test asserted a WARNING here, on the theory that
    anything else meant the guard had silently divided the problem away. That
    was wrong: emitting a confident 'needs 6000s' from a 10-token sample is the
    false-positive this gate exists to stop.
    """
    assert _warn(
        router, caplog, elapsed=1, timeout=60, max_tokens=8192,
        result=_result(prompt=900_000, completion=10),
    ) == ""


def test_zero_timeout_is_not_treated_as_a_deadline(router, caplog) -> None:
    assert _warn(
        router, caplog, elapsed=999, timeout=0, max_tokens=8192,
        result=_result(prompt=100, completion=100),
    ) == ""


# --------------------------------------------------------------------------- #
# Short replies. Found live: the check fired on `table_concept_grouping`
# returning 67 tokens in 7s, reporting "10 tok/s, needs 200s" for a task that
# needs 13.7 tok/s and comfortably clears it. Almost all of those 7s were
# time-to-first-token -- a FIXED per-request cost, not a per-token one.
# --------------------------------------------------------------------------- #
def test_tiny_reply_does_not_produce_a_rate(router, caplog) -> None:
    """The exact live false positive: 815 in / 67 out in 7s."""
    assert _warn(
        router, caplog, elapsed=7, timeout=150, max_tokens=2048,
        result=_result(prompt=815, completion=67),
    ) == "", "TTFT divided across 67 tokens is not a throughput measurement"


def test_short_reply_stays_silent_even_when_slow_in_wall_terms(router, caplog) -> None:
    """A 30s call returning 50 tokens still says nothing about throughput."""
    assert _warn(
        router, caplog, elapsed=30, timeout=90, max_tokens=4096,
        result=_result(prompt=2_000, completion=50),
    ) == ""


def test_the_check_activates_once_output_is_long_enough_to_measure(router, caplog) -> None:
    """Skipping short replies must not disable the check. The first reply big
    enough to time is also the first one where length can cause a timeout."""
    text = _warn(
        router, caplog, elapsed=60, timeout=90, max_tokens=8192,
        result=_result(prompt=1_000, completion=600),
    )
    assert "under-budgeted" in text


def test_boundary_reply_length_is_evaluated(router, caplog) -> None:
    """At exactly the token threshold, with a decode span well past its own
    threshold, the sample is usable and must be judged."""
    from backend.app.services.llm_router import _MIN_TOKENS_TO_RATE

    text = _warn(
        router, caplog, elapsed=120, timeout=150, max_tokens=8192,
        result=_result(prompt=500, completion=_MIN_TOKENS_TO_RATE),
    )
    assert "under-budgeted" in text


def test_long_reply_over_a_short_decode_span_is_rejected(router, caplog) -> None:
    """Many tokens is not sufficient on its own. 900 tokens delivered in ~1s of
    decoding is a fine result, but the span is too short for the overhead
    estimate to be a small share of it, so no rate should be claimed."""
    assert _warn(
        router, caplog, elapsed=3, timeout=180, max_tokens=8192,
        result=_result(prompt=1_000, completion=900),
    ) == ""


def test_fixed_overhead_is_not_charged_to_decoding(router, caplog) -> None:
    """A healthy call whose measured rate only looks marginal once ~2s of
    request overhead is wrongly attributed to token generation."""
    # 3000 tokens in 32s -> 94 tok/s raw; the overhead correction raises it.
    assert _warn(
        router, caplog, elapsed=32, timeout=150, max_tokens=8192,
        result=_result(prompt=3_000, completion=3_000),
    ) == ""
