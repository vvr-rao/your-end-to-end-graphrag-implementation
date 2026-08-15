"""Output-size accounting in the cost report.

max_tokens and timeout are both sized against the LARGEST reply a task can
produce, and that number was assumed rather than measured -- which is how ~25
tasks per preset ended up demanding a decode rate nobody had checked. Recording
observed completion lengths makes the next sizing pass evidence-based.
"""
from backend.app.services.llm_router import LLMRouter


def _router(max_tokens=4096):
    r = LLMRouter.__new__(LLMRouter)
    r.reset_counters()
    r._cost_by_task, r._calls_by_task, r._out_tokens_by_task = {}, {}, {}
    r._tasks = {"t": {"model": "m", "provider": "p", "max_tokens": max_tokens}}
    return r


def test_headroom_flags_a_task_budgeted_for_output_it_never_produces():
    r = _router(max_tokens=8192)
    r._out_tokens_by_task["t"] = [40, 55, 38]
    s = r._output_stats("t")
    assert s["out_tokens_max"] == 55
    assert s["headroom"] == round(8192 / 55, 1)


def test_headroom_near_one_means_replies_may_be_truncated():
    r = _router(max_tokens=1024)
    r._out_tokens_by_task["t"] = [1020, 1024]
    assert r._output_stats("t")["headroom"] == 1.0


def test_p95_ignores_the_single_worst_outlier():
    r = _router()
    r._out_tokens_by_task["t"] = [10] * 99 + [4000]
    s = r._output_stats("t")
    assert s["out_tokens_max"] == 4000
    assert s["out_tokens_p95"] == 10, "p95 must not be dragged by one outlier"


def test_single_sample_does_not_crash_percentile_indexing():
    r = _router()
    r._out_tokens_by_task["t"] = [77]
    assert r._output_stats("t")["out_tokens_p95"] == 77


def test_task_with_no_observations_reports_nothing():
    """Absent, not zero -- a zero would read as 'produced no output'."""
    assert _router()._output_stats("t") == {}


def test_missing_max_tokens_omits_headroom_rather_than_dividing_by_zero():
    r = _router(max_tokens=0)
    r._out_tokens_by_task["t"] = [100]
    s = r._output_stats("t")
    assert "headroom" not in s and s["max_tokens_configured"] is None
