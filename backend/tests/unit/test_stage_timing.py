"""Stage wall-clock accounting.

Motivated by a concrete gap: after a multi-hour 30-document run, "how long did
dedup take?" could only be answered by diffing timestamps on the log lines that
happened to bracket it -- a 15-20 minute window. Cost was attributable per task;
time was not attributable at all.
"""

from __future__ import annotations

import json

from backend.app.services.stage_timing import StageTimer, format_duration


def test_duration_is_human_scaled_across_magnitudes() -> None:
    """'5027s' is not a readable answer to 'how long did it take'."""
    assert format_duration(4.2) == "4.2s"
    assert format_duration(95) == "1m 35s"
    assert format_duration(5027) == "1h 23m"


def test_seconds_boundary_does_not_render_as_zero_minutes() -> None:
    assert format_duration(59.9) == "59.9s"
    assert format_duration(60) == "1m 00s"


def test_stage_records_elapsed_and_prints_it() -> None:
    t = StageTimer()
    with t.stage("stage3-dedup"):
        pass
    assert "stage3-dedup" in t.as_dict()["stages"]


def test_a_stage_that_raises_still_reports_its_duration() -> None:
    """The stage that died after 40 minutes is precisely the one whose
    duration you need; swallowing the timing on the error path would lose it.
    The exception must still propagate."""
    t = StageTimer()
    try:
        with t.stage("stage2-propose"):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    else:
        raise AssertionError("exception must propagate")
    assert "stage2-propose" in t.as_dict()["stages"]


def test_repeated_stage_accumulates_rather_than_overwrites() -> None:
    """Table extraction runs per batch; reporting only the final batch would
    understate it by however many batches there were."""
    t = StageTimer()
    for _ in range(3):
        with t.stage("table-extract"):
            pass
    assert t.as_dict()["stages"]["table-extract"]["passes"] == 3


def test_record_accepts_durations_measured_elsewhere() -> None:
    """Stage 4 is timed with explicit start/record instead of a `with` block."""
    t = StageTimer()
    t.record("stage4-apply", 12.5)
    assert t.as_dict()["stages"]["stage4-apply"]["seconds"] == 12.5


def test_report_is_sorted_slowest_first() -> None:
    """The long pole is the only row most readers will act on, so it goes
    first -- insertion order would bury it."""
    t = StageTimer()
    t.record("summarize", 100.0)
    t.record("stage3-dedup", 900.0)
    t.record("stage1-classify", 10.0)
    body = [l for l in t.report_lines() if l.startswith("[wall]   ")]
    assert [l.split()[1] for l in body] == ["stage3-dedup", "summarize", "stage1-classify"]


def test_percentages_are_of_total_elapsed_not_of_stage_sum() -> None:
    """Stages overlap and some time is unattributed. Normalising to the sum of
    stages would inflate every row to a fake 100% and hide the gap."""
    t = StageTimer()
    t.record("a", 1.0)
    t.record("b", 1.0)
    pcts = [
        float(l.split("%")[0].split()[-1])
        for l in t.report_lines()
        if l.startswith("[wall]   ")
    ]
    # Total elapsed is ~0 here, so shares are enormous -- the point is only
    # that they are NOT forced to 50/50 by normalising against the stage sum.
    assert sum(pcts) > 150

def test_empty_timer_reports_nothing_rather_than_a_header() -> None:
    """A --dry-run does no stages; an empty table with a total is noise."""
    assert StageTimer().report_lines() == []


def test_as_dict_is_json_serialisable() -> None:
    """It gets written to timings.json in the run folder."""
    t = StageTimer()
    t.record("summarize", 3.0)
    json.dumps(t.as_dict())
