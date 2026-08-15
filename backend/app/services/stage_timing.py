"""Wall-clock timing for the long pipeline stages.

Why this exists: the cost report (`[cost]` at the end of a run) breaks spend
down per task, so it is easy to see that `class_proposal` was 65% of the bill.
There was no equivalent for TIME. When a 30-document run took hours, the only
way to answer "how long did dedup take?" was to diff timestamps on surrounding
log lines and bracket it -- for a stage with no boundary logging at all, that
bracket was 15-20 minutes wide.

That matters because time and cost do NOT rank the same. Dedup is 22% of spend
but a single sequential barrier between two fan-outs; table extraction is
almost free but was the long pole at ~100 minutes. You cannot decide which
concurrency knob to raise without knowing where the wall time actually went.

Timings are printed per stage as they complete (so a detached run shows
progress) and again as a sorted table at the end, next to the cost report.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator


def format_duration(seconds: float) -> str:
    """Human-scaled duration: seconds under a minute, then m/s, then h/m.

    A bare seconds count is unreadable at pipeline scale -- '5027s' does not
    land the way '1h 23m' does, and these numbers exist to be read.
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60):02d}s"
    return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60):02d}m"


class StageTimer:
    """Accumulates wall time per named stage, in completion order.

    Re-entering a name adds to it rather than replacing, so a stage that runs
    in several passes (table extraction per batch, say) reports one total
    instead of only its final slice.
    """

    def __init__(self) -> None:
        self._elapsed: dict[str, float] = {}
        self._counts: dict[str, int] = {}
        self._started = time.monotonic()

    @contextmanager
    def stage(self, name: str, *, quiet: bool = False) -> Iterator[None]:
        """Time a block and print `wall=...` on exit.

        The line is printed even when the block raises, because a stage that
        died after 40 minutes is exactly the one whose duration you want. The
        exception propagates untouched.
        """
        t0 = time.monotonic()
        try:
            yield
        finally:
            dt = time.monotonic() - t0
            self._elapsed[name] = self._elapsed.get(name, 0.0) + dt
            self._counts[name] = self._counts.get(name, 0) + 1
            if not quiet:
                print(f"[{name}] wall={format_duration(dt)}", flush=True)

    def record(self, name: str, seconds: float) -> None:
        """Add a duration measured elsewhere (e.g. inside a subprocess pool)."""
        self._elapsed[name] = self._elapsed.get(name, 0.0) + seconds
        self._counts[name] = self._counts.get(name, 0) + 1

    @property
    def total_seconds(self) -> float:
        return time.monotonic() - self._started

    def report_lines(self) -> list[str]:
        """Stages sorted slowest-first, with each one's share of wall time.

        The share is of TOTAL elapsed, not of the sum of stages: stages overlap
        (table extraction runs while nothing else does, but summarization and
        chunking interleave), and normalising to the sum would inflate every
        row and hide unaccounted time. A column that does not add to 100% is
        the honest one here.
        """
        if not self._elapsed:
            return []
        total = self.total_seconds or 1.0
        width = max(len(n) for n in self._elapsed)
        lines = ["", f"[wall] total: {format_duration(total)}"]
        for name, dt in sorted(self._elapsed.items(), key=lambda kv: -kv[1]):
            calls = self._counts.get(name, 1)
            suffix = f"  ({calls} passes)" if calls > 1 else ""
            lines.append(
                f"[wall]   {name:<{width}}  {format_duration(dt):>9}"
                f"  {100.0 * dt / total:5.1f}%{suffix}"
            )
        return lines

    def print_report(self) -> None:
        for line in self.report_lines():
            print(line)

    def as_dict(self) -> dict[str, object]:
        """Serialisable form, for writing next to the run's stats/cost report."""
        return {
            "total_seconds": round(self.total_seconds, 2),
            "stages": {
                name: {
                    "seconds": round(dt, 2),
                    "human": format_duration(dt),
                    "passes": self._counts.get(name, 1),
                }
                for name, dt in sorted(self._elapsed.items(), key=lambda kv: -kv[1])
            },
        }
