"""Every task's timeout must fit a maximum-length reply, with margin.

`max_tokens / timeout` is not a style preference -- it is the decode rate the
model must SUSTAIN for a full-length reply to arrive before the deadline. Below
it the request cannot succeed at all: it times out, then retries, and the run
burns wall time on generations it throws away. That is what turned one 42k-token
document into 45 minutes.

An earlier pass fixed the worst offenders by normalising every task to 27.3
tok/s. That removed the outliers but left no headroom: the rate actually
MEASURED on gpt-4.1-mini was 27 tok/s, so the entire config sat a rounding error
below its own requirement, and two dozen tasks per preset were one slow minute
from a timeout. This test pins a real margin instead of a boundary.

Streamed tasks are exempt: streaming turns a whole-request deadline into a
between-chunk one, so a long-but-healthy generation cannot trip it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

_CONFIG_DIR = Path(__file__).resolve().parents[3] / "config"
_PRESETS = sorted(_CONFIG_DIR.glob("models*.yaml"))

# Measured decode rate on gpt-4.1-mini during the 30-document run.
_MEASURED_TOK_S = 27.0
# Required rate must stay at or below this, i.e. ~45% headroom.
_MAX_REQUIRED_TOK_S = 20.0


def _tasks(path: Path) -> dict:
    doc = yaml.safe_load(path.read_text()) or {}
    return doc.get("tasks", doc)


def test_presets_were_found() -> None:
    """A glob that silently matches nothing would make every test below pass."""
    assert _PRESETS, f"no models*.yaml under {_CONFIG_DIR}"


@pytest.mark.parametrize("path", _PRESETS, ids=lambda p: p.name)
def test_no_task_demands_an_unachievable_decode_rate(path: Path) -> None:
    offenders = []
    for name, spec in _tasks(path).items():
        if not isinstance(spec, dict) or spec.get("stream"):
            continue
        mt, to = spec.get("max_tokens"), spec.get("timeout")
        if not mt or not to:
            continue
        rate = mt / to
        if rate > _MAX_REQUIRED_TOK_S:
            offenders.append(f"{name}: {mt}/{to} needs {rate:.1f} tok/s")
    assert not offenders, (
        f"{path.name} demands more than {_MAX_REQUIRED_TOK_S} tok/s "
        f"(measured {_MEASURED_TOK_S}):\n  " + "\n  ".join(offenders)
    )


@pytest.mark.parametrize("path", _PRESETS, ids=lambda p: p.name)
def test_every_task_declares_both_halves_of_the_budget(path: Path) -> None:
    """A task with max_tokens but no timeout inherits the router's default and
    escapes the check above -- the exact way this regressed before."""
    missing = [
        name for name, spec in _tasks(path).items()
        if isinstance(spec, dict) and spec.get("max_tokens") and not spec.get("timeout")
    ]
    assert not missing, f"{path.name}: max_tokens without timeout: {missing}"


@pytest.mark.parametrize("path", _PRESETS, ids=lambda p: p.name)
def test_long_output_tasks_are_streamed_or_generously_budgeted(path: Path) -> None:
    """At >=16k max_tokens even a healthy rate needs many minutes, so these
    must either stream or carry a timeout that reflects the real arithmetic."""
    for name, spec in _tasks(path).items():
        if not isinstance(spec, dict):
            continue
        mt = spec.get("max_tokens") or 0
        if mt < 16_384 or spec.get("stream"):
            continue
        assert spec.get("timeout", 0) >= mt / _MAX_REQUIRED_TOK_S, (
            f"{path.name}:{name} emits up to {mt} tokens un-streamed with a "
            f"{spec.get('timeout')}s timeout"
        )


def test_presets_agree_on_which_tasks_exist() -> None:
    """A task present in one preset and missing from another means switching
    provider mode raises KeyError mid-run, after paid work has happened."""
    names = {p.name: set(_tasks(p)) for p in _PRESETS}
    baseline_name, baseline = next(iter(names.items()))
    for other, task_set in names.items():
        assert task_set == baseline, (
            f"{other} and {baseline_name} differ: "
            f"only in {other}: {sorted(task_set - baseline)}; "
            f"missing from {other}: {sorted(baseline - task_set)}"
        )
