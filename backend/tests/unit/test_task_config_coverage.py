"""Every LLM task named in code must be defined in every preset.

This exists because `retrieval_rounds_plan` was called at retrieval.py:753 and
defined in NO preset. `router.chat()` raised `KeyError: Unknown task`, the
caller's `except Exception` fail-safe swallowed it, and two-round retrieval
silently never ran -- for every provider mode, since it was written. The output
was indistinguishable from the planner legitimately deciding one round sufficed,
so nothing ever surfaced it.

Two distinct failures are covered here, and neither is caught by the other:

  * A task missing from ALL presets -- dead code that looks live.
  * A task missing from ONE preset -- works until someone switches provider
    mode, then raises mid-run after paid work has already happened.

Tasks whose names are built at runtime (chosen by artifact type, or by a
summary-vs-artifact branch) cannot be found by scanning literals, so they are
listed explicitly below. That list is itself asserted against the config, so a
renamed task still fails here rather than in production.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

_ROOT = Path(__file__).resolve().parents[3]
_PRESETS = sorted((_ROOT / "config").glob("models*.yaml"))

# Task names assembled at runtime rather than passed as literals:
#   db_artifact_gen.py:288-291   task_name = "artifact_chunk_extract[_with_entities]"
#   db_artifact_rollup.py:321    _fam = "summary_merge" if atype == "Summary" else "artifact_merge"
_DYNAMIC_TASKS = {
    "artifact_chunk_extract",
    "artifact_chunk_extract_with_entities",
    "artifact_merge",
    "artifact_merge_evaluate",
    "artifact_merge_revise",
    "summary_merge",
    "summary_merge_evaluate",
    "summary_merge_revise",
}

_CHAT_LITERAL = re.compile(r"""\.chat\(\s*\n?\s*["']([a-z_]+)["']""")


def _tasks(path: Path) -> dict:
    doc = yaml.safe_load(path.read_text()) or {}
    return doc.get("tasks", doc)


def _called_tasks() -> set[str]:
    found: set[str] = set()
    for py in (_ROOT / "backend" / "app").rglob("*.py"):
        found |= {m.group(1) for m in _CHAT_LITERAL.finditer(py.read_text())}
    return found | _DYNAMIC_TASKS


def test_scan_actually_finds_call_sites() -> None:
    """A regex that silently matches nothing would make every test below
    vacuously pass -- the same class of bug being fixed here."""
    called = _called_tasks()
    assert len(called) > 20, f"only found {len(called)} tasks; the scan is broken"
    assert "entity_extract" in called, "known literal call site not found"


@pytest.mark.parametrize("path", _PRESETS, ids=lambda p: p.name)
def test_every_called_task_is_defined(path: Path) -> None:
    missing = sorted(_called_tasks() - set(_tasks(path)))
    assert not missing, (
        f"{path.name} is missing tasks that code calls: {missing}. "
        "router.chat() raises KeyError on these, which callers with a broad "
        "`except` will swallow -- the feature then silently never runs."
    )


def test_retrieval_rounds_plan_is_present_everywhere() -> None:
    """Named explicitly, not just covered by the sweep above, because this is
    the regression: two-round retrieval was dead in all four presets."""
    for path in _PRESETS:
        spec = _tasks(path).get("retrieval_rounds_plan")
        assert spec, f"{path.name} lost retrieval_rounds_plan"
        assert spec.get("response_format") == "json_object", (
            f"{path.name}: the planner's reply is parsed as JSON"
        )


def test_planner_is_sized_like_its_sibling() -> None:
    """It emits the same shape as query_decompose -- a short JSON object of
    sub-questions. A large max_tokens here would just be wasted budget, and a
    tiny one would truncate the second question."""
    for path in _PRESETS:
        t = _tasks(path)
        assert t["retrieval_rounds_plan"]["max_tokens"] == t["query_decompose"]["max_tokens"]


def test_planner_stays_on_a_cheap_model() -> None:
    """It runs on EVERY deep_research query, so it must not be routed to a
    synthesis-class model."""
    expensive = ("gpt-4.1", "gpt-5.4", "claude-opus", "claude-sonnet")
    for path in _PRESETS:
        model = _tasks(path)["retrieval_rounds_plan"]["model"]
        assert not model.startswith(expensive), f"{path.name}: planner on {model}"
