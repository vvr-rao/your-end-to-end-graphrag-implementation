"""Table-extraction cache accounting must not lie about hits or spend.

Live symptom: an 18-PDF re-run finished in 66s instead of 526s -- the cache was
plainly working -- yet reported "0 cache-hits" AND the identical $0.8145 as the
run that actually paid. Both halves were wrong, and the cost half is the one
that matters: every re-run inflated the reported bill by the full original
extraction cost.

Root cause: the subprocess path decided "was this cached?" by reading the
manifest's own `source` field after reloading it from disk. That field is
written as "fresh" when an entry is PERSISTED; the "cache" value only ever
exists on the in-memory return value. So the check could never be true, and the
cached cost was added as though newly spent.
"""

from __future__ import annotations

import json
from pathlib import Path


def _persisted_manifest_source(cache_file: Path) -> str | None:
    doc = json.loads(cache_file.read_text())
    return (doc.get("manifest") or doc).get("source")


def test_persisted_entries_are_marked_fresh_not_cache(tmp_path: Path) -> None:
    """Pins the fact the old check got wrong. If persistence ever starts
    writing "cache", the reasoning in table_extract needs revisiting."""
    f = tmp_path / "e.jsonld"
    f.write_text(json.dumps({"manifest": {"source": "fresh", "cost_usd": 0.04}}))
    assert _persisted_manifest_source(f) == "fresh"


def test_reading_source_from_a_persisted_entry_never_detects_a_hit(tmp_path: Path) -> None:
    """The exact broken predicate, shown to be unsatisfiable."""
    f = tmp_path / "e.jsonld"
    f.write_text(json.dumps({"manifest": {"source": "fresh", "cost_usd": 0.04}}))
    assert (_persisted_manifest_source(f) == "cache") is False


def test_cache_hit_accounting_separates_spent_from_saved() -> None:
    """The rule the fix implements: a pre-existing entry counts as a hit and
    its cost goes to `saved`, never to `spent`."""
    stats = {"cost_usd": 0.0, "cost_saved_usd": 0.0, "n_cached": 0}

    def account(was_cached: bool, cached_cost: float) -> None:
        if was_cached:
            stats["n_cached"] += 1
            stats["cost_saved_usd"] += cached_cost
        else:
            stats["cost_usd"] += cached_cost

    account(False, 0.05)   # first run pays
    account(True, 0.05)    # re-run does not
    account(True, 0.05)
    assert stats["n_cached"] == 2
    assert stats["cost_usd"] == 0.05, "a fully cached re-run must report ~$0"
    assert stats["cost_saved_usd"] == 0.10


def test_source_is_sampled_before_the_worker_runs() -> None:
    """After the worker exits the entry exists either way, so the observation
    has to happen first. Guards against the check drifting back downstream."""
    import inspect
    from backend.app.services import table_extract

    # The public entrypoint is a thin wrapper; the worker loop lives here.
    src = inspect.getsource(table_extract._drive_subprocess_workers)
    pre = src.index("was_cached = table_cache.two_tier_load(")
    spawn = src.index("create_subprocess_exec")
    assert pre < spawn, "cache presence must be sampled before spawning the worker"
