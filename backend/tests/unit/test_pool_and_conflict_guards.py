"""Guards for two failures that each destroyed a paid extraction run.

Both were found on 2026-09-20 and together burned $1.94 on a 30-document
corpus while writing nothing and exiting 0.
"""
from __future__ import annotations

import inspect

from backend.app.services import db_entity_extract as dee


def test_entity_insert_uses_on_conflict_do_nothing():
    """A bare INSERT let ONE colliding identifier roll back the whole batch.

    `_entity_iri` hashes (canonical_name, class_iri) while the reuse check
    keys on (normalized_name, class_id), and name collapse renames entities
    AFTER that check -- so a rename onto a name a previous run stored
    collided, and every LLM call in the run was lost. That made the project's
    own "smoke-test with --limit N, then scale up" workflow impossible.
    """
    src = inspect.getsource(dee.extract_entities)
    assert "on_conflict_do_nothing" in src, (
        "entity insert must tolerate a duplicate entity_identifier; a bare "
        "INSERT loses the entire batch on one collision"
    )
    assert 'index_elements=["entity_identifier"]' in src


def test_conflicting_mints_are_counted_not_silent():
    """A reused mint must be reported, not silently folded away."""
    src = inspect.getsource(dee.extract_entities)
    assert "_conflicts" in src
    assert "already existed by" in src


def test_request_wins_when_the_pool_can_still_be_sized(monkeypatch):
    """`--concurrency 64` must get 64 connections, not a quiet reduction.

    The pool exists to serve the work. Capping an explicit request would also
    break the documented "CLI flag always wins" contract
    (test_cli_concurrency.test_cli_flag_always_wins).
    """
    from backend.app.cli import main as cli

    monkeypatch.setattr(
        "backend.app.db.engine.set_pool_minimum", lambda n: True, raising=False
    )
    assert cli._cap_to_pool(64) == 64


def test_capped_only_once_the_pool_is_fixed(monkeypatch):
    """Once the engine exists its pool size cannot change, so the only honest
    option left is to run fewer workers than connections."""
    from backend.app.cli import main as cli

    monkeypatch.setattr(
        "backend.app.db.engine.set_pool_minimum", lambda n: False, raising=False
    )
    monkeypatch.setattr(
        "backend.app.db.engine.pool_capacity", lambda: 32, raising=False
    )
    assert cli._cap_to_pool(64) == 32
    assert cli._cap_to_pool(8) == 8


def test_cap_to_pool_fails_open(monkeypatch):
    """If capacity cannot be read, never block the run."""
    from backend.app.cli import main as cli

    def boom(*_a):
        raise RuntimeError("no engine")

    monkeypatch.setattr(
        "backend.app.db.engine.set_pool_minimum", boom, raising=False
    )
    assert cli._cap_to_pool(64) == 64


def test_pool_is_not_a_fixed_four():
    """The pool must scale with configured concurrency, not sit at 4 + 4."""
    import backend.app.db.engine as eng

    src = inspect.getsource(eng.get_engine)
    assert "pool_size=4," not in src, "pool must not be hardcoded to 4"
    assert "concurrency" in src, "pool should derive from configured concurrency"
    assert "_MAX_POOL" in src, "auto-derived pool must stay capped"
    assert "_pool_minimum" in src, "a declared worker count must size the pool"


# ---------------------------------------------------------------------------
# An explicit database.pool_size is the SERVER'S limit, not a suggestion.
# Found 2026-09-27, before an end-to-end run, by checking rather than assuming.
# ---------------------------------------------------------------------------

def _fake_db_cfg(monkeypatch, db: dict, concurrency: dict | None = None):
    from backend.app.db import engine as eng

    cfg = {"database": db, "concurrency": concurrency or {"entity_extraction": 64}}
    monkeypatch.setattr(
        eng, "get_settings",
        lambda: type("S", (), {"app_config": cfg})(), raising=True,
    )
    monkeypatch.setattr(eng, "_pool_minimum", 0, raising=False)
    return eng


def test_explicit_pool_size_is_a_hard_ceiling(monkeypatch):
    """`set_pool_minimum` must refuse to grow the pool past a declared size.

    Supabase free tier in session mode caps the PROJECT at 15 client
    connections. `database.pool_size: 10` was silently overridden by
    `set_pool_minimum(64)` -- which `_resolve_concurrency` calls for the
    config's own `concurrency.entity_extraction: 64`, no CLI flag involved --
    and the engine opened a 64-connection pool, reproducing the very
    EMAXCONNSESSION failure the pin was added to prevent.
    """
    eng = _fake_db_cfg(monkeypatch, {"pool_size": 10, "max_overflow": 2})
    assert eng.configured_pool_ceiling() == 12
    assert eng.set_pool_minimum(64) is False, (
        "a request beyond the declared ceiling must be refused so the CALLER "
        "reduces concurrency; growing the pool past it fails at the server"
    )
    assert eng.set_pool_minimum(12) is True


def test_no_explicit_pool_size_still_scales(monkeypatch):
    """A self-hosted / docker Postgres has no such cap -- do not cripple it."""
    eng = _fake_db_cfg(monkeypatch, {"pool_size": None, "max_overflow": None})
    assert eng.configured_pool_ceiling() is None
    assert eng.set_pool_minimum(64) is True


def test_only_db_bound_stages_are_capped():
    """Capping a stage that holds no DB session slows it for nothing.

    Measured 2026-09-26 on one 15-client Supabase DB: register-documents ran
    42 docs at concurrency 64 with zero pool errors, while extract-entities at
    32 lost 62 of 71 chunks to EMAXCONNSESSION.
    """
    from backend.app.cli import main as cli

    assert "entity_extraction" in cli._POOL_BOUND_STAGES
    assert "artifact_generation" in cli._POOL_BOUND_STAGES
    assert "summarization" not in cli._POOL_BOUND_STAGES, (
        "summarization is LLM+disk work whose DB writes happen after fan-in; "
        "capping it to the pool slows the long pole of ingestion for no gain"
    )
