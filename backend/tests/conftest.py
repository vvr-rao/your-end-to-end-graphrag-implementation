"""Top-level conftest — provides minimum env vars so Settings() loads in tests."""

from __future__ import annotations

import os

os.environ.setdefault("DATABASE_URL", "postgresql://x:y@localhost:5432/db")
os.environ.setdefault("BEARER_TOKEN", "test-bearer-token")


import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _db_url_switch_clears_cached_settings(monkeypatch):
    """Tests that point at a real database (ONTOLOGY_BUILD_TEST_DSN) do
    ``monkeypatch.setenv("DATABASE_URL", DSN)`` then ``reset_engine_cache()``.
    The engine reads the lru_cached ``get_settings()``, so once any earlier test
    had built Settings with the placeholder URL above, those tests connected as
    user "x" -- they only passed when run alone. Within tests,
    ``reset_engine_cache`` therefore also drops the cached Settings; and after a
    test that used it, both caches are cleared again so the next test neither
    reuses the DSN-bound engine (``set_pool_minimum`` refuses once an engine
    exists) nor the DSN settings."""
    from backend.app.core import config as _cfg
    from backend.app.db import engine as _eng

    original = _eng.reset_engine_cache
    used = []

    def _reset() -> None:
        used.append(True)
        _cfg.get_settings.cache_clear()
        original()

    monkeypatch.setattr(_eng, "reset_engine_cache", _reset)
    yield
    if used:
        _cfg.get_settings.cache_clear()
        original()
