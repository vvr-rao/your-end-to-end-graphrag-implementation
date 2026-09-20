"""Async SQLAlchemy engine + session factory.

Built once per process. The engine is sized small for the Supabase
Session Pooler (`aws-*.pooler.supabase.com`), which already does its
own internal pooling to the backend Postgres -- we just need a few
client-side slots to avoid head-of-line blocking on long queries.
"""
from __future__ import annotations

from functools import lru_cache

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from backend.app.core.config import get_settings

# Ceiling on the auto-derived pool: a Supabase project has a finite
# client-connection budget shared by every process.
_MAX_POOL = 16

# Hard ceiling, including an explicit request. Past this a run is more likely
# to exhaust the project's connection budget than to go faster.
_ABS_MAX_POOL = 64

# Raised by `set_pool_minimum` before the engine is built, so a caller that
# knows it will run N concurrent workers gets a pool that can serve them.
_pool_minimum: int = 0


def set_pool_minimum(n: int) -> bool:
    """Ask for a pool able to serve `n` concurrent workers.

    Returns True if the request will be honoured. Must be called BEFORE the
    engine is created -- `get_engine` is lru_cached, so once a pool exists its
    size is fixed for the process and the caller must reduce concurrency
    instead.
    """
    global _pool_minimum
    if get_engine.cache_info().currsize:        # engine already built
        return False
    _pool_minimum = max(_pool_minimum, min(int(n), _ABS_MAX_POOL))
    return int(n) <= _ABS_MAX_POOL


def _ssl_context_for(dsn: str):
    """Build an SSL context for the DSN if the host is Supabase. The
    pooler presents a self-signed-chain cert from the sandbox's
    perspective (we lack Supabase's root CA), so we skip cert
    verification while keeping TLS 1.3 encryption + cipher negotiation
    on. The same approach the user's other Supabase clients use."""
    import ssl as _ssl

    if not any(s in dsn for s in (".supabase.co", ".supabase.com")):
        return None
    ctx = _ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = _ssl.CERT_NONE
    return ctx


@lru_cache(maxsize=1)
def get_engine() -> AsyncEngine:
    """Single shared engine for the whole process. The connection-string
    validator in core/config.py has already normalized the scheme to
    `postgresql+asyncpg://` and force-appended `sslmode=require` on
    Supabase hosts. We strip `sslmode` (libpq-only param) here and pass
    a real `ssl=` SSLContext via connect_args instead."""
    settings = get_settings()
    raw_dsn = str(settings.database_url)
    # asyncpg doesn't accept libpq-style `sslmode` -- strip it.
    if "?" in raw_dsn:
        head, _, query = raw_dsn.partition("?")
        params = [
            kv for kv in query.split("&")
            if kv and not kv.startswith("sslmode=") and not kv.startswith("ssl=")
        ]
        dsn = head + ("?" + "&".join(params) if params else "")
    else:
        dsn = raw_dsn

    connect_args: dict = {}
    ssl_ctx = _ssl_context_for(dsn)
    if ssl_ctx is not None:
        connect_args["ssl"] = ssl_ctx

    # Pool sizing. The old fixed 4 + 4 was chosen for the API, where a request
    # holds a connection briefly. The batch CLI is the opposite: every worker
    # holds a session across several slow LLM calls, so with
    # `concurrency.entity_extraction: 64` against 8 slots, 36 of 62 chunks died
    # on `QueuePool limit of size 4 overflow 4 reached ... timeout 30.00`
    # (measured 2026-09-20). A 20-chunk smoke test passed clean, which is why
    # the mismatch stayed hidden.
    #
    # `database.pool_size` / `database.max_overflow` make it explicit, and the
    # default now follows the largest configured concurrency rather than a
    # constant -- capped, because a Supabase project has a finite client
    # budget and one runaway process should not consume it.
    _db_cfg = (settings.app_config.get("database", {}) or {})
    _conc = (settings.app_config.get("concurrency", {}) or {})
    _want = max([4, *(int(v) for v in _conc.values()
                      if isinstance(v, (int, float)) and v)])
    _derived = min(_MAX_POOL, _want)
    _pool = int(_db_cfg.get("pool_size") or _derived)
    _overflow = int(_db_cfg.get("max_overflow") or _pool)
    # A caller that declared its worker count wins over the derived default:
    # the pool exists to serve the work, not the other way round.
    if _pool_minimum and (_pool + _overflow) < _pool_minimum:
        _pool = min(_ABS_MAX_POOL, _pool_minimum)
        _overflow = 0
    return create_async_engine(
        dsn,
        connect_args=connect_args,
        pool_size=_pool,
        max_overflow=_overflow,
        pool_timeout=float(_db_cfg.get("pool_timeout") or 60),
        pool_pre_ping=True,           # quickly detect dropped pooler connections
        pool_recycle=300,             # recycle conns after 5 min idle
        echo=False,
        future=True,
    )


def pool_capacity() -> int:
    """Total connections this process may hold (pool + overflow).

    The CLI caps worker concurrency at this, because a worker holds its
    session across several slow LLM calls: more workers than connections does
    not go faster, it just queues until `pool_timeout` and then fails the
    chunk.
    """
    e = get_engine()
    return int(e.pool.size()) + int(getattr(e.pool, "_max_overflow", 0) or 0)


@lru_cache(maxsize=1)
def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    """Factory for AsyncSession; one per request / per CLI invocation."""
    return async_sessionmaker(
        get_engine(),
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )


def reset_engine_cache() -> None:
    """Clear the engine + sessionmaker caches so the next get_engine()
    builds a fresh engine bound to the current event loop.

    Use this BETWEEN asyncio.run() boundaries (e.g. multi-step CLI
    commands that interleave Alembic with async ops). DOES NOT await
    engine.dispose() -- the old engine's loop is already closed at
    that point, and trying to await on it raises 'Event loop is
    closed'. Underlying asyncpg connections leak briefly but the
    Supabase pooler reclaims them via idle timeout."""
    get_engine.cache_clear()
    get_sessionmaker.cache_clear()


async def dispose_engine() -> None:
    """Tear down the global engine cleanly (test fixtures + graceful
    shutdown WITHIN the same event loop). For CLI use that crosses
    loop boundaries, call `reset_engine_cache()` instead."""
    if get_engine.cache_info().currsize > 0:
        engine = get_engine()
        await engine.dispose()
    get_engine.cache_clear()
    get_sessionmaker.cache_clear()
