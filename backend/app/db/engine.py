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

    return create_async_engine(
        dsn,
        connect_args=connect_args,
        # Supabase Session Pooler already pools server-side; we just
        # need a handful of client slots.
        pool_size=4,
        max_overflow=4,
        pool_pre_ping=True,           # quickly detect dropped pooler connections
        pool_recycle=300,             # recycle conns after 5 min idle
        echo=False,
        future=True,
    )


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
