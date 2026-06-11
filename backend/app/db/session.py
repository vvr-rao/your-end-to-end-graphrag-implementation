"""Async session helpers + FastAPI dependency.

Two ways to obtain a session:

  1. As a FastAPI dependency: `session: AsyncSession = Depends(get_session)`.
  2. As a context manager in service / CLI code:
     `async with session_scope() as session: ...`

Both go through the same factory; the context manager wraps in a
transaction with auto-rollback on exception.
"""
from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db.engine import get_sessionmaker


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency: yields a session, commits if the handler
    succeeds, rolls back on any exception."""
    async with get_sessionmaker()() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    """Service / CLI context manager. Same commit-on-success +
    rollback-on-exception contract as the FastAPI dependency."""
    async with get_sessionmaker()() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
