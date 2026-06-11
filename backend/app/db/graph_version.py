"""Singleton graph-version counter.

Stamps every ingested artifact + relationship with a monotonic
integer so time-bounded queries ("show me everything before
graph_version 42") work without scanning timestamps. Bumps once per
register-document / update-document / delete-document invocation.
"""
from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


async def current_version(session: AsyncSession) -> int:
    """Return the current graph_version. Reads the singleton row."""
    row = await session.execute(
        text("SELECT current_value FROM graphrag.graph_version_state WHERE id = 1")
    )
    val = row.scalar_one_or_none()
    if val is None:
        raise RuntimeError(
            "graph_version_state singleton row missing -- did the migration run?"
        )
    return int(val)


async def bump_version(session: AsyncSession) -> int:
    """Increment the singleton counter and return the new value.

    Uses a single UPDATE ... RETURNING so concurrent bumps serialize at
    the row-lock level (the table has exactly one row; this is
    effectively a global lock). Callers MUST be inside an explicit
    transaction so they observe their own bump."""
    row = await session.execute(
        text(
            "UPDATE graphrag.graph_version_state "
            "SET current_value = current_value + 1 "
            "WHERE id = 1 "
            "RETURNING current_value"
        )
    )
    new_val = row.scalar_one()
    return int(new_val)
