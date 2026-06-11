"""Postgres size monitor for the graphrag schema.

Prints per-table size + total. Exits non-zero if total > 400 MB so
shell wrappers can guard against the 500 MB Supabase free-tier cap.
"""
from __future__ import annotations

import sys
from sqlalchemy import text

from backend.app.db.session import session_scope

# Hard cap: 500 MB. Warn at 80%.
_CAP_BYTES = 500 * 1024 * 1024
_WARN_AT_BYTES = 400 * 1024 * 1024


def _fmt(b: int) -> str:
    """Human-readable bytes."""
    if b < 1024:
        return f"{b} B"
    if b < 1024 * 1024:
        return f"{b/1024:.1f} KB"
    return f"{b/1024/1024:.1f} MB"


async def report_db_size() -> int:
    """Query pg_total_relation_size for every graphrag table.
    Returns the total bytes; exits 1 if > _WARN_AT_BYTES."""
    sql = text("""
        SELECT
            tablename AS table_name,
            pg_total_relation_size('graphrag.' || quote_ident(tablename)) AS bytes
        FROM pg_tables
        WHERE schemaname = 'graphrag'
        ORDER BY bytes DESC
    """)

    async with session_scope() as session:
        rows = (await session.execute(sql)).all()

    total = sum(int(r.bytes) for r in rows)

    print("=" * 64)
    print("GRAPHRAG DB SIZE REPORT")
    print("=" * 64)
    if not rows:
        print("  (no graphrag tables yet -- run `alembic upgrade head`)")
        return 0

    for row in rows:
        print(f"  {row.table_name:32s} {_fmt(int(row.bytes)):>12}")
    print("-" * 64)
    print(f"  {'TOTAL':32s} {_fmt(total):>12} / 500.0 MB cap")
    headroom = _CAP_BYTES - total
    print(f"  {'HEADROOM':32s} {_fmt(max(0, headroom)):>12}")

    if total > _WARN_AT_BYTES:
        pct = 100 * total / _CAP_BYTES
        print(f"\n  WARNING: at {pct:.1f}% of cap. Consider archiving STALE artifacts.")
        sys.exit(1)
    print(f"\n  STATUS: OK ({100*total/_CAP_BYTES:.1f}% of 500 MB cap)")
    return 0
