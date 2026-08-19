"""Phase 2 DB status report: graph_version, alembic revision, row counts."""
from __future__ import annotations

from sqlalchemy import text

from backend.app.db.session import session_scope

_REPORT_ORDER = [
    "ontology_classes",
    "ontology_object_properties",
    "ontology_data_properties",
    "ontology_instances",
    "documents",
    "chunks",
    "entities",
    "time_instances",
    "intelligence_artifacts",
    "artifact_sources",
    "graph_relationships",
    "conversations",
    "conversation_turns",
    "retrieval_runs",
    "retrieval_evidence",
    "term_aliases",
]


def _head_revision() -> str | None:
    """The newest migration on disk, for comparison against the DB's.

    Returns None if alembic's script directory can't be read (e.g. the
    CLI is being run from outside the repo), in which case the caller
    just skips the comparison.
    """
    try:
        from alembic.config import Config
        from alembic.script import ScriptDirectory

        return ScriptDirectory.from_config(Config("alembic.ini")).get_current_head()
    except Exception:
        return None


async def report_db_status() -> int:
    """Pretty-print Phase 2 DB status. Returns 0 on success."""
    async with session_scope() as session:
        # Alembic revision
        rev = await session.execute(
            text("SELECT version_num FROM graphrag.alembic_version LIMIT 1")
        )
        revision = rev.scalar_one_or_none() or "(none)"

        # graph_version
        gv = await session.execute(
            text("SELECT current_value FROM graphrag.graph_version_state WHERE id = 1")
        )
        graph_version = gv.scalar_one_or_none()

        # Per-table row counts. A table missing entirely (the DB predates
        # the migration that adds it) reports as "-" rather than blowing up
        # the whole status report.
        counts: dict[str, int | None] = {}
        for tbl in _REPORT_ORDER:
            try:
                r = await session.execute(
                    text(f"SELECT count(*) FROM graphrag.{tbl}")
                )
                counts[tbl] = int(r.scalar_one())
            except Exception:
                await session.rollback()
                counts[tbl] = None

    head = _head_revision()
    behind = head is not None and revision != head

    print("=" * 64)
    print("GRAPHRAG DB STATUS")
    print("=" * 64)
    print(f"  alembic revision   : {revision}")
    print(f"  graph_version      : {graph_version}")
    print()
    print("  Row counts:")
    for tbl in _REPORT_ORDER:
        n = counts[tbl]
        print(f"    {tbl:32s} {'-' if n is None else format(n, '>10,')}")
    # Pending migrations degrade features silently rather than loudly --
    # a missing `term_aliases`, for instance, just turns synonym expansion
    # off and logs a warning nobody reads. Say so where it will be seen.
    if behind:
        print()
        print(f"  !! MIGRATIONS PENDING: db is at {revision}, code ships {head}.")
        print("     Features added by the newer revisions are inactive until you run:")
        print("       uv run python -m backend.app.cli db-migrate")
    print("=" * 64)
    return 0
