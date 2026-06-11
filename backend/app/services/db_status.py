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
]


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

        # Per-table row counts
        counts: dict[str, int] = {}
        for tbl in _REPORT_ORDER:
            r = await session.execute(text(f"SELECT count(*) FROM graphrag.{tbl}"))
            counts[tbl] = int(r.scalar_one())

    print("=" * 64)
    print("GRAPHRAG DB STATUS")
    print("=" * 64)
    print(f"  alembic revision   : {revision}")
    print(f"  graph_version      : {graph_version}")
    print()
    print("  Row counts:")
    for tbl in _REPORT_ORDER:
        n = counts[tbl]
        print(f"    {tbl:32s} {n:>10,}")
    print("=" * 64)
    return 0
