"""Wipe the ontology side of the graphrag schema.

Used by `db-init --mode replace` when the user re-imports an ontology
they've modified and wants stale rows removed (an UPSERT wouldn't
catch IRIs that disappeared from the source).

Refuses to run if dependent tables have rows (entities, relationships,
artifacts, etc.). Phase 2 milestones B+ populate those; the user has
to delete documents first via the document-lifecycle CLI.
"""
from __future__ import annotations

from sqlalchemy import text

from backend.app.db.session import session_scope


# Tables we will wipe in --mode replace, in dependency order.
_ONTOLOGY_TABLES: tuple[str, ...] = (
    "ontology_instances",
    "ontology_data_properties",
    "ontology_object_properties",
    "ontology_classes",
)

# Tables that depend on the ontology rows. If any of these have data,
# we refuse to wipe (would either FK-cascade-delete user data or
# leave dangling references).
_DEPENDENT_TABLES: tuple[str, ...] = (
    "entities",
    "graph_relationships",
    "intelligence_artifacts",
    "artifact_sources",
    "documents",
    "chunks",
    "retrieval_runs",
    "retrieval_evidence",
    "conversations",
    "conversation_turns",
)

# Rows that belong to the ontology itself rather than to user data, and so
# must not block an ontology replace.
#
# clear-corpus deliberately PRESERVES ontology-sourced edges
# (`DELETE ... WHERE relationship_source <> 'ONTOLOGY'`), but this guard
# counted those same preserved rows and refused the wipe -- making
# clear-corpus followed by `db-init --mode replace` a dead end, with the
# suggested recovery ("delete the dependent rows first") having no command
# behind it. The replace truncates the ontology tables anyway, so these rows
# are its own to remove.
#
# IS DISTINCT FROM rather than <>: the latter is NULL-unsafe. The column is
# NOT NULL today, so this is free insurance rather than a live fix.
_DEP_COUNT_FILTER: dict[str, str] = {
    "graph_relationships": " WHERE relationship_source IS DISTINCT FROM 'ONTOLOGY'",
}


async def wipe_ontology_tables(*, confirm: bool = False) -> dict[str, int]:
    """Truncate the 4 ontology tables. Returns row counts before wipe.

    Refuses to run unless `confirm=True` and the dependent tables are
    empty. Raises RuntimeError on either guard violation. The caller
    should prompt the user before passing confirm=True."""
    async with session_scope() as session:
        # Guard 1: dependent tables must be empty.
        nonempty_deps: list[tuple[str, int]] = []
        for tbl in _DEPENDENT_TABLES:
            r = await session.execute(
                text(f"SELECT count(*) FROM graphrag.{tbl}{_DEP_COUNT_FILTER.get(tbl, '')}")
            )
            n = int(r.scalar_one())
            if n > 0:
                nonempty_deps.append((tbl, n))
        if nonempty_deps:
            details = ", ".join(f"{t}={n}" for t, n in nonempty_deps)
            raise RuntimeError(
                f"refusing to wipe ontology tables -- dependent tables "
                f"have data ({details}). Run `clear-corpus --yes` to remove "
                f"documents, chunks, entities and artifacts first."
            )

        # Guard 2: caller must have confirmed.
        if not confirm:
            raise RuntimeError(
                "wipe requires confirmation; pass --yes to db-init."
            )

        # Count current rows for the report.
        before: dict[str, int] = {}
        for tbl in _ONTOLOGY_TABLES:
            r = await session.execute(text(f"SELECT count(*) FROM graphrag.{tbl}"))
            before[tbl] = int(r.scalar_one())

        print(
            f"[db-wipe] truncating: "
            + ", ".join(f"{t}={n}" for t, n in before.items())
        )

        # Truncate in one go. CASCADE is safe here because the
        # dependent-tables guard above confirmed all FK-targeting tables
        # are empty -- CASCADE wouldn't have anything to drop. Without
        # CASCADE, Postgres refuses TRUNCATE on any table referenced by
        # an FK even when the referencing table is empty.
        await session.execute(text(
            "TRUNCATE TABLE "
            + ", ".join(f"graphrag.{t}" for t in _ONTOLOGY_TABLES)
            + " CASCADE"
        ))

        # Reset graph_version to 0 so the next import is "version 1".
        await session.execute(text(
            "UPDATE graphrag.graph_version_state SET current_value = 0 "
            "WHERE id = 1"
        ))

    print("[db-wipe] DONE; graph_version reset to 0")
    return before
