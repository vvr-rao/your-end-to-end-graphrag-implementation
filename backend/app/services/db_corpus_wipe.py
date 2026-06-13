"""Wipe the corpus-derived side of the graphrag schema.

Removes every row that was produced by ingestion + downstream passes
(register-documents, enrich-time, extract-entities, generate-artifacts,
conversations, retrieval). LEAVES the ontology side untouched
(ontology_classes / properties / instances + their materialized edges
tagged source='ONTOLOGY').

Used by `clear-corpus --yes` when the user wants a clean re-run of
the corpus pipeline without re-importing the ontology.

`graph_version_state` is NOT reset -- the counter keeps incrementing
across the wipe boundary so audits see a single monotonic timeline.
"""
from __future__ import annotations

from sqlalchemy import text

from backend.app.db.session import session_scope


# Tables we'll wipe, in dependency order. ON DELETE CASCADE handles
# most M2M rows automatically (artifact_sources cascades from
# intelligence_artifacts; chunks cascade from documents). We still
# DELETE each parent explicitly so the row-count report is meaningful.
_CORPUS_TABLES: tuple[str, ...] = (
    "retrieval_evidence",
    "retrieval_runs",
    "conversation_turns",
    "conversations",
    "intelligence_artifacts",   # cascades artifact_sources
    "entities",
    "time_instances",
    "chunks",                   # cascades from documents in DDL, but explicit DELETE is clearer
    "documents",
)


async def wipe_corpus(*, confirm: bool = False) -> dict[str, int]:
    """Delete all corpus-derived rows. Returns the before-counts.

    Refuses unless `confirm=True`. Also refuses if `ontology_classes`
    is empty -- that's the sanity check; if the ontology side is
    already gone you almost certainly meant `db-init --mode replace`
    not this command.
    """
    async with session_scope() as session:
        # Sanity: ontology must be present.
        r = await session.execute(
            text("SELECT count(*) FROM graphrag.ontology_classes")
        )
        n_classes = int(r.scalar_one())
        if n_classes == 0:
            raise RuntimeError(
                "refusing to clear corpus -- ontology_classes is empty. "
                "Use `db-init --mode replace` if you want to re-import "
                "the ontology too."
            )

        if not confirm:
            raise RuntimeError(
                "clear-corpus requires confirmation; pass --yes."
            )

        # Snapshot counts BEFORE so the report is meaningful.
        before: dict[str, int] = {}
        for tbl in _CORPUS_TABLES:
            r = await session.execute(
                text(f"SELECT count(*) FROM graphrag.{tbl}")
            )
            before[tbl] = int(r.scalar_one())

        # graph_relationships split by source -- only wipe non-ONTOLOGY.
        r = await session.execute(text(
            "SELECT relationship_source, count(*) FROM graphrag.graph_relationships "
            "GROUP BY relationship_source"
        ))
        edge_before = {src: int(n) for src, n in r.all()}
        before["graph_relationships_NON_ONTOLOGY"] = sum(
            n for src, n in edge_before.items() if src != "ONTOLOGY"
        )
        before["graph_relationships_ONTOLOGY_keep"] = edge_before.get("ONTOLOGY", 0)

        print(
            "[clear-corpus] before: "
            + ", ".join(f"{t}={n}" for t, n in before.items() if n > 0)
        )

        # Wipe corpus-derived edges (keep ONTOLOGY).
        await session.execute(text(
            "DELETE FROM graphrag.graph_relationships "
            "WHERE relationship_source <> 'ONTOLOGY'"
        ))

        # Wipe each corpus table. Order matters for FK constraints
        # not handled by ON DELETE CASCADE.
        for tbl in _CORPUS_TABLES:
            await session.execute(text(f"DELETE FROM graphrag.{tbl}"))

    # Verify after.
    after: dict[str, int] = {}
    async with session_scope() as session:
        for tbl in _CORPUS_TABLES:
            r = await session.execute(
                text(f"SELECT count(*) FROM graphrag.{tbl}")
            )
            after[tbl] = int(r.scalar_one())
        r = await session.execute(text(
            "SELECT count(*) FROM graphrag.graph_relationships"
        ))
        after["graph_relationships_total"] = int(r.scalar_one())

    print(
        "[clear-corpus] after:  "
        + ", ".join(f"{t}={n}" for t, n in after.items())
    )
    print(
        "[clear-corpus] DONE; ontology side untouched; "
        "graph_version_state preserved."
    )
    return before
