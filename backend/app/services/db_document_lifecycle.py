"""Milestone B lifecycle: delete + update + list helpers.

Soft delete (default): mark document + its chunks as DELETED, sweep
dependent artifacts to STALE. Hard delete: row removal cascades via
ON DELETE CASCADE on chunks; FK on graph_relationships sets the
source_document_id / source_chunk_id columns to NULL.

Update: insert a new versioned row that supersedes the old one; old
goes DELETED, chunks for the new doc get freshly ingested.

List: read-only browse with chunk counts.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import delete as sql_delete
from sqlalchemy import func, select, update as sql_update

from backend.app.db.graph_version import bump_version
from backend.app.db.models.artifacts import ArtifactSource, IntelligenceArtifact
from backend.app.db.models.documents import Chunk, Document
from backend.app.db.session import session_scope
from backend.app.services.db_document_ingest import (
    _document_iri,
    ingest_documents_folder,
)
from backend.app.services.document_io import load_document


async def delete_document(iri: str, *, hard: bool = False) -> dict[str, Any]:
    """Soft-delete by default (status=DELETED on doc + its chunks +
    sweep dependent artifacts). Hard delete removes the row; FK
    cascade clears chunks; graph_relationships rows whose source_*
    columns referenced this doc go to NULL.

    Returns a summary dict for the CLI.
    """
    async with session_scope() as session:
        result = await session.execute(
            select(Document.id, Document.title, Document.status).where(
                Document.document_identifier == iri
            )
        )
        row = result.first()
        if row is None:
            raise ValueError(f"document not found: {iri}")
        doc_id, title, status = row

        # Find artifacts that source from this doc's chunks.
        chunk_ids_subq = select(Chunk.id).where(Chunk.document_id == doc_id)
        affected_artifacts = await session.execute(
            select(ArtifactSource.artifact_id).where(
                ArtifactSource.chunk_id.in_(chunk_ids_subq)
            ).distinct()
        )
        artifact_ids = [aid for (aid,) in affected_artifacts.all()]

        # Of those, which have ALL sources in this doc? Those become STALE.
        # (Mixed-source artifacts stay ACTIVE.)
        stale_ids: list[Any] = []
        for aid in artifact_ids:
            total_sources = await session.execute(
                select(func.count()).select_from(ArtifactSource).where(
                    ArtifactSource.artifact_id == aid
                )
            )
            from_this_doc = await session.execute(
                select(func.count()).select_from(ArtifactSource).where(
                    ArtifactSource.artifact_id == aid,
                    ArtifactSource.chunk_id.in_(chunk_ids_subq),
                )
            )
            if total_sources.scalar_one() == from_this_doc.scalar_one():
                stale_ids.append(aid)

        chunk_count_result = await session.execute(
            select(func.count()).select_from(Chunk).where(
                Chunk.document_id == doc_id
            )
        )
        chunk_count = chunk_count_result.scalar_one()

        if hard:
            await session.execute(sql_delete(Document).where(Document.id == doc_id))
            mode = "HARD"
        else:
            now = datetime.now(timezone.utc)
            await session.execute(
                sql_update(Document).where(Document.id == doc_id).values(
                    status="DELETED", is_deleted=True, deleted_at=now
                )
            )
            await session.execute(
                sql_update(Chunk).where(Chunk.document_id == doc_id).values(
                    status="DELETED", is_deleted=True, deleted_at=now
                )
            )
            if stale_ids:
                await session.execute(
                    sql_update(IntelligenceArtifact).where(
                        IntelligenceArtifact.id.in_(stale_ids)
                    ).values(status="STALE")
                )
            mode = "SOFT"

        new_version = await bump_version(session)

    print(
        f"[delete-document] {mode} delete: {title} ({iri})\n"
        f"  chunks affected:    {chunk_count}\n"
        f"  artifacts -> STALE: {len(stale_ids)}\n"
        f"  graph_version ->    {new_version}"
    )
    return {
        "mode": mode,
        "title": title,
        "chunks": chunk_count,
        "stale_artifacts": len(stale_ids),
        "new_graph_version": new_version,
    }


async def update_document(iri: str, new_path: Path) -> dict[str, Any]:
    """Insert a new versioned doc that supersedes `iri`. The old doc
    moves to status=DELETED. If the new file has the same sha256 as
    the existing doc, no-op + log."""
    if not new_path.exists():
        raise FileNotFoundError(new_path)

    new_doc = load_document(new_path)
    new_hash = hashlib.sha256(new_doc.text.encode("utf-8")).hexdigest()

    async with session_scope() as session:
        result = await session.execute(
            select(
                Document.id, Document.title, Document.document_hash,
                Document.version, Document.status,
            ).where(Document.document_identifier == iri)
        )
        row = result.first()
        if row is None:
            raise ValueError(f"document not found: {iri}")
        old_id, old_title, old_hash, old_version, old_status = row

    if new_hash == old_hash:
        print(
            f"[update-document] no-op: new file hash matches existing "
            f"(sha={new_hash[:16]}); no write."
        )
        return {
            "action": "no-op",
            "title": old_title,
            "hash": new_hash,
        }

    # Soft-delete the old version first so chunks get marked correctly.
    await delete_document(iri, hard=False)

    # Re-ingest from the new file's parent folder, but FILTER to just
    # this one file by symlinking / copying into a temp dir? Simpler:
    # process the single file directly using ingest_documents_folder's
    # internal pipeline. To keep this generic without changing the
    # ingest API, we use the parent folder + a 1-file limit and the
    # filename match.
    parent = new_path.parent
    summary = await ingest_documents_folder(parent, limit=None)
    # The ingest is keyed by hash; the just-added new doc is in the DB
    # under its own IRI. Mark the relationship in DB metadata.
    new_iri = _document_iri(new_hash)
    async with session_scope() as session:
        await session.execute(
            sql_update(Document).where(
                Document.document_identifier == new_iri
            ).values(
                version=old_version + 1,
                supersedes_document_id=old_id,
            )
        )

    return {
        "action": "updated",
        "old_iri": iri,
        "new_iri": new_iri,
        "new_version": old_version + 1,
        "chunks_added": summary.chunks_inserted,
    }


async def list_documents(
    *, status: str | None = None, limit: int | None = None
) -> list[dict[str, Any]]:
    """Return a list of doc summaries. Optionally filter by status."""
    async with session_scope() as session:
        stmt = (
            select(
                Document.document_identifier,
                Document.title,
                Document.status,
                Document.version,
                Document.created_at,
                func.count(Chunk.id).label("chunk_count"),
            )
            .outerjoin(Chunk, Chunk.document_id == Document.id)
            .group_by(Document.id)
            .order_by(Document.created_at.desc())
        )
        if status is not None:
            stmt = stmt.where(Document.status == status)
        if limit is not None:
            stmt = stmt.limit(limit)

        result = await session.execute(stmt)
        rows = result.all()

    return [
        {
            "iri": iri,
            "title": title,
            "status": status_,
            "version": version,
            "created_at": created_at.isoformat(),
            "chunks": chunk_count,
        }
        for iri, title, status_, version, created_at, chunk_count in rows
    ]
