"""Refuse to mix vectors from two embedding models in one database.

Every vector column is `vector(1024)`, so a different model at the same
dimension (e.g. Qwen3-Embedding @ 1024 vs text-embedding-3-small @ 1024)
writes and queries without error -- and every distance, and every threshold
calibrated on those distances, silently becomes meaningless.

`graph_version_state.embedding_model` (migration 0010) records which model
wrote the database's vectors. NULL means "never stamped": a database built
before 0010 that already holds vectors is treated as text-embedding-3-small,
the only model that existed then; an empty one adopts whatever is configured.
"""
from __future__ import annotations

from sqlalchemy import text

from backend.app.db.session import session_scope

LEGACY_EMBEDDING_MODEL = "text-embedding-3-small"

# Tables whose vectors pin the model. Checking for ANY row is enough.
_VECTOR_TABLES = (
    "ontology_classes", "documents", "chunks", "entities",
    "intelligence_artifacts", "graph_relationships",
)

# Process-level memo: model names already verified against this database.
_verified: set[str] = set()


class EmbeddingModelMismatchError(RuntimeError):
    """The configured embedding model differs from the one in the database."""


def resolve_stamp(stamped: str | None, has_vectors: bool, configured: str) -> tuple[str, bool]:
    """Return (database model, whether to write `configured` as the stamp).

    Pure so the decision table is unit-testable without a database.
    """
    if stamped:
        return stamped, False
    if has_vectors:
        return LEGACY_EMBEDDING_MODEL, configured == LEGACY_EMBEDDING_MODEL
    return configured, True


async def ensure_embedding_model(embedder: object) -> None:
    """Raise EmbeddingModelMismatchError unless `embedder.model` matches the DB.

    Called before any code path that writes or queries vectors. Test doubles
    that are not a real `Embedder` are skipped: they never reach a database.
    """
    from backend.app.services.embeddings import Embedder

    if not isinstance(embedder, Embedder):
        return
    configured = str(embedder.model)
    if configured in _verified:
        return
    async with session_scope() as session:
        stamped = (
            await session.execute(
                text("SELECT embedding_model FROM graphrag.graph_version_state WHERE id = 1")
            )
        ).scalar_one_or_none()
        has_vectors = False
        if not stamped:
            probes = " UNION ALL ".join(
                f"(SELECT 1 FROM graphrag.{t} WHERE embedding IS NOT NULL LIMIT 1)"
                for t in _VECTOR_TABLES
            )
            has_vectors = (await session.execute(text(probes))).first() is not None
        db_model, write = resolve_stamp(stamped, has_vectors, configured)
        if db_model != configured:
            raise EmbeddingModelMismatchError(
                f"this database's vectors were written by '{db_model}', but the "
                f"active models.yaml embeds with '{configured}'. Mixing models "
                "makes every similarity meaningless. Switch the `embeddings:` "
                "entry back, or use a separate/wiped database for this model."
            )
        if write:
            await session.execute(
                text(
                    "UPDATE graphrag.graph_version_state SET embedding_model = :m "
                    "WHERE id = 1 AND embedding_model IS NULL"
                ),
                {"m": configured},
            )
    _verified.add(configured)
