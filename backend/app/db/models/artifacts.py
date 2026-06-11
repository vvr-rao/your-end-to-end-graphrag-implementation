"""Intelligence artifacts + artifact_sources M2M for traceability.

Each row is an OWL individual under a VIAO class (Summary, Claim,
Finding, Observation, Insight, Recommendation). The `artifact_type`
column mirrors the VIAO subclass name; the IRI in
`artifact_identifier` follows `viao:<Type>_<uuid>` convention so
the artifact can be exported to RDF directly.

`ArtifactSource` is the M2M traceability link: artifact ← derived from → chunk.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import CheckConstraint, ForeignKey, Integer, Numeric, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.db.base import (
    Base,
    metadata_jsonb_column,
    timestamp_column,
    uuid_pk_column,
)
from backend.app.db.models.ontology import EMBEDDING_DIM


class IntelligenceArtifact(Base):
    __tablename__ = "intelligence_artifacts"
    __table_args__ = (
        CheckConstraint(
            "artifact_type IN ('Summary','Claim','Finding','Observation','Insight','Recommendation')",
            name="intel_artifacts_type_check",
        ),
        CheckConstraint(
            "status IN ('ACTIVE','STALE','RETIRED','DELETED')",
            name="intel_artifacts_status_check",
        ),
    )

    id: Mapped[uuid.UUID] = uuid_pk_column()
    artifact_identifier: Mapped[str] = mapped_column(
        Text, unique=True, nullable=False
    )  # viao:<Type>_<uuid> IRI
    artifact_type: Mapped[str] = mapped_column(Text, nullable=False)
    title: Mapped[str | None] = mapped_column(Text)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[Decimal | None] = mapped_column(Numeric)
    model_name: Mapped[str | None] = mapped_column(Text)
    prompt_version: Mapped[str | None] = mapped_column(Text)
    generation_run_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM))
    status: Mapped[str] = mapped_column(
        Text, nullable=False, default="ACTIVE", server_default="ACTIVE"
    )
    graph_version: Mapped[int] = mapped_column(Integer, nullable=False)
    extra_metadata: Mapped[dict[str, Any]] = metadata_jsonb_column()
    created_at: Mapped[datetime] = timestamp_column()


class ArtifactSource(Base):
    """M2M: artifact ← derived-from → chunk. The composite PK enforces
    that the same chunk can't be linked twice to the same artifact."""

    __tablename__ = "artifact_sources"

    artifact_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("intelligence_artifacts.id", ondelete="CASCADE"),
        primary_key=True,
    )
    chunk_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("chunks.id", ondelete="CASCADE"),
        primary_key=True,
    )
