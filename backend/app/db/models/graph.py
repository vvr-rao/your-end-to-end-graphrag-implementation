"""Single canonical edge table for the knowledge graph.

Polymorphic: `source_node_type` + `source_node_id` (UUID) point at one
of the entity tables (ontology_class / document / chunk / entity /
time_instance / intelligence_artifact). Same for target. The
`predicate_iri` holds the VIAO/FOAF/ORG/SKOS/W3C predicate.

`relationship_source` distinguishes provenance:
- ONTOLOGY            -- imported from a source ontology's TBox
- DOCUMENT_EXTRACTION -- pulled out of a chunk by the LLM
- LLM_INFERENCE       -- generated alongside an artifact
- TIME_ENRICHMENT     -- created by the temporal-enrichment pass
- MANUAL              -- inserted via API by a user
"""
from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import Boolean, CheckConstraint, ForeignKey, Integer, Numeric, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.db.base import (
    Base,
    metadata_jsonb_column,
    timestamp_column,
    uuid_pk_column,
)


class GraphRelationship(Base):
    __tablename__ = "graph_relationships"
    __table_args__ = (
        CheckConstraint(
            "source_node_type IN ('ontology_class','document','chunk','entity','time_instance','intelligence_artifact')",
            name="rel_source_node_type_check",
        ),
        CheckConstraint(
            "target_node_type IN ('ontology_class','document','chunk','entity','time_instance','intelligence_artifact')",
            name="rel_target_node_type_check",
        ),
        CheckConstraint(
            "relationship_source IN ('ONTOLOGY','DOCUMENT_EXTRACTION','LLM_INFERENCE','TIME_ENRICHMENT','MANUAL')",
            name="rel_relationship_source_check",
        ),
    )

    id: Mapped[uuid.UUID] = uuid_pk_column()
    source_node_type: Mapped[str] = mapped_column(Text, nullable=False)
    source_node_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    target_node_type: Mapped[str] = mapped_column(Text, nullable=False)
    target_node_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    predicate_iri: Mapped[str] = mapped_column(Text, nullable=False)
    predicate_label: Mapped[str | None] = mapped_column(Text)
    relationship_type: Mapped[str | None] = mapped_column(Text)
    relationship_source: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[Decimal | None] = mapped_column(Numeric)
    is_authoritative: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True, server_default="true"
    )
    source_document_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="SET NULL")
    )
    source_chunk_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("chunks.id", ondelete="SET NULL")
    )
    source_artifact_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("intelligence_artifacts.id", ondelete="SET NULL"),
    )
    generation_run_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    graph_version: Mapped[int] = mapped_column(Integer, nullable=False)
    extra_metadata: Mapped[dict[str, Any]] = metadata_jsonb_column()
    created_at: Mapped[datetime] = timestamp_column()
