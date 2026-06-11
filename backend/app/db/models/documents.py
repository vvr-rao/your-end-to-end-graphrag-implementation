"""Documents + chunks. Instances of viao:Document + viao:Chunk."""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import CheckConstraint, ForeignKey, Integer, Text, Boolean, DateTime
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.db.base import (
    Base,
    metadata_jsonb_column,
    timestamp_column,
    uuid_pk_column,
)
from backend.app.db.models.ontology import EMBEDDING_DIM


class Document(Base):
    __tablename__ = "documents"
    __table_args__ = (
        CheckConstraint(
            "status IN ('ACTIVE','STALE','DELETED')",
            name="documents_status_check",
        ),
    )

    id: Mapped[uuid.UUID] = uuid_pk_column()
    document_identifier: Mapped[str] = mapped_column(
        Text, unique=True, nullable=False
    )  # viao:Document IRI
    title: Mapped[str] = mapped_column(Text, nullable=False)
    file_name: Mapped[str | None] = mapped_column(Text)
    file_path: Mapped[str | None] = mapped_column(Text)
    file_type: Mapped[str | None] = mapped_column(Text)
    source_system: Mapped[str | None] = mapped_column(Text)
    source_uri: Mapped[str | None] = mapped_column(Text)
    document_hash: Mapped[str] = mapped_column(Text, nullable=False)
    text_summary: Mapped[str | None] = mapped_column(Text)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM))
    status: Mapped[str] = mapped_column(
        Text, nullable=False, default="ACTIVE", server_default="ACTIVE"
    )
    version: Mapped[int] = mapped_column(
        Integer, nullable=False, default=1, server_default="1"
    )
    is_deleted: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="false"
    )
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    supersedes_document_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id")
    )
    extra_metadata: Mapped[dict[str, Any]] = metadata_jsonb_column()
    created_at: Mapped[datetime] = timestamp_column()
    updated_at: Mapped[datetime] = timestamp_column(updates=True)

    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class Chunk(Base):
    __tablename__ = "chunks"
    __table_args__ = (
        CheckConstraint(
            "status IN ('ACTIVE','STALE','DELETED')",
            name="chunks_status_check",
        ),
    )

    id: Mapped[uuid.UUID] = uuid_pk_column()
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    chunk_identifier: Mapped[str] = mapped_column(
        Text, unique=True, nullable=False
    )  # viao:Chunk IRI
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    page_number: Mapped[int | None] = mapped_column(Integer)
    section_title: Mapped[str | None] = mapped_column(Text)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM))
    status: Mapped[str] = mapped_column(
        Text, nullable=False, default="ACTIVE", server_default="ACTIVE"
    )
    is_deleted: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="false"
    )
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    extra_metadata: Mapped[dict[str, Any]] = metadata_jsonb_column()
    created_at: Mapped[datetime] = timestamp_column()

    document: Mapped["Document"] = relationship(back_populates="chunks")
