"""Corpus-extracted entities + temporal nodes.

`Entity` is an individual mentioned in a chunk (Honda, Vietnam,
Donald Trump). Its `class_id` FK points at an `OntologyClass` --
in line with the Phase 1 "no new classes in Phase 2" rule, an entity
ALWAYS instantiates an existing ontology class.

`TimeInstance` is a normalized temporal node (YEAR_2024 / Q1_2024 /
MONTH_2024_01) created by the temporal-enrichment pass in Milestone D.
"""
from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import CheckConstraint, Date, ForeignKey, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.db.base import (
    Base,
    metadata_jsonb_column,
    timestamp_column,
    uuid_pk_column,
)
from backend.app.db.models.ontology import EMBEDDING_DIM


class Entity(Base):
    __tablename__ = "entities"
    __table_args__ = (
        CheckConstraint(
            "status IN ('ACTIVE','STALE','DELETED')",
            name="entities_status_check",
        ),
    )

    id: Mapped[uuid.UUID] = uuid_pk_column()
    entity_identifier: Mapped[str] = mapped_column(
        Text, unique=True, nullable=False
    )  # IRI under .../entities#<slug>
    name: Mapped[str] = mapped_column(Text, nullable=False)
    normalized_name: Mapped[str] = mapped_column(
        Text, nullable=False
    )  # lowercased + punctuation-stripped for trgm dedup
    class_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("ontology_classes.id"),
        nullable=False,
    )
    iri: Mapped[str] = mapped_column(
        Text, nullable=False
    )  # mirrors entity_identifier; kept distinct so we can rebrand later
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM))
    status: Mapped[str] = mapped_column(
        Text, nullable=False, default="ACTIVE", server_default="ACTIVE"
    )
    extra_metadata: Mapped[dict[str, Any]] = metadata_jsonb_column()
    created_at: Mapped[datetime] = timestamp_column()


class TimeInstance(Base):
    __tablename__ = "time_instances"
    __table_args__ = (
        CheckConstraint(
            "time_level IN ('year','quarter','month','day','interval')",
            name="time_instances_level_check",
        ),
    )

    id: Mapped[uuid.UUID] = uuid_pk_column()
    time_identifier: Mapped[str] = mapped_column(
        Text, unique=True, nullable=False
    )  # YEAR_2024 / Q1_2024 / MONTH_2024_01
    time_level: Mapped[str] = mapped_column(Text, nullable=False)
    start_date: Mapped[date] = mapped_column(Date, nullable=False)
    end_date: Mapped[date] = mapped_column(Date, nullable=False)
    display_label: Mapped[str] = mapped_column(Text, nullable=False)
    parent_time_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("time_instances.id")
    )
    extra_metadata: Mapped[dict[str, Any]] = metadata_jsonb_column()
    created_at: Mapped[datetime] = timestamp_column()
