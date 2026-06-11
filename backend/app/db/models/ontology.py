"""Ontology side of the schema: classes + properties + instances.

These tables hold the canonical ontology imported from the Phase-1
merge folder via `import-ontology`. They are EFFECTIVELY READ-ONLY
during Phase 2 (per the spec: "no new ontology classes shall be
created during Phase 2"). New entities go in `entities`; new
relationships go in `graph_relationships`.

The full owlready2-style dict-of-dicts entry is preserved in the
`metadata` JSONB column so we can round-trip back to OWL via
`ontology_export.write_owl`.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import Boolean, Text
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.db.base import (
    Base,
    metadata_jsonb_column,
    timestamp_column,
    uuid_pk_column,
)

EMBEDDING_DIM = 1024


class OntologyClass(Base):
    __tablename__ = "ontology_classes"

    id: Mapped[uuid.UUID] = uuid_pk_column()
    iri: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    label: Mapped[str | None] = mapped_column(Text)
    description: Mapped[str | None] = mapped_column(Text)
    namespace: Mapped[str] = mapped_column(Text, nullable=False)
    source_ontology: Mapped[str | None] = mapped_column(Text)
    is_viao_class: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="false"
    )
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM))
    extra_metadata: Mapped[dict[str, Any]] = metadata_jsonb_column()
    created_at: Mapped[datetime] = timestamp_column()


class OntologyObjectProperty(Base):
    __tablename__ = "ontology_object_properties"

    id: Mapped[uuid.UUID] = uuid_pk_column()
    iri: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    label: Mapped[str | None] = mapped_column(Text)
    description: Mapped[str | None] = mapped_column(Text)
    namespace: Mapped[str] = mapped_column(Text, nullable=False)
    source_ontology: Mapped[str | None] = mapped_column(Text)
    extra_metadata: Mapped[dict[str, Any]] = metadata_jsonb_column()
    created_at: Mapped[datetime] = timestamp_column()


class OntologyDataProperty(Base):
    __tablename__ = "ontology_data_properties"

    id: Mapped[uuid.UUID] = uuid_pk_column()
    iri: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    label: Mapped[str | None] = mapped_column(Text)
    description: Mapped[str | None] = mapped_column(Text)
    namespace: Mapped[str] = mapped_column(Text, nullable=False)
    source_ontology: Mapped[str | None] = mapped_column(Text)
    extra_metadata: Mapped[dict[str, Any]] = metadata_jsonb_column()
    created_at: Mapped[datetime] = timestamp_column()


class OntologyInstance(Base):
    """Individuals declared in the source ontologies (e.g. time:January
    as an instance of time:MonthOfYear). DISTINCT from `entities` which
    holds individuals extracted from the corpus."""

    __tablename__ = "ontology_instances"

    id: Mapped[uuid.UUID] = uuid_pk_column()
    iri: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    label: Mapped[str | None] = mapped_column(Text)
    description: Mapped[str | None] = mapped_column(Text)
    namespace: Mapped[str] = mapped_column(Text, nullable=False)
    type_iri: Mapped[str | None] = mapped_column(Text)
    source_ontology: Mapped[str | None] = mapped_column(Text)
    extra_metadata: Mapped[dict[str, Any]] = metadata_jsonb_column()
    created_at: Mapped[datetime] = timestamp_column()
