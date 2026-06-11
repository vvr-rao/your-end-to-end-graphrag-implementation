"""SQLAlchemy 2.0 declarative base + shared column patterns.

Every model in `backend/app/db/models/` inherits from `Base`. Tables
live in the `graphrag` Postgres schema. We pin a custom MetaData with
that schema name so SQLAlchemy emits the right qualified DDL by default.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, MetaData, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

GRAPHRAG_SCHEMA = "graphrag"


class Base(DeclarativeBase):
    """All ORM models inherit from this. Tables go under `graphrag` schema."""

    metadata = MetaData(schema=GRAPHRAG_SCHEMA)


def uuid_pk_column() -> Mapped[uuid.UUID]:
    """Reusable UUID primary-key column with server-side default
    `gen_random_uuid()`. Use as `id: Mapped[uuid.UUID] = uuid_pk_column()`."""
    return mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=func.gen_random_uuid(),
    )


def timestamp_column(*, updates: bool = False) -> Mapped[datetime]:
    """A timestamptz column with NOT NULL + server default of `now()`.
    Pass `updates=True` to also bump on UPDATE."""
    kwargs: dict[str, Any] = {
        "nullable": False,
        "server_default": func.now(),
    }
    if updates:
        kwargs["onupdate"] = func.now()
    return mapped_column(DateTime(timezone=True), **kwargs)


def metadata_jsonb_column() -> Mapped[dict[str, Any]]:
    """A JSONB column for free-form metadata. Column name (and Python
    attribute name) is `extra_metadata` -- we can't name it `metadata`
    because SQLAlchemy reserves that attribute on DeclarativeBase, and
    upserts through the ORM CRUD path resolve string keys via
    Base.metadata and choke. Defaults to `{}` on insert."""
    return mapped_column(
        JSONB,
        nullable=False,
        server_default="{}",
        default=dict,
    )
