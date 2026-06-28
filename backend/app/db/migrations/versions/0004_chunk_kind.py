"""add chunks.kind ('summary' | 'fulltext')

Adds a discriminator column to `graphrag.chunks` so a document can carry both
summary chunks (the default, summarized text that gets embedded) and optional
full-text chunks (verbatim original text, written by
`register-documents --full-text-chunks`). Forward-compatible: existing rows
backfill to 'summary' via server_default, so existing DBs stay valid and every
consumer (entity/artifact extraction, retrieval) behaves exactly as before.

Revision ID: 0004_chunk_kind
Revises: 0003_event_artifact
Create Date: 2026-06-27
"""
from __future__ import annotations

from typing import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = "0004_chunk_kind"
down_revision: str | Sequence[str] | None = "0003_event_artifact"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMA = "graphrag"
CHECK_NAME = "chunks_kind_check"


def upgrade() -> None:
    op.add_column(
        "chunks",
        sa.Column(
            "kind",
            sa.Text(),
            nullable=False,
            server_default="summary",
        ),
        schema=SCHEMA,
    )
    op.execute(
        f"ALTER TABLE {SCHEMA}.chunks "
        f"ADD CONSTRAINT {CHECK_NAME} CHECK (kind IN ('summary','fulltext'))"
    )


def downgrade() -> None:
    op.execute(f"ALTER TABLE {SCHEMA}.chunks DROP CONSTRAINT {CHECK_NAME}")
    op.drop_column("chunks", "kind", schema=SCHEMA)
