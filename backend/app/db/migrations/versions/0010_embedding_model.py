"""record which embedding model wrote the database's vectors

Adds a nullable `graph_version_state.embedding_model`. Every vector column is
`vector(1024)`, so two models at the same dimension mix without error and
silently corrupt every similarity; `backend/app/db/embedding_guard.py` stamps
this column on first use and refuses a mismatched model afterwards. NULL on
an existing database means "pre-0010", read as text-embedding-3-small.

Revision ID: 0010_embedding_model
Revises: 0009_rel_embedding_index
Create Date: 2026-10-12
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0010_embedding_model"
down_revision: str | Sequence[str] | None = "0009_rel_embedding_index"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMA = "graphrag"


def upgrade() -> None:
    op.add_column(
        "graph_version_state",
        sa.Column("embedding_model", sa.Text(), nullable=True),
        schema=SCHEMA,
    )


def downgrade() -> None:
    op.drop_column("graph_version_state", "embedding_model", schema=SCHEMA)
