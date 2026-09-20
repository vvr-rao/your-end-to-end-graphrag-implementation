"""add graphrag.graph_relationships.embedding (vector 1024)

Extracted relationships already carry the sentence that supports them in
`extra_metadata.evidence`, and `relatedTo` edges carry the free-text
`relation` phrase, but neither was ever embedded -- so retrieval could
only use an edge as adjacency ("A touches B"), never as meaning ("A was
accused of defrauding B"). Every edge was worth the same 0.7^hop.

This column stores one vector per edge over
`<source> <relation> <target>. <evidence>` so that:

  - a question's parsed relationship can be matched against stored edges
    (relationship-aware hopping), and
  - a vague question ("who was accused of fraud") can rank the members of
    a class by how well their edges match the relation phrase.

Nullable and backfilled out of band (`embed-relationships`): an older
database keeps working, retrieval falls back to the broad walk wherever
the column is NULL.

Revision ID: 0008_relationship_embedding
Revises: 0007_manual_aliases
Create Date: 2026-09-20
"""
from __future__ import annotations

from typing import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector


revision: str = "0008_relationship_embedding"
down_revision: str | Sequence[str] | None = "0007_manual_aliases"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMA = "graphrag"
EMB = 1024  # text-embedding-3-small @ 1024-dim, as everywhere else


def upgrade() -> None:
    op.add_column(
        "graph_relationships",
        sa.Column("embedding", Vector(EMB), nullable=True),
        schema=SCHEMA,
    )
    # Partial: only extracted edges are ever embedded, and on a large
    # graph the ONTOLOGY/derivedFromChunk rows are the bulk of the table.
    op.execute(
        f"CREATE INDEX graph_relationships_embedding_idx "
        f"ON {SCHEMA}.graph_relationships "
        f"USING hnsw (embedding vector_l2_ops) "
        f"WHERE embedding IS NOT NULL"
    )


def downgrade() -> None:
    op.execute(
        f"DROP INDEX IF EXISTS {SCHEMA}.graph_relationships_embedding_idx"
    )
    op.drop_column("graph_relationships", "embedding", schema=SCHEMA)
