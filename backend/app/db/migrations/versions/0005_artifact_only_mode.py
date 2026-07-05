"""widen retrieval_runs.retrieval_mode CHECK to include 'artifact_only'

`artifact_only` is a new retrieval mode that runs the GraphRAG logic over ALL
intelligence artifacts (entity-linked + a global vector pass over every active
artifact) with the chunk arm disabled, then synthesizes a deep-research-shaped
answer purely from artifact evidence. Persisting such a run failed the existing
CHECK constraint, so this migration widens the allowed value set. Additive and
forward-compatible: existing rows are unaffected.

Revision ID: 0005_artifact_only_mode
Revises: 0004_chunk_kind
Create Date: 2026-07-05
"""
from __future__ import annotations

from typing import Sequence

from alembic import op


revision: str = "0005_artifact_only_mode"
down_revision: str | Sequence[str] | None = "0004_chunk_kind"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMA = "graphrag"
CHECK_NAME = "retrieval_runs_mode_check"
_OLD = "('simple_qa','summarize','deep_research','insights','knowledge_gaps')"
_NEW = "('simple_qa','summarize','deep_research','insights','knowledge_gaps','artifact_only')"


def upgrade() -> None:
    op.execute(f"ALTER TABLE {SCHEMA}.retrieval_runs DROP CONSTRAINT {CHECK_NAME}")
    op.execute(
        f"ALTER TABLE {SCHEMA}.retrieval_runs "
        f"ADD CONSTRAINT {CHECK_NAME} CHECK (retrieval_mode IN {_NEW})"
    )


def downgrade() -> None:
    op.execute(f"ALTER TABLE {SCHEMA}.retrieval_runs DROP CONSTRAINT {CHECK_NAME}")
    op.execute(
        f"ALTER TABLE {SCHEMA}.retrieval_runs "
        f"ADD CONSTRAINT {CHECK_NAME} CHECK (retrieval_mode IN {_OLD})"
    )
