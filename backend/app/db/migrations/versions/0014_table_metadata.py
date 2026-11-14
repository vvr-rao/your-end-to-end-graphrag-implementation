"""widen intelligence_artifacts.artifact_type CHECK to include 'TableMetadata'

Structured-data support (``extract-entities --structured-data``) stores one
metadata node per loaded data table as an intelligence artifact of type
``TableMetadata``: the LLM-written table/column descriptions plus its
metrics, filters, queries and lookups (Postgres SQL) in ``extra_metadata``.
The data tables themselves are created at load time, not by a migration.
Additive and forward-compatible: existing rows are unaffected.

Revision ID: 0014_table_metadata
Revises: 0013_term_alias_judgements
Create Date: 2026-11-13
"""
from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0014_table_metadata"
down_revision: str | Sequence[str] | None = "0013_term_alias_judgements"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMA = "graphrag"
CHECK_NAME = "intel_artifacts_type_check"
_OLD = ("('Summary','Claim','Finding','Observation','Insight','Recommendation',"
        "'StructuredTable','Event')")
_NEW = ("('Summary','Claim','Finding','Observation','Insight','Recommendation',"
        "'StructuredTable','Event','TableMetadata')")


def upgrade() -> None:
    op.execute(f"ALTER TABLE {SCHEMA}.intelligence_artifacts DROP CONSTRAINT {CHECK_NAME}")
    op.execute(f"ALTER TABLE {SCHEMA}.intelligence_artifacts "
               f"ADD CONSTRAINT {CHECK_NAME} CHECK (artifact_type IN {_NEW})")


def downgrade() -> None:
    # Fails while TableMetadata rows exist -- remove them (and the data tables)
    # first, e.g. with clear-corpus.
    op.execute(f"ALTER TABLE {SCHEMA}.intelligence_artifacts DROP CONSTRAINT {CHECK_NAME}")
    op.execute(f"ALTER TABLE {SCHEMA}.intelligence_artifacts "
               f"ADD CONSTRAINT {CHECK_NAME} CHECK (artifact_type IN {_OLD})")
