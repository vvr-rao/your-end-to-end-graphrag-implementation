"""phase 2a: add StructuredTable to intelligence_artifacts.artifact_type

Widens the CHECK constraint on `graphrag.intelligence_artifacts.artifact_type`
from the original 6-value set to include `'StructuredTable'`. Forward-
compatible: existing rows remain valid. The reverse migration narrows
the constraint back; it will fail (correctly) if any StructuredTable
rows exist when downgrading.

No new tables. The JSON-LD payload for an extracted table is stored in
the existing `extra_metadata` JSONB column on the same row.

Revision ID: 0002_structured_tables
Revises: 0001_init_phase2
Create Date: 2026-06-14
"""
from __future__ import annotations

from typing import Sequence

from alembic import op


revision: str = "0002_structured_tables"
down_revision: str | Sequence[str] | None = "0001_init_phase2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMA = "graphrag"
CHECK_NAME = "intel_artifacts_type_check"

_ALLOWED_PHASE2 = (
    "'Summary','Claim','Finding','Observation','Insight','Recommendation'"
)
_ALLOWED_PHASE2A = _ALLOWED_PHASE2 + ",'StructuredTable'"


def upgrade() -> None:
    op.execute(
        f"ALTER TABLE {SCHEMA}.intelligence_artifacts "
        f"DROP CONSTRAINT {CHECK_NAME}"
    )
    op.execute(
        f"ALTER TABLE {SCHEMA}.intelligence_artifacts "
        f"ADD CONSTRAINT {CHECK_NAME} "
        f"CHECK (artifact_type IN ({_ALLOWED_PHASE2A}))"
    )


def downgrade() -> None:
    op.execute(
        f"ALTER TABLE {SCHEMA}.intelligence_artifacts "
        f"DROP CONSTRAINT {CHECK_NAME}"
    )
    op.execute(
        f"ALTER TABLE {SCHEMA}.intelligence_artifacts "
        f"ADD CONSTRAINT {CHECK_NAME} "
        f"CHECK (artifact_type IN ({_ALLOWED_PHASE2}))"
    )
