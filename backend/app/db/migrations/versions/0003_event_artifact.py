"""phase 2a follow-up: add Event to intelligence_artifacts.artifact_type

Widens the CHECK constraint on `graphrag.intelligence_artifacts.artifact_type`
from the Phase 2a 7-value set to include `'Event'`. Forward-compatible:
existing rows remain valid. The reverse migration narrows back; it will
fail (correctly) if any Event rows exist when downgrading.

No new tables. Event date metadata (event_date / event_start_date /
event_end_date / event_category) is stored in the existing
`extra_metadata` JSONB column on the same row.

Revision ID: 0003_event_artifact
Revises: 0002_structured_tables
Create Date: 2026-06-15
"""
from __future__ import annotations

from typing import Sequence

from alembic import op


revision: str = "0003_event_artifact"
down_revision: str | Sequence[str] | None = "0002_structured_tables"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMA = "graphrag"
CHECK_NAME = "intel_artifacts_type_check"

_ALLOWED_PHASE2A = (
    "'Summary','Claim','Finding','Observation','Insight','Recommendation',"
    "'StructuredTable'"
)
_ALLOWED_PHASE2A_PLUS_EVENT = _ALLOWED_PHASE2A + ",'Event'"


def upgrade() -> None:
    op.execute(
        f"ALTER TABLE {SCHEMA}.intelligence_artifacts "
        f"DROP CONSTRAINT {CHECK_NAME}"
    )
    op.execute(
        f"ALTER TABLE {SCHEMA}.intelligence_artifacts "
        f"ADD CONSTRAINT {CHECK_NAME} "
        f"CHECK (artifact_type IN ({_ALLOWED_PHASE2A_PLUS_EVENT}))"
    )


def downgrade() -> None:
    op.execute(
        f"ALTER TABLE {SCHEMA}.intelligence_artifacts "
        f"DROP CONSTRAINT {CHECK_NAME}"
    )
    op.execute(
        f"ALTER TABLE {SCHEMA}.intelligence_artifacts "
        f"ADD CONSTRAINT {CHECK_NAME} "
        f"CHECK (artifact_type IN ({_ALLOWED_PHASE2A}))"
    )
