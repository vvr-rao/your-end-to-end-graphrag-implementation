"""term_alias_judgements: cached LLM verdicts on mined synonym pairs

`mine-aliases` now asks a judge model (task `alias_judge`) whether each
mined pair really names one thing; only SAME pairs reach `term_aliases`.
The refresh runs after every ingest and rebuilds `term_aliases` from
scratch, so the verdicts live in their own table and survive it: a pair is
judged once per (judge model, prompt version), not on every ingest.

Keyed by model + version so switching LLM mode, or changing the prompt,
re-judges with the new judge instead of trusting another model's verdicts.

Revision ID: 0013_term_alias_judgements
Revises: 0012_term_aliases_byte_order
Create Date: 2026-10-22
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0013_term_alias_judgements"
down_revision: str | Sequence[str] | None = "0012_term_aliases_byte_order"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMA = "graphrag"


def upgrade() -> None:
    op.create_table(
        "term_alias_judgements",
        # Normalized and ordered exactly as in term_aliases (byte order).
        sa.Column("term_a", sa.Text(), nullable=False),
        sa.Column("term_b", sa.Text(), nullable=False),
        sa.Column("judge_model", sa.Text(), nullable=False),
        sa.Column("judge_version", sa.Text(), nullable=False),
        sa.Column("same", sa.Boolean(), nullable=False),
        # What the judge saw, so a verdict can be audited later.
        sa.Column("surface_a", sa.Text(), nullable=False),
        sa.Column("surface_b", sa.Text(), nullable=False),
        sa.Column("context", sa.Text(), nullable=True),
        sa.Column(
            "judged_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.PrimaryKeyConstraint(
            "term_a", "term_b", "judge_model", "judge_version",
            name="term_alias_judgements_pk",
        ),
        schema=SCHEMA,
    )


def downgrade() -> None:
    op.drop_table("term_alias_judgements", schema=SCHEMA)
