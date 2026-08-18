"""allow evidence_kind='manual' in graphrag.term_aliases

`mine-aliases` treats `term_aliases` as a DERIVED view of the corpus: a
full scan replaces it and prunes any row it did not just mine. That is
right for mined rows -- it is what makes tightened precision guards clean
up after themselves -- but it silently deletes anything a human added by
hand, and since the refresh now runs automatically after every ingest,
a hand-curated pair would vanish without a trace.

Widening the CHECK to admit 'manual' gives operators a durable slot. The
mining pass excludes that kind from both its scan and its prune, so
hand-curated pairs survive every refresh.

Reversible: the downgrade deletes manual rows before narrowing the
constraint back, so it cannot fail on a populated table (the failure mode
0005's downgrade has).

Revision ID: 0007_manual_aliases
Revises: 0006_term_aliases
Create Date: 2026-08-17
"""
from __future__ import annotations

from typing import Sequence

from alembic import op


revision: str = "0007_manual_aliases"
down_revision: str | Sequence[str] | None = "0006_term_aliases"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMA = "graphrag"
CHECK_NAME = "term_aliases_kind_check"


def upgrade() -> None:
    op.execute(
        f"ALTER TABLE {SCHEMA}.term_aliases DROP CONSTRAINT {CHECK_NAME}"
    )
    op.execute(
        f"ALTER TABLE {SCHEMA}.term_aliases ADD CONSTRAINT {CHECK_NAME} "
        f"CHECK (evidence_kind IN "
        f"('parenthetical','phrase','ontology','manual'))"
    )


def downgrade() -> None:
    # Drop the rows the narrowed constraint would reject, or the ALTER
    # fails outright on any database where someone curated a pair.
    op.execute(
        f"DELETE FROM {SCHEMA}.term_aliases WHERE evidence_kind = 'manual'"
    )
    op.execute(
        f"ALTER TABLE {SCHEMA}.term_aliases DROP CONSTRAINT {CHECK_NAME}"
    )
    op.execute(
        f"ALTER TABLE {SCHEMA}.term_aliases ADD CONSTRAINT {CHECK_NAME} "
        f"CHECK (evidence_kind IN ('parenthetical','phrase','ontology'))"
    )
