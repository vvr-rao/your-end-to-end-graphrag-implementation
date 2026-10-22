"""term_aliases: compare the pair order byte-wise, as the miner does

`alias_mining.make_pair` orders each pair with Python's `<` (code points);
the CHECK compared `term_a < term_b` in the database collation. They differ:
en_US.utf8 ignores spaces and punctuation at the first level, so
`'dax oil refinery' < 'daxoil'` holds in Python and fails in Postgres. On the
2026-10-19 web-search build that one pair of 834 raised a CheckViolation,
rolled back the whole alias refresh and left term_aliases empty -- silently,
because register-documents reports the refresh as a warning. `COLLATE "C"`
is byte order, which for UTF-8 is code-point order: exactly Python's. Every
row written so far satisfies it (each was ordered by Python).

Revision ID: 0012_term_aliases_byte_order
Revises: 0011_ontology_build_staging
Create Date: 2026-10-22
"""
from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0012_term_aliases_byte_order"
down_revision: str | Sequence[str] | None = "0011_ontology_build_staging"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMA = "graphrag"
NAME = "term_aliases_order_check"


def upgrade() -> None:
    op.drop_constraint(NAME, "term_aliases", schema=SCHEMA, type_="check")
    op.create_check_constraint(NAME, "term_aliases", 'term_a COLLATE "C" < term_b COLLATE "C"', schema=SCHEMA)


def downgrade() -> None:
    op.drop_constraint(NAME, "term_aliases", schema=SCHEMA, type_="check")
    op.create_check_constraint(NAME, "term_aliases", "term_a < term_b", schema=SCHEMA)
