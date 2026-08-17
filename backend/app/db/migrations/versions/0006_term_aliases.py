"""add graphrag.term_aliases (corpus-mined synonym pairs)

Backs the `mine-aliases` pass, which extracts synonym pairs
deterministically from the corpus -- parenthetical apposition
("MOUNJARO (tirzepatide) injection"), explicit alias phrases
("also known as"), and the skos:altLabel / oboInOwl:hasExactSynonym
annotations already imported into `ontology_classes.extra_metadata`.
Retrieval then expands query terms with the mined aliases, which fixes
the brand/generic vocabulary split (a query saying "tirzepatide" could
not reach chunks that only ever say "MOUNJARO").

Additive by construction: creates ONE new table and ALTERs nothing, so
the downgrade is a clean DROP TABLE. Deliberate -- 0005's downgrade is
already known-broken against a populated DB, and this must not add to
that.

Aliases cannot hang off `entities`: entity extraction mines only
`chunks.kind='summary'`, so a name that appears solely in full-text
chunks (MOUNJARO, in the reported failure) may have no entity row at
all. The pairs are also symmetric and many-to-many, which a table
models and an array column does not.

Revision ID: 0006_term_aliases
Revises: 0005_artifact_only_mode
Create Date: 2026-08-17
"""
from __future__ import annotations

from typing import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision: str = "0006_term_aliases"
down_revision: str | Sequence[str] | None = "0005_artifact_only_mode"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMA = "graphrag"


def upgrade() -> None:
    op.create_table(
        "term_aliases",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.func.gen_random_uuid(),
        ),
        # term_a / term_b are normalized (lowercased, punctuation-stripped)
        # and stored in lexical order (term_a < term_b) so ONE row serves
        # lookups from either direction.
        sa.Column("term_a", sa.Text(), nullable=False),
        sa.Column("term_b", sa.Text(), nullable=False),
        # The surface forms as they actually appeared, preserved for probe
        # text -- "MOUNJARO" embeds differently from "mounjaro".
        sa.Column("surface_a", sa.Text(), nullable=False),
        sa.Column("surface_b", sa.Text(), nullable=False),
        sa.Column("evidence_kind", sa.Text(), nullable=False),
        # Chunk IRI or ontology class IRI -- makes every pair auditable
        # back to the text that produced it.
        sa.Column("evidence_ref", sa.Text(), nullable=True),
        sa.Column(
            "occurrences", sa.Integer(), nullable=False, server_default="1"
        ),
        sa.Column(
            "graph_version", sa.Integer(), nullable=False, server_default="0"
        ),
        sa.Column(
            "extra_metadata",
            postgresql.JSONB,
            nullable=False,
            server_default="{}",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.CheckConstraint(
            "evidence_kind IN ('parenthetical','phrase','ontology')",
            name="term_aliases_kind_check",
        ),
        sa.CheckConstraint("term_a < term_b", name="term_aliases_order_check"),
        sa.UniqueConstraint(
            "term_a", "term_b", "evidence_kind", name="term_aliases_pair_uq"
        ),
        schema=SCHEMA,
    )
    # Both directions are indexed: query-time lookup is a UNION of
    # (term_a = :norm) and (term_b = :norm).
    op.create_index(
        "term_aliases_a_idx", "term_aliases", ["term_a"], schema=SCHEMA
    )
    op.create_index(
        "term_aliases_b_idx", "term_aliases", ["term_b"], schema=SCHEMA
    )


def downgrade() -> None:
    op.drop_index("term_aliases_b_idx", table_name="term_aliases", schema=SCHEMA)
    op.drop_index("term_aliases_a_idx", table_name="term_aliases", schema=SCHEMA)
    op.drop_table("term_aliases", schema=SCHEMA)
