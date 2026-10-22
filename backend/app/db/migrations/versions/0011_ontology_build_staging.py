"""ontology_build schema: staging store for class-proposal cleanup at scale

Prune-expand used to hold every Stage-2 proposal in Python dicts and dedup
them with an LLM that saw ~200 items per call. On the 416k-token web-search
corpus that meant 20,175 proposals, two OOM kills on the 2.7 GB dev box, and
a dedup that removed nothing beyond identical labels. Corpora of tens of
millions of tokens mean 10^5-10^6 proposals, so the proposals now live here:

  label_vectors   one row per DISTINCT folded label per run and kind
                  (class / relation / existing_class), with occurrences,
                  parent/description, a status and an embedding under an
                  HNSW cosine index -- nearest-neighbour candidates are n*k
                  index lookups, never an n x n matrix.
  pair_decisions  every candidate pair considered and what was decided
                  (auto_merge / judged_same / judged_distinct / skipped):
                  the audit trail the duplicate-rate measurement samples.

A separate schema so a build never touches the graphrag (Q&A) tables; rows
are deleted per run_id when a build finishes.

Revision ID: 0011_ontology_build_staging
Revises: 0010_embedding_model
Create Date: 2026-10-18
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision: str = "0011_ontology_build_staging"
down_revision: str | Sequence[str] | None = "0010_embedding_model"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMA = "ontology_build"
EMB = 1024


def upgrade() -> None:
    op.execute(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}")
    op.create_table(
        "label_vectors",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("run_id", sa.Text, nullable=False),
        sa.Column("kind", sa.Text, nullable=False),
        sa.Column("label_key", sa.Text, nullable=False),
        sa.Column("label", sa.Text, nullable=False),
        sa.Column("parent", sa.Text),
        sa.Column("description", sa.Text),
        sa.Column("iri", sa.Text),
        sa.Column("occurrences", sa.Integer, nullable=False, server_default="1"),
        sa.Column("chunk_refs", postgresql.ARRAY(sa.Integer), nullable=False, server_default="{}"),
        sa.Column("status", sa.Text, nullable=False, server_default="kept"),
        sa.Column("status_reason", sa.Text),
        sa.Column("merged_into", sa.BigInteger),
        sa.Column("extra", postgresql.JSONB, nullable=False, server_default="{}"),
        sa.Column("embedding", Vector(EMB)),
        sa.CheckConstraint("kind IN ('class', 'relation', 'existing_class')", name="label_vectors_kind_check"),
        sa.CheckConstraint("status IN ('kept', 'junk', 'instance', 'merged')", name="label_vectors_status_check"),
        sa.UniqueConstraint("run_id", "kind", "label_key", name="label_vectors_run_kind_key"),
        schema=SCHEMA,
    )
    op.create_index("label_vectors_run_kind_status_idx", "label_vectors",
                    ["run_id", "kind", "status"], schema=SCHEMA)
    # Candidate search runs on the first 256 dimensions (the Qwen3 / OpenAI
    # embeddings are Matryoshka-trained, so a prefix is itself an embedding):
    # a 4x smaller index and 4x cheaper distances. Candidates are re-scored on
    # the full 1024 dimensions, so the cosine used for decisions is exact.
    # A STORED column, not an expression index: the 1 KB prefix stays inline
    # while the 4 KB vector is TOASTed, so exact scans (small existing-class
    # sets, where the planner skips HNSW) ran 4.6x faster (4.6 -> 1.0 ms).
    op.execute(
        f"ALTER TABLE {SCHEMA}.label_vectors ADD COLUMN emb256 vector(256) "
        f"GENERATED ALWAYS AS (subvector(embedding, 1, 256)::vector(256)) STORED"
    )
    # No table-wide vector index: HNSW maintenance on every embedding write
    # was 77% of write time (10k rows: 67 s with the index, 7.9 s without;
    # one bulk build afterwards 19 s). Each build creates its own partial
    # index per kind once its vectors are in (BuildStore.ensure_index) and
    # drops it in clear(). knn_batch therefore puts run and kind into the
    # query as literals, so the planner can match that partial index.
    #
    # Nearest neighbours for a whole batch of ids in ONE round trip. Measured
    # on the 2-CPU dev box (10k labels): the same lookup issued per id from
    # Python cost 15 ms (session + SET LOCALs + query); looped server-side it
    # costs 5-7 ms. A plain LATERAL join was ~10x slower per row: the planner
    # stops using the HNSW index once the query vector comes from the outer
    # row, so the loop runs one index query per id. Prefix search on 256
    # dimensions over-fetching `over`, re-scored on the full vector.
    op.execute(f"""
        CREATE FUNCTION {SCHEMA}.knn_batch(ids bigint[], p_run text, p_kind text, k int, over int, ef int)
        RETURNS TABLE(a_id bigint, b_id bigint, cos float8) LANGUAGE plpgsql AS $$
        DECLARE a bigint; q vector; q256 vector(256); sql text;
        BEGIN
          PERFORM set_config('hnsw.iterative_scan', 'relaxed_order', true);
          PERFORM set_config('hnsw.ef_search', ef::text, true);
          sql := format(
            'SELECT $3, c.id, 1 - (c.embedding <=> $1) AS cs FROM ('
            '  SELECT b.id, b.embedding FROM {SCHEMA}.label_vectors b'
            '  WHERE b.run_id = %L AND b.kind = %L AND b.status = %L'
            '    AND b.id <> $3 AND b.embedding IS NOT NULL'
            '  ORDER BY b.emb256 <=> $2 LIMIT %s) c '
            'ORDER BY cs DESC LIMIT %s', p_run, p_kind, 'kept', over, k);
          FOREACH a IN ARRAY ids LOOP
            SELECT v.embedding, v.emb256 INTO q, q256
            FROM {SCHEMA}.label_vectors v WHERE v.id = a;
            CONTINUE WHEN q IS NULL;
            RETURN QUERY EXECUTE sql USING q, q256, a;
          END LOOP;
        END $$
    """)
    op.create_table(
        "pair_decisions",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("run_id", sa.Text, nullable=False),
        sa.Column("a_id", sa.BigInteger, nullable=False),
        sa.Column("b_id", sa.BigInteger, nullable=False),
        sa.Column("cosine", sa.Float, nullable=False),
        sa.Column("decision", sa.Text, nullable=False),
        sa.Column("reason", sa.Text),
        sa.CheckConstraint(
            "decision IN ('auto_merge', 'judged_same', 'judged_distinct', 'skipped')",
            name="pair_decisions_decision_check"),
        sa.UniqueConstraint("run_id", "a_id", "b_id", name="pair_decisions_run_pair_key"),
        schema=SCHEMA,
    )
    op.create_index("pair_decisions_run_decision_idx", "pair_decisions",
                    ["run_id", "decision"], schema=SCHEMA)


def downgrade() -> None:
    op.drop_table("pair_decisions", schema=SCHEMA)
    op.execute(f"DROP FUNCTION IF EXISTS {SCHEMA}.knn_batch(bigint[], text, text, int, int, int)")
    op.drop_table("label_vectors", schema=SCHEMA)
    op.execute(f"DROP SCHEMA IF EXISTS {SCHEMA}")
