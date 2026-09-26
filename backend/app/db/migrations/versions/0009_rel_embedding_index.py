"""narrow graphrag.graph_relationships' HNSW index to the rows retrieval queries

0008 created the index as

    USING hnsw (embedding vector_l2_ops) WHERE embedding IS NOT NULL

and it was never used by anything. Two reasons, both fixed here and in
`retrieval_sql`:

1. The three relationship searches were written with `<=>` (cosine) while the
   opclass serves `<->` (L2) -- they are the only cosine queries in the file.
   pgvector cannot answer a cosine ordering from an L2 index, so every one of
   them fell back to a sequential scan. `EXPLAIN ANALYZE` on the 66-document
   build showed `Bitmap Index Scan on rel_source_idx` (the node-type btree)
   with `Rows Removed by Filter: 5262`.

   The queries now use `<->`. That is not an approximation: the embeddings are
   unit-normalised (`text-embedding-3-small` @ 1024; measured mean norm
   0.999996 over 200 rows), so `l2 = sqrt(2 - 2*cos)` exactly -- verified at
   0.942705 vs 0.942740 on a real pair -- and ordering by L2 ascending is
   identical to ordering by cosine descending.

2. The partial predicate did not match what the queries filter on. They all add
   `source_node_type = 'entity' AND target_node_type = 'entity'`, which the
   index could not satisfy, so HNSW would have returned the k nearest rows
   OVERALL and left the node-type filter to discard an unknown number of them --
   silently returning fewer than k. Folding those two predicates into the index
   makes it exactly cover the query.

Ontology TBox rows (subClassOf, rdf:type) are the bulk of this table -- 29,075
rows against 1,481 embedded on the current build -- so a predicate that keeps
only embedded entity->entity rows also keeps the index small.

Revision ID: 0009_rel_embedding_index
Revises: 0008_relationship_embedding
Create Date: 2026-09-26
"""
from __future__ import annotations

from typing import Sequence

from alembic import op


revision: str = "0009_rel_embedding_index"
down_revision: str | Sequence[str] | None = "0008_relationship_embedding"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMA = "graphrag"
IDX = "graph_relationships_embedding_idx"
_ENTITY_ONLY = (
    "embedding IS NOT NULL "
    "AND source_node_type = 'entity' "
    "AND target_node_type = 'entity'"
)


def upgrade() -> None:
    op.execute(f"DROP INDEX IF EXISTS {SCHEMA}.{IDX}")
    op.execute(
        f"CREATE INDEX {IDX} ON {SCHEMA}.graph_relationships "
        f"USING hnsw (embedding vector_l2_ops) WHERE ({_ENTITY_ONLY})"
    )


def downgrade() -> None:
    op.execute(f"DROP INDEX IF EXISTS {SCHEMA}.{IDX}")
    op.execute(
        f"CREATE INDEX {IDX} ON {SCHEMA}.graph_relationships "
        f"USING hnsw (embedding vector_l2_ops) WHERE (embedding IS NOT NULL)"
    )
