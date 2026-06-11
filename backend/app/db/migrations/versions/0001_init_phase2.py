"""init phase 2: graphrag schema with all 16 tables + extensions

Revision ID: 0001_init_phase2
Revises:
Create Date: 2026-06-10
"""
from __future__ import annotations

from typing import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

# Alembic identifiers
revision: str = "0001_init_phase2"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMA = "graphrag"
EMB = 1024  # text-embedding-3-small @ 1024-dim


def upgrade() -> None:
    # ---------------- schema + extensions ----------------
    op.execute(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}")
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")

    # ---------------- singleton: graph_version_state ----------------
    op.create_table(
        "graph_version_state",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("current_value", sa.Integer, nullable=False, server_default="0"),
        sa.CheckConstraint("id = 1", name="graph_version_state_singleton"),
        schema=SCHEMA,
    )
    op.execute(
        f"INSERT INTO {SCHEMA}.graph_version_state (id, current_value) VALUES (1, 0)"
    )

    # ---------------- ontology side ----------------
    for tbl, with_embedding, extra_cols in [
        ("ontology_classes", True, [sa.Column("is_viao_class", sa.Boolean, nullable=False, server_default="false")]),
        ("ontology_object_properties", False, []),
        ("ontology_data_properties", False, []),
        ("ontology_instances", False, [sa.Column("type_iri", sa.Text)]),
    ]:
        cols = [
            sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True,
                      server_default=sa.func.gen_random_uuid()),
            sa.Column("iri", sa.Text, unique=True, nullable=False),
            sa.Column("label", sa.Text),
            sa.Column("description", sa.Text),
            sa.Column("namespace", sa.Text, nullable=False),
            sa.Column("source_ontology", sa.Text),
            *extra_cols,
            sa.Column("extra_metadata", postgresql.JSONB, nullable=False, server_default="{}"),
            sa.Column("created_at", sa.DateTime(timezone=True),
                      nullable=False, server_default=sa.func.now()),
        ]
        if with_embedding:
            cols.insert(-2, sa.Column("embedding", Vector(EMB)))
        op.create_table(tbl, *cols, schema=SCHEMA)
        op.create_index(f"{tbl}_namespace_idx", tbl, ["namespace"], schema=SCHEMA)
        if with_embedding:
            op.execute(
                f"CREATE INDEX {tbl}_embedding_idx ON {SCHEMA}.{tbl} "
                f"USING hnsw (embedding vector_l2_ops)"
            )
            op.execute(
                f"CREATE INDEX {tbl}_label_trgm ON {SCHEMA}.{tbl} "
                f"USING gin (label gin_trgm_ops)"
            )

    # ---------------- documents ----------------
    op.create_table(
        "documents",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.func.gen_random_uuid()),
        sa.Column("document_identifier", sa.Text, unique=True, nullable=False),
        sa.Column("title", sa.Text, nullable=False),
        sa.Column("file_name", sa.Text),
        sa.Column("file_path", sa.Text),
        sa.Column("file_type", sa.Text),
        sa.Column("source_system", sa.Text),
        sa.Column("source_uri", sa.Text),
        sa.Column("document_hash", sa.Text, nullable=False),
        sa.Column("text_summary", sa.Text),
        sa.Column("embedding", Vector(EMB)),
        sa.Column("status", sa.Text, nullable=False, server_default="ACTIVE"),
        sa.Column("version", sa.Integer, nullable=False, server_default="1"),
        sa.Column("is_deleted", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("deleted_at", sa.DateTime(timezone=True)),
        sa.Column("supersedes_document_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey(f"{SCHEMA}.documents.id")),
        sa.Column("extra_metadata", postgresql.JSONB, nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True),
                  nullable=False, server_default=sa.func.now()),
        sa.CheckConstraint("status IN ('ACTIVE','STALE','DELETED')",
                           name="documents_status_check"),
        schema=SCHEMA,
    )
    op.execute(
        f"CREATE INDEX documents_status_idx ON {SCHEMA}.documents (status) "
        f"WHERE NOT is_deleted"
    )
    op.execute(
        f"CREATE INDEX documents_embedding_idx ON {SCHEMA}.documents "
        f"USING hnsw (embedding vector_l2_ops)"
    )
    op.execute(
        f"CREATE UNIQUE INDEX documents_hash_active_idx ON {SCHEMA}.documents "
        f"(document_hash) WHERE NOT is_deleted"
    )

    # ---------------- chunks ----------------
    op.create_table(
        "chunks",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.func.gen_random_uuid()),
        sa.Column("document_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey(f"{SCHEMA}.documents.id", ondelete="CASCADE"),
                  nullable=False),
        sa.Column("chunk_identifier", sa.Text, unique=True, nullable=False),
        sa.Column("chunk_index", sa.Integer, nullable=False),
        sa.Column("text", sa.Text, nullable=False),
        sa.Column("token_count", sa.Integer, nullable=False),
        sa.Column("page_number", sa.Integer),
        sa.Column("section_title", sa.Text),
        sa.Column("embedding", Vector(EMB)),
        sa.Column("status", sa.Text, nullable=False, server_default="ACTIVE"),
        sa.Column("is_deleted", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("deleted_at", sa.DateTime(timezone=True)),
        sa.Column("extra_metadata", postgresql.JSONB, nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  nullable=False, server_default=sa.func.now()),
        sa.CheckConstraint("status IN ('ACTIVE','STALE','DELETED')",
                           name="chunks_status_check"),
        schema=SCHEMA,
    )
    op.create_index("chunks_doc_idx", "chunks", ["document_id", "chunk_index"], schema=SCHEMA)
    op.execute(
        f"CREATE INDEX chunks_embedding_idx ON {SCHEMA}.chunks "
        f"USING hnsw (embedding vector_l2_ops)"
    )

    # ---------------- entities ----------------
    op.create_table(
        "entities",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.func.gen_random_uuid()),
        sa.Column("entity_identifier", sa.Text, unique=True, nullable=False),
        sa.Column("name", sa.Text, nullable=False),
        sa.Column("normalized_name", sa.Text, nullable=False),
        sa.Column("class_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey(f"{SCHEMA}.ontology_classes.id"), nullable=False),
        sa.Column("iri", sa.Text, nullable=False),
        sa.Column("embedding", Vector(EMB)),
        sa.Column("status", sa.Text, nullable=False, server_default="ACTIVE"),
        sa.Column("extra_metadata", postgresql.JSONB, nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  nullable=False, server_default=sa.func.now()),
        sa.CheckConstraint("status IN ('ACTIVE','STALE','DELETED')",
                           name="entities_status_check"),
        schema=SCHEMA,
    )
    op.create_index("entities_class_idx", "entities", ["class_id"], schema=SCHEMA)
    op.execute(
        f"CREATE INDEX entities_normalized_trgm ON {SCHEMA}.entities "
        f"USING gin (normalized_name gin_trgm_ops)"
    )
    op.execute(
        f"CREATE INDEX entities_embedding_idx ON {SCHEMA}.entities "
        f"USING hnsw (embedding vector_l2_ops)"
    )

    # ---------------- time_instances ----------------
    op.create_table(
        "time_instances",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.func.gen_random_uuid()),
        sa.Column("time_identifier", sa.Text, unique=True, nullable=False),
        sa.Column("time_level", sa.Text, nullable=False),
        sa.Column("start_date", sa.Date, nullable=False),
        sa.Column("end_date", sa.Date, nullable=False),
        sa.Column("display_label", sa.Text, nullable=False),
        sa.Column("parent_time_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey(f"{SCHEMA}.time_instances.id")),
        sa.Column("extra_metadata", postgresql.JSONB, nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  nullable=False, server_default=sa.func.now()),
        sa.CheckConstraint(
            "time_level IN ('year','quarter','month','day','interval')",
            name="time_instances_level_check",
        ),
        schema=SCHEMA,
    )
    op.create_index("time_instances_parent_idx", "time_instances",
                    ["parent_time_id"], schema=SCHEMA)
    op.create_index("time_instances_dates_idx", "time_instances",
                    ["start_date", "end_date"], schema=SCHEMA)

    # ---------------- intelligence_artifacts ----------------
    op.create_table(
        "intelligence_artifacts",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.func.gen_random_uuid()),
        sa.Column("artifact_identifier", sa.Text, unique=True, nullable=False),
        sa.Column("artifact_type", sa.Text, nullable=False),
        sa.Column("title", sa.Text),
        sa.Column("text", sa.Text, nullable=False),
        sa.Column("confidence", sa.Numeric),
        sa.Column("model_name", sa.Text),
        sa.Column("prompt_version", sa.Text),
        sa.Column("generation_run_id", postgresql.UUID(as_uuid=True)),
        sa.Column("embedding", Vector(EMB)),
        sa.Column("status", sa.Text, nullable=False, server_default="ACTIVE"),
        sa.Column("graph_version", sa.Integer, nullable=False),
        sa.Column("extra_metadata", postgresql.JSONB, nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  nullable=False, server_default=sa.func.now()),
        sa.CheckConstraint(
            "artifact_type IN ('Summary','Claim','Finding','Observation','Insight','Recommendation')",
            name="intel_artifacts_type_check",
        ),
        sa.CheckConstraint(
            "status IN ('ACTIVE','STALE','RETIRED','DELETED')",
            name="intel_artifacts_status_check",
        ),
        schema=SCHEMA,
    )
    op.create_index("intel_artifacts_type_idx", "intelligence_artifacts",
                    ["artifact_type"], schema=SCHEMA)
    op.create_index("intel_artifacts_status_idx", "intelligence_artifacts",
                    ["status"], schema=SCHEMA)
    op.create_index("intel_artifacts_gv_idx", "intelligence_artifacts",
                    ["graph_version"], schema=SCHEMA)
    op.execute(
        f"CREATE INDEX intel_artifacts_embedding_idx ON {SCHEMA}.intelligence_artifacts "
        f"USING hnsw (embedding vector_l2_ops)"
    )

    # ---------------- artifact_sources (M2M) ----------------
    op.create_table(
        "artifact_sources",
        sa.Column("artifact_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey(f"{SCHEMA}.intelligence_artifacts.id", ondelete="CASCADE"),
                  primary_key=True),
        sa.Column("chunk_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey(f"{SCHEMA}.chunks.id", ondelete="CASCADE"),
                  primary_key=True),
        schema=SCHEMA,
    )
    op.create_index("artifact_sources_chunk_idx", "artifact_sources",
                    ["chunk_id"], schema=SCHEMA)

    # ---------------- graph_relationships ----------------
    op.create_table(
        "graph_relationships",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.func.gen_random_uuid()),
        sa.Column("source_node_type", sa.Text, nullable=False),
        sa.Column("source_node_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("target_node_type", sa.Text, nullable=False),
        sa.Column("target_node_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("predicate_iri", sa.Text, nullable=False),
        sa.Column("predicate_label", sa.Text),
        sa.Column("relationship_type", sa.Text),
        sa.Column("relationship_source", sa.Text, nullable=False),
        sa.Column("confidence", sa.Numeric),
        sa.Column("is_authoritative", sa.Boolean, nullable=False, server_default="true"),
        sa.Column("source_document_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey(f"{SCHEMA}.documents.id", ondelete="SET NULL")),
        sa.Column("source_chunk_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey(f"{SCHEMA}.chunks.id", ondelete="SET NULL")),
        sa.Column("source_artifact_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey(f"{SCHEMA}.intelligence_artifacts.id", ondelete="SET NULL")),
        sa.Column("generation_run_id", postgresql.UUID(as_uuid=True)),
        sa.Column("graph_version", sa.Integer, nullable=False),
        sa.Column("extra_metadata", postgresql.JSONB, nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  nullable=False, server_default=sa.func.now()),
        sa.CheckConstraint(
            "source_node_type IN ('ontology_class','document','chunk','entity','time_instance','intelligence_artifact')",
            name="rel_source_node_type_check"),
        sa.CheckConstraint(
            "target_node_type IN ('ontology_class','document','chunk','entity','time_instance','intelligence_artifact')",
            name="rel_target_node_type_check"),
        sa.CheckConstraint(
            "relationship_source IN ('ONTOLOGY','DOCUMENT_EXTRACTION','LLM_INFERENCE','TIME_ENRICHMENT','MANUAL')",
            name="rel_relationship_source_check"),
        schema=SCHEMA,
    )
    op.create_index("rel_source_idx", "graph_relationships",
                    ["source_node_type", "source_node_id"], schema=SCHEMA)
    op.create_index("rel_target_idx", "graph_relationships",
                    ["target_node_type", "target_node_id"], schema=SCHEMA)
    op.create_index("rel_predicate_idx", "graph_relationships",
                    ["predicate_iri"], schema=SCHEMA)
    op.create_index("rel_gv_idx", "graph_relationships",
                    ["graph_version"], schema=SCHEMA)
    op.execute(
        f"CREATE INDEX rel_chunk_idx ON {SCHEMA}.graph_relationships "
        f"(source_chunk_id) WHERE source_chunk_id IS NOT NULL"
    )
    op.execute(
        f"CREATE INDEX rel_artifact_idx ON {SCHEMA}.graph_relationships "
        f"(source_artifact_id) WHERE source_artifact_id IS NOT NULL"
    )

    # ---------------- conversations + turns ----------------
    op.create_table(
        "conversations",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.func.gen_random_uuid()),
        sa.Column("conversation_identifier", sa.Text, unique=True, nullable=False),
        sa.Column("extra_metadata", postgresql.JSONB, nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  nullable=False, server_default=sa.func.now()),
        schema=SCHEMA,
    )

    op.create_table(
        "conversation_turns",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.func.gen_random_uuid()),
        sa.Column("conversation_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey(f"{SCHEMA}.conversations.id", ondelete="CASCADE"),
                  nullable=False),
        sa.Column("turn_index", sa.Integer, nullable=False),
        sa.Column("user_question", sa.Text, nullable=False),
        sa.Column("resolved_question", sa.Text),
        sa.Column("retrieval_mode", sa.Text, nullable=False),
        sa.Column("answer_text", sa.Text),
        sa.Column("extra_metadata", postgresql.JSONB, nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  nullable=False, server_default=sa.func.now()),
        schema=SCHEMA,
    )
    op.create_index("conv_turns_idx", "conversation_turns",
                    ["conversation_id", "turn_index"], schema=SCHEMA)

    # ---------------- retrieval_runs + retrieval_evidence ----------------
    op.create_table(
        "retrieval_runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.func.gen_random_uuid()),
        sa.Column("conversation_turn_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey(f"{SCHEMA}.conversation_turns.id", ondelete="CASCADE")),
        sa.Column("resolved_query", sa.Text, nullable=False),
        sa.Column("retrieval_mode", sa.Text, nullable=False),
        sa.Column("matched_classes", postgresql.JSONB, nullable=False, server_default="[]"),
        sa.Column("matched_entities", postgresql.JSONB, nullable=False, server_default="[]"),
        sa.Column("matched_time_instances", postgresql.JSONB, nullable=False, server_default="[]"),
        sa.Column("graph_hops", sa.Integer, nullable=False),
        sa.Column("retrieval_plan", postgresql.JSONB, nullable=False, server_default="{}"),
        sa.Column("graph_version", sa.Integer, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  nullable=False, server_default=sa.func.now()),
        sa.CheckConstraint(
            "retrieval_mode IN ('simple_qa','summarize','deep_research','insights','knowledge_gaps')",
            name="retrieval_runs_mode_check"),
        schema=SCHEMA,
    )

    op.create_table(
        "retrieval_evidence",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.func.gen_random_uuid()),
        sa.Column("retrieval_run_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey(f"{SCHEMA}.retrieval_runs.id", ondelete="CASCADE"),
                  nullable=False),
        sa.Column("evidence_kind", sa.Text, nullable=False),
        sa.Column("evidence_uuid", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("evidence_iri", sa.Text),
        sa.Column("rank", sa.Integer, nullable=False),
        sa.Column("score", sa.Float),
        sa.Column("snippet", sa.Text),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  nullable=False, server_default=sa.func.now()),
        sa.CheckConstraint(
            "evidence_kind IN ('chunk','class','entity','artifact','relationship','time_instance')",
            name="retrieval_evidence_kind_check"),
        schema=SCHEMA,
    )
    op.create_index("retrieval_evidence_run_idx", "retrieval_evidence",
                    ["retrieval_run_id", "rank"], schema=SCHEMA)


def downgrade() -> None:
    # Drop in reverse FK order.
    for tbl in (
        "retrieval_evidence", "retrieval_runs",
        "conversation_turns", "conversations",
        "graph_relationships", "artifact_sources",
        "intelligence_artifacts", "time_instances",
        "entities", "chunks", "documents",
        "ontology_instances", "ontology_data_properties",
        "ontology_object_properties", "ontology_classes",
        "graph_version_state",
    ):
        op.drop_table(tbl, schema=SCHEMA)
    # NB: leave the schema + extensions in place -- other apps may use them.
