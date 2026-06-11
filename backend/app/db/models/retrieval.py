"""Retrieval runs + evidence rows for traceability.

A RetrievalRun is created for every QA invocation. It records the
resolved query, mode, what got matched at parse + concept-expansion
time, and which graph_version the run executed against. The result
set lives in retrieval_evidence as one row per evidence item with a
rank.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import CheckConstraint, Float, ForeignKey, Integer, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.db.base import (
    Base,
    timestamp_column,
    uuid_pk_column,
)


class RetrievalRun(Base):
    __tablename__ = "retrieval_runs"
    __table_args__ = (
        CheckConstraint(
            "retrieval_mode IN ('simple_qa','summarize','deep_research','insights','knowledge_gaps')",
            name="retrieval_runs_mode_check",
        ),
    )

    id: Mapped[uuid.UUID] = uuid_pk_column()
    conversation_turn_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversation_turns.id", ondelete="CASCADE"),
    )
    resolved_query: Mapped[str] = mapped_column(Text, nullable=False)
    retrieval_mode: Mapped[str] = mapped_column(Text, nullable=False)
    matched_classes: Mapped[list[Any]] = mapped_column(
        JSONB, nullable=False, default=list, server_default="[]"
    )
    matched_entities: Mapped[list[Any]] = mapped_column(
        JSONB, nullable=False, default=list, server_default="[]"
    )
    matched_time_instances: Mapped[list[Any]] = mapped_column(
        JSONB, nullable=False, default=list, server_default="[]"
    )
    graph_hops: Mapped[int] = mapped_column(Integer, nullable=False)
    retrieval_plan: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict, server_default="{}"
    )
    graph_version: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = timestamp_column()


class RetrievalEvidence(Base):
    __tablename__ = "retrieval_evidence"
    __table_args__ = (
        CheckConstraint(
            "evidence_kind IN ('chunk','class','entity','artifact','relationship','time_instance')",
            name="retrieval_evidence_kind_check",
        ),
    )

    id: Mapped[uuid.UUID] = uuid_pk_column()
    retrieval_run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("retrieval_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    evidence_kind: Mapped[str] = mapped_column(Text, nullable=False)
    evidence_uuid: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    evidence_iri: Mapped[str | None] = mapped_column(Text)
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    score: Mapped[float | None] = mapped_column(Float)
    snippet: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = timestamp_column()
