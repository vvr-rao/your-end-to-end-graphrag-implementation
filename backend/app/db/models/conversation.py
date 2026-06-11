"""Multi-turn QA: conversations + conversation_turns."""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import ForeignKey, Integer, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.db.base import (
    Base,
    metadata_jsonb_column,
    timestamp_column,
    uuid_pk_column,
)


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[uuid.UUID] = uuid_pk_column()
    conversation_identifier: Mapped[str] = mapped_column(
        Text, unique=True, nullable=False
    )
    extra_metadata: Mapped[dict[str, Any]] = metadata_jsonb_column()
    created_at: Mapped[datetime] = timestamp_column()

    turns: Mapped[list["ConversationTurn"]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="ConversationTurn.turn_index",
    )


class ConversationTurn(Base):
    __tablename__ = "conversation_turns"

    id: Mapped[uuid.UUID] = uuid_pk_column()
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
    )
    turn_index: Mapped[int] = mapped_column(Integer, nullable=False)
    user_question: Mapped[str] = mapped_column(Text, nullable=False)
    resolved_question: Mapped[str | None] = mapped_column(Text)
    retrieval_mode: Mapped[str] = mapped_column(Text, nullable=False)
    answer_text: Mapped[str | None] = mapped_column(Text)
    extra_metadata: Mapped[dict[str, Any]] = metadata_jsonb_column()
    created_at: Mapped[datetime] = timestamp_column()

    conversation: Mapped["Conversation"] = relationship(back_populates="turns")
