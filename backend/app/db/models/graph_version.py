"""Singleton row holding the monotonic graph_version counter.

Inserted by the initial migration. The graph_version helpers
(`backend/app/db/graph_version.py`) read/update it via SQL — this
model exists for Alembic autogen + introspection."""
from __future__ import annotations

from sqlalchemy import CheckConstraint, Integer
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.db.base import Base


class GraphVersionState(Base):
    __tablename__ = "graph_version_state"
    __table_args__ = (
        CheckConstraint("id = 1", name="graph_version_state_singleton"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    current_value: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )
