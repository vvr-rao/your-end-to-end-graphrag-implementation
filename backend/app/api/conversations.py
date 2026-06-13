"""Conversation routes (Milestone G over HTTP).

POST   /conversations                -- start
POST   /conversations/{iri}/turns    -- add a turn (follow-up resolved automatically)
GET    /conversations/{iri}          -- replay
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Path, status
from pydantic import BaseModel, Field

from backend.app.services.db_conversation import (
    add_turn,
    replay_conversation,
    start_conversation,
)

router = APIRouter(prefix="/conversations", tags=["conversations"])


class StartRequest(BaseModel):
    title: str | None = None


class StartResponse(BaseModel):
    iri: str
    id: str
    title: str | None


class TurnRequest(BaseModel):
    question: str = Field(..., description="User question (may be a follow-up).")
    mode: str = Field(
        "deep_research",
        description=(
            "'deep_research' (default) returns a structured 6-section "
            "answer; 'simple_qa' returns a tight 1-3 sentence direct "
            "answer."
        ),
    )
    top_k: int | None = Field(
        None, ge=1, le=100,
        description="Defaults to 30 for deep_research, 20 for simple_qa.",
    )
    hops: int = 2
    max_cost_usd: float = Field(0.20, gt=0.0, le=10.0)
    decompose: bool = True
    max_probes: int = 5
    history_window: int = Field(3, ge=0, le=10)


class TurnResponse(BaseModel):
    conversation_iri: str
    conversation_turn_id: str
    turn_index: int
    follow_up_resolved: bool
    user_question: str
    resolved_question: str
    mode: str
    answer: str | None
    evidence: list[dict[str, Any]]
    retrieval_run_id: str | None
    cost_usd: float
    wall_seconds: float


class ConversationView(BaseModel):
    iri: str
    title: str | None
    created_at: str
    turn_count: int
    turns: list[dict[str, Any]]


@router.post("", response_model=StartResponse, operation_id="conversation_start")
async def conversation_start(req: StartRequest) -> StartResponse:
    out = await start_conversation(title=req.title)
    return StartResponse(iri=out["iri"], id=out["id"], title=out.get("title"))


@router.post(
    "/{iri:path}/turns",
    response_model=TurnResponse,
    operation_id="conversation_turn",
)
async def conversation_turn(
    iri: str = Path(..., description="Conversation IRI."),
    req: TurnRequest | None = None,
) -> TurnResponse:
    if req is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="request body required",
        )
    try:
        out = await add_turn(
            conversation_iri=iri,
            question=req.question,
            mode=req.mode,
            top_k=req.top_k,
            hops=req.hops,
            max_cost_usd=req.max_cost_usd,
            decompose=req.decompose,
            max_probes=req.max_probes,
            history_window=req.history_window,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc),
        )
    return TurnResponse(**out)


@router.get(
    "/{iri:path}",
    response_model=ConversationView,
    operation_id="conversation_show",
)
async def conversation_show(iri: str) -> ConversationView:
    try:
        out = await replay_conversation(conversation_iri=iri)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc),
        )
    return ConversationView(**out)
