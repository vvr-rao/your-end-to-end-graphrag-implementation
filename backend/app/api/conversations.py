"""Conversation routes (Milestone G over HTTP).

GET    /conversations                -- list past conversations (paginated)
POST   /conversations                -- start
POST   /conversations/{iri}/turns    -- add a turn (follow-up resolved automatically)
GET    /conversations/{iri}          -- replay
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Path, Query, status
from pydantic import BaseModel, Field

from backend.app.services.db_conversation import (
    add_turn,
    list_conversations,
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
            "'deep_research' (default) returns a structured 7-section "
            "answer; 'simple_qa' returns a tight 1-3 sentence direct "
            "answer."
        ),
    )
    top_k: int | None = Field(
        None, ge=1, le=100,
        description="Defaults to 30 for deep_research, 20 for simple_qa.",
    )
    # Bounds mirror QARequest (qa.py); TurnRequest was missing them.
    hops: int = Field(2, ge=0, le=4)
    max_cost_usd: float = Field(0.20, gt=0.0, le=10.0)
    decompose: bool = True
    max_probes: int = Field(5, ge=1, le=8)
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


class ConversationListItem(BaseModel):
    iri: str
    title: str | None
    created_at: str
    turn_count: int
    last_turn_at: str | None


@router.get(
    "",
    response_model=list[ConversationListItem],
    operation_id="conversation_list",
    summary="List conversations",
    description=(
        "List existing conversations, newest first, with their IRIs and turn "
        "counts. Use this to find a conversation IRI to resume via "
        "conversation_show or conversation_turn."
    ),
)
async def conversation_list(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> list[ConversationListItem]:
    rows = await list_conversations(limit=limit, offset=offset)
    return [ConversationListItem(**row) for row in rows]


@router.post(
    "",
    response_model=StartResponse,
    operation_id="conversation_start",
    summary="Start a new conversation",
    description=(
        "Create a conversation and return its IRI. Pass that IRI to "
        "conversation_turn to ask questions with follow-up resolution against "
        "the conversation's history."
    ),
)
async def conversation_start(req: StartRequest) -> StartResponse:
    out = await start_conversation(title=req.title)
    return StartResponse(iri=out["iri"], id=out["id"], title=out.get("title"))


@router.post(
    "/{iri:path}/turns",
    response_model=TurnResponse,
    operation_id="conversation_turn",
    summary="Ask a question within a conversation",
    description=(
        "Ask a question in an existing conversation. Unlike qa_ask, follow-ups "
        "are resolved against the conversation's history, so pronouns and "
        "elliptical questions ('what about 2024?') work. `iri` is the FULL "
        "conversation IRI including its '#' fragment, from conversation_start "
        "or conversation_list. `question` is required."
    ),
)
async def conversation_turn(
    # `req` must NOT be Optional. An optional body makes FastAPI emit
    # {"anyOf": [{"$ref": ...}, {"type": "null"}]}, and fastapi-mcp only reads
    # body properties from a schema with a top-level "properties" key
    # (openapi/convert.py:181) -- anyOf has none. The result was an MCP tool
    # exposing only {"iri"}, with question/mode/top_k missing entirely, so an
    # agent had no way to ask anything. qa_ask works precisely because its body
    # is required, which emits a plain $ref that the resolver expands in place.
    req: TurnRequest,
    iri: str = Path(..., description="Conversation IRI (full IRI, including '#')."),
) -> TurnResponse:
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
    summary="Replay a conversation's turns",
    description=(
        "Return a conversation with all its turns -- questions, answers and "
        "citations -- in order. `iri` is the FULL conversation IRI including "
        "its '#' fragment."
    ),
)
async def conversation_show(iri: str) -> ConversationView:
    try:
        out = await replay_conversation(conversation_iri=iri)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc),
        )
    return ConversationView(**out)
