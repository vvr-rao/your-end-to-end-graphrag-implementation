"""POST /qa -- one-shot retrieval against the corpus.

Wraps `services.retrieval.retrieve_and_answer`. Same shape as the
CLI's `query --json` output.

Two modes (2026-06-13 redesign):
  - simple_qa     -- tight 1-3 sentence direct answer.
  - deep_research -- structured 7-section output (SPECIFICS /
                     ANALYSIS / ANSWER / CONTRADICTIONS / CLAIMS /
                     DATA COVERAGE MISMATCH / KEY INSIGHTS). DEFAULT.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from backend.app.services.retrieval import retrieve_and_answer

router = APIRouter(prefix="/qa", tags=["qa"])


_VALID_MODES = ("simple_qa", "deep_research")


class QARequest(BaseModel):
    question: str = Field(..., description="User question.")
    mode: str = Field(
        "deep_research",
        description=(
            "Retrieval mode. 'deep_research' (default) returns a "
            "structured 7-section answer; 'simple_qa' returns a tight "
            "1-3 sentence direct answer."
        ),
    )
    top_k: int | None = Field(
        None, ge=1, le=100,
        description=(
            "Candidates surfaced as evidence. Defaults to 30 for "
            "deep_research, 20 for simple_qa."
        ),
    )
    hops: int = Field(2, ge=0, le=4)
    max_cost_usd: float = Field(1.0, gt=0.0, le=10.0)
    decompose: bool = Field(True, description="Run step-9a query decomposition.")
    max_probes: int = Field(5, ge=1, le=8)


class QAEvidenceItem(BaseModel):
    kind: str
    iri: str
    rank: int
    score: float
    text: str | None = None
    document_iri: str | None = None
    document_title: str | None = None
    artifact_type: str | None = None
    confidence: float | None = None


class QAResponse(BaseModel):
    answer: str | None
    mode: str
    resolved_query: str
    evidence: list[QAEvidenceItem]
    retrieval_run_id: str | None
    parsed: dict[str, Any]
    cost_usd: float
    wall_seconds: float
    graph_version: int


@router.post("", response_model=QAResponse, operation_id="qa_ask")
async def qa_ask(req: QARequest) -> QAResponse:
    if req.mode not in _VALID_MODES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"mode must be one of {_VALID_MODES}. The "
                "summarize/insights/knowledge_gaps/exhaustive_search "
                "modes were removed -- use deep_research for structured "
                "answers or simple_qa for direct factoids."
            ),
        )
    result = await retrieve_and_answer(
        req.question,
        mode=req.mode,
        top_k=req.top_k,
        hops=req.hops,
        max_cost_usd=req.max_cost_usd,
        decompose=req.decompose,
        max_probes=req.max_probes,
    )
    return QAResponse(
        answer=result.answer,
        mode=result.mode,
        resolved_query=result.resolved_query,
        evidence=[QAEvidenceItem(**ev) for ev in result.evidence],
        retrieval_run_id=str(result.retrieval_run_id) if result.retrieval_run_id else None,
        parsed=result.parsed,
        cost_usd=result.cost_usd,
        wall_seconds=result.wall_seconds,
        graph_version=result.graph_version,
    )
