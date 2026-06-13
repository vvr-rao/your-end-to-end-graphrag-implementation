"""GET /retrieval_runs/{id} -- full evidence chain for any past answer.

Returns the retrieval_runs row + its retrieval_evidence children,
ready for a UI to render the answer -> artifact -> chunk -> document
traceability chain.
"""
from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import text as sql_text

from backend.app.db.session import session_scope

router = APIRouter(prefix="/retrieval_runs", tags=["trace"])


@router.get("/{run_id}", operation_id="trace_retrieval_run")
async def trace_retrieval_run(run_id: str) -> dict[str, Any]:
    try:
        ruid = uuid.UUID(run_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="run_id must be a valid UUID",
        )
    async with session_scope() as session:
        r = await session.execute(
            sql_text("""
            SELECT id, conversation_turn_id, resolved_query, retrieval_mode,
                   matched_classes, matched_entities, matched_time_instances,
                   graph_hops, retrieval_plan, graph_version, created_at
              FROM graphrag.retrieval_runs WHERE id = :id
            """),
            {"id": ruid},
        )
        row = r.first()
        if row is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"retrieval_run {run_id} not found",
            )
        rid, conv_turn, query, mode, mc, me, mt, hops, plan, gv, created = row

        r = await session.execute(
            sql_text("""
            SELECT evidence_kind, evidence_iri, rank, score, snippet, created_at
              FROM graphrag.retrieval_evidence
             WHERE retrieval_run_id = :id
             ORDER BY rank
            """),
            {"id": ruid},
        )
        evidence = [
            {"kind": k, "iri": iri, "rank": rnk, "score": s,
             "snippet": snip, "created_at": ts.isoformat()}
            for k, iri, rnk, s, snip, ts in r.all()
        ]

    return {
        "retrieval_run_id": str(rid),
        "conversation_turn_id": str(conv_turn) if conv_turn else None,
        "resolved_query": query,
        "mode": mode,
        "matched_classes": mc,
        "matched_entities": me,
        "matched_time_instances": mt,
        "graph_hops": hops,
        "retrieval_plan": plan,
        "graph_version": gv,
        "created_at": created.isoformat(),
        "evidence_count": len(evidence),
        "evidence": evidence,
    }
