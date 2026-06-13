"""Milestone G: conversation-aware QA with follow-up resolution.

Three driver functions:
  start_conversation(title)         -> {iri, id}
  add_turn(conv_iri, q, mode, ...)  -> RetrievalResult (with conversation_turn_id)
  replay_conversation(conv_iri)     -> list of turn dicts

Each turn:
  1. Fetch the last 3 turns of THIS conversation.
  2. If any prior turn exists, run `follow_up_resolution` (gpt-4o-mini)
     to rewrite the new question into a standalone form. Otherwise
     `resolved_query = user_question`.
  3. Run the F retrieval pipeline against `resolved_query`.
  4. Persist a `conversation_turns` row with user_question +
     resolved_question + mode + answer_text + retrieval_run_id (via
     retrieval_runs.conversation_turn_id linkage).

Reuses `retrieve_and_answer` from retrieval.py -- no new pipeline.
"""
from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import select, text as sql_text

from backend.app.db.models.conversation import Conversation, ConversationTurn
from backend.app.db.session import session_scope
from backend.app.services.db_artifact_gen import _extract_json
from backend.app.services.llm_router import LLMRouter
from backend.app.services.prompts import PROMPTS
from backend.app.services.retrieval import retrieve_and_answer, RetrievalResult


_CONVERSATIONS_NS = "https://veerla-ramrao.ai/ontology/conversations"


def _conversation_iri(uid: uuid.UUID) -> str:
    return f"{_CONVERSATIONS_NS}#Conversation_{uid.hex[:16]}"


async def start_conversation(
    *, title: str | None = None
) -> dict[str, Any]:
    """Insert a conversations row. Returns {iri, id, title}."""
    cuid = uuid.uuid4()
    iri = _conversation_iri(cuid)
    async with session_scope() as session:
        await session.execute(
            sql_text("""
            INSERT INTO graphrag.conversations (
              id, conversation_identifier, extra_metadata, created_at
            ) VALUES (
              :id, :iri, CAST(:meta AS jsonb), now()
            )
            """),
            {
                "id": cuid,
                "iri": iri,
                "meta": '{"title": ' + (
                    f'"{title}"' if title else "null"
                ) + "}",
            },
        )
    return {"iri": iri, "id": str(cuid), "title": title}


async def _resolve_follow_up(
    router: LLMRouter,
    new_question: str,
    prior_turns: list[tuple[str, str, str]],
) -> tuple[str, bool]:
    """If `prior_turns` is non-empty, ask gpt-4o-mini to rewrite the
    new question into a standalone form. Each prior_turn is
    (user_question, resolved_question, answer_text). Including the
    answer is critical -- it's what lets the rewriter pull named
    entities from prior answers (e.g. 'what frameworks?' -> 'what are
    the FAO Hand-in-Hand Initiative, Zero Hunger 2030, ...').
    Returns (resolved, did_call)."""
    if not prior_turns:
        return new_question, False
    system, user = PROMPTS["follow_up_resolution"](new_question, prior_turns)
    if not system:
        return new_question, False
    try:
        out = await router.chat(
            "follow_up_resolution", system=system, user=user
        )
        rewritten = (out.text or "").strip()
        return (rewritten or new_question), True
    except Exception as exc:
        print(f"[conversation] follow-up resolution failed ({exc}); "
              "using new_question as-is")
        return new_question, False


async def _fetch_prior_evidence_iris(
    conversation_uid: uuid.UUID, history_window: int = 3
) -> dict[str, list[str]]:
    """Pull the top evidence IRIs from the last N turns' retrieval_runs.
    Returns {'chunks': [...], 'artifacts': [...]} -- IRI strings, ranked.
    Used to seed the current turn's candidate pool so docs the prior
    answers came from don't fall off the list when the new question
    is narrower."""
    async with session_scope() as session:
        r = await session.execute(
            sql_text("""
            SELECT re.evidence_kind, re.evidence_iri, re.rank
              FROM graphrag.conversation_turns ct
              JOIN graphrag.retrieval_runs rr ON rr.conversation_turn_id = ct.id
              JOIN graphrag.retrieval_evidence re ON re.retrieval_run_id = rr.id
             WHERE ct.conversation_id = :cid
             ORDER BY ct.turn_index DESC, re.rank
             LIMIT :lim
            """),
            {"cid": conversation_uid, "lim": history_window * 30},
        )
        rows = r.all()
    out: dict[str, list[str]] = {"chunk": [], "artifact": []}
    seen: set[str] = set()
    for kind, iri, rank in rows:
        if not iri or iri in seen:
            continue
        seen.add(iri)
        if kind in out:
            out[kind].append(iri)
    return out


async def add_turn(
    *,
    conversation_iri: str,
    question: str,
    mode: str = "simple_qa",
    top_k: int = 20,
    hops: int = 2,
    max_cost_usd: float = 1.0,
    decompose: bool = True,
    max_probes: int = 5,
    exhaustive_limit: int = 100,
    history_window: int = 3,
    verbose: bool = False,
) -> dict[str, Any]:
    """Add one turn to a conversation. Returns the same envelope as
    `query`, plus `conversation_turn_id`, `resolved_question`,
    `turn_index`, `follow_up_resolved` (bool)."""
    # 1. Look up the conversation + last N turns.
    async with session_scope() as session:
        r = await session.execute(
            select(Conversation.id).where(
                Conversation.conversation_identifier == conversation_iri
            )
        )
        conv_uid = r.scalar_one_or_none()
        if conv_uid is None:
            raise ValueError(f"conversation not found: {conversation_iri}")

        # Determine next turn_index.
        r = await session.execute(
            sql_text("""
            SELECT coalesce(max(turn_index), -1) + 1
              FROM graphrag.conversation_turns
             WHERE conversation_id = :id
            """),
            {"id": conv_uid},
        )
        next_turn_index = int(r.scalar_one())

        # Pull last N (asked, resolved, answer) triples for follow-up
        # resolution + conversation-aware synthesis.
        r = await session.execute(
            sql_text("""
            SELECT user_question, resolved_question, answer_text
              FROM graphrag.conversation_turns
             WHERE conversation_id = :id
               AND answer_text IS NOT NULL
               AND answer_text <> ''
             ORDER BY turn_index DESC
             LIMIT :n
            """),
            {"id": conv_uid, "n": history_window},
        )
        prior = [
            (asked, resolved or asked, answer or "")
            for asked, resolved, answer in r.all()
        ]
    prior.reverse()  # oldest first

    # 2. Follow-up resolution (only if prior turns exist).
    router = LLMRouter()
    cost_before = router.total_cost_usd
    resolved_query, did_resolve = await _resolve_follow_up(
        router, question, prior
    )
    if verbose and did_resolve:
        print(f"[conversation] follow-up: '{question}'\n"
              f"  resolved -> '{resolved_query}'")

    # 3. Pre-insert a conversation_turns row so retrieval can link to it
    #    via its conversation_turn_id (F's persistence layer expects
    #    this FK).
    turn_uid = uuid.uuid4()
    async with session_scope() as session:
        await session.execute(
            sql_text("""
            INSERT INTO graphrag.conversation_turns (
              id, conversation_id, turn_index, user_question,
              resolved_question, retrieval_mode, answer_text,
              extra_metadata, created_at
            ) VALUES (
              :id, :cid, :idx, :uq, :rq, :mode, NULL,
              CAST(:meta AS jsonb), now()
            )
            """),
            {
                "id": turn_uid,
                "cid": conv_uid,
                "idx": next_turn_index,
                "uq": question,
                "rq": resolved_query,
                "mode": mode,
                "meta": '{"follow_up_resolved": '
                        + ("true" if did_resolve else "false")
                        + "}",
            },
        )

    # 4. Run the F pipeline (retrieves evidence + draft answer).
    result: RetrievalResult = await retrieve_and_answer(
        question,  # original for any potential UI display
        mode=mode,
        top_k=top_k,
        hops=hops,
        max_cost_usd=max_cost_usd,
        decompose=decompose,
        max_probes=max_probes,
        exhaustive_limit=exhaustive_limit,
        conversation_turn_id=turn_uid,
        resolved_query=resolved_query,
        verbose=verbose,
    )

    # 4b. If prior turns exist, re-synthesize the answer with full
    #     conversation context in scope. The F pipeline only saw THIS
    #     turn's evidence; we want the synthesizer to also know what
    #     was asked + answered before so it can build on prior context
    #     instead of treating each turn as a fresh search.
    if prior and result.answer is not None and mode != "exhaustive_search":
        prior_qa: list[tuple[str, str]] = [
            (resolved or asked, ans)
            for asked, resolved, ans in prior
            if ans
        ]
        try:
            sys_p, user_p = PROMPTS["answer_conversation_turn"](
                resolved_query, result.evidence, prior_qa, base_mode=mode,
            )
            out = await router.chat(
                "answer_conversation_turn", system=sys_p, user=user_p,
            )
            convo_answer = (out.text or "").strip()
            if convo_answer:
                if verbose:
                    print(
                        "[conversation] re-synthesized answer with "
                        f"{len(prior_qa)} prior turn(s) in scope"
                    )
                result.answer = convo_answer
        except Exception as exc:
            print(
                f"[conversation] conv-aware synth failed ({exc}); "
                "keeping the F-pipeline draft answer"
            )

    # 5. Update the turn row with the (possibly re-synthesized) answer.
    async with session_scope() as session:
        await session.execute(
            sql_text("""
            UPDATE graphrag.conversation_turns
               SET answer_text = :ans
             WHERE id = :id
            """),
            {"ans": result.answer or "", "id": turn_uid},
        )

    total_cost = router.total_cost_usd - cost_before + result.cost_usd
    return {
        "conversation_iri": conversation_iri,
        "conversation_turn_id": str(turn_uid),
        "turn_index": next_turn_index,
        "follow_up_resolved": did_resolve,
        "user_question": question,
        "resolved_question": resolved_query,
        "mode": result.mode,
        "answer": result.answer,
        "exhaustive_results": result.exhaustive_results,
        "evidence": result.evidence,
        "retrieval_run_id": (
            str(result.retrieval_run_id) if result.retrieval_run_id else None
        ),
        "cost_usd": total_cost,
        "wall_seconds": result.wall_seconds,
    }


async def replay_conversation(
    conversation_iri: str,
) -> dict[str, Any]:
    """Return the conversation header + ordered turns with answers."""
    async with session_scope() as session:
        r = await session.execute(
            select(Conversation.id, Conversation.extra_metadata, Conversation.created_at).where(
                Conversation.conversation_identifier == conversation_iri
            )
        )
        row = r.first()
        if row is None:
            raise ValueError(f"conversation not found: {conversation_iri}")
        conv_uid, meta, created_at = row

        r = await session.execute(
            sql_text("""
            SELECT turn_index, user_question, resolved_question,
                   retrieval_mode, answer_text, extra_metadata, created_at
              FROM graphrag.conversation_turns
             WHERE conversation_id = :id
             ORDER BY turn_index
            """),
            {"id": conv_uid},
        )
        turns = [
            {
                "turn_index": ti,
                "user_question": uq,
                "resolved_question": rq,
                "mode": mode,
                "answer": ans,
                "follow_up_resolved": (em or {}).get("follow_up_resolved", False),
                "created_at": ts.isoformat(),
            }
            for ti, uq, rq, mode, ans, em, ts in r.all()
        ]

    return {
        "iri": conversation_iri,
        "title": (meta or {}).get("title"),
        "created_at": created_at.isoformat(),
        "turn_count": len(turns),
        "turns": turns,
    }
