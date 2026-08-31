"""Milestone F: retrieval pipeline + answer synthesis.

Driver function `retrieve_and_answer(question, mode, ...)` runs the
12-step pipeline as designed in the plan:

  1. (G only) follow-up resolution
  2. mode selected by caller
  3. question_parse -> entities/classes/time/intent
  4. ontology match (vector + trgm)
  5. concept expansion
  6. seed nodes union
  7. graph BFS (depth `hops`)
  8. candidate retrieval (chunks + artifacts)
  9. multi-probe vector rerank
     a. query_decompose -> sub-questions
     b. embed all probes
     c. per-probe SQL rerank against the step-8 candidate set
     d. per-candidate, per-probe scores
 10. RRF fusion of probe rankings + graph signals
 11. context engineering: pack top-K
 12. answer generation (mode-specific prompt)

Persists `retrieval_runs` (1 row) + `retrieval_evidence` (top-K rows).

Two modes (2026-06-13 redesign):
  - simple_qa     -- tight 1-3 sentence direct answer; top_k=20.
  - deep_research -- structured 7-section output (SPECIFICS / ANALYSIS
                     / ANSWER / CONTRADICTIONS / CLAIMS /
                     DATA COVERAGE MISMATCH / KEY INSIGHTS); top_k=30;
                     default mode.

The summarize/insights/knowledge_gaps/exhaustive_search modes were
removed in the redesign in favor of the structured deep_research
output. The two remaining modes share the same pipeline up through
step 10; they diverge at steps 11 + 12 only on prompt choice + model.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select, text as sql_text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.config import get_settings
from backend.app.db.graph_version import current_version
from backend.app.db.models.documents import Chunk, Document
from backend.app.db.models.entities import Entity, TimeInstance
from backend.app.db.models.ontology import OntologyClass
from backend.app.db.session import session_scope
from backend.app.services.alias_mining import is_acronym, normalize_term
from backend.app.services.db_artifact_gen import _extract_json
from backend.app.services.embeddings import Embedder
from backend.app.services.llm_router import LLMRouter
from backend.app.services.prompts import PROMPTS
from backend.app.services import retrieval_sql
from backend.app.services.retrieval_ranking import cap_per_source, rrf_fuse
from backend.app.services.retrieval_sql import _vec_str

log = logging.getLogger(__name__)

_DEFAULT_HOPS_FALLBACK = 3


def default_hops() -> int:
    """Graph-BFS depth (step 7) default, sourced from config.yaml's
    `qa.hops`. Falls back to 3 if the key is missing or config can't be
    read. This is the single source of truth -- the CLI and the
    conversation path both pass `hops=None` so they inherit it."""
    try:
        value = get_settings().app_config.get("qa", {}).get("hops")
        return int(value) if value is not None else _DEFAULT_HOPS_FALLBACK
    except Exception:
        return _DEFAULT_HOPS_FALLBACK


def _qa_cfg(key: str, default: Any) -> Any:
    """Read one `qa.*` config value, falling back on any config error.

    Every knob added here must keep today's behavior when unset -- these
    features are additive and a missing key must never change results.
    """
    try:
        value = get_settings().app_config.get("qa", {}).get(key)
        return default if value is None else value
    except Exception:
        return default


def _resolve_doc_cap(value: Any, top_k: int) -> int:
    """How many evidence chunks one document may supply.

    Accepts either form, so a single setting works across corpora:

      0.75  -> a FRACTION of top_k. 75% of 30 = 22, guaranteeing at least
               a quarter of the evidence comes from elsewhere.
      8     -> an absolute count (any value >= 1).
      0     -> disabled.

    The fraction is the better default because it scales with `top_k` and
    is corpus-independent. A fixed count cannot be: 8 breaks the 30-of-30
    monopoly on a corpus of small labels, but dilutes a question about one
    200-page filing across seven documents, and there is no single integer
    that is right for both. A fraction bounds the WORST case concentration
    without dictating how concentrated a legitimately single-source answer
    may be.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0
    if v <= 0:
        return 0
    if v < 1:                       # fraction of the evidence budget
        return max(1, int(top_k * v))
    return int(v)                   # absolute count


_VALID_MODES = ("simple_qa", "deep_research", "artifact_only")
# deep_research is the default workhorse mode; simple_qa is the tight
# direct-answer mode. artifact_only runs the graph pipeline over ALL
# intelligence artifacts (entity-linked + a global vector pass), never chunks,
# and synthesizes with the deep_research structured prompt.
# Other modes (summarize, insights, knowledge_gaps, exhaustive_search) were
# removed 2026-06-13 in favor of the structured deep_research output.


@dataclass
class RetrievalResult:
    answer: str | None
    mode: str
    resolved_query: str
    evidence: list[dict[str, Any]] = field(default_factory=list)
    retrieval_run_id: uuid.UUID | None = None
    parsed: dict[str, Any] = field(default_factory=dict)
    # Citations the synthesis emitted that resolve to nothing in `evidence`.
    # Stripped from `answer`; surfaced here (and in retrieval_plan) so a
    # regression is measurable instead of silent.
    invalid_citations: list[str] = field(default_factory=list)
    cost_usd: float = 0.0
    wall_seconds: float = 0.0
    graph_version: int = 0


# Matches inline evidence citation tokens the synthesis emits, e.g.
# "[viao:Chunk_180a994f...]", "[viao:Claim_c91c...]", a bracketed URL, or the
# kind-prefixed form the evidence block used to render ("[chunk https://...]").
# The body is captured so `_validate_citations` can check what was cited;
# leading whitespace is consumed so removal doesn't leave a double space.
_CITATION_TOKEN_RE = re.compile(
    r"\s*\[(?:(?:chunk|artifact|evidence)\s+)?(viao:[^\]]+|https?://[^\]]+)\]"
)


def _strip_citation_tokens(text: str) -> str:
    """Remove inline [viao:...] / [http...] citation tokens from prose."""
    return _CITATION_TOKEN_RE.sub("", text)


def _citation_key(raw: str) -> str:
    """Reduce a cited identifier to the fragment both forms share.

    The model may write the `viao:`-prefixed short form it is asked for,
    or echo a full URL from the evidence block. Both reduce to the part
    after the last '#' (or after 'viao:'), which is what actually
    identifies the row.

    Prose inside the bracket is discarded. The deep_research CLAIMS
    section is specified as `"<Claim>. [Stated by <source> in <doc>]"`,
    so citations legitimately arrive as
    `[viao:Claim_cd262250c7af4c13 in the corpus]` -- and keeping that tail
    made the key miss, stripping a citation whose row had genuinely been
    retrieved. Losing real traceability is worse than the fabricated
    citations this guard exists to catch, so the identifier is taken as
    the first whitespace-delimited token.
    """
    s = raw.strip()
    if "#" in s:
        s = s.rsplit("#", 1)[-1]
    elif s.lower().startswith("viao:"):
        s = s[5:]
    return s.strip().split()[0] if s.strip() else ""


def _validate_citations(
    answer: str, evidence: list[dict[str, Any]]
) -> tuple[str, list[str]]:
    """Drop inline citations that resolve to nothing in `evidence`.

    The synthesis occasionally cites an id that does not exist, or copies
    a real one incorrectly -- an audit of 21 questions found 4 such
    citations out of 264, all in one answer. The claims themselves were
    true and separately cited, so no reader was misled on fact, but a
    citation resolving to nothing breaks the traceability chain the whole
    artifact layer exists to provide.

    Returns `(cleaned_answer, invalid_ids)`. The evidence set is already
    in hand at synthesis time, so this costs one pass over the text and
    no I/O.
    """
    if not answer or not evidence:
        return answer, []
    allowed = {
        _citation_key(ev["iri"]) for ev in evidence if ev.get("iri")
    }
    if not allowed:
        return answer, []

    invalid: list[str] = []

    def _keep(m: re.Match[str]) -> str:
        # One bracket may carry SEVERAL ids -- the synthesis writes
        # "[viao:Chunk_x; viao:Claim_y; viao:Claim_z]" because the rules
        # tell it multiple citations on one fact are fine. Validate each
        # part and rebuild from the survivors; treating the whole body as
        # a single id would discard valid citations along with the bad
        # one, which is worse than the defect this guards against.
        parts = [p.strip() for p in re.split(r"[;,]", m.group(1)) if p.strip()]
        good = [p for p in parts if _citation_key(p) in allowed]
        bad = [p for p in parts if _citation_key(p) not in allowed]
        invalid.extend(_citation_key(p) for p in bad)
        if not good:
            return ""
        if not bad:
            return m.group(0)          # untouched, preserves original spacing
        lead = m.group(0)[: m.group(0).index("[")]
        return f"{lead}[{'; '.join(good)}]"

    cleaned = _CITATION_TOKEN_RE.sub(_keep, answer)
    if invalid:
        # Removing a token mid-sentence leaves seams -- a doubled space, or
        # a space stranded before the full stop. Tidy those; don't attempt
        # to repair the prose itself.
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
        cleaned = re.sub(r"[ \t]+([.,;:!?)])", r"\1", cleaned)
    return cleaned, invalid


async def retrieve_and_answer(
    question: str,
    *,
    mode: str = "deep_research",
    top_k: int | None = None,
    hops: int | None = None,
    max_cost_usd: float = 1.0,
    decompose: bool = True,
    max_probes: int = 3,
    rerank: bool | None = None,
    conversation_turn_id: uuid.UUID | None = None,
    resolved_query: str | None = None,
    verbose: bool = False,
    multi_round: bool = True,
    persist: bool = True,
    _round_depth: int = 0,
) -> RetrievalResult:
    """Driver. `resolved_query` skips the follow-up resolution step
    (the caller -- typically Milestone G -- already resolved it).

    `top_k` defaults to 30 for deep_research (more breadth for the
    seven-section output) and 20 for simple_qa.

    `persist=False` skips the `retrieval_runs` / `retrieval_evidence`
    writes, leaving `retrieval_run_id` None. Lets the CLI be pointed at a
    deployed database for testing without leaving rows in it; the API
    path always persists.
    """
    if mode not in _VALID_MODES:
        raise ValueError(
            f"invalid mode: {mode}; must be one of {_VALID_MODES}. "
            "The summarize/insights/knowledge_gaps/exhaustive_search "
            "modes were removed -- use deep_research for structured "
            "answers or simple_qa for direct factoids."
        )
    if top_k is None:
        # artifacts are tiny (~35 tokens) so we can afford many more of them.
        top_k = 40 if mode == "artifact_only" else (30 if mode == "deep_research" else 20)
    if hops is None:
        hops = default_hops()
    t0 = time.time()
    router = LLMRouter()
    cost_before = router.total_cost_usd

    resolved_query = resolved_query or question

    # -------- iterative (2-round) retrieval, deep_research only --------
    # Some questions depend on entities that must be discovered first (e.g.
    # "compare Ozempic to its competitors, which is fastest?" needs the
    # competitor list before it can retrieve each competitor's attributes). A
    # planner LLM detects that; if so we run a cheap BRIDGE round (simple_qa) to
    # find those entities, then a FINAL deep_research round whose question is
    # enriched with them -- so the normal pipeline seeds on the discovered
    # entities and the synthesis can use them. Capped at 2 rounds: sub-rounds run
    # with `_round_depth=1` / `multi_round=False`, so they never re-plan.
    if mode in ("deep_research", "artifact_only") and multi_round and _round_depth == 0:
        plan = await _plan_rounds(router, resolved_query)
        if plan["needs_second_round"] and plan["round1_question"] and plan["round2_question"]:
            if verbose:
                print(
                    f"[query] 2-round retrieval: "
                    f"bridge={plan['round1_question']!r} final={plan['round2_question']!r}"
                )
            # Round 1 (bridge): cheap simple_qa to discover the entities.
            r1 = await retrieve_and_answer(
                plan["round1_question"], mode="simple_qa",
                hops=hops, max_cost_usd=max_cost_usd, decompose=decompose,
                max_probes=max_probes, rerank=rerank, verbose=verbose,
                multi_round=False, persist=persist, _round_depth=1,
            )
            # Round 2 (final): enrich the question with round-1 findings so the
            # pipeline seeds on the discovered entities AND the synthesis uses them.
            # For artifact_only the round-1 bridge is a chunk-based *discovery*
            # step; strip its citation tokens from the prior-research block so the
            # final artifact-only synthesis can't echo chunk citations it never
            # retrieved (keeps the mode's "answer from artifacts only" contract).
            prior = _strip_citation_tokens(r1.answer) if mode == "artifact_only" else r1.answer
            enriched_final = (
                f"{plan['round2_question']}\n\n"
                f"[Prior research -- {plan['round1_question']}]\n{prior}"
            )
            # Round 2 (final): stay in the caller's mode so artifact_only keeps
            # synthesizing from artifacts only (just now enriched with the
            # discovered competitor entities), while deep_research stays as-is.
            r2 = await retrieve_and_answer(
                enriched_final, mode=mode, top_k=top_k,
                hops=hops, max_cost_usd=max_cost_usd, decompose=decompose,
                max_probes=max_probes, rerank=rerank,
                conversation_turn_id=conversation_turn_id,
                verbose=verbose, multi_round=False, persist=persist,
                _round_depth=1,
            )
            # Merge: r2 is the final answer; fold in round-1 evidence + all cost.
            # artifact_only only ever surfaces artifact evidence, so skip folding
            # in round-1's chunk evidence (it would reintroduce chunk rows into an
            # artifact-only result); still merge any artifact rows round 1 found.
            seen = {ev.get("node_id") for ev in r2.evidence}
            for ev in r1.evidence:
                if mode == "artifact_only" and ev.get("kind") != "artifact":
                    continue
                if ev.get("node_id") not in seen:
                    r2.evidence.append(ev)
            planning_cost = router.total_cost_usd - cost_before
            r2.cost_usd += r1.cost_usd + planning_cost
            r2.resolved_query = resolved_query  # report the user's original query
            r2.parsed["rounds"] = {
                "round1_question": plan["round1_question"],
                "round2_question": plan["round2_question"],
                "round1_answer": r1.answer,
            }
            r2.wall_seconds = time.time() - t0
            return r2

    # -------- step 3: parse the question --------
    parsed = await _question_parse(router, resolved_query)
    if verbose:
        print(f"[query] parsed: {parsed}")

    # -------- step 3.5: corpus-mined alias expansion --------
    # Pure SQL lookup against `term_aliases` (populated by `mine-aliases`).
    # No LLM: the synonymy comes from the corpus stating it, e.g. the FDA
    # label line "MOUNJARO (tirzepatide) injection". Without this, a query
    # saying "tirzepatide" cannot reach chunks that only ever say
    # "MOUNJARO" -- they are unrelated in embedding space. Empty table =>
    # empty dict => every downstream step behaves exactly as before.
    aliases = await _expand_aliases(parsed.get("entities") or [])
    # The parser decides entity-vs-class, and it routinely files acronyms as
    # CLASSES -- so `mrhd` and `ace`, both present in term_aliases, never
    # fired in a 21-question audit. That gated a fully deterministic lookup
    # behind an LLM's judgement call. Expand class terms too, but only accept
    # ACRONYM pairs from them: class terms are common nouns, and an ungated
    # lookup would let a class of "hormone" pull in "angiotensin II".
    class_aliases = await _expand_aliases(
        parsed.get("classes") or [], acronyms_only=True
    )
    for term, alts in class_aliases.items():
        if term not in aliases:
            aliases[term] = alts
    # Last resort, and the only parser-independent one: match the alias
    # vocabulary against the raw question text. Catches multi-word phrases
    # the parser reduced to a head noun ("maximum recommended human dose"
    # -> "dose"), which is why that question and its acronym form retrieved
    # different evidence.
    for term, alts in (await _expand_aliases_from_question(resolved_query)).items():
        if term not in aliases:
            aliases[term] = alts
    parsed["aliases"] = aliases
    if verbose and aliases:
        print(
            "[query] mined aliases: "
            + "; ".join(f"{t} -> {', '.join(a)}" for t, a in aliases.items())
        )

    # -------- step 4: ontology match --------
    embedder = Embedder()
    qvec = (await embedder.embed([resolved_query]))[0]
    seeds, matched_classes, matched_entities, matched_times = await _ontology_match(
        parsed, qvec, aliases=aliases
    )
    if verbose:
        print(
            f"[query] matched: classes={len(matched_classes)} "
            f"entities={len(matched_entities)} times={len(matched_times)} "
            f"seeds={len(seeds)}"
        )

    # -------- step 5: concept expansion --------
    expanded_class_ids = await _concept_expansion(
        router, resolved_query, matched_classes
    )
    seeds.extend((cid, "ontology_class") for cid in expanded_class_ids)
    seeds = list(set(seeds))   # dedupe

    # -------- step 6+7: BFS --------
    async with session_scope() as session:
        bfs_nodes = await retrieval_sql.bfs_expand(
            session, seeds, max_hops=hops, decay=0.7
        )
    expanded_entity_ids = [
        nid for (nid, ntype), _ in bfs_nodes.items() if ntype == "entity"
    ]
    # Seeds may include entities already; merge them in.
    for nid, ntype in seeds:
        if ntype == "entity" and nid not in expanded_entity_ids:
            expanded_entity_ids.append(nid)
    expanded_time_ids = [
        nid for (nid, ntype), _ in bfs_nodes.items() if ntype == "time_instance"
    ]
    for nid, ntype in seeds:
        if ntype == "time_instance" and nid not in expanded_time_ids:
            expanded_time_ids.append(nid)
    if verbose:
        print(
            f"[query] BFS yielded {len(bfs_nodes)} nodes "
            f"({len(expanded_entity_ids)} entities, "
            f"{len(expanded_time_ids)} time_instances)"
        )

    # -------- step 8: candidate retrieval --------
    async with session_scope() as session:
        ent_chunks = await retrieval_sql.fetch_candidate_chunks_for_entities(
            session, expanded_entity_ids, limit=500,
        )
        time_chunks = await retrieval_sql.fetch_candidate_chunks_for_time_instances(
            session, expanded_time_ids, limit=200,
        )
        artifact_candidates = await retrieval_sql.fetch_candidate_artifacts_for_entities(
            session, expanded_entity_ids, limit=200,
        )

    # Document-level recall arm. `documents.embedding` has been written and
    # HNSW-indexed since 0001 but nothing ever read it. A document that is
    # ABOUT the query topic surfaces here even when no individual chunk
    # matches the query's vocabulary -- an independent path to the same
    # recall gap brand/generic names open at chunk level, and one that
    # needs no mined alias pair. `qa.k_documents: 0` disables it.
    _k_docs = int(_qa_cfg("k_documents", 10))
    doc_chunks: list[tuple[uuid.UUID, float]] = []
    if _k_docs > 0:
        async with session_scope() as session:
            _doc_ids = await retrieval_sql.vector_search_documents(
                session, qvec, top_k=_k_docs,
            )
            if _doc_ids:
                doc_chunks = await retrieval_sql.fetch_chunks_for_documents(
                    session, _doc_ids, limit=300,
                    per_document_limit=int(
                        _qa_cfg("max_fulltext_chunks_per_document", 50)
                    ),
                )
        if verbose:
            print(
                f"[query] document arm: {len(_doc_ids)} document(s) -> "
                f"{len(doc_chunks)} chunk(s)"
            )

    candidate_chunk_ids = list(
        {cid for cid, _ in ent_chunks + time_chunks + doc_chunks}
    )
    candidate_artifact_ids = [aid for aid, _ in artifact_candidates]

    # Document-mediated table inclusion: pull every StructuredTable
    # artifact derived from a document that owns at least one of our
    # candidate chunks. Without this step, revenue / financial-data
    # tables that don't repeat the company name inside their cells get
    # missed -- the entity-anchored artifact path only catches tables
    # with direct entity edges (~10-15% of tables in practice). With
    # this step, asking "BHP's 2025 revenue" pulls every table from the
    # BHP 10-K into the candidate pool, where vector rerank then
    # surfaces the actually-relevant revenue table by similarity to the
    # query probes. Bounded at 200 tables to keep Stage-9 vector-rerank
    # cheap; the limit only bites for queries that match very many
    # documents simultaneously.
    if candidate_chunk_ids:
        async with session_scope() as session:
            doc_table_candidates = (
                await retrieval_sql.fetch_table_artifacts_for_chunks(
                    session, candidate_chunk_ids, limit=200,
                )
            )
        # Avoid duplicates if a table was already in candidate_artifact_ids
        # via a direct entity edge.
        already_in = set(candidate_artifact_ids)
        for aid, _ in doc_table_candidates:
            if aid not in already_in:
                candidate_artifact_ids.append(aid)
                already_in.add(aid)
        if verbose:
            n_added = len(candidate_artifact_ids) - len(artifact_candidates)
            print(
                f"[query] document-mediated tables added: {n_added} "
                f"StructuredTable artifact(s) (now {len(candidate_artifact_ids)} "
                f"total artifact candidates)"
            )

    # -------- artifact_only: run graphrag over ALL artifacts, no chunks --------
    if mode == "artifact_only":
        # Keep the entity-linked artifacts (graph arm, already in
        # candidate_artifact_ids) and add a GLOBAL vector pass over every
        # artifact -- so Insights / Recommendations / Summaries that carry no
        # entity edge are also in scope. Then disable the chunk arm entirely:
        # answers are synthesized from artifacts only.
        async with session_scope() as session:
            global_arts = await retrieval_sql.vector_search_all_artifacts(
                session, qvec, top_k=max(top_k * 3, 120),
            )
        _seen_a = set(candidate_artifact_ids)
        for aid in global_arts:
            if aid not in _seen_a:
                candidate_artifact_ids.append(aid)
                _seen_a.add(aid)
        candidate_chunk_ids = []   # no chunk arm
        ent_chunks = []            # -> empty graph_chunk_ranking; no chunk leakage
        if verbose:
            print(
                f"[query] artifact_only: {len(candidate_artifact_ids)} artifact "
                f"candidate(s) (entity-linked + global vector); chunk arm off"
            )

    # -------- step 8.5: full-text bridge --------
    # When documents owning our candidate (summary) chunks also carry verbatim
    # full-text chunks (ingested with --full-text-chunks), swap them into the
    # candidate pool so retrieval runs over verbatim text (better recall + exact
    # citations). The graph entered via summary-chunk entity edges; here we exit
    # into full text, document-mediated (mirrors the table bridge above). The
    # graph-coverage signal is propagated to the fulltext chunks by document.
    # No-op when no full-text chunks exist → identical to prior behavior.
    graph_chunk_ranking: list[uuid.UUID] = [cid for cid, _ in ent_chunks]
    if candidate_chunk_ids:
        ft_rows: list[tuple[uuid.UUID, float, uuid.UUID]] = []
        chunk_doc_map: dict[uuid.UUID, uuid.UUID] = {}
        async with session_scope() as session:
            ft_rows = await retrieval_sql.fetch_fulltext_chunks_for_chunks(
                session, candidate_chunk_ids, limit=500,
                per_document_limit=int(
                    _qa_cfg("max_fulltext_chunks_per_document", 50)
                ),
            )
            if ft_rows:
                chunk_doc_map = await retrieval_sql.fetch_chunk_document_ids(
                    session, candidate_chunk_ids,
                )
        if ft_rows:
            ft_doc_ids = {did for _, _, did in ft_rows}
            # Keep summary chunks only for docs that have NO full-text chunks.
            summary_keep = [
                cid for cid in candidate_chunk_ids
                if chunk_doc_map.get(cid) not in ft_doc_ids
            ]
            ft_ids = [cid for cid, _, _ in ft_rows]
            candidate_chunk_ids = summary_keep + ft_ids
            # Propagate graph scores: fulltext chunks inherit their document's
            # candidate-chunk count; kept summary chunks keep their entity score.
            ent_score = {cid: sc for cid, sc in ent_chunks}
            scored = [(cid, ent_score.get(cid, 0.0)) for cid in summary_keep]
            scored += [(cid, hits) for cid, hits, _ in ft_rows]
            scored.sort(key=lambda t: t[1], reverse=True)
            graph_chunk_ranking = [cid for cid, _ in scored]
            if verbose:
                print(
                    f"[query] full-text bridge: {len(ft_ids)} fulltext chunk(s) "
                    f"across {len(ft_doc_ids)} doc(s); candidate pool now "
                    f"{len(candidate_chunk_ids)}"
                )

    if not candidate_chunk_ids and not candidate_artifact_ids:
        if verbose:
            print("[query] zero candidates from graph; falling back to global vector search")
        # Pure vector fallback. Helps when the query has no entity/class
        # match (e.g. abstract questions about the corpus). Prefer full-text
        # chunks globally when any exist (verbatim citations); else summary.
        async with session_scope() as session:
            r = await session.execute(
                sql_text("""
                SELECT id FROM graphrag.chunks
                 WHERE embedding IS NOT NULL AND status='ACTIVE'
                   AND (kind = 'fulltext' OR NOT EXISTS (
                         SELECT 1 FROM graphrag.chunks
                          WHERE kind = 'fulltext' AND status = 'ACTIVE'))
                 ORDER BY embedding <-> CAST(:probe AS vector)
                 LIMIT :limit
                """),
                {"probe": _vec_str(qvec), "limit": top_k * 2},
            )
            candidate_chunk_ids = [row[0] for row in r.all()]

    if verbose:
        print(
            f"[query] candidates: chunks={len(candidate_chunk_ids)} "
            f"artifacts={len(candidate_artifact_ids)}"
        )

    # -------- step 9: multi-probe vector rerank --------
    probes = [resolved_query]
    if decompose:
        _intent = (parsed.get("intent") or "").lower()
        # Use the PARSED entity list (clean, distinct names the LLM extracted --
        # e.g. Ozempic, Mounjaro, Trulicity, Victoza) rather than matched_entities,
        # which holds many surface-form variants of the same drug (Ozempic pen,
        # OZEMPIC, MOUNJARO KwikPen, ...). Dedupe case-insensitively, keep order.
        _ent_names: list[str] = []
        _seen_ent: set[str] = set()
        for _n in (parsed.get("entities") or []):
            _k = (_n or "").strip().lower()
            if _k and _k not in _seen_ent:
                _seen_ent.add(_k)
                _ent_names.append(_n.strip())
        _max_ent = int(
            get_settings().app_config.get("qa", {}).get("max_entity_probes", 8)
        )
        # Comparison/enumeration over many entities: fan out ONE probe per entity
        # so rerank surfaces each competitor evenly (not just the 1-2 a generic
        # decompose named). Else fall back to the generic decompose.
        if _use_entity_probes(_intent, len(_ent_names), _max_ent):
            for p in await _entity_probes(router, resolved_query, _ent_names[:_max_ent]):
                if p.strip() and p.strip() != resolved_query.strip():
                    probes.append(p)
            if verbose:
                print(
                    f"[query] entity-probe fan-out ({_intent}, "
                    f"{len(_ent_names)} entities): {len(probes) - 1} per-entity probe(s)"
                )
        else:
            sub_qs = await _query_decompose(router, resolved_query)
            for sq in sub_qs:  # keep up to max_probes total; always include original
                if len(probes) >= max_probes:
                    break
                if sq.strip() and sq.strip() != resolved_query.strip():
                    probes.append(sq)

    # Alias injection, applied to whichever probe set was built above.
    # The chunk that broke the reported query says "MOUNJARO" and never
    # "tirzepatide"; without the brand name in a probe, no probe can
    # reach it.
    probes = _apply_aliases_to_probes(probes, aliases, verbose=verbose)

    if verbose:
        print(f"[query] probes ({len(probes)}): {probes}")

    probe_vecs = await embedder.embed(probes)

    # Rerank all probes on ONE session so the per-connection query-plan cache /
    # warm pgvector state is reused (first rerank ~cold, the rest are ~5x faster).
    # NOTE: do NOT parallelize with a session-per-probe here -- on a CPU-limited
    # pooled Postgres (Supabase) concurrent queries serialize AND each fresh
    # session pays the cold cost, which is strictly slower. The id-filtered
    # rerank can't use the HNSW index (exact scan), so the real levers are fewer
    # probes (max_probes) and a smaller candidate set, not concurrency.
    chunk_rankings: list[list[uuid.UUID]] = []
    artifact_rankings: list[list[uuid.UUID]] = []
    async with session_scope() as session:
        for probe_text, pvec in zip(probes, probe_vecs, strict=False):
            ranked = await retrieval_sql.vector_rerank_chunks(
                session, candidate_chunk_ids, pvec, top_k=top_k * 3,
            )
            chunk_rankings.append(
                _trim_by_distance(ranked, label=probe_text, verbose=verbose)
            )
            if candidate_artifact_ids:
                ranked_a = await retrieval_sql.vector_rerank_artifacts(
                    session, candidate_artifact_ids, pvec, top_k=top_k,
                )
                artifact_rankings.append(_trim_by_distance(ranked_a))

    # Graph-distance ranking: BFS score per chunk (via its entities). Uses the
    # entity-coverage ranking from step 8, propagated to full-text chunks by the
    # step-8.5 bridge when present (else identical to `ent_chunks`).
    chunk_rankings.append(graph_chunk_ranking)
    # Document-similarity ranking as its own RRF voter. Filtered to the
    # surviving candidate set -- the step-8.5 bridge may have swapped a
    # document's summary chunks out for its full-text ones, and voting for
    # a chunk that is no longer a candidate would burn a top_k slot on a
    # row that gets dropped again at context-packing time.
    if doc_chunks:
        _cand = set(candidate_chunk_ids)
        _doc_ranking = [cid for cid, _ in doc_chunks if cid in _cand]
        if _doc_ranking:
            chunk_rankings.append(_doc_ranking)
    if artifact_candidates:
        artifact_rankings.append([aid for aid, _ in artifact_candidates])

    # Records whether step 10b ran, for retrieval_plan (see below).
    _rerank_note: dict[str, Any] | None = None

    # -------- step 10: RRF fusion --------
    if mode == "artifact_only":
        fused_chunks: list[tuple[uuid.UUID, float]] = []
    else:
        _fused_all = rrf_fuse(chunk_rankings)
        # -------- step 10b: optional vector rerank of the fused pool --------
        # RRF fuses rank ORDER and throws the distances away (see
        # `_trim_by_distance`), so a chunk that merely placed well across
        # several probes can outrank one that is genuinely closer to the
        # question. This re-sorts the fused pool by L2 distance to the
        # ORIGINAL question embedding, which is the signal fusion discarded.
        #
        # It runs on a pool WIDER than top_k and BEFORE the cap + truncation,
        # so it can swap a better chunk into the final set rather than merely
        # reorder the ones RRF already chose -- reordering alone would not
        # move precision, which is the point of the flag.
        #
        # Each chunk keeps its ORIGINAL RRF score: `cap_per_source` reads that
        # score as higher-is-better, and it is persisted to
        # `retrieval_evidence.score`. Substituting a distance (lower-is-better)
        # would silently invert both. The consequence is that score is no
        # longer monotonic with rank on a reranked run, so the run records
        # `retrieval_plan.rerank` to say the ordering key is not the score.
        _rerank_on = (
            bool(_qa_cfg("rerank_after_fusion", False)) if rerank is None
            else bool(rerank)
        )
        if _rerank_on and _fused_all:
            _pool_mult = max(1, int(_qa_cfg("rerank_pool_multiplier", 3)))
            _pool = _fused_all[: top_k * _pool_mult]
            _pool_ids = [cid for cid, _ in _pool]
            async with session_scope() as session:
                _reranked = await retrieval_sql.vector_rerank_chunks(
                    session, _pool_ids, qvec, top_k=len(_pool_ids)
                )
            if _reranked:
                _rrf_score = dict(_pool)
                _order = [cid for cid, _ in _reranked]
                # Anything the rerank did not return (no embedding) keeps its
                # RRF position behind the reranked head, rather than vanishing.
                _seen = set(_order)
                _tail = [cid for cid, _ in _pool if cid not in _seen]
                _fused_all = (
                    [(cid, _rrf_score[cid]) for cid in _order + _tail]
                    + _fused_all[top_k * _pool_mult :]
                )
                _rerank_note = {
                    "applied": True,
                    "pool": len(_pool_ids),
                    "pool_multiplier": _pool_mult,
                    "note": "ordered by distance to the question embedding; "
                            "score column remains the RRF score",
                }
                if verbose:
                    print(
                        f"[query] reranked {len(_pool_ids)} fused chunk(s) "
                        f"against the question embedding"
                    )
        # Cap one document's share BEFORE truncating to top_k. RRF's usual
        # defence is agreement between independent lists, but all three
        # voters here run over the same candidate pool -- when that pool is
        # concentrated they agree rather than cross-check, and one document
        # took all 30 evidence slots while 20 documents held the relevant
        # section. Capping after fusion keeps relevance in charge of the
        # ordering and only trims the monopoly.
        _cap = _resolve_doc_cap(_qa_cfg("max_chunks_per_document", 0.75), top_k)
        if _cap > 0:
            async with session_scope() as session:
                _doc_of = await retrieval_sql.fetch_chunk_document_ids(
                    session, [cid for cid, _ in _fused_all[: top_k * 4]]
                )
            _fused_all = cap_per_source(
                _fused_all, _doc_of, _cap,
                relevance_ratio=float(_qa_cfg("diversity_relevance_ratio", 0.0)),
            )
        fused_chunks = _fused_all[:top_k]
    # artifact_only packs a full artifact context (not top_k//2).
    _art_k = top_k if mode == "artifact_only" else top_k // 2
    fused_artifacts = rrf_fuse(artifact_rankings)[:_art_k]

    # -------- step 11: pack context --------
    _fused_chunk_ids = [cid for cid, _ in fused_chunks]
    async with session_scope() as session:
        chunk_rows = await retrieval_sql.fetch_chunk_text(session, _fused_chunk_ids)
        artifact_rows = await retrieval_sql.fetch_artifact_rows(
            session, [aid for aid, _ in fused_artifacts]
        )
        # Attribution: which entities each surviving chunk is actually
        # about. One query over the final top-k, so the synthesis can be
        # told that a passage concerns dulaglutide and must not be quoted
        # as tirzepatide's dose.
        chunk_entities = await retrieval_sql.fetch_chunk_entity_names(
            session, _fused_chunk_ids
        )
        # Same idea for time: a date-scoped question ("by 2025") can only be
        # honoured if each passage carries the period it concerns.
        chunk_times = await retrieval_sql.fetch_chunk_time_labels(
            session, _fused_chunk_ids
        )

    evidence: list[dict[str, Any]] = []
    for rank, (cid, score) in enumerate(fused_chunks, start=1):
        info = chunk_rows.get(cid)
        if info is None:
            continue
        evidence.append({
            "kind": "chunk",
            # `node_id` is the chunk's primary-key UUID; persisted as
            # retrieval_evidence.evidence_uuid so downstream consumers can
            # JOIN back to chunks.id without parsing the IRI string.
            "node_id": cid,
            "iri": info["iri"],
            "rank": rank,
            "score": score,
            "text": info["text"],
            "document_iri": info["document_iri"],
            "document_title": info["document_title"],
            "entities": chunk_entities.get(cid, []),
            # Pre-rendered span string, not a list: placing both endpoints
            # requires seeing all of a chunk's periods (see
            # retrieval_sql._render_time_span).
            "times": chunk_times.get(cid, ""),
        })
    for rank, (aid, score) in enumerate(fused_artifacts, start=1):
        info = artifact_rows.get(aid)
        if info is None:
            continue
        evidence.append({
            "kind": "artifact",
            # `node_id` is the artifact's primary-key UUID; persisted as
            # retrieval_evidence.evidence_uuid so JOINs against
            # intelligence_artifacts.id work directly.
            "node_id": aid,
            "iri": info["iri"],
            "artifact_type": info["type"],
            "rank": rank + len(fused_chunks),
            "score": score,
            "text": info["text"],
            "confidence": info["confidence"],
        })

    # -------- step 12: answer generation --------
    answer_task_name = _ANSWER_TASK_BY_MODE[mode]
    answer_prompt_fn = PROMPTS[answer_task_name]
    _evidence_cap = int(
        get_settings().app_config.get("qa", {}).get("evidence_char_cap", 600)
    )
    # Hand the mined aliases to the synthesis too, not just to retrieval.
    # Finding the evidence and understanding it are separate problems: the
    # model was being told alternate names exist without being told WHICH,
    # so it had to re-derive "MOUNJARO is tirzepatide" from whatever
    # happened to be in the context window.
    # `time_terms` likewise: the parser extracts a date constraint and, until
    # now, only used it to seed graph nodes. The synthesis never saw it, so
    # "by 2025" was answered with a 2026 approval.
    sys_p, user_p = answer_prompt_fn(
        resolved_query,
        evidence,
        _evidence_cap,
        parsed.get("aliases") or {},
        parsed.get("time_terms") or [],
    )
    try:
        out = await router.chat(answer_task_name, system=sys_p, user=user_p)
        answer = out.text.strip()
    except Exception as exc:
        answer = f"(answer generation failed: {exc})"

    # Every cited id must resolve to something we actually retrieved.
    answer, invalid_citations = _validate_citations(answer, evidence)
    if invalid_citations:
        log.warning(
            "stripped %d unresolvable citation(s) from the answer: %s",
            len(invalid_citations), ", ".join(invalid_citations[:8]),
        )

    cost = router.total_cost_usd - cost_before
    result = RetrievalResult(
        answer=answer,
        mode=mode,
        resolved_query=resolved_query,
        evidence=evidence,
        parsed=parsed,
        invalid_citations=invalid_citations,
        cost_usd=cost,
        wall_seconds=time.time() - t0,
    )
    if persist:
        await _persist_run(
            result, mode=mode, resolved_query=resolved_query,
            matched_classes=matched_classes, matched_entities=matched_entities,
            matched_times=matched_times, graph_hops=hops,
            conversation_turn_id=conversation_turn_id,
            rerank_note=_rerank_note,
        )
    else:
        async with session_scope() as session:
            result.graph_version = await current_version(session)
    return result


_ANSWER_TASK_BY_MODE = {
    "simple_qa": "answer_simple_qa",
    "deep_research": "answer_deep_research",
    # artifact_only reuses the deep_research structured 7-section synthesis.
    "artifact_only": "answer_deep_research",
}


# --------------- helper steps ---------------


async def _question_parse(router: LLMRouter, q: str) -> dict[str, Any]:
    system, user = PROMPTS["question_parse"](q)
    try:
        out = await router.chat("question_parse", system=system, user=user)
        parsed = _extract_json(out.text) or {}
    except Exception:
        parsed = {}
    return {
        "entities": parsed.get("entities") or [],
        "classes": parsed.get("classes") or [],
        "time_terms": parsed.get("time_terms") or [],
        "intent": parsed.get("intent") or "factoid",
    }


async def _expand_aliases(
    entity_terms: list[str], *, acronyms_only: bool = False
) -> dict[str, list[str]]:
    """Corpus-mined synonyms for the parsed query terms.

    Keyed by the ORIGINAL surface term as it appeared in the question, so
    probe rewriting can substitute in place. Terms are normalized with
    the same normalizer used at ingest before lookup -- note the existing
    entity trigram match at `_ontology_match` only lowercases, so
    punctuation is handled inconsistently there; this path uses the real
    one.

    `acronyms_only` keeps just those aliases where the query term or its
    alias is an acronym. Used for parsed CLASS terms, which are common
    nouns: it recovers the acronym expansions the parser misfiled as
    classes (MRHD, ACE) without letting a class of "hormone" drag in
    "angiotensin II".

    Fails soft: any error (table missing because 0006 has not been
    applied yet, DB hiccup) yields {} and retrieval proceeds unchanged.
    """
    if not entity_terms:
        return {}
    max_aliases = int(_qa_cfg("max_aliases", 6))
    if max_aliases <= 0:
        return {}
    norm_to_original: dict[str, str] = {}
    for term in entity_terms[:10]:
        norm = normalize_term(term or "")
        if norm and norm not in norm_to_original:
            norm_to_original[norm] = term.strip()
    if not norm_to_original:
        return {}
    try:
        async with session_scope() as session:
            by_norm = await retrieval_sql.fetch_aliases(
                session, list(norm_to_original), per_term_limit=max_aliases
            )
    except Exception as exc:
        log.warning(
            "alias lookup failed (%s: %s); continuing without alias "
            "expansion. Has `mine-aliases` been run?",
            type(exc).__name__, exc,
        )
        return {}
    out: dict[str, list[str]] = {}
    for norm, surfaces in by_norm.items():
        if norm not in norm_to_original or not surfaces:
            continue
        original = norm_to_original[norm]
        term_is_acronym = is_acronym(original.strip())
        # The SQL allows up to 6 words so acronym expansions survive; tighten
        # back to 3 for everything else, where a long surface is boilerplate
        # rather than a name and would poison the probe it rides in.
        surfaces = [
            s for s in surfaces
            if term_is_acronym or is_acronym(s) or len(s.split()) <= 3
        ]
        if acronyms_only:
            # Class terms are common nouns. Keep the pair only if this is
            # genuinely an acronym relationship, or a class of "hormone"
            # would drag in "angiotensin II".
            surfaces = [
                s for s in surfaces if term_is_acronym or is_acronym(s)
            ]
        if not surfaces:
            continue
        out[original] = surfaces
    return out


_MIN_LITERAL_ALIAS_CHARS = 4


async def _expand_aliases_from_question(question: str) -> dict[str, list[str]]:
    """Mined aliases for terms that appear VERBATIM in the question.

    The parser is the weak link in alias expansion: it reduces multi-word
    technical phrases to a head noun, so "What is the maximum recommended
    human dose...?" parses to classes ['dose', 'animal studies'] and the
    `maximum recommended human dose <-> MRHD` pair -- which is sitting
    right there in the table -- is never looked up. Asking the same
    question by its acronym DID expand, so the two phrasings diverged for
    no reason other than how an LLM chose to chop the sentence.

    This scan does not consult the parser at all. It matches the alias
    vocabulary against the raw question on whole-word boundaries, longest
    first so "maximum recommended human dose" wins over any substring of
    itself.

    Fails soft, like every other alias path.
    """
    q_norm = normalize_term(question or "")
    if not q_norm:
        return {}
    max_aliases = int(_qa_cfg("max_aliases", 6))
    if max_aliases <= 0:
        return {}
    try:
        async with session_scope() as session:
            pairs = await retrieval_sql.fetch_all_alias_terms(session)
    except Exception as exc:
        log.warning(
            "literal alias scan failed (%s: %s); continuing without it",
            type(exc).__name__, exc,
        )
        return {}

    # `normalize_term` DELETES hyphens rather than replacing them with a
    # space, so "angiotensin-converting enzyme" normalizes to
    # "angiotensinconverting enzyme" and never matches the stored
    # "angiotensin converting enzyme". Changing the normalizer would shift
    # entity matching everywhere (and it is pinned to the ingest-time one
    # by test), so widen the haystack here instead: scan both forms.
    haystacks = {f" {q_norm} "}
    hyphen_split = normalize_term(re.sub(r"[-‐-―]", " ", question or ""))
    if hyphen_split and hyphen_split != q_norm:
        haystacks.add(f" {hyphen_split} ")

    hits: list[tuple[int, str, str]] = []   # (length, surface_found, alias)
    for term_a, term_b, surface_a, surface_b in pairs:
        for term, surface, alias in (
            (term_a, surface_a, surface_b),
            (term_b, surface_b, surface_a),
        ):
            # Short terms are too collision-prone for a substring scan;
            # acronyms are exempt because that is the whole point.
            if len(term.replace(" ", "")) < _MIN_LITERAL_ALIAS_CHARS and not is_acronym(
                surface.strip()
            ):
                continue
            if any(f" {term} " in h for h in haystacks):
                hits.append((len(term), surface, alias))

    out: dict[str, list[str]] = {}
    for _len, surface, alias in sorted(hits, key=lambda h: -h[0]):
        if len(out) >= max_aliases:
            break
        bucket = out.setdefault(surface, [])
        if alias not in bucket and len(bucket) < max_aliases:
            bucket.append(alias)
    return out


async def _ontology_match(
    parsed: dict[str, Any],
    qvec: list[float],
    *,
    aliases: dict[str, list[str]] | None = None,
) -> tuple[
    list[tuple[uuid.UUID, str]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Vector + trgm match the parsed terms against ontology + entity
    rows. Returns (seeds, matched_classes, matched_entities,
    matched_times)."""
    seeds: list[tuple[uuid.UUID, str]] = []
    matched_classes: list[dict[str, Any]] = []
    matched_entities: list[dict[str, Any]] = []
    matched_times: list[dict[str, Any]] = []

    async with session_scope() as session:
        # Vector-match classes against the resolved-query embedding.
        r = await session.execute(
            sql_text("""
            SELECT id, iri, label, (embedding <-> CAST(:probe AS vector)) AS dist
              FROM graphrag.ontology_classes
             WHERE embedding IS NOT NULL
             ORDER BY dist LIMIT 30
            """),
            {"probe": _vec_str(qvec)},
        )
        for cid, iri, label, dist in r.all():
            matched_classes.append(
                {"id": cid, "iri": iri, "label": label or "", "dist": float(dist)}
            )
            seeds.append((cid, "ontology_class"))

        # Vector + trgm entity match per parsed entity term. Mined aliases
        # are matched alongside the question's own wording, so a graph
        # entity minted as "MOUNJARO" is seeded by a question that only
        # ever says "tirzepatide". Costs no rerank pass -- this widens the
        # candidate pool, it does not add probes.
        _terms: list[str] = list(parsed.get("entities", [])[:10])
        for _alias_list in (aliases or {}).values():
            _terms.extend(_alias_list)
        for term in _terms:
            if not term.strip():
                continue
            r = await session.execute(
                sql_text("""
                SELECT id, name,
                       similarity(normalized_name, lower(:t)) AS sim
                  FROM graphrag.entities
                 WHERE similarity(normalized_name, lower(:t)) >= 0.4
                 ORDER BY sim DESC LIMIT 10
                """),
                {"t": term},
            )
            for eid, name, sim in r.all():
                matched_entities.append(
                    {"id": eid, "name": name, "sim": float(sim)}
                )
                seeds.append((eid, "entity"))

        # If no entity term parsed out, still try a global vector pass.
        if not matched_entities:
            r = await session.execute(
                sql_text("""
                SELECT id, name, (embedding <-> CAST(:probe AS vector)) AS dist
                  FROM graphrag.entities
                 WHERE embedding IS NOT NULL
                 ORDER BY dist LIMIT 15
                """),
                {"probe": _vec_str(qvec)},
            )
            for eid, name, dist in r.all():
                matched_entities.append(
                    {"id": eid, "name": name, "sim": 1.0 - float(dist) / 2}
                )
                seeds.append((eid, "entity"))

        # Time term matching: try YEAR_YYYY / MONTH_YYYY_MM etc identifiers.
        for term in parsed.get("time_terms", [])[:5]:
            m = re.search(r"\b(19|20)\d{2}\b", term)
            if not m:
                continue
            year = m.group(0)
            ident = f"YEAR_{year}"
            r = await session.execute(
                sql_text(
                    "SELECT id, display_label FROM graphrag.time_instances "
                    "WHERE time_identifier = :ident"
                ),
                {"ident": ident},
            )
            row = r.first()
            if row:
                matched_times.append(
                    {"id": row[0], "display_label": row[1]}
                )
                seeds.append((row[0], "time_instance"))

    return seeds, matched_classes, matched_entities, matched_times


async def _concept_expansion(
    router: LLMRouter, q: str, matched_classes: list[dict[str, Any]]
) -> list[uuid.UUID]:
    """LLM picks 0-15 additional class IRIs from the matched list."""
    if not matched_classes:
        return []
    listing = [(c["iri"], c["label"]) for c in matched_classes[:20]]
    sys_p, user_p = PROMPTS["concept_expansion"](q, listing)
    try:
        out = await router.chat("concept_expansion", system=sys_p, user=user_p)
        parsed = _extract_json(out.text) or {}
    except Exception:
        return []
    related = parsed.get("related_classes") or []
    if not related:
        return []
    # Resolve IRIs back to IDs in our matched set (we don't trust
    # the LLM to invent IRIs).
    iri_to_id = {c["iri"]: c["id"] for c in matched_classes}
    return [iri_to_id[iri] for iri in related if iri in iri_to_id]


async def _query_decompose(router: LLMRouter, q: str) -> list[str]:
    sys_p, user_p = PROMPTS["query_decompose"](q)
    try:
        out = await router.chat("query_decompose", system=sys_p, user=user_p)
        parsed = _extract_json(out.text) or {}
    except Exception:
        return []
    sqs = parsed.get("sub_questions") or []
    return [s for s in sqs if isinstance(s, str) and s.strip()]


def _trim_by_distance(
    ranked: list[tuple[uuid.UUID, float]],
    *,
    label: str | None = None,
    verbose: bool = False,
) -> list[uuid.UUID]:
    """Drop a probe's weak tail before RRF sees it.

    RRF fuses rank ORDER and discards distance, so a probe that matched
    nothing still casts full-strength rank-1..N votes -- indistinguishable
    from a probe that found exactly the right passage. In the reported
    failure the tirzepatide probe had no real match, yet its best-of-
    nothing results were fused as confidently as the dulaglutide probes'
    genuine hits, and dulaglutide passages swept the answer context.

    Two guards, both from `qa.*` and both no-ops when unset:

      probe_dist_floor -- if even the BEST hit is farther than this, the
                          probe found nothing; contribute no votes at all.
      probe_dist_band  -- keep only results within this distance of the
                          probe's own best hit.

    Distances are L2 over normalized OpenAI embeddings, so they are
    comparable across probes.
    """
    if not ranked:
        return []
    floor = _qa_cfg("probe_dist_floor", None)
    band = _qa_cfg("probe_dist_band", None)
    if floor is None and band is None:
        return [cid for cid, _ in ranked]

    best = min(dist for _, dist in ranked)
    if floor is not None and best > float(floor):
        if verbose:
            print(
                f"[query] probe dropped (best dist {best:.4f} > floor "
                f"{float(floor):.2f}): {label!r}"
            )
        return []
    if band is None:
        return [cid for cid, _ in ranked]
    cutoff = best + float(band)
    kept = [cid for cid, dist in ranked if dist <= cutoff]
    if verbose and len(kept) < len(ranked):
        print(
            f"[query] probe trimmed {len(ranked)} -> {len(kept)} "
            f"(best {best:.4f}, cutoff {cutoff:.4f}): {label!r}"
        )
    return kept


def _apply_aliases_to_probes(
    probes: list[str],
    aliases: dict[str, list[str]],
    *,
    verbose: bool = False,
) -> list[str]:
    """Inject corpus-mined aliases into the probe list.

    Mode comes from `qa.alias_probe_mode`:

      blended  -- rewrite in place: "Dosing schedule of tirzepatide"
                  becomes "Dosing schedule of tirzepatide (Mounjaro)".
                  Costs NO extra rerank pass.
      separate -- add a dedicated probe per alias:
                  "Dosing schedule of Mounjaro". Strictly better recall
                  (the measured evidence is that a dedicated brand-name
                  probe moved the target chunk from rank #886 to #1),
                  at one sequential id-filtered rerank each.
      off      -- aliases used for graph seeding only.

    Default is `blended`. If it proves too weak on a given corpus, flip
    the knob rather than editing code.

    No aliases (or an unrecognized mode) returns the input list
    unchanged, so behavior is identical to before this feature existed.
    """
    if not aliases or not probes:
        return probes
    mode = str(_qa_cfg("alias_probe_mode", "blended")).strip().lower()
    if mode not in ("blended", "separate"):
        return probes

    max_alias_probes = int(_qa_cfg("max_alias_probes", 4))
    out: list[str] = []
    added = 0
    seen = {p.strip().lower() for p in probes}

    for probe in probes:
        rewritten = probe
        lowered = probe.lower()
        for term, surfaces in aliases.items():
            if not term or term.lower() not in lowered:
                continue
            # Don't repeat an alias the probe already names.
            fresh = [s for s in surfaces if s.lower() not in lowered]
            if not fresh:
                continue
            if mode == "blended":
                # Substitute via a callable so the matched text is kept
                # verbatim -- a case-insensitive match on "TIRZEPATIDE"
                # must not rewrite it to the lowercased parse output.
                #
                # 3, not 2: semaglutide legitimately has three brands
                # (Ozempic, Wegovy, Rybelsus) and a 2-cap left Rybelsus out
                # of every probe, so "which drugs contain semaglutide?"
                # answered with two of the three.
                _n_blend = int(_qa_cfg("alias_blend_count", 3))
                _suffix = f" ({', '.join(fresh[:_n_blend])})"
                rewritten = re.sub(
                    re.escape(term),
                    lambda m, s=_suffix: m.group(0) + s,
                    rewritten,
                    count=1,
                    flags=re.IGNORECASE,
                )
                lowered = rewritten.lower()
        out.append(rewritten)

    if mode == "separate":
        for probe in probes:
            lowered = probe.lower()
            for term, surfaces in aliases.items():
                if not term or term.lower() not in lowered:
                    continue
                for surface in surfaces:
                    if added >= max_alias_probes:
                        break
                    if surface.lower() in lowered:
                        continue
                    variant = re.sub(
                        re.escape(term), surface, probe, count=1, flags=re.IGNORECASE
                    )
                    key = variant.strip().lower()
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append(variant)
                    added += 1

    if verbose and out != probes:
        # blended rewrites in place (0 added) while separate appends, so
        # report both numbers rather than just the length delta.
        rewritten = sum(
            1 for a, b in zip(probes, out[: len(probes)]) if a != b
        )
        print(
            f"[query] alias probe mode={mode}: "
            f"{rewritten} probe(s) rewritten, {len(out) - len(probes)} added"
        )
    return out


def _use_entity_probes(intent: str, n_entities: int, max_entity_probes: int) -> bool:
    """Trigger for per-entity probe fan-out: a comparison/enumeration query over
    several entities. Otherwise the generic decompose path is used."""
    return (
        max_entity_probes > 0
        and (intent or "").lower() in ("comparison", "enumeration")
        and n_entities >= 3
    )


async def _entity_probes(
    router: LLMRouter, q: str, entity_names: list[str]
) -> list[str]:
    """One focused probe per entity (comparison/enumeration fan-out), so vector
    rerank surfaces each entity's evidence evenly. Returns [] on failure ->
    caller falls back to the generic decompose probes."""
    if not entity_names:
        return []
    sys_p, user_p = PROMPTS["entity_probes"](q, entity_names)
    try:
        out = await router.chat("entity_probes", system=sys_p, user=user_p)
        parsed = _extract_json(out.text) or {}
    except Exception:
        return []
    probes = parsed.get("probes") or []
    return [p for p in probes if isinstance(p, str) and p.strip()]


async def _plan_rounds(router: LLMRouter, q: str) -> dict[str, Any]:
    """Decide whether `q` needs a 2nd (dependent) retrieval round. Conservative
    by design -- only genuine bridge questions (must discover entities first, or
    a multi-part question whose 2nd part depends on the 1st) get two rounds; the
    common case returns needs_second_round=False. Fails safe to one round."""
    sys_p, user_p = PROMPTS["retrieval_rounds_plan"](q)
    # The task must EXIST. This lookup used to sit inside the try below, so when
    # `retrieval_rounds_plan` was missing from every preset the KeyError was
    # swallowed by the fail-safe and two-round retrieval silently never ran --
    # indistinguishable from the planner legitimately choosing one round.
    # "Fail safe to one round" is right for a bad LLM response; it is wrong for
    # a misconfiguration, which is permanent and invisible.
    router.task_spec("retrieval_rounds_plan")
    try:
        out = await router.chat("retrieval_rounds_plan", system=sys_p, user=user_p)
        parsed = _extract_json(out.text) or {}
    except Exception as exc:
        log.warning(
            "round planner call failed (%s: %s); falling back to a single "
            "retrieval round", type(exc).__name__, exc,
        )
        return {"needs_second_round": False, "round1_question": "", "round2_question": ""}
    r1 = (parsed.get("round1_question") or "").strip()
    r2 = (parsed.get("round2_question") or "").strip()
    # Guard: only treat as multi-round if BOTH sub-questions are present and the
    # bridge question is actually distinct from the original (avoids spurious
    # 2-round runs the planner sometimes proposes for simple questions).
    needs = bool(parsed.get("needs_second_round")) and bool(r1) and bool(r2) \
        and r1.strip().lower() != q.strip().lower()
    return {
        "needs_second_round": needs,
        "round1_question": r1,
        "round2_question": r2,
        "reason": parsed.get("reason") or "",
    }


# --------------- persistence ---------------


async def _persist_run(
    result: RetrievalResult,
    *,
    mode: str,
    resolved_query: str,
    matched_classes: list[dict[str, Any]],
    matched_entities: list[dict[str, Any]],
    matched_times: list[dict[str, Any]],
    graph_hops: int,
    conversation_turn_id: uuid.UUID | None,
    rerank_note: dict[str, Any] | None = None,
) -> None:
    """INSERT retrieval_runs + retrieval_evidence rows."""
    run_id = uuid.uuid4()
    async with session_scope() as session:
        gv = await current_version(session)
        result.graph_version = gv
        await session.execute(
            sql_text("""
            INSERT INTO graphrag.retrieval_runs (
              id, conversation_turn_id, resolved_query, retrieval_mode,
              matched_classes, matched_entities, matched_time_instances,
              graph_hops, retrieval_plan, graph_version, created_at
            ) VALUES (
              :id, :ctid, :q, :mode,
              CAST(:mc AS jsonb), CAST(:me AS jsonb), CAST(:mt AS jsonb),
              :hops, CAST(:plan AS jsonb), :gv, now()
            )
            """),
            {
                "id": run_id, "ctid": conversation_turn_id,
                "q": resolved_query, "mode": mode,
                "mc": json.dumps([
                    {"iri": c["iri"], "label": c.get("label", "")}
                    for c in matched_classes[:20]
                ]),
                "me": json.dumps([
                    {"name": e["name"]} for e in matched_entities[:20]
                ]),
                "mt": json.dumps([
                    {"label": t["display_label"]} for t in matched_times[:10]
                ]),
                "hops": graph_hops,
                # Aliases are recorded so a bad expansion is diagnosable
                # after the fact -- otherwise a wrong mined pair would be
                # invisible in the run history.
                "plan": json.dumps({
                    "intent": result.parsed.get("intent", ""),
                    "aliases": result.parsed.get("aliases", {}),
                    "time_terms": result.parsed.get("time_terms", []),
                    # Persisted so citation quality is trendable across runs
                    # rather than only visible in a log line.
                    "invalid_citations": result.invalid_citations,
                    # Present only on a reranked run. Says the evidence
                    # ORDER came from distance to the question embedding
                    # while `score` is still the RRF score -- so score is
                    # not monotonic with rank here, and sorting evidence
                    # by score does NOT reproduce what was delivered.
                    **({"rerank": rerank_note} if rerank_note else {}),
                }),
                "gv": gv,
            },
        )

        # Evidence rows. evidence_uuid carries the source node's primary
        # key (chunk_id for kind='chunk', artifact_id for kind='artifact')
        # so downstream queries can JOIN re.evidence_uuid against
        # graphrag.chunks.id / graphrag.intelligence_artifacts.id without
        # parsing the IRI string. Falls back to a fresh UUID only if a
        # caller produced an evidence dict without `node_id` (legacy).
        ev_rows = []
        for ev in result.evidence[:50]:
            node_id = ev.get("node_id")
            if node_id is None:
                node_id = uuid.uuid4()
            ev_rows.append({
                "id": uuid.uuid4(),
                "retrieval_run_id": run_id,
                "evidence_kind": ev["kind"],
                "evidence_uuid": node_id,
                "evidence_iri": ev["iri"],
                "rank": ev["rank"],
                "score": float(ev["score"]),
                "snippet": (ev.get("text") or "")[:500],
            })
        if ev_rows:
            await session.execute(
                sql_text("""
                INSERT INTO graphrag.retrieval_evidence (
                  id, retrieval_run_id, evidence_kind, evidence_uuid,
                  evidence_iri, rank, score, snippet, created_at
                ) VALUES (
                  :id, :retrieval_run_id, :evidence_kind, :evidence_uuid,
                  :evidence_iri, :rank, :score, :snippet, now()
                )
                """),
                ev_rows,
            )

    result.retrieval_run_id = run_id
