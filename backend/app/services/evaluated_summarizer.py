"""Evaluated (near-lossless) document summarizer.

Ported from the standalone reference summarizer into the async, LLMRouter-based
platform. Per source window it runs an evaluator feedback loop:

    summarize -> generate source-derived questions -> evaluate the summary
    against them -> revise on gaps -> repeat (default `eval_rounds` rounds)

then a deterministic section-coverage check. We stop at the per-chunk evaluated
summaries (the reference's `summary.chunks.md`) -- NO final LLM merge, which is
lossy.

Contract for ingestion: `evaluated_summarize_documents_async` returns one
`EvaluatedDocSummary` per input document, carrying:
  - `chunk_summaries`: the list of STORED summary chunks (each <=
    `summary_chunk_max_tokens`; a large per-window summary is split on its
    section headers so retrieval embeddings stay sharp and the synthesis prompt
    isn't truncated away -- lossless, all content retained across pieces),
  - `combined`: the concatenation (stored as documents.text_summary),
  - `audit`: per-window evaluator details (rounds, missing items, coverage).

Small documents (<= `threshold_tokens`) are NOT summarized: their original text
is chunked normally (via chunking.chunk_documents) and returned as the summary
chunks, matching the legacy single-pass behavior.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tiktoken

from backend.app.services.chunking import TextChunk, chunk_documents
from backend.app.services.db_artifact_gen import _extract_json
from backend.app.services.document_io import LoadedDocument
from backend.app.services.llm_router import LLMRouter
from backend.app.services.prompts import PROMPTS, REQUIRED_SECTIONS

# Cache namespace -- bump when the loop/prompts change so stale summaries drop.
_EVAL_SUMMARY_VERSION = "v1"


def get_encoder(encoding_name: str = "o200k_base"):
    try:
        return tiktoken.get_encoding(encoding_name)
    except (KeyError, ValueError):
        return tiktoken.get_encoding("o200k_base")


@dataclass
class EvaluatedDocSummary:
    path: Path
    chunk_summaries: list[str]
    combined: str
    summarized: bool
    audit: dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.path.name


def evaluated_result_to_chunks(
    result: EvaluatedDocSummary, enc=None
) -> list[TextChunk]:
    """Flatten one document's evaluated summary into TextChunks -- exactly one
    chunk per stored summary piece (already size-capped + section-split by the
    summarizer). Shared by register-documents (db_document_ingest) and the
    prune-expand streaming path so both produce identical summary chunks."""
    if enc is None:
        enc = get_encoder()
    return [
        TextChunk(
            index=i,
            text=t,
            token_count=len(enc.encode(t)),
            source_name=result.name,
        )
        for i, t in enumerate(result.chunk_summaries)
    ]


# --------------------------------------------------------------------------- #
# Deterministic section-coverage check (complements the LLM evaluator).
# --------------------------------------------------------------------------- #
def check_section_coverage(summary_text: str) -> dict[str, bool]:
    """For each required section, whether its header appears in the summary.
    A section may legitimately be present but say 'None identified in this
    chunk.' -- this only checks the header is accounted for."""
    coverage: dict[str, bool] = {}
    for canonical, aliases in REQUIRED_SECTIONS:
        found = False
        for alias in aliases:
            pattern = r"(?mi)^\s*[#>*\-\s]*" + re.escape(alias) + r"\b"
            if re.search(pattern, summary_text):
                found = True
                break
        coverage[canonical] = found
    return coverage


# Header line matcher used to split a section-structured summary into pieces.
_ALL_SECTION_ALIASES = [a for _, aliases in REQUIRED_SECTIONS for a in aliases]
_SECTION_HEADER_RE = re.compile(
    r"(?mi)^\s*[#>*\-\s]*(?:" + "|".join(re.escape(a) for a in _ALL_SECTION_ALIASES) + r")\b.*$"
)


def _count_tokens(text: str, enc) -> int:
    return len(enc.encode(text))


def _token_split(text: str, max_tokens: int, enc) -> list[str]:
    toks = enc.encode(text)
    if len(toks) <= max_tokens:
        return [text]
    return [enc.decode(toks[i : i + max_tokens]) for i in range(0, len(toks), max_tokens)]


def _split_on_sections(summary: str, max_tokens: int, enc) -> list[str]:
    """Split one evaluated chunk summary into pieces <= `max_tokens`, preferring
    the section-header boundaries so CLAIMS/EVIDENCE/... blocks stay coherent.
    Lossless: every character ends up in exactly one piece. Falls back to a
    token split when there are no section headers or a single section is too big."""
    if _count_tokens(summary, enc) <= max_tokens:
        return [summary.strip()] if summary.strip() else []

    # Find header line offsets; carve the summary into [header .. next header) blocks.
    starts = [m.start() for m in _SECTION_HEADER_RE.finditer(summary)]
    if not starts:
        return [p for p in _token_split(summary, max_tokens, enc) if p.strip()]
    if starts[0] > 0:
        starts = [0, *starts]
    blocks: list[str] = []
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else len(summary)
        blk = summary[s:e].strip()
        if blk:
            blocks.append(blk)

    pieces: list[str] = []
    cur: list[str] = []
    cur_tok = 0
    for blk in blocks:
        bt = _count_tokens(blk, enc)
        if bt > max_tokens:
            # Flush accumulated, then hard-split the oversized block.
            if cur:
                pieces.append("\n\n".join(cur))
                cur, cur_tok = [], 0
            pieces.extend(p for p in _token_split(blk, max_tokens, enc) if p.strip())
            continue
        if cur_tok + bt > max_tokens and cur:
            pieces.append("\n\n".join(cur))
            cur, cur_tok = [], 0
        cur.append(blk)
        cur_tok += bt
    if cur:
        pieces.append("\n\n".join(cur))
    return [p for p in pieces if p.strip()]


def _token_windows(text: str, max_tokens: int, overlap: int, enc) -> list[str]:
    """Overlapping token windows over the source text (the reference's page
    chunker, adapted to already-extracted text)."""
    toks = enc.encode(text)
    if len(toks) <= max_tokens:
        return [text]
    step = max(1, max_tokens - overlap)
    return [enc.decode(toks[i : i + max_tokens]) for i in range(0, len(toks), step)]


# --------------------------------------------------------------------------- #
# Per-chunk evaluator loop.
# --------------------------------------------------------------------------- #
async def _summarize_chunk_with_eval(
    router: LLMRouter,
    source_text: str,
    *,
    num_questions: int,
    eval_rounds: int,
) -> tuple[str, dict[str, Any]]:
    """Summarize one source window with the question/evaluate/revise loop.
    Returns (summary, audit)."""
    sys_p, usr_p = PROMPTS["evaluated_summary_chunk"]()
    # Source goes as cache_prefix (cached leading block) -- reused across this
    # window's summarize/question-gen/revise calls at cache-read price.
    out = await router.chat(
        "evaluated_summary_chunk", system=sys_p, user=usr_p, cache_prefix=source_text
    )
    summary = (out.text or "").strip()

    # Generate the question set once; reuse across rounds so each round checks
    # whether the previous revision actually closed the gaps.
    qsys, qusr = PROMPTS["summary_question_gen"](num_questions)
    try:
        q_out = await router.chat(
            "summary_question_gen", system=qsys, user=qusr, cache_prefix=source_text
        )
        questions = _extract_json(q_out.text) or {"questions": []}
    except Exception as exc:
        return summary, {"passed": None, "rounds": 0, "error": f"question_gen: {exc!r}"}

    q_json = json.dumps(questions, ensure_ascii=False)
    passed = False
    rounds_run = 0
    missing_total = 0
    # One extra evaluation beyond the revision budget so the final revision is
    # itself verified.
    for round_idx in range(1, eval_rounds + 2):
        rounds_run = round_idx
        esys, eusr = PROMPTS["summary_evaluate"](q_json, summary)
        try:
            e_out = await router.chat("summary_evaluate", system=esys, user=eusr)
            evaluation = _extract_json(e_out.text) or {}
        except Exception:
            break
        passed = bool(evaluation.get("passed", False))
        missing = evaluation.get("missing_items") or []
        section_issues = evaluation.get("section_issues") or []
        if passed or (not missing and not section_issues):
            passed = True
            break
        if round_idx > eval_rounds:  # out of revision budget
            missing_total += len(missing)
            break
        missing_total += len(missing)
        rsys, rusr = PROMPTS["summary_revise"](
            summary, json.dumps(evaluation, ensure_ascii=False)
        )
        try:
            r_out = await router.chat(
                "summary_revise", system=rsys, user=rusr, cache_prefix=source_text
            )
            revised = (r_out.text or "").strip()
            if revised:
                summary = revised
        except Exception:
            break

    coverage = check_section_coverage(summary)
    return summary, {
        "passed": passed,
        "rounds": rounds_run,
        "missing_items": missing_total,
        "missing_sections": [k for k, v in coverage.items() if not v],
    }


# --------------------------------------------------------------------------- #
# Cache (per-document JSON of chunk summaries).
# --------------------------------------------------------------------------- #
def _cache_dir() -> Path:
    root = Path.home() / ".cache" / "your-end-to-end-graphrag-implementation" / "eval_summaries"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _cache_key(text: str, model: str, eval_rounds: int, max_chunk_tokens: int,
               summary_chunk_max_tokens: int) -> str:
    h = hashlib.sha256()
    for part in (_EVAL_SUMMARY_VERSION, model, str(eval_rounds),
                 str(max_chunk_tokens), str(summary_chunk_max_tokens), text):
        h.update(part.encode("utf-8"))
        h.update(b"|")
    return h.hexdigest()


def _cache_load(path: Path) -> list[str] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, list) and data else None


def _cache_save(path: Path, chunk_summaries: list[str]) -> None:
    tmp = path.with_suffix(f".tmp.{os.getpid()}")
    try:
        tmp.write_text(json.dumps(chunk_summaries, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
    except OSError:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


# --------------------------------------------------------------------------- #
# Driver.
# --------------------------------------------------------------------------- #
async def evaluated_summarize_documents_async(
    documents: list[LoadedDocument],
    router: LLMRouter,
    *,
    threshold_tokens: int = 2000,
    eval_rounds: int = 3,
    questions_per_chunk: int = 12,
    max_chunk_tokens: int = 12000,
    overlap_tokens: int = 500,
    summary_chunk_max_tokens: int = 1200,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    encoding_name: str = "o200k_base",
    concurrency: int = 4,
    use_cache: bool = True,
    max_cost_usd: float = 1000.0,
) -> list[EvaluatedDocSummary]:
    """Evaluated per-chunk summarization for each document. See module docstring."""
    enc = get_encoder(encoding_name)
    model = router.task_spec("evaluated_summary_chunk").get("model", "")
    cost_before = router.total_cost_usd
    cost_hit = asyncio.Event()
    sem = asyncio.Semaphore(concurrency)
    cache_dir = _cache_dir() if use_cache else None

    async def _one(doc: LoadedDocument) -> EvaluatedDocSummary:
        n_tok = len(enc.encode(doc.text))
        # Small docs: not summarized -- chunk the original text normally.
        if n_tok <= threshold_tokens or threshold_tokens <= 0:
            chunks = [c.text for c in chunk_documents(
                [doc], chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                encoding_name=encoding_name,
            )]
            chunks = [c for c in chunks if c.strip()]
            return EvaluatedDocSummary(
                path=doc.path, chunk_summaries=chunks,
                combined=doc.text, summarized=False,
                audit={"summarized": False, "source_tokens": n_tok},
            )

        # Cache hit?
        if cache_dir is not None:
            key = _cache_key(doc.text, model, eval_rounds, max_chunk_tokens,
                             summary_chunk_max_tokens)
            cpath = cache_dir / f"{key}.json"
            cached = _cache_load(cpath)
            if cached is not None:
                return EvaluatedDocSummary(
                    path=doc.path, chunk_summaries=cached,
                    combined="\n\n".join(cached), summarized=True,
                    audit={"summarized": True, "cache": "hit", "source_tokens": n_tok},
                )

        windows = _token_windows(doc.text, max_chunk_tokens, overlap_tokens, enc)
        window_summaries: list[str] = []
        audits: list[dict[str, Any]] = []
        for w in windows:
            if cost_hit.is_set():
                break
            async with sem:
                if cost_hit.is_set():
                    break
                summ, audit = await _summarize_chunk_with_eval(
                    router, w, num_questions=questions_per_chunk, eval_rounds=eval_rounds,
                )
            if summ.strip():
                window_summaries.append(summ)
                audits.append(audit)
            if router.total_cost_usd - cost_before > max_cost_usd and not cost_hit.is_set():
                cost_hit.set()
                print(f"[evaluated-summary] HALT: cost cap ${max_cost_usd:.2f} reached")

        # Size-cap: split each window summary on section headers into stored chunks.
        stored: list[str] = []
        for ws in window_summaries:
            stored.extend(_split_on_sections(ws, summary_chunk_max_tokens, enc))
        if not stored:  # all windows failed -> fall back to original-text chunks
            stored = [c.text for c in chunk_documents(
                [doc], chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                encoding_name=encoding_name,
            ) if c.text.strip()]
            return EvaluatedDocSummary(
                path=doc.path, chunk_summaries=stored, combined=doc.text,
                summarized=False, audit={"summarized": False, "reason": "all-windows-failed"},
            )

        combined = "\n\n".join(window_summaries)
        if cache_dir is not None:
            _cache_save(cache_dir / f"{key}.json", stored)
        return EvaluatedDocSummary(
            path=doc.path, chunk_summaries=stored, combined=combined,
            summarized=True,
            audit={
                "summarized": True, "source_tokens": n_tok,
                "windows": len(windows), "stored_chunks": len(stored),
                "windows_passed": sum(1 for a in audits if a.get("passed")),
                "per_window": audits,
            },
        )

    t0 = time.time()
    results = await asyncio.gather(*[_one(d) for d in documents])
    n_sum = sum(1 for r in results if r.summarized)
    print(
        f"[evaluated-summary] {len(results)} doc(s): {n_sum} summarized "
        f"(eval_rounds={eval_rounds}), cost=${router.total_cost_usd - cost_before:.4f}, "
        f"wall={time.time() - t0:.1f}s"
    )
    return list(results)
