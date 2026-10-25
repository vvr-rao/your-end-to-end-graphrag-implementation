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

from backend.app.core.config import cache_root
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


def _line_split(text: str, max_tokens: int, enc) -> list[str]:
    """Pack whole lines into pieces <= `max_tokens`; only a single line longer
    than the cap is token-split.

    A plain token split cut every `max_tokens` tokens wherever it landed: on
    the LGES statements one piece ended "Total Liabilities: 13,543" and the
    next began ",750", so each piece misstated the figure, and "LGES Wroclaw"
    was separated from "(EBRD): 47,636". A line is the smallest unit a
    summary states a fact in, so it is never cut.
    """
    pieces: list[str] = []
    cur: list[str] = []
    cur_tok = 0
    for line in text.splitlines(keepends=True):
        lt = _count_tokens(line, enc)
        if lt > max_tokens:
            if cur:
                pieces.append("".join(cur))
                cur, cur_tok = [], 0
            pieces.extend(_token_split(line, max_tokens, enc))
            continue
        if cur_tok + lt > max_tokens and cur:
            pieces.append("".join(cur))
            cur, cur_tok = [], 0
        cur.append(line)
        cur_tok += lt
    if cur:
        pieces.append("".join(cur))
    return [p.strip() for p in pieces if p.strip()]


def _split_on_sections(summary: str, max_tokens: int, enc) -> list[str]:
    """Split one evaluated chunk summary into pieces <= `max_tokens`, preferring
    the section-header boundaries so CLAIMS/EVIDENCE/... blocks stay coherent.
    Lossless: every character ends up in exactly one piece. Falls back to whole
    lines (`_line_split`) when there are no section headers or a single section
    is too big."""
    if _count_tokens(summary, enc) <= max_tokens:
        return [summary.strip()] if summary.strip() else []

    # Find header line offsets; carve the summary into [header .. next header) blocks.
    starts = [m.start() for m in _SECTION_HEADER_RE.finditer(summary)]
    if not starts:
        return _line_split(summary, max_tokens, enc)
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
            pieces.extend(_line_split(blk, max_tokens, enc))
            continue
        if cur_tok + bt > max_tokens and cur:
            pieces.append("\n\n".join(cur))
            cur, cur_tok = [], 0
        cur.append(blk)
        cur_tok += bt
    if cur:
        pieces.append("\n\n".join(cur))
    return [p for p in pieces if p.strip()]


_HEX_STEM_RE = re.compile(r"(?i)^[0-9a-f]{8,}(?:[-_][0-9a-f]+)*$")


def _summary_subject(window_summaries: list[str], doc_path: Path) -> str | None:
    """Who and what the document is about, for the context line on every piece.

    The summary's first line is the evaluated summarizer's title ("LG ENERGY
    SOLUTION, LTD. Interim Condensed Separate Financial Statements (Q1 2025)").
    When it is a section header or missing, fall back to the file name; a
    hash-like name ("24220c53...") says nothing, so then there is no line.
    """
    for ws in window_summaries[:1]:
        for line in ws.splitlines():
            if not line.strip():
                continue
            t = " ".join(line.strip(" \t*#>_-").split())
            if 8 <= len(t) <= 200 and not _SECTION_HEADER_RE.match(line) and re.search(r"[A-Za-z]{3}", t):
                return t
            break
    stem = " ".join(re.sub(r"[_\-]+", " ", doc_path.stem).split())
    if len(stem) >= 4 and not _HEX_STEM_RE.match(doc_path.stem) and re.search(r"[A-Za-z]{3}", stem):
        return stem
    return None


_SUBJECT_OPENING_TOKENS = 3000


async def _document_subject(
    router: LLMRouter, doc: LoadedDocument, enc, sem: asyncio.Semaphore | None = None
) -> dict[str, Any] | None:
    """One `document_subject` call: who/what the document is about, read from
    the first `_SUBJECT_OPENING_TOKENS` of the SOURCE (cover page, report
    header). Summaries from some models have no title line at all (every
    gpt-4.1-mini summary on the bench starts at CLAIMS), and file names say
    "LGES" or nothing ("24220c53-....pdf").

    Optional and soft: no such task in models.yaml, or any failure, returns
    None and the caller falls back to `_summary_subject`.
    """
    try:
        router.task_spec("document_subject")
    except (KeyError, AttributeError):
        return None
    opening = enc.decode(enc.encode(doc.text)[:_SUBJECT_OPENING_TOKENS])
    sys_p, usr_p = PROMPTS["document_subject"](opening, doc.name)
    try:
        async with (sem if sem is not None else _NoLimit()):
            out = await router.chat("document_subject", system=sys_p, user=usr_p)
        parsed = _extract_json(out.text) or {}
    except Exception as exc:
        print(f"[evaluated-summary] {doc.name}: document_subject failed: {exc}"[:200])
        return None
    subject = " ".join(str(parsed.get("subject") or "").split())
    if not 4 <= len(subject) <= 200:
        return None
    info = {"subject": subject}
    for k in ("organization", "document_type", "period"):
        v = parsed.get(k)
        info[k] = " ".join(v.split()) if isinstance(v, str) and v.strip() else None
    return info


def _split_with_context(
    window_summaries: list[str], subject: str | None, max_tokens: int, enc
) -> list[str]:
    """Split each window summary into stored pieces and start every piece with
    the document's context line.

    Only a window's first piece carried the title, so the LGES revenue piece
    named no company: entity extraction could not link it to LG Energy
    Solution, and its embedding said nothing about whose figures they were.
    The line is budgeted inside `max_tokens`.
    """
    if not subject:
        return [p for ws in window_summaries for p in _split_on_sections(ws, max_tokens, enc)]
    prefix = f"[Document: {subject}]\n"
    budget = max(1, max_tokens - _count_tokens(prefix, enc))
    return [prefix + p for ws in window_summaries
            for p in _split_on_sections(ws, budget, enc)]


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
class _NoLimit:
    """Reusable no-op async context manager, for callers that self-throttle.

    Must be REUSABLE: the chain below enters its limiter once per LLM call, and
    an @asynccontextmanager generator can only be entered once -- the second
    `async with` raises, which the chain's except-blocks would swallow, silently
    truncating the summary. asyncio.Semaphore is reusable, so this must be too.
    """

    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, *exc: object) -> None:
        return None


async def _summarize_chunk_with_eval(
    router: LLMRouter,
    source_text: str,
    *,
    num_questions: int,
    eval_rounds: int,
    sem: asyncio.Semaphore | None = None,
) -> tuple[str, dict[str, Any]]:
    """Summarize one source window with the question/evaluate/revise loop.
    Returns (summary, audit).

    `sem` throttles EACH LLM call rather than the whole chain. The chain is 9
    strictly sequential calls (summarize -> question-gen -> evaluate/revise
    rounds), and almost all of that time is spent waiting on the network. When
    the caller held one slot for the entire chain, a concurrency of N allowed
    only N windows in flight -- so on a 139-window corpus, 4 slots meant 35
    sequential rounds and a ~4-hour run at under 1% of the account's TPM.
    Releasing between calls lets the same N slots cover far more windows.
    """
    limit: Any = sem if sem is not None else _NoLimit()

    sys_p, usr_p = PROMPTS["evaluated_summary_chunk"]()
    # Source goes as cache_prefix (cached leading block) -- reused across this
    # window's summarize/question-gen/revise calls at cache-read price.
    async with limit:
        out = await router.chat(
            "evaluated_summary_chunk", system=sys_p, user=usr_p, cache_prefix=source_text
        )
    summary = (out.text or "").strip()

    # Generate the question set once; reuse across rounds so each round checks
    # whether the previous revision actually closed the gaps.
    qsys, qusr = PROMPTS["summary_question_gen"](num_questions)
    try:
        async with limit:
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
            async with limit:
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
            async with limit:
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
    root = cache_root() / "eval_summaries"
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


def _rejoin_legacy_pieces(pieces: list[str], max_tokens: int, enc) -> str:
    """Undo the old split. A piece the token split cut is exactly `max_tokens`
    long (give or take a decode round-trip), so it joins to the next one with
    no separator; every other boundary was a block boundary ("\n\n")."""
    out = pieces[0] if pieces else ""
    for prev, cur in zip(pieces, pieces[1:]):
        cut = _count_tokens(prev, enc) >= max_tokens - 2 and not _SECTION_HEADER_RE.match(cur)
        out += ("" if cut else "\n\n") + cur
    return out


def _cache_load(
    path: Path, max_tokens: int, enc
) -> tuple[list[str], dict[str, Any] | None] | None:
    """(window summaries, document subject or None). The summaries are stored
    unsplit, so the split can change without re-summarizing. Old entries hold
    the split pieces; they are rejoined into one summary."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    if isinstance(data, dict) and data.get("format") == 2:
        windows = data.get("windows")
        subject = data.get("subject") if isinstance(data.get("subject"), dict) else None
        return (windows, subject) if isinstance(windows, list) and windows else None
    if isinstance(data, list) and data:
        return [_rejoin_legacy_pieces(data, max_tokens, enc)], None
    return None


def _cache_save(
    path: Path, window_summaries: list[str], subject: dict[str, Any] | None = None
) -> None:
    tmp = path.with_suffix(f".tmp.{os.getpid()}")
    try:
        tmp.write_text(json.dumps({"format": 2, "windows": window_summaries,
                                   "subject": subject},
                                  ensure_ascii=False), encoding="utf-8")
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
    cache_read_before = router.cache_read_tokens
    input_before = router.total_input_tokens
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
            hit = _cache_load(cpath, summary_chunk_max_tokens, enc)
            if hit is not None:
                cached, info = hit
                if info is None:
                    # Summarized before the subject call existed: pay for it once.
                    info = await _document_subject(router, doc, enc, sem)
                    if info is not None:
                        _cache_save(cpath, cached, info)
                subject = info["subject"] if info else _summary_subject(cached, doc.path)
                return EvaluatedDocSummary(
                    path=doc.path,
                    chunk_summaries=_split_with_context(
                        cached, subject, summary_chunk_max_tokens, enc),
                    combined="\n\n".join(cached), summarized=True,
                    audit={"summarized": True, "cache": "hit", "source_tokens": n_tok,
                           "subject": info or {"subject": subject, "source": "fallback"}},
                )

        windows = _token_windows(doc.text, max_chunk_tokens, overlap_tokens, enc)
        n_win = len(windows)
        print(f"[evaluated-summary] {doc.name}: {n_tok:,} tok -> {n_win} window(s), summarizing...")
        # Windows are independent: each is summarized standalone and the results are
        # concatenated in source order below. Run them concurrently (throttled by the
        # shared `sem`) rather than sequentially, so a single large doc can occupy
        # several concurrency slots instead of one -- the difference between a 40-window
        # doc taking hours (serial) vs minutes. Nothing here defends against the process
        # being SUSPENDED mid-call (asyncio timers use a monotonic clock that freezes
        # with the process); speed + visible progress just make that state obvious and
        # short-lived rather than a silent multi-hour stall.
        summ_by_idx: dict[int, str] = {}
        audit_by_idx: dict[int, dict[str, Any]] = {}
        failed_windows = 0
        done = 0

        async def _run_window(i: int, w: str) -> None:
            nonlocal failed_windows, done
            if cost_hit.is_set():
                return
            err: Exception | None = None
            summ, audit = "", {}
            # The semaphore is passed DOWN so it throttles each LLM call rather
            # than being held across the whole 9-call chain. Holding it for the
            # chain capped in-flight windows at `concurrency`, most of that time
            # idle on network wait; releasing between calls lets the same limit
            # cover many more windows.
            try:
                summ, audit = await _summarize_chunk_with_eval(
                    router, w, num_questions=questions_per_chunk,
                    eval_rounds=eval_rounds, sem=sem,
                )
            except Exception as exc:
                # A window whose every retry failed must not take down the document
                # or the surrounding gather. It is counted instead: the doc is left
                # incomplete below and therefore NOT cached, so a re-run retries it
                # rather than serving a summary that silently dropped this window's
                # source content.
                err = exc
                failed_windows += 1
            if err is None and summ.strip():
                summ_by_idx[i] = summ
                audit_by_idx[i] = audit
            # `done` counts every window that reaches an outcome (ok or failed), so
            # progress is a single monotonic X/N regardless of completion order.
            done += 1
            spent = router.total_cost_usd - cost_before
            status = (f"FAILED (src window #{i + 1}): {type(err).__name__}: {err}"[:120]
                      if err is not None else "done")
            print(f"[evaluated-summary] {doc.name}: window {done}/{n_win} {status} "
                  f"(${spent:.2f} so far)")
            if spent > max_cost_usd and not cost_hit.is_set():
                cost_hit.set()
                print(f"[evaluated-summary] HALT: cost cap ${max_cost_usd:.2f} reached")

        await asyncio.gather(*[_run_window(i, w) for i, w in enumerate(windows)])
        # Reassemble in source order (gather completes out of order).
        window_summaries: list[str] = [summ_by_idx[i] for i in sorted(summ_by_idx)]
        audits: list[dict[str, Any]] = [audit_by_idx[i] for i in sorted(audit_by_idx)]

        # Size-cap: split each window summary on section headers into stored
        # chunks, each starting with the document's context line.
        info = await _document_subject(router, doc, enc, sem) if window_summaries else None
        subject = info["subject"] if info else _summary_subject(window_summaries, doc.path)
        stored = _split_with_context(
            window_summaries, subject, summary_chunk_max_tokens, enc)
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
        # Persist only a COMPLETE doc. A partial summary -- a window that failed every
        # retry, or a cost-cap halt that broke the loop early -- is indistinguishable
        # from a good one once written, so caching it would make the loss permanent:
        # every later run takes the cache hit and never re-summarizes the doc.
        complete = failed_windows == 0 and len(window_summaries) == len(windows)
        if cache_dir is not None and complete:
            _cache_save(cache_dir / f"{key}.json", window_summaries, info)
        return EvaluatedDocSummary(
            path=doc.path, chunk_summaries=stored, combined=combined,
            summarized=True,
            audit={
                "summarized": True, "source_tokens": n_tok,
                "windows": len(windows), "stored_chunks": len(stored),
                "subject": info or {"subject": subject, "source": "fallback"},
                "windows_passed": sum(1 for a in audits if a.get("passed")),
                "failed_windows": failed_windows,
                "complete": complete, "cached": complete,
                "per_window": audits,
            },
        )

    t0 = time.time()
    results = await asyncio.gather(*[_one(d) for d in documents])
    n_sum = sum(1 for r in results if r.summarized)
    cache_read = router.cache_read_tokens - cache_read_before
    input_delta = router.total_input_tokens - input_before
    hit_pct = (100.0 * cache_read / input_delta) if input_delta else 0.0
    print(
        f"[evaluated-summary] {len(results)} doc(s): {n_sum} summarized "
        f"(eval_rounds={eval_rounds}), cost=${router.total_cost_usd - cost_before:.4f}, "
        f"prompt-cache-hit={hit_pct:.0f}% ({cache_read:,}/{input_delta:,} input tok), "
        f"wall={time.time() - t0:.1f}s"
    )
    return list(results)
