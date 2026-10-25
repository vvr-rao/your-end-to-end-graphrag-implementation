#!/usr/bin/env python3
"""Report your provider rate limits and recommend a concurrency setting.

    uv run python scripts/tpm_check.py [<documents-dir>] [--selection F]
    uv run python scripts/tpm_check.py --apply            # write the suggestions
    uv run python scripts/tpm_check.py --set entity_extraction=32 --set dedup=8

Pass a corpus directory to also size `streaming_batch_size`, which silently
CAPS effective concurrency: only the current batch's windows exist, so a
semaphore larger than that has nothing to schedule.

Reads each provider's authoritative rate-limit headers (one ~10-token probe per
model the active models.yaml routes to -- OpenAI, Anthropic and Groq) and turns
them into a per-STAGE suggestion for `concurrency:` in config/config.yaml. A
stage's suggestion is the minimum over every task it runs, so a stage that
mixes a cheap model with a tight one is sized by the tight one, and the report
names the task that binds.

Nothing is written unless --apply (write every suggestion) or --set
stage=N (write your own value) is passed; the skills ask the user first.

WHY THIS EXISTS: the shipped default of 4 was tuned for a low tier. On a
tier-3+ account it leaves most of the allowance unused -- a 1.6M-token
corpus spent ~4 hours summarizing at under 1% of a 10M TPM limit. Raising
concurrency cuts wall time close to linearly. It does NOT reduce cost: the
same tokens are sent either way.

Reads only whether keys are present, never prints key values.
"""
from __future__ import annotations

import asyncio
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app.core.config import get_settings  # noqa: E402
from backend.app.services.llm_router import (  # noqa: E402
    _openai_uses_completion_tokens,
)

# Stay well under the ceiling: bursts, retries and other traffic share the bucket.
TARGET_UTILISATION = 0.25


def _num(v) -> int | None:
    if v is None:
        return None
    s = str(v).strip()
    try:
        return int(s)
    except ValueError:
        pass
    mult = {"k": 1_000, "m": 1_000_000}
    if s and s[-1].lower() in mult:
        try:
            return int(float(s[:-1]) * mult[s[-1].lower()])
        except ValueError:
            return None
    return None


# Every concurrency knob that bounds LLM calls, and the models.yaml tasks that
# run under it. A stage is sized by the TIGHTEST of its tasks: one semaphore
# covers all of them, so a mini-model stage that also makes gpt-4.1 calls
# (extract-entities' orphan check; generate-artifacts' Insight / rollup) can
# only go as wide as gpt-4.1 allows.
#
# test_tpm_check.py asserts every task listed here exists in the shipped
# models presets, and that every task the stage's own modules call is listed.
STAGE_TASKS: dict[str, tuple[str, ...]] = {
    "summarization": (
        "evaluated_summary_chunk", "summary_question_gen", "summary_evaluate",
        "summary_revise", "document_subject",
    ),
    "chunk_classification": ("chunk_classification",),
    "class_proposal": ("class_proposal",),
    "dedup": ("match_dedup", "class_synonym_judge"),
    "table_mining": ("table_concept_grouping",),
    "entity_extraction": (
        "entity_extract", "concept_extract", "entity_validate",
        "relationship_extract", "relationship_verify",
        "relationship_orphan_check", "relationship_repair",
    ),
    "artifact_generation": (
        "artifact_chunk_extract", "artifact_chunk_extract_with_entities",
        "artifact_document_summary", "insight_gen",
        "artifact_merge", "artifact_merge_evaluate", "artifact_merge_revise",
        "summary_merge", "summary_merge_evaluate", "summary_merge_revise",
    ),
    "evaluation": (
        "judge_comprehensiveness", "judge_no_hallucination",
        "judge_gap_detection", "judge_consistency",
        "question_parse", "query_decompose", "entity_probes",
        "answer_simple_qa", "answer_deep_research",
    ),
}

# Stages that touch Postgres. They are sized by TPM/RPM like the rest; the DB
# only bounds the POOL (see db_pool_report), except for `_POOL_BOUND_STAGES`.
PHASE2_STAGES = ("entity_extraction", "artifact_generation", "evaluation")

# The rough input side of one call (a source window plus the prompt).
INPUT_TOKENS_PER_CALL = 12_000
MAX_SUGGESTION = 128
# Tasks with a 32k output budget (class_proposal, match_dedup on gpt-4.1)
# throttle when many run at once even with TPM to spare -- models.yaml warns of
# it, and the measured-safe range is 12-16. TPM alone would suggest 128 on a
# 30M-TPM account. Raise past this only incrementally, watching for 429s.
LARGE_CALL_OUTPUT_TOKENS = 32_768
LARGE_CALL_CAP = 16


def _task_specs() -> dict[str, dict]:
    return (get_settings().models_config.get("tasks") or {})


def _models_in_use() -> list[tuple[str, str]]:
    """Every distinct (provider, chat model) a stage in STAGE_TASKS routes to."""
    specs = _task_specs()
    seen: list[tuple[str, str]] = []
    for tasks in STAGE_TASKS.values():
        for t in tasks:
            spec = specs.get(t) or {}
            key = (spec.get("provider"), spec.get("model"))
            if key[0] and key[1] and key not in seen:
                seen.append(key)
    return seen


async def _probe_openai_compatible(client, provider: str, model: str) -> dict:
    # gpt-5.x / o-series reject `max_tokens` and non-default temperature.
    #
    # They are also REASONING models: a budget of 1 is spent entirely on
    # reasoning tokens and the call 400s with "Could not finish the message
    # because max_tokens or model output limit was reached", so the probe used
    # to report gpt-5.x as an error and silently omit its TPM/RPM from the gate
    # -- which matters, because summary_merge (the Summary rollup that runs in
    # generate-artifacts even without --rollup) is one of them. 16 is enough to
    # get a reply and still costs ~nothing. Measured 2026-09-27: gpt-5.4 then
    # reports 40M TPM / 15k RPM.
    kw = ({"max_completion_tokens": 16}
          if provider == "openai" and _openai_uses_completion_tokens(model)
          else {"max_tokens": 1})
    raw = await client.chat.completions.with_raw_response.create(
        model=model, messages=[{"role": "user", "content": "hi"}], **kw,
    )
    h = raw.headers
    return {
        "tpm": _num(h.get("x-ratelimit-limit-tokens")),
        # Groq's x-ratelimit-limit-requests is per DAY, not per minute --
        # dividing it as RPM would overstate headroom 1440x. Leave RPM unknown.
        "rpm": None if provider == "groq" else _num(h.get("x-ratelimit-limit-requests")),
    }


async def _probe_anthropic(client, model: str) -> dict:
    raw = await client.messages.with_raw_response.create(
        model=model, max_tokens=1, messages=[{"role": "user", "content": "hi"}],
    )
    h = raw.headers
    # Anthropic meters input and output tokens separately (ITPM / OTPM).
    return {
        "itpm": _num(h.get("anthropic-ratelimit-input-tokens-limit")),
        "otpm": _num(h.get("anthropic-ratelimit-output-tokens-limit")),
        "tpm": _num(h.get("anthropic-ratelimit-tokens-limit")),
        "rpm": _num(h.get("anthropic-ratelimit-requests-limit")),
    }


async def probe(provider: str, model: str) -> dict:
    """One tiny call; returns the model's limits, or an `error`."""
    s = get_settings()
    row: dict = {"provider": provider, "model": model}
    try:
        if provider == "openai":
            if not s.openai_api_key:
                return {**row, "error": "OPENAI_API_KEY not set"}
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=s.openai_api_key, max_retries=0)
            row.update(await _probe_openai_compatible(client, provider, model))
        elif provider == "groq":
            if not s.groq_api_key:
                return {**row, "error": "GROQ_API_KEY not set"}
            from groq import AsyncGroq
            client = AsyncGroq(api_key=s.groq_api_key, max_retries=0)
            row.update(await _probe_openai_compatible(client, provider, model))
        elif provider == "anthropic":
            if not s.anthropic_api_key:
                return {**row, "error": "ANTHROPIC_API_KEY not set"}
            from anthropic import AsyncAnthropic
            client = AsyncAnthropic(api_key=s.anthropic_api_key, max_retries=0)
            row.update(await _probe_anthropic(client, model))
        elif provider == "deepinfra":
            # Dedicated deployments have no rate limit (GPUs are rented), and
            # serverless sends no limit headers. Sized by deepinfra_sizing.py.
            return {**row, "error": "no limit headers: use deepinfra_sizing.py"}
        else:
            return {**row, "error": f"unknown provider {provider!r}"}
    except Exception as exc:
        return {**row, "error": f"{type(exc).__name__}: {str(exc)[:70]}"}
    return row


def suggest(limits: dict, output_tokens: int) -> int | None:
    """Workers one model's limits support for calls of this shape.

    Assumes each worker issues about one call a minute (heavy calls take tens
    of seconds) and aims at 25% of the limit, leaving room for bursts, retries
    and other traffic on the same key. Returns None when no token limit is
    known -- an unknown limit must not look like an unlimited one.
    """
    per_call = INPUT_TOKENS_PER_CALL + output_tokens
    bounds: list[int] = []
    if limits.get("itpm"):
        bounds.append(int(limits["itpm"] * TARGET_UTILISATION / INPUT_TOKENS_PER_CALL))
    if limits.get("otpm"):
        bounds.append(int(limits["otpm"] * TARGET_UTILISATION / max(1, output_tokens)))
    if limits.get("tpm") and not (limits.get("itpm") or limits.get("otpm")):
        bounds.append(int(limits["tpm"] * TARGET_UTILISATION / per_call))
    if not bounds:
        return None
    if limits.get("rpm"):
        bounds.append(int(limits["rpm"] * TARGET_UTILISATION))
    if output_tokens >= LARGE_CALL_OUTPUT_TOKENS:
        bounds.append(LARGE_CALL_CAP)
    return max(1, min(min(bounds), MAX_SUGGESTION))


def stage_suggestions(limits_by_model: dict[tuple[str, str], dict]) -> dict[str, dict]:
    """Per stage: the suggestion and the task/model that binds it."""
    specs = _task_specs()
    out: dict[str, dict] = {}
    for stage, tasks in STAGE_TASKS.items():
        best: dict | None = None
        unknown: list[str] = []
        for t in tasks:
            spec = specs.get(t)
            if not spec:
                continue  # task not routed in this preset
            key = (spec.get("provider"), spec.get("model"))
            lim = limits_by_model.get(key) or {}
            n = None if lim.get("error") else suggest(lim, int(spec.get("max_tokens") or 8192))
            if n is None:
                unknown.append(f"{t} ({key[0]}/{key[1]})")
                continue
            if best is None or n < best["value"]:
                best = {"value": n, "task": t, "provider": key[0], "model": key[1]}
        out[stage] = {**(best or {"value": None}), "unknown": unknown}
    return out


def apply_concurrency(path: Path, values: dict[str, int]) -> list[str]:
    """Set `concurrency.<stage>: N` in a YAML file, keeping every comment.

    A line edit, not a YAML round-trip: config.yaml is mostly comments that
    record measurements, and a dump would erase them. Missing keys are added at
    the end of the block. Returns a line per change.
    """
    lines = path.read_text().splitlines(keepends=True)
    try:
        start = next(i for i, ln in enumerate(lines) if ln.rstrip() == "concurrency:")
    except StopIteration:
        lines.append("\nconcurrency:\n")
        start = len(lines) - 1
    end = start + 1
    while end < len(lines) and (
        not lines[end].strip() or lines[end].startswith((" ", "\t"))
    ):
        end += 1
    changes: list[str] = []
    pending = dict(values)
    for i in range(start + 1, end):
        m = re.match(r"^(\s+)([a-z_]+):\s*(\S+)(.*)$", lines[i].rstrip("\n"))
        if m and m.group(2) in pending:
            new = pending.pop(m.group(2))
            if str(new) != m.group(3):
                changes.append(f"concurrency.{m.group(2)}: {m.group(3)} -> {new}")
            lines[i] = f"{m.group(1)}{m.group(2)}: {new}{m.group(4)}\n"
    insert_at = end
    while insert_at > start + 1 and not lines[insert_at - 1].strip():
        insert_at -= 1
    for k, v in pending.items():
        lines.insert(insert_at, f"  {k}: {v}\n")
        insert_at += 1
        changes.append(f"concurrency.{k}: (unset) -> {v}")
    path.write_text("".join(lines))
    return changes


def _mem_linux() -> tuple[int, int]:
    vals = {}
    for line in open("/proc/meminfo"):
        k, _, rest = line.partition(":")
        if k in ("MemAvailable", "MemTotal"):
            vals[k] = int(rest.split()[0]) // 1024
    return vals.get("MemAvailable", -1), vals.get("MemTotal", -1)


def parse_vm_stat(text: str, page_size: int) -> int:
    """MB genuinely reclaimable, from macOS `vm_stat`.

    free + inactive + speculative: inactive pages are evictable, so counting
    only 'free' understates available memory several-fold on macOS.
    """
    total_pages = 0
    for key in ("Pages free", "Pages inactive", "Pages speculative"):
        for line in text.splitlines():
            if line.startswith(key + ":"):
                total_pages += int(line.split(":")[1].strip().rstrip("."))
                break
    return (total_pages * page_size) // (1024 * 1024)


def _mem_macos() -> tuple[int, int]:
    import subprocess

    total = int(subprocess.run(["sysctl", "-n", "hw.memsize"],
                               capture_output=True, text=True).stdout.strip())
    vm = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    page = 4096
    if "page size of" in vm:
        page = int(vm.split("page size of")[1].split("bytes")[0].strip())
    return parse_vm_stat(vm, page), total // (1024 * 1024)


def _mem_windows() -> tuple[int, int]:
    import ctypes

    class _MemStatus(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    st = _MemStatus()
    st.dwLength = ctypes.sizeof(_MemStatus)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))  # type: ignore[attr-defined]
    mb = 1024 * 1024
    return st.ullAvailPhys // mb, st.ullTotalPhys // mb


def _win_mem_raw() -> tuple[int, int]:
    """(total physical, total page file) in MB. Windows only."""
    import ctypes

    class _M(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    st = _M()
    st.dwLength = ctypes.sizeof(_M)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))  # type: ignore[attr-defined]
    mb = 1024 * 1024
    return st.ullTotalPhys // mb, st.ullTotalPageFile // mb


def _mem_mb() -> tuple[int, int]:
    """(available, total) MB on Linux, macOS or Windows. (-1,-1) if unknown.

    Prefers psutil when the user happens to have it; otherwise uses each
    platform's own interface rather than adding a dependency for one function.
    """
    try:
        import psutil  # optional, not a project dependency

        vm = psutil.virtual_memory()
        return vm.available // (1024 * 1024), vm.total // (1024 * 1024)
    except Exception:
        pass
    try:
        if sys.platform.startswith("linux"):
            return _mem_linux()
        if sys.platform == "darwin":
            return _mem_macos()
        if sys.platform.startswith("win"):
            return _mem_windows()
    except Exception:
        pass
    return -1, -1


def _has_swap() -> bool:
    """Whether the OS can page out. False means an over-commit is a HARD KILL."""
    try:
        import psutil

        return psutil.swap_memory().total > 0
    except Exception:
        pass
    try:
        if sys.platform.startswith("linux"):
            with open("/proc/swaps") as fh:
                return len(fh.read().strip().splitlines()) > 1
        if sys.platform == "darwin":
            import subprocess

            out = subprocess.run(["sysctl", "-n", "vm.swapusage"],
                                 capture_output=True, text=True).stdout
            return "total = 0.00M" not in out and bool(out.strip())
        if sys.platform.startswith("win"):
            # ullTotalPageFile is physical RAM PLUS the page file, so swap
            # exists only when it exceeds physical. Windows usually has a page
            # file, but it can be disabled on perf-tuned machines -- and that
            # is precisely when the "no swap = hard kill" warning matters, so
            # this must actually be measured rather than assumed.
            total_phys, total_page = _win_mem_raw()
            return total_page > total_phys * 1.05
    except Exception:
        pass
    return False


# Measured on this project: one table_extract_worker subprocess (125-page
# pharma label, vision ON) peaks around 152 MB RSS. 200 MB is that plus margin,
# since a PDF with large page rasters costs several times the average.
_TABLE_WORKER_MB = 200
# Never plan to consume the last of RAM: the pipeline itself, Postgres client
# buffers and the OS page cache all need room, and on a swapless box an OOM is
# a hard kill rather than a slowdown.
_RESERVE_MB = 600
# Whole-page vision calls in flight inside ONE worker
# (concurrency.table_page_vision). A waiting call holds its base64 page PNG
# (~0.05 MB at 150 dpi) plus request/response buffers; pages are rendered one
# at a time per worker, so the ~51 MB render spike does not multiply. 5 MB per
# call is that with a wide margin. Capped at 8: serverless Qwen3-VL kept ~33
# tok/s per stream at 8 in flight (2026-10-24), not measured beyond.
_PAGE_VISION_MB = 5
_PAGE_VISION_MAX = 8
_TABLE_WORKERS_MAX = 8


def table_concurrency_for_budget(budget_mb: int) -> tuple[int, int]:
    """(concurrency.table_extraction, concurrency.table_page_vision) for a
    memory budget. Each worker costs _TABLE_WORKER_MB plus _PAGE_VISION_MB per
    page call it keeps in flight; the most pages per worker that still leave
    room for one worker wins, then as many such workers as fit."""
    for pages in (_PAGE_VISION_MAX, 4, 2, 1):
        workers = budget_mb // (_TABLE_WORKER_MB + pages * _PAGE_VISION_MB)
        if workers >= 1:
            return min(_TABLE_WORKERS_MAX, workers), pages
    return 1, 1


# --------------------------------------------------------------------------- #
# Build memory: the PIPELINE's own peak on THIS machine (the one the skills
# and the detached run use), projected from the corpus. Concurrency is a small
# share of it; the peak is Stage 4 + the OWL export holding the whole expanded
# ontology at once. Any mode -- the stages are the same.
# Measured 2026-10-19: Qwen v5 on the 415.7k-token web-search corpus proposed
# 11.9 distinct classes, 32.2 instances and 13.5 relations per 1k tokens; the
# 30k-class / 90k-instance / 20k-relation export test peaked at 706 MB over a
# ~112 MB base, 4.3 KB per item. Linear in tokens on purpose: vocabularies
# saturate (fitted exponent 0.75-0.91), so a large corpus lands BELOW this.
# --------------------------------------------------------------------------- #
BUILD_ITEMS_PER_1K_TOKENS = {"classes": 11.9, "instances": 32.2, "relations": 13.5}
BUILD_PEAK_KB_PER_ITEM = 4.4
BUILD_BASE_MB = 120


def build_memory_check(corpus_tokens: int, mem: tuple[int, int] | None = None) -> dict:
    """Projected peak MB of a build of `corpus_tokens` vs this machine's RAM.
    verdict: fits (<= 80% of usable), tight (<= usable), does_not_fit, or
    unknown (memory unreadable). `mem` = (available, total) MB, for tests."""
    k = corpus_tokens / 1000
    items = {name: int(rate * k) for name, rate in BUILD_ITEMS_PER_1K_TOKENS.items()}
    peak = int(BUILD_BASE_MB + sum(items.values()) * BUILD_PEAK_KB_PER_ITEM / 1024)
    avail, total = mem if mem is not None else _mem_mb()
    out = {"corpus_tokens": int(corpus_tokens), **items, "peak_mb": peak,
           "available_mb": int(avail), "total_mb": int(total),
           # RAM a machine needs for this to "fit": peak within 80% of usable.
           "needs_total_mb": int(peak / 0.8 + _RESERVE_MB)}
    if avail < 0:
        out["verdict"] = "unknown"
        return out
    budget = max(0, avail - _RESERVE_MB)
    out["budget_mb"] = budget
    out["verdict"] = "fits" if peak <= 0.8 * budget else ("tight" if peak <= budget else "does_not_fit")
    return out


def print_build_memory(check: dict) -> None:
    v = check["verdict"]
    print(f"\n  Build memory (this machine): projected peak ~{check['peak_mb']:,} MB for "
          f"{check['corpus_tokens']:,} tokens (~{check['classes']:,} classes, {check['instances']:,} "
          f"instances, {check['relations']:,} relations -- an upper bound)")
    if v == "unknown":
        print("    Memory could not be read here; check it with the OS command before a large run.")
        return
    print(f"    available {check['available_mb']:,} MB, usable {check['budget_mb']:,} MB -> "
          + {"fits": "FITS", "tight": "TIGHT -- close other apps before the run",
             "does_not_fit": "DOES NOT FIT"}[v])
    if v != "fits":
        print(f"    For headroom use a machine with ~{check['needs_total_mb'] / 1024:.1f} GB free, or build a "
              f"subset first (--select-subset / --limit).")


def advise_memory(batch_size: int, table_conc: int, page_conc: int = 1) -> dict[str, int]:
    """Recommend memory-bound settings from what this machine actually has.
    Returns the table knobs it suggests ({} when memory cannot be read), which
    --apply writes alongside the rate-limit suggestions."""
    avail, total = _mem_mb()
    if avail < 0:
        print("\n\nMemory: could not be read on this platform "
              f"({sys.platform}) -- skipping memory-based advice. "
              "Linux, macOS and Windows are supported; check the OS command "
              "manually (see the run-prune-expand skill).")
        return {}
    swap = _has_swap()
    print(f"\n\nMachine memory: {avail:,} MB available of {total:,} MB total"
          f"  |  swap: {'yes' if swap else 'NONE'}")
    if not swap:
        print("  No swap: exceeding RAM is a HARD KILL mid-run, not a slowdown.")

    budget = max(0, avail - _RESERVE_MB)
    rec_table, rec_pages = table_concurrency_for_budget(budget)
    print(f"\n  usable budget: {budget:,} MB "
          f"(available minus a {_RESERVE_MB} MB reserve)")

    print(f"\n  concurrency.table_extraction: {table_conc}  ->  suggest {rec_table}")
    print(f"    Each PDF extracts in its own subprocess (~{_TABLE_WORKER_MB} MB "
          f"budgeted, 152 MB measured).")
    print(f"    Serial extraction is often the long pole: 18 PDFs at 1 = ~100 min.")
    if rec_table > table_conc:
        print(f"    -> raising to {rec_table} could cut that to ~{100 // rec_table} min.")
    elif rec_table < table_conc:
        print(f"    -> LOWER it: not enough free memory for {table_conc} workers.")

    print(f"\n  concurrency.table_page_vision: {page_conc}  ->  suggest {rec_pages}")
    print(f"    Whole-page vision calls in flight per PDF (~{_PAGE_VISION_MB} MB each, "
          f"counted in the worker budget above).")
    print("    Each page is a decode-bound call (serverless Qwen3-VL: 40-150 s); 8 in")
    print("    flight finished 8 pages in 150 s vs ~8x that one at a time. Output is identical.")

    # streaming_batch_size holds N documents' extracted text, not their files.
    # Even large labels are a few hundred KB of text, so this is cheap; the
    # config's own note puts 16 docs at ~50 MB.
    rec_batch = 32 if budget >= 1200 else (16 if budget >= 600 else 8)
    print(f"\n  chunking.streaming_batch_size: {batch_size}  ->  suggest {rec_batch}")
    print(f"    Holds N documents' extracted TEXT (~50 MB at 16 docs), so it is")
    print(f"    cheap -- but it CAPS effective concurrency (see the corpus table).")
    if avail < 1200:
        print(f"\n  NOTE: under ~1.2 GB free. Close other applications (an IDE can")
        print(f"    hold 1.5+ GB) before a long paid run.")
    return {"table_extraction": rec_table, "table_page_vision": rec_pages}


def _selected_paths(selection_path: str) -> set[str] | None:
    """Resolved paths of the documents a selection.json marked selected.

    Returns None (meaning "no filter") if the file is missing or malformed --
    sizing the whole corpus is a harmless over-estimate, whereas refusing to
    run would block the mandatory pre-flight gate.
    """
    import json
    from pathlib import Path

    try:
        report = json.loads(Path(selection_path).read_text(encoding="utf-8"))
        chosen = {
            str(Path(d["path"]).resolve())
            for d in report.get("documents", [])
            if d.get("selected")
        }
    except (OSError, ValueError, KeyError) as exc:
        print(f"  (could not read {selection_path}: {exc}; sizing the FULL corpus)")
        return None
    if not chosen:
        print(f"  ({selection_path} selected nothing; sizing the FULL corpus)")
        return None
    return chosen


def analyse_corpus(
    docs_dir: str, batch_size: int, concurrency: int, selection_path: str | None = None
) -> None:
    """Explain how batch size and concurrency interact for THIS corpus.

    streaming_batch_size is a stop-the-world barrier: pipeline_llm awaits each
    batch before loading the next, so windows from later documents do not exist
    yet. Effective concurrency is therefore
        min(concurrency, windows_in_the_current_batch)
    and a semaphore bigger than the batch's window count is simply idle.
    """
    from pathlib import Path

    from backend.app.services.document_io import load_documents
    from backend.app.services.evaluated_summarizer import get_encoder

    print(f"\n\nCorpus plan for {docs_dir}")
    print("  (extracting + tokenising -- may take a minute on a large corpus)")
    enc = get_encoder()
    docs = list(load_documents(Path(docs_dir)))
    if selection_path:
        chosen = _selected_paths(selection_path)
        if chosen is not None:
            before = len(docs)
            docs = [d for d in docs if str(d.path.resolve()) in chosen]
            print(f"  sizing the SELECTED subset only: {len(docs)} of {before} document(s)")
    if not docs:
        print("  no documents found.")
        return
    toks = [len(enc.encode(d.text)) for d in docs]
    print_build_memory(build_memory_check(sum(toks)))
    wins = [max(1, -(-t // 11500)) for t in toks]   # 12k window - 500 overlap
    n, total_w = len(docs), sum(wins)

    print(f"\n  {n} document(s), {sum(toks):,} tokens -> ~{total_w} summarization windows")
    print(f"  largest doc: {max(toks):,} tok ({max(wins)} windows)")

    print(f"\n{'batch_size':>11} {'batches':>8} {'windows/batch':>14} {'effective conc.':>16}  note")
    print("  " + "-" * 68)
    best = batch_size
    for b in sorted({4, 8, 16, 32, n}):
        if b < 1:
            continue
        nb = -(-n // b)
        per = total_w / nb
        eff = min(concurrency, per)
        note = "saturates" if per >= concurrency else f"CAPS concurrency at ~{per:.0f}"
        star = " <- current" if b == batch_size else ""
        print(f"{b:>11} {nb:>8} {per:>14.0f} {eff:>16.0f}  {note}{star}")
        if per >= concurrency and b <= n and (best == batch_size or b < best):
            best = b

    print(f"\n  Current: streaming_batch_size={batch_size}, concurrency.summarization={concurrency}")
    per_now = total_w / max(1, -(-n // batch_size))
    if per_now < concurrency:
        print(f"  -> Only ~{per_now:.0f} windows exist per batch, so {concurrency - per_now:.0f} "
              f"of your {concurrency} concurrency slots sit IDLE.")
        print(f"  -> RECOMMEND chunking.streaming_batch_size: {best} "
              f"(memory cost is small: ~50 MB at 16 docs).")
        print(f"     Raising concurrency alone will NOT help until you do.")
    else:
        print(f"  -> Batch size is not the constraint here; concurrency "
              f"{concurrency} is fully usable.")
    print("\n  Batches are SEQUENTIAL (a barrier), so each batch also runs at the")
    print("  speed of its slowest document -- one 6-window doc holds up its whole batch.")


def db_pool_report(mini_suggestion: int | None) -> None:
    """The THIRD constraint on concurrency, after TPM/RPM and memory.

    The pool cannot outgrow what the server accepts: Supabase free tier in
    SESSION mode caps the whole PROJECT at 15 client connections, shared by
    every process. Measured 2026-09-26, `extract-entities --concurrency 32` lost
    62 of its first 71 chunks to
    `(EMAXCONNSESSION) max clients reached in session mode` -- because the pool
    was GROWN to 32, not because workers held connections.

    Worker concurrency is a separate question. Only stages whose workers hold a
    connection across an LLM call (`_POOL_BOUND_STAGES`) are capped to the pool;
    extraction and artifact generation take a connection only for millisecond
    reads, so they run at their configured concurrency against a small pool.
    """
    try:
        from backend.app.cli.main import _POOL_BOUND_STAGES
        from backend.app.db.engine import configured_pool_ceiling
    except Exception as exc:                                   # pragma: no cover
        print(f"\nDATABASE pool: could not inspect ({exc}); skipping.")
        return

    cfg = get_settings().app_config
    db = (cfg.get("database") or {})
    conc = (cfg.get("concurrency") or {})
    ceiling = configured_pool_ceiling()

    print("\nDATABASE connection pool (the third limit -- NOT a rate limit)")
    dsn = get_settings().database_url or ""
    host = dsn.split("@")[-1].split("/")[0].split(":")[0]
    is_supabase = "supabase" in host
    if is_supabase:
        print("  target: Supabase  ->  SESSION MODE CAPS THE PROJECT AT 15 CLIENTS")
    elif host:
        print("  target: self-hosted / docker  ->  no fixed client cap")

    if ceiling is None:
        derived = min(16, max([4, *(int(v) for v in conc.values()
                                    if isinstance(v, (int, float)) and v)]))
        print("  database.pool_size is UNSET -> pool derived from the largest")
        print(f"  concurrency value: {derived} + {derived} overflow = {derived * 2},")
        print("  and a stage may raise it to 64. Fine on docker; on Supabase set")
        print("  database.pool_size explicitly or a big stage will exceed the cap.")
        cap = None
    else:
        print(f"  database.pool_size {db.get('pool_size')} + max_overflow "
              f"{db.get('max_overflow')} = {ceiling} connections (a HARD ceiling)")
        if is_supabase and ceiling > 15:
            print(f"  *** {ceiling} EXCEEDS the 15-client cap -- lower pool_size ***")
        cap = ceiling

    print("  Pool-capped stages (a worker holds its connection across LLM calls):")
    for k in sorted(_POOL_BOUND_STAGES):
        now = conc.get(k, "(unset)")
        note = ""
        if cap and isinstance(now, int) and now > cap:
            note = f"  -> CAPPED to {cap} at runtime"
        print(f"    concurrency.{k:<22} config {now}{note}")
    if cap and mini_suggestion and mini_suggestion > cap:
        print(f"  NOTE: the rate-limit suggestion above ({mini_suggestion}) exceeds the")
        print(f"  pool ceiling ({cap}), so DB-bound stages will run at {cap} however high")
        print("  you set them. Raising pool_size only helps if the SERVER allows it.")
    print("  NOT capped: summarization, entity_extraction and artifact_generation")
    print("  (their workers hold no DB session while waiting on the LLM, so they share")
    print("  the pool in millisecond slices), and every prune-expand stage, which")
    print("  never touches the database at all.")


_TABLE_KNOBS = ("table_extraction", "table_page_vision")


def _parse_sets(argv: list[str]) -> dict[str, int]:
    """`--set stage=N` pairs (repeatable). Unknown stages are rejected."""
    out: dict[str, int] = {}
    for i, a in enumerate(argv):
        if a == "--set" and i + 1 < len(argv):
            k, _, v = argv[i + 1].partition("=")
            if k not in STAGE_TASKS and k not in _TABLE_KNOBS:
                raise SystemExit(f"--set: unknown stage {k!r}; "
                                 f"one of {sorted([*STAGE_TASKS, *_TABLE_KNOBS])}")
            out[k] = int(v)
    return out


async def main() -> int:
    argv = sys.argv[1:]
    manual = _parse_sets(argv)
    do_apply = "--apply" in argv

    models = _models_in_use()
    if any(p == "deepinfra" for p, _ in models):
        print("This models.yaml routes to DeepInfra. A dedicated deployment is sized by")
        print("GPU memory and throughput, not rate limits -- run instead:")
        print("  uv run python scripts/deepinfra_sizing.py <docs-dir>\n")
    rows = await asyncio.gather(*[probe(p, m) for p, m in models])
    limits = {(r["provider"], r["model"]): r for r in rows}

    print("Provider rate limits (from each provider's own headers)\n")
    print(f"{'provider/model':<34} {'TPM (in/out)':>22} {'RPM':>8}")
    print("-" * 66)
    for r in rows:
        name = f"{r['provider']}/{r['model']}"
        if r.get("error"):
            print(f"{name:<34} {r['error'][:40]}")
            continue
        tok = (f"{r.get('itpm') or '?'}/{r.get('otpm') or '?'}"
               if r.get("itpm") or r.get("otpm") else f"{r.get('tpm') or '?'}")
        print(f"{name:<34} {tok:>22} {r.get('rpm') or '?':>8}")

    sugg = stage_suggestions(limits)
    cur = (get_settings().app_config.get("concurrency") or {})
    print("\nSuggested concurrency per stage (bound by its TIGHTEST task):\n")
    print(f"  {'stage':<22} {'now':>6} {'suggest':>8}   binding task (model)")
    for stage, sg in sugg.items():
        now = cur.get(stage, "unset")
        val = sg["value"] if sg["value"] is not None else "?"
        bind = (f"{sg['task']} ({sg['model']})" if sg.get("task")
                else "no limits known -- keep current")
        mark = ""
        if isinstance(now, int) and isinstance(sg["value"], int):
            mark = " <- raise" if now < sg["value"] // 2 else (
                " <- LOWER" if now > sg["value"] else "")
        print(f"  {stage:<22} {now!s:>6} {val!s:>8}   {bind}{mark}")
        if sg["unknown"]:
            print(f"  {'':<22} {'':>6} {'':>8}   (unprobed: {', '.join(sg['unknown'][:3])})")

    print("\n  Phase-2 stages (" + ", ".join(PHASE2_STAGES) + ") are sized by the")
    print("  RATE LIMIT above like every other stage. The database bounds the POOL,")
    print("  reported below, not their worker count -- except `evaluation`.")
    print("  class_proposal / dedup send 32k-token gpt-4.1 requests; models.yaml warns")
    print("  concurrent large calls throttle, so raise those INCREMENTALLY and watch")
    print("  for 429s. class_proposal is ~65% of prune-expand cost; dedup ~22%.")
    sel = (get_settings().app_config.get("corpus_selection") or {}).get(
        "label_concurrency", 8)
    mini = sugg.get("chunk_classification", {}).get("value")
    if mini:
        print(f"\n  corpus_selection.label_concurrency now {sel}: pass "
              f"--selection-concurrency {mini} to --select-subset runs.")
    print("\nThis buys SPEED, not savings. Wall time falls close to linearly;")
    print("total cost is unchanged because the same tokens are sent either way.")
    print("A higher value does slightly lower the prompt-cache hit rate")
    print("(measured 69% -> 49% going 4 -> 32, about +2% spend).")

    db_pool_report(sugg.get("evaluation", {}).get("value"))

    cfg0 = get_settings().app_config
    mem_sugg = advise_memory(
        int((cfg0.get("chunking") or {}).get("streaming_batch_size", 8)),
        int((cfg0.get("concurrency") or {}).get("table_extraction", 1)),
        int((cfg0.get("concurrency") or {}).get("table_page_vision", 1)),
    )

    if do_apply or manual:
        values = {k: v["value"] for k, v in sugg.items()
                  if do_apply and v["value"] is not None}
        if do_apply:
            values.update(mem_sugg)          # table workers + pages, from free memory
        values.update(manual)
        path = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
        changes = apply_concurrency(path, values)
        print(f"\nWROTE {path.relative_to(path.parent.parent)}:")
        for c in changes or ["(no changes -- already at these values)"]:
            print(f"  {c}")


    # Minimal arg handling: first positional is the corpus, optional
    # --selection <selection.json> narrows it to a chosen subset.
    set_values = {argv[i + 1] for i, a in enumerate(argv)
                  if a == "--set" and i + 1 < len(argv)}
    positional = [a for a in argv if not a.startswith("-") and a not in set_values]
    selection = None
    if "--selection" in sys.argv:
        i = sys.argv.index("--selection")
        if i + 1 < len(sys.argv):
            selection = sys.argv[i + 1]
            if selection in positional:
                positional.remove(selection)

    if positional:
        cfg2 = get_settings().app_config
        analyse_corpus(
            positional[0],
            int((cfg2.get("chunking") or {}).get("streaming_batch_size", 8)),
            int((cfg2.get("concurrency") or {}).get("summarization", 4)),
            selection_path=selection,
        )
    else:
        print("\nTip: pass a documents directory to also size "
              "chunking.streaming_batch_size,\n     which caps effective concurrency: "
              "uv run python scripts/tpm_check.py <docs-dir>")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
