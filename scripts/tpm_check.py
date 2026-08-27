#!/usr/bin/env python3
"""Report your provider rate limits and recommend a concurrency setting.

    uv run python scripts/tpm_check.py [<documents-dir>]

Pass a corpus directory to also size `streaming_batch_size`, which silently
CAPS effective concurrency: only the current batch's windows exist, so a
semaphore larger than that has nothing to schedule.

Reads OpenAI's authoritative rate-limit headers (one ~10-token probe per
model) and turns them into a concrete suggestion for `concurrency:` in
config/config.yaml.

WHY THIS EXISTS: the shipped default of 4 was tuned for a low tier. On a
tier-3+ account it leaves most of the allowance unused -- a 1.6M-token
corpus spent ~4 hours summarizing at under 1% of a 10M TPM limit. Raising
concurrency cuts wall time close to linearly. It does NOT reduce cost: the
same tokens are sent either way.

Reads only whether keys are present, never prints key values.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app.core.config import get_settings  # noqa: E402
from backend.app.services.llm_router import (  # noqa: E402
    _openai_uses_completion_tokens,
)

# Rough per-call shape of the heavy stages: a ~12k-token source window plus
# up to 8k of output, taking ~60s. Used only to size the suggestion.
TOKENS_PER_CALL = 20_000
SECONDS_PER_CALL = 60
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


def _models_in_use() -> list[str]:
    """Every distinct OpenAI model the active models.yaml routes to."""
    tasks = (get_settings().models_config.get("tasks") or {})
    seen: list[str] = []
    for spec in tasks.values():
        if spec.get("provider") != "openai":
            continue
        m = spec.get("model")
        if m and m not in seen and not str(m).startswith("text-embedding"):
            seen.append(m)
    return seen or ["gpt-4.1", "gpt-4.1-mini"]


async def probe(client, model: str) -> dict:
    # gpt-5.x / o-series reject `max_tokens` and non-default temperature.
    kw = {"max_completion_tokens": 1} if _openai_uses_completion_tokens(model) else {"max_tokens": 1}
    try:
        raw = await client.chat.completions.with_raw_response.create(
            model=model, messages=[{"role": "user", "content": "hi"}], **kw,
        )
    except Exception as exc:
        return {"model": model, "error": f"{type(exc).__name__}: {str(exc)[:70]}"}
    h = raw.headers
    return {
        "model": model,
        "tpm": _num(h.get("x-ratelimit-limit-tokens")),
        "rpm": _num(h.get("x-ratelimit-limit-requests")),
        "remaining": _num(h.get("x-ratelimit-remaining-tokens")),
    }


def suggest(tpm: int | None, rpm: int | None) -> int | None:
    if not tpm:
        return None
    by_tokens = int((tpm * TARGET_UTILISATION) /
                    (TOKENS_PER_CALL * (60 / SECONDS_PER_CALL)))
    by_requests = int((rpm or 10_000) * TARGET_UTILISATION * (SECONDS_PER_CALL / 60))
    return max(1, min(by_tokens, by_requests, 128))


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


def advise_memory(batch_size: int, table_conc: int) -> None:
    """Recommend memory-bound settings from what this machine actually has."""
    avail, total = _mem_mb()
    if avail < 0:
        print("\n\nMemory: could not be read on this platform "
              f"({sys.platform}) -- skipping memory-based advice. "
              "Linux, macOS and Windows are supported; check the OS command "
              "manually (see the run-prune-expand skill).")
        return
    swap = _has_swap()
    print(f"\n\nMachine memory: {avail:,} MB available of {total:,} MB total"
          f"  |  swap: {'yes' if swap else 'NONE'}")
    if not swap:
        print("  No swap: exceeding RAM is a HARD KILL mid-run, not a slowdown.")

    budget = max(0, avail - _RESERVE_MB)
    rec_table = max(1, min(8, budget // _TABLE_WORKER_MB))
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


async def main() -> int:
    s = get_settings()
    if not s.openai_api_key:
        print("OPENAI_API_KEY not set — cannot probe rate limits.")
        return 1
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=s.openai_api_key, max_retries=0)
    models = _models_in_use()
    rows = await asyncio.gather(*[probe(client, m) for m in models])

    print("Provider rate limits (from OpenAI's own headers)\n")
    print(f"{'model':<18} {'TPM limit':>12} {'RPM limit':>10} {'suggested conc.':>16}")
    print("-" * 60)
    # Bucket by tier: the mini-class models drive the cheap high-volume stages;
    # the big models drive Stage 1/2 and sit on a much tighter limit. Taking one
    # minimum across both would pin the cheap stages to the tight limit -- the
    # exact mistake these separate knobs exist to prevent.
    mini: list[int] = []
    big: list[int] = []
    for r in rows:
        if r.get("error"):
            print(f"{r['model']:<18} {r['error'][:44]}")
            continue
        sug = suggest(r["tpm"], r["rpm"])
        if sug:
            (mini if (r["tpm"] or 0) >= 5_000_000 else big).append(sug)
        print(f"{r['model']:<18} {r['tpm'] or '?':>12} {r['rpm'] or '?':>10} {sug or '?':>16}")

    cfg = get_settings().app_config
    cur = (cfg.get("concurrency") or {})
    print("\nCurrent config/config.yaml:")
    for k in ("summarization", "entity_extraction", "artifact_generation",
              "table_mining", "evaluation"):
        print(f"  concurrency.{k:<20} {cur.get(k, '(unset -> 4)')}")
    print(f"  expansion.max_concurrent_llm_calls  "
          f"{(cfg.get('expansion') or {}).get('max_concurrent_llm_calls', '(unset -> 4)')}"
          "   <- keep LOW (gpt-4.1 @ 32k max_tokens on the tighter tier)")

    if mini or big:
        cur = (get_settings().app_config.get("concurrency") or {})
        if mini:
            m = min(mini)
            print(f"\nMINI-MODEL stages (~10M TPM) -- suggested {m}:")
            for k in ("summarization", "chunk_classification", "entity_extraction",
                      "artifact_generation", "table_mining"):
                now = cur.get(k, "(unset)")
                flag = "  <- raise" if isinstance(now, int) and now < m // 2 else ""
                print(f"    concurrency.{k:<22} now {now}{flag}")
            print("    chunk_classification is Stage 1: it ran at 4 for a long time only")
            print("    because it shared Stage 2's semaphore. It is independent now.")
            sel_now = (get_settings().app_config.get("corpus_selection") or {}).get(
                "label_concurrency", 8)
            print(f"\n    corpus_selection.label_concurrency  now {sel_now}"
                  f"{'  <- raise' if isinstance(sel_now, int) and sel_now < m // 2 else ''}")
            print("    Drives --select-subset's doc-type labelling + embed compression")
            print("    (cheap model, so it belongs in this tier). Pass it as")
            print(f"    --selection-concurrency {m}; the SUMMARIZATION inside selection")
            print("    takes --summarization-concurrency and is the real long pole.")
        if big:
            b = min(big)
            print(f"\nBIG-MODEL stages (~2M TPM, 32k-token requests) -- ceiling ~{b}:")
            for k in ("class_proposal", "dedup"):
                print(f"    concurrency.{k:<22} now {cur.get(k, '(unset)')}")
            print("    MEASURED: class_proposal used only 6.2% of the 2M tier at 4,")
            print("    so 12-16 is likely safe. But models.yaml warns that concurrent")
            print("    large gpt-4.1 calls throttle -- raise INCREMENTALLY, watch 429s.")
            print("    class_proposal is ~65% of prune-expand cost; dedup ~22%.")
        print("\nThis buys SPEED, not savings. Wall time falls close to linearly;")
        print("total cost is unchanged because the same tokens are sent either way.")
        print("A higher value does slightly lower the prompt-cache hit rate")
        print("(measured 69% -> 49% going 4 -> 32, about +2% spend).")

    cfg1 = get_settings().app_config
    advise_memory(
        int((cfg1.get("chunking") or {}).get("streaming_batch_size", 8)),
        int((cfg1.get("concurrency") or {}).get("table_extraction", 1)),
    )

    # Minimal arg handling: first positional is the corpus, optional
    # --selection <selection.json> narrows it to a chosen subset.
    positional = [a for a in sys.argv[1:] if not a.startswith("-")]
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
