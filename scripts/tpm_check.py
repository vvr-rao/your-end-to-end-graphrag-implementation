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


def analyse_corpus(docs_dir: str, batch_size: int, concurrency: int) -> None:
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
        if mini:
            print(f"\nSuggested concurrency for the mini-model stages: {min(mini)}")
            print("  concurrency.summarization / entity_extraction /")
            print("  artifact_generation / table_mining")
        if big:
            print(f"\nSuggested expansion.max_concurrent_llm_calls: {min(big)}")
            print("  (Stage 1/2 class_proposal -- the tighter tier; the shipped 4-8")
            print("   is deliberately below this, since it sends 32k-token requests)")
        print("\nThis buys SPEED, not savings. Wall time falls close to linearly;")
        print("total cost is unchanged because the same tokens are sent either way.")
        print("A higher value does slightly lower the prompt-cache hit rate")
        print("(measured 69% -> 49% going 4 -> 32, about +2% spend).")

    if len(sys.argv) > 1:
        cfg2 = get_settings().app_config
        analyse_corpus(
            sys.argv[1],
            int((cfg2.get("chunking") or {}).get("streaming_batch_size", 8)),
            int((cfg2.get("concurrency") or {}).get("summarization", 4)),
        )
    else:
        print("\nTip: pass a documents directory to also size "
              "chunking.streaming_batch_size,\n     which caps effective concurrency: "
              "uv run python scripts/tpm_check.py <docs-dir>")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
