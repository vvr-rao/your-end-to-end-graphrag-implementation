"""Sample OpenAI's own rate-limit headers to measure real TPM utilisation.

Our token counters only tell us what WE sent. OpenAI returns the
authoritative view on every response:

    x-ratelimit-limit-tokens      the bucket size (your tier's TPM)
    x-ratelimit-remaining-tokens  what's left in the current window
    x-ratelimit-reset-tokens      when it refills

Utilisation = (limit - remaining) / limit, sampled while the sweep runs.
Limits are PER MODEL, so each model is probed separately. Each probe costs
one ~10-token call.
"""
import asyncio
import sys
import time
from datetime import datetime

from openai import AsyncOpenAI

from backend.app.core.config import get_settings

MODELS = ["gpt-4.1", "gpt-4.1-mini", "gpt-4o-mini"]
INTERVAL = int(sys.argv[1]) if len(sys.argv) > 1 else 20
DURATION = int(sys.argv[2]) if len(sys.argv) > 2 else 1800


def _num(v):
    """Header values look like '10000000' or '9.98M' or '1998000'."""
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


async def probe(client: AsyncOpenAI, model: str) -> dict | None:
    try:
        raw = await client.chat.completions.with_raw_response.create(
            model=model,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1,
        )
    except Exception as exc:
        return {"model": model, "error": f"{type(exc).__name__}: {str(exc)[:80]}"}
    h = raw.headers
    lim_t = _num(h.get("x-ratelimit-limit-tokens"))
    rem_t = _num(h.get("x-ratelimit-remaining-tokens"))
    lim_r = _num(h.get("x-ratelimit-limit-requests"))
    rem_r = _num(h.get("x-ratelimit-remaining-requests"))
    used = (lim_t - rem_t) if (lim_t and rem_t is not None) else None
    return {
        "model": model,
        "limit_tpm": lim_t,
        "remaining": rem_t,
        "used": used,
        "pct": (100.0 * used / lim_t) if (used is not None and lim_t) else None,
        "limit_rpm": lim_r,
        "remaining_rpm": rem_r,
        "reset": h.get("x-ratelimit-reset-tokens"),
    }


async def main() -> None:
    client = AsyncOpenAI(api_key=get_settings().openai_api_key, max_retries=0)
    print(f"{'time':<9} {'model':<14} {'limit TPM':>12} {'used':>12} {'util':>7} "
          f"{'RPM left':>10}  reset", flush=True)
    t_end = time.monotonic() + DURATION
    peak: dict[str, float] = {}
    while time.monotonic() < t_end:
        rows = await asyncio.gather(*[probe(client, m) for m in MODELS])
        stamp = datetime.now().strftime("%H:%M:%S")
        for r in rows:
            if r is None:
                continue
            if r.get("error"):
                print(f"{stamp:<9} {r['model']:<14} {r['error']}", flush=True)
                continue
            pct = r["pct"]
            if pct is not None:
                peak[r["model"]] = max(peak.get(r["model"], 0.0), pct)
            print(f"{stamp:<9} {r['model']:<14} {str(r['limit_tpm']):>12} "
                  f"{str(r['used']):>12} {('%.2f%%' % pct) if pct is not None else '?':>7} "
                  f"{str(r['remaining_rpm']):>10}  {r['reset']}", flush=True)
        print(f"{'':9} peak so far: " +
              ", ".join(f"{m}={p:.2f}%" for m, p in sorted(peak.items())), flush=True)
        await asyncio.sleep(INTERVAL)


asyncio.run(main())
