"""Sample OpenAI rate-limit headers + local memory while a run is in flight.

remaining-tokens is BURST headroom (the bucket refills at limit/60 per second),
so a steady load barely dents it. Printed alongside memory so one line answers
both "am I near the rate limit" and "am I near an OOM".
"""
import asyncio, sys, time
from datetime import datetime
from openai import AsyncOpenAI
from backend.app.core.config import get_settings

MODELS = ["gpt-4.1", "gpt-4.1-mini", "gpt-4o-mini"]

def _num(v):
    try: return int(str(v).strip())
    except Exception: return None

def avail_mb():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable"): return int(line.split()[1]) // 1024
    return -1

async def probe(c, m):
    try:
        r = await c.chat.completions.with_raw_response.create(
            model=m, messages=[{"role":"user","content":"hi"}], max_tokens=1)
    except Exception as e:
        return m, None, None, f"{type(e).__name__}"
    h = r.headers
    lim, rem = _num(h.get("x-ratelimit-limit-tokens")), _num(h.get("x-ratelimit-remaining-tokens"))
    used = (lim - rem) if (lim and rem is not None) else None
    return m, lim, used, None

async def main():
    interval = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    c = AsyncOpenAI(api_key=get_settings().openai_api_key, max_retries=0)
    peak = {}
    while True:
        rows = await asyncio.gather(*[probe(c, m) for m in MODELS])
        parts = []
        for m, lim, used, err in rows:
            if err: parts.append(f"{m}=ERR"); continue
            pct = (100.0*used/lim) if (used is not None and lim) else 0.0
            peak[m] = max(peak.get(m, 0.0), pct)
            parts.append(f"{m.replace('gpt-','')}={pct:.2f}%")
        print(f"{datetime.now():%H:%M:%S}  TPM used: {'  '.join(parts)}   "
              f"| peak {max(peak.values()):.2f}%  | mem {avail_mb()} MB", flush=True)
        await asyncio.sleep(interval)

asyncio.run(main())
