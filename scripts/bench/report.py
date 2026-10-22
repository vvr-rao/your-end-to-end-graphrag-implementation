#!/usr/bin/env python3
"""Collect a benchmark folder into a comparison report (free; reads files).

    uv run python scripts/bench/report.py benchmarks/<date> [--ledger .build/gpu_ledger.jsonl]

Per corpus and arm: step outcomes + wall time, per-stage LLM tokens / cost /
call seconds (from GRAPHRAG_COST_REPORT_DIR dumps), DB size, and every eval
summary. Then the hosting-mode analysis:

  (a) dedicated Flash-Next: actual GPU-hours x rate (ledger) vs what the
      SAME creation tokens would cost per-token on serverless Qwen3.8-Flash;
  (b) serverless Qwen3.8-27B retrieval: measured cost/query vs a 1x H100
      dedicated deployment, with the break-even query rate.

Writes <dir>/report.md and <dir>/report.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

# Serverless reference prices, USD per 1M tokens (deepinfra.com, 2026-10-12).
SERVERLESS_FLASH = {"in": 0.113, "out": 0.382}   # Qwen/Qwen3.8-Flash (nearest listing)
SERVERLESS_27B = {"in": 0.15, "out": 1.875}
H100_USD_HR = 2.20                               # 27B BF16 (55.6 GB) fits one H100

RETRIEVAL_TASKS = {
    "question_parse", "question_relations", "concept_expansion", "query_decompose",
    "retrieval_rounds_plan", "entity_probes", "chunk_relevance_filter", "answer_simple_qa",
    "answer_deep_research", "follow_up_resolution", "answer_conversation_turn", "qa_synthesis",
}


def merge_costs(cost_dir: Path) -> dict[str, dict]:
    """Sum every dumped cost_report's by_task rows."""
    tasks: dict[str, dict] = {}
    for f in sorted(cost_dir.glob("*.json")):
        if f.name == "corpus.json":
            continue
        for t, v in json.loads(f.read_text()).get("by_task", {}).items():
            a = tasks.setdefault(t, {"calls": 0, "cost_usd": 0.0, "in_tokens": 0, "out_tokens": 0,
                                     "call_seconds": 0.0, "model": v.get("model"),
                                     "provider": v.get("provider")})
            for k in ("calls", "cost_usd", "in_tokens", "out_tokens", "call_seconds"):
                a[k] += v.get(k) or 0
    return tasks


def collect(bench: Path) -> dict:
    out: dict = {}
    # Any arm directory (openai, qwen*, sl-<model>) holding corpus subfolders.
    for arm_dir in sorted(p for p in bench.iterdir() if p.is_dir() and any(c.is_dir() for c in p.iterdir())):
        for c in sorted(p for p in arm_dir.iterdir() if p.is_dir()):
            row: dict = {"steps": [], "tasks": {}, "evals": []}
            if (c / "steps.json").exists():
                row["steps"] = json.loads((c / "steps.json").read_text())
            if (c / "costs").exists():
                row["tasks"] = merge_costs(c / "costs")
                cj = c / "costs" / "corpus.json"
                row["corpus"] = json.loads(cj.read_text()) if cj.exists() else {}
            for e in sorted((c / "eval").glob("*.json")) if (c / "eval").exists() else []:
                d = json.loads(e.read_text())
                row["evals"].append({"file": e.name, "summary": d.get("summary") or _judge_summary(d)})
            out.setdefault(c.name, {})[arm_dir.name] = row
    return out


def _judge_summary(d: dict) -> dict:
    """evaluate-queries JSON -> its aggregate block, whatever it is named."""
    for k in ("aggregate", "summary", "metrics", "totals"):
        if k in d:
            return d[k]
    return {k: v for k, v in d.items() if not isinstance(v, list)}


def hosting_analysis(results: dict, ledger: Path | None) -> dict:
    created_in = created_out = 0
    retr_cost = retr_in = retr_out = retr_calls = retr_secs = 0.0
    for arms in results.values():
        for t, v in (arms.get("qwen", {}).get("tasks") or {}).items():
            if t in RETRIEVAL_TASKS:
                retr_cost += v["cost_usd"]
                retr_in += v["in_tokens"]
                retr_out += v["out_tokens"]
                retr_calls += v["calls"]
                retr_secs += v["call_seconds"]
            elif v.get("provider") == "deepinfra" and t != "embeddings":
                created_in += v["in_tokens"]
                created_out += v["out_tokens"]
    a = {"creation_in_tokens": created_in, "creation_out_tokens": created_out,
         "serverless_flash_equiv_usd": round(created_in / 1e6 * SERVERLESS_FLASH["in"]
                                             + created_out / 1e6 * SERVERLESS_FLASH["out"], 2)}
    if ledger and ledger.exists():
        from deepinfra_deploy import gpu_hours

        rows = gpu_hours(ledger)
        a["dedicated_gpu_hours"] = round(sum(r["hours"] or 0 for r in rows), 2)
        a["dedicated_usd"] = round(sum(r["usd"] or 0 for r in rows), 2)
        a["still_running"] = [r["deploy_id"] for r in rows if r["hours"] is None]
    # Retrieval: serverless spend per LLM call-hour vs renting an H100.
    if retr_secs:
        a["retrieval_serverless_usd"] = round(retr_cost, 4)
        a["retrieval_llm_hours"] = round(retr_secs / 3600, 3)
        usd_per_busy_hour = retr_cost / (retr_secs / 3600)
        # A dedicated H100 pays off only once it is busy enough that serverless
        # would cost more per wall-clock hour than the rental.
        a["retrieval_breakeven_utilisation"] = round(H100_USD_HR / usd_per_busy_hour, 2) if usd_per_busy_hour else None
    return a


def render(results: dict, hosting: dict) -> str:
    lines = ["# DeepInfra/Qwen vs OpenAI-only benchmark", ""]
    for corpus, arms in results.items():
        lines += [f"## {corpus}", "", "| metric | " + " | ".join(arms) + " |", "|---|" + "---|" * len(arms)]

        def cell(arm, fn, arms=arms):
            try:
                return fn(arms[arm])
            except Exception:
                return "-"

        metrics = {
            "steps ok": lambda r: f"{sum(s['exit'] == 0 for s in r['steps'])}/{len(r['steps'])}",
            "pipeline wall (min)": lambda r: f"{sum(s['seconds'] for s in r['steps']) / 60:.1f}",
            "LLM cost (per-token $)": lambda r: f"{sum(v['cost_usd'] for v in r['tasks'].values()):.2f}",
            "LLM calls": lambda r: f"{sum(v['calls'] for v in r['tasks'].values()):,}",
            "in / out tokens (M)": lambda r: (f"{sum(v['in_tokens'] for v in r['tasks'].values()) / 1e6:.2f} / "
                                              f"{sum(v['out_tokens'] for v in r['tasks'].values()) / 1e6:.2f}"),
        }
        for name, fn in metrics.items():
            lines.append(f"| {name} | " + " | ".join(cell(a, fn) for a in arms) + " |")
        lines.append("")
        for arm, r in arms.items():
            for e in r["evals"]:
                lines.append(f"- **{arm}** `{e['file']}`: `{json.dumps(e['summary'])[:400]}`")
        lines.append("")
    lines += ["## Hosting mode", "", "```json", json.dumps(hosting, indent=2), "```", ""]
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    bench = Path(argv[0])
    ledger = Path(argv[argv.index("--ledger") + 1]) if "--ledger" in argv else ROOT / ".build" / "gpu_ledger.jsonl"
    results = collect(bench)
    hosting = hosting_analysis(results, ledger)
    (bench / "report.json").write_text(json.dumps({"results": results, "hosting": hosting}, indent=2))
    (bench / "report.md").write_text(render(results, hosting))
    print(f"wrote {bench / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
