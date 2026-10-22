#!/usr/bin/env python3
"""Re-run RETRIEVAL for an arm on the pharma question sets, without paying for
the final deep_research synthesis. PAID (query parsing / decomposition / round
planning / the round-1 simple_qa bridge answer still call the arm's models).

    uv run python scripts/bench/retrieval_only.py --arm openai-qwenemb --corpus pharma12 \
        [--out-sub eval] [--concurrency 4] [--qa-overrides '{"graph_score_weighting": true}'] \
        [--query-prefix TEXT] [--llm-cache PATH] --go

  --qa-overrides  JSON of qa.* knobs for this run only (QA_OVERRIDES env).
  --query-prefix  replaces the embedder's query instruction ("" = none) via a
                  per-run copy of the arm's models.yaml.
  --llm-cache     record/replay every chat call by (task, system, user). The
                  first run records; later runs with the same file replay, so
                  parse/decompose/relations are IDENTICAL across variants and a
                  difference is the lever's, not LLM run-to-run noise. A prompt
                  that changes (e.g. round 2 carries a different round-1
                  answer) misses and is called live.

The final `answer_deep_research` call is replaced by a stub: everything that
SHAPES retrieval (including the round-1 answer that seeds round 2) runs for
real, so the evidence lists match a full run. Writes the same files the full
eval writes, so scripts/bench/retrieval_quality.py reads them unchanged:
  <out>/<arm>/<corpus>/<out-sub>/audit_deep_research.jsonl
  <out>/<arm>/<corpus>/<out-sub>/pharma_test_{simple_qa,deep_research}.json
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "bench"))

STUB = "(retrieval-only run: synthesis skipped)"


def _arg(argv, name, default=None):
    return argv[argv.index(name) + 1] if name in argv and argv.index(name) + 1 < len(argv) else default


CACHE: dict[str, str] = {}
CACHE_STATS = {"hit": 0, "miss": 0}


def _stub_synthesis(use_cache: bool) -> None:
    import hashlib

    from backend.app.services.llm_router import ChatResult, LLMRouter

    real = LLMRouter.chat

    async def chat(self, task, *a, **kw):
        if task == "answer_deep_research":
            return ChatResult(text=STUB, model="stub", provider="stub")
        if not use_cache:
            return await real(self, task, *a, **kw)
        key = hashlib.sha256(json.dumps([task, kw.get("system"), kw.get("user"), [str(x) for x in a]],
                                        sort_keys=True).encode()).hexdigest()
        if key in CACHE:
            CACHE_STATS["hit"] += 1
            return ChatResult(text=CACHE[key], model="cache", provider="cache", cost_usd=0.0)
        CACHE_STATS["miss"] += 1
        out = await real(self, task, *a, **kw)
        CACHE[key] = out.text
        return out

    LLMRouter.chat = chat


async def run(base: Path, conc: int) -> float:
    from backend.app.services.eval_judge import parse_question_file
    from backend.app.services.retrieval import retrieve_and_answer
    from run_audit_questions import QUESTIONS

    sem = asyncio.Semaphore(conc)
    cost = [0.0]

    async def ask(q: str, mode: str):
        async with sem:
            t0 = time.monotonic()
            res = await retrieve_and_answer(q, mode=mode, persist=False)
            cost[0] += res.cost_usd
            return res, time.monotonic() - t0

    audit = await asyncio.gather(*[ask(q, "deep_research") for _, q, _ in QUESTIONS])
    with (base / "audit_deep_research.jsonl").open("w") as fh:
        for (qid, q, exp), (res, wall) in zip(QUESTIONS, audit, strict=True):
            fh.write(json.dumps({"id": qid, "question": q, "expectation": exp, "answer": res.answer,
                                 "evidence_iris": [e.get("iri") for e in res.evidence],
                                 "invalid_citations": [], "cost_usd": res.cost_usd, "wall_seconds": wall,
                                 "distinct_documents": len({e.get("document_title") for e in res.evidence
                                                            if e.get("kind") == "chunk" and e.get("document_title")})},
                                default=str) + "\n")
    print(f"audit done (${cost[0]:.3f} so far)", flush=True)
    qs = parse_question_file(ROOT / "eval_questions" / "pharma_test.txt")
    for mode in ("simple_qa", "deep_research"):
        rows = await asyncio.gather(*[ask(q, mode) for q, _ in qs for _ in range(2)])
        results = []
        for i, (q, gap) in enumerate(qs):
            runs = [{"answer": r.answer, "evidence": r.evidence, "wall_seconds": w, "cost_usd": r.cost_usd}
                    for r, w in rows[2 * i:2 * i + 2]]
            results.append({"question": q, "expected_gap": gap, "runs": runs})
        (base / f"pharma_test_{mode}.json").write_text(json.dumps({"results": results}, default=str))
        print(f"pharma_test {mode} done (${cost[0]:.3f} so far)", flush=True)
    return cost[0]


def main(argv: list[str]) -> int:
    arm, corpus = _arg(argv, "--arm"), _arg(argv, "--corpus", "pharma12")
    if not arm:
        print(__doc__)
        return 2
    from run_arm import arm_env

    out = ROOT / "benchmarks" / _arg(argv, "--date", "2026-10-17") / arm / corpus
    base = out / _arg(argv, "--out-sub", "eval")
    if "--go" not in argv:
        print(f"(dry run) retrieval-only for {arm} -> {base}; add --go (PAID)")
        return 0
    base.mkdir(parents=True, exist_ok=True)
    env = arm_env(arm, corpus, out)
    if "--query-prefix" in argv:
        import yaml

        doc = yaml.safe_load(Path(env["MODELS_CONFIG_PATH"]).read_text())
        doc["tasks"]["embeddings"]["query_prefix"] = _arg(argv, "--query-prefix", "").replace("\\n", "\n")
        variant = base / "models.yaml"
        variant.write_text(yaml.safe_dump(doc, sort_keys=False))
        env["MODELS_CONFIG_PATH"] = str(variant)
    if _arg(argv, "--qa-overrides"):
        json.loads(_arg(argv, "--qa-overrides"))  # fail fast on bad JSON
        env["QA_OVERRIDES"] = _arg(argv, "--qa-overrides")
    (base / "run_config.json").write_text(json.dumps(
        {"arm": arm, "qa_overrides": env.get("QA_OVERRIDES"), "query_prefix": _arg(argv, "--query-prefix"),
         "models": env["MODELS_CONFIG_PATH"], "llm_cache": _arg(argv, "--llm-cache")}, indent=2))
    os.environ.update(env)
    cache_path = Path(_arg(argv, "--llm-cache")) if _arg(argv, "--llm-cache") else None
    if cache_path and cache_path.exists():
        CACHE.update(json.loads(cache_path.read_text()))
    _stub_synthesis(cache_path is not None)
    try:
        cost = asyncio.run(run(base, int(_arg(argv, "--concurrency", "4"))))
    finally:
        if cache_path:
            # Merge with whatever another run saved meanwhile (parallel variants).
            on_disk = json.loads(cache_path.read_text()) if cache_path.exists() else {}
            on_disk.update(CACHE)
            cache_path.write_text(json.dumps(on_disk))
    print(f"total retrieval cost ${cost:.3f} -> {base}; llm cache {CACHE_STATS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
