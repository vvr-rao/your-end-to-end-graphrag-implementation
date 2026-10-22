#!/usr/bin/env python3
"""Evaluate one arm's knowledge graph. Every mode is PAID (needs --go).

    # MultiHop-RAG gold (web-search corpus): answer accuracy + gold-doc retrieval
    uv run python scripts/bench/eval_arm.py multihop --arm qwen --corpus web \
        --docs source_documents/multihop-rag-subset [--n 60] [--mode simple_qa] --go

    # Judge-scored question file (any corpus): wraps `evaluate-queries`
    uv run python scripts/bench/eval_arm.py judge --arm qwen --corpus finance \
        --questions eval_questions/finance_bench.txt [--mode deep_research] [--runs 2] --go

Judges stay on gpt-4.1 in BOTH arms (the deepinfra preset keeps judge_* on
OpenAI), so scores are comparable. Results go to <out>/<arm>/<corpus>/eval/.
"""
from __future__ import annotations

import asyncio
import glob
import json
import random
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "bench"))

MHRAG_GLOB = str(Path.home() / ".cache/huggingface/hub/datasets--yixuantt--MultiHopRAG/snapshots/*/MultiHopRAG.json")
INSUFFICIENT = "insufficient information"


def _arg(argv, name, default=None):
    return argv[argv.index(name) + 1] if name in argv and argv.index(name) + 1 < len(argv) else default


# --------------------------------------------------------------------------- #
# MultiHop gold set (pure helpers are unit-tested)
# --------------------------------------------------------------------------- #

def doc_titles(docs_dir: Path) -> dict[str, str]:
    """{normalised article title: file stem} from each file's `Title:` header."""
    out: dict[str, str] = {}
    for p in sorted(docs_dir.glob("*.txt")):
        head = p.read_text(errors="replace")[:600]
        m = re.search(r"^Title:\s*(.+)$", head, re.M)
        if m:
            out[_norm(m.group(1))] = p.stem
    return out


def _norm(s: str) -> str:
    return re.sub(r"\W+", " ", s.lower()).strip()


def gold_questions(questions: list[dict], titles: dict[str, str]) -> list[dict]:
    """Questions whose evidence lies ENTIRELY inside the corpus subset.

    Null queries (no evidence; the right answer is to decline) are excluded:
    all 301 would pass this filter vacuously. Add a slice with --null N.
    """
    out = []
    for q in questions:
        ev = q.get("evidence_list") or []
        if not ev:
            continue
        stems = [titles.get(_norm(e["title"])) for e in ev]
        if all(stems):
            out.append({**q, "gold_docs": sorted(set(stems))})
    return out


def answer_correct(pred: str | None, gold: str, qtype: str) -> bool:
    p = _norm(pred or "")
    g = _norm(gold)
    if qtype == "null_query" or g == INSUFFICIENT:
        return any(k in p for k in ("insufficient", "not enough information", "does not contain",
                                    "no information", "cannot be determined"))
    if g in ("yes", "no"):
        first = p.split(" ", 1)[0] if p else ""
        return first == g
    return g in p


def retrieval_scores(evidence: list[dict], gold_docs: list[str], titles: dict[str, str]) -> dict:
    """Rank of gold documents in the evidence list (deduplicated by document)."""
    stems_by_title = {k: v for k, v in titles.items()}
    order: list[str] = []
    for e in evidence:
        t = e.get("document_title") or ""
        stem = stems_by_title.get(_norm(t)) or (t if t in gold_docs else None) or Path(t).stem
        if stem and stem not in order:
            order.append(stem)
    if not gold_docs:
        return {"recall": None, "mrr": None, "gold_at5": None}
    found = [order.index(g) + 1 for g in gold_docs if g in order]
    return {
        "recall": len(found) / len(gold_docs),
        "mrr": 1 / min(found) if found else 0.0,
        "gold_at5": sum(1 for r in found if r <= 5),
    }


async def _run_multihop(sample: list[dict], titles: dict[str, str], mode: str, conc: int) -> list[dict]:
    from backend.app.services.retrieval import retrieve_and_answer

    sem = asyncio.Semaphore(conc)

    async def one(q: dict) -> dict:
        async with sem:
            t0 = time.monotonic()
            try:
                r = await retrieve_and_answer(q["query"], mode=mode, persist=False, max_cost_usd=1.0)
                ans, ev, cost = r.answer, r.evidence, r.cost_usd
                err = None
            except Exception as exc:
                ans, ev, cost, err = None, [], 0.0, f"{type(exc).__name__}: {exc}"[:200]
            return {
                "query": q["query"], "type": q["question_type"], "gold": q["answer"],
                "pred": ans, "correct": answer_correct(ans, q["answer"], q["question_type"]),
                **retrieval_scores(ev, q["gold_docs"], titles),
                "cost_usd": cost, "seconds": round(time.monotonic() - t0, 1), "error": err,
            }

    return await asyncio.gather(*[one(q) for q in sample])


def summarise(rows: list[dict]) -> dict:
    def mean(k, rs):
        v = [r[k] for r in rs if r.get(k) is not None]
        return round(sum(v) / len(v), 4) if v else None

    by_type = {}
    for t in sorted({r["type"] for r in rows}):
        rs = [r for r in rows if r["type"] == t]
        by_type[t] = {"n": len(rs), "accuracy": mean("correct", rs)}
    return {"n": len(rows), "accuracy": mean("correct", rows), "recall": mean("recall", rows),
            "mrr": mean("mrr", rows), "gold_at5": mean("gold_at5", rows),
            "cost_usd": round(sum(r["cost_usd"] or 0 for r in rows), 4),
            "p50_seconds": sorted(r["seconds"] for r in rows)[len(rows) // 2] if rows else None,
            "errors": sum(1 for r in rows if r["error"]), "by_type": by_type}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv: list[str]) -> int:
    kind = argv[0] if argv else ""
    arm, corpus = _arg(argv, "--arm"), _arg(argv, "--corpus")
    if kind not in ("multihop", "judge") or not (arm and corpus):
        print(__doc__)
        return 2
    base = Path(_arg(argv, "--out", str(ROOT / "benchmarks" / time.strftime("%Y-%m-%d")))) / arm / corpus
    mode = _arg(argv, "--mode", "simple_qa" if kind == "multihop" else "deep_research")
    if "--go" not in argv:
        print(f"(dry run) {kind} eval of arm={arm} corpus={corpus} mode={mode} -> {base / 'eval'}; add --go (PAID)")
        return 0

    from run_arm import arm_env

    env = arm_env(arm, corpus, base)
    (base / "eval").mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%H%M%S")

    if kind == "judge":
        qfile = _arg(argv, "--questions")
        stem = f"judge-{Path(qfile).stem}-{mode}-{stamp}"
        cmd = ["uv", "run", "python", "-m", "backend.app.cli", "evaluate-queries", "--questions", qfile,
               "--mode", mode, "--runs-per-question", _arg(argv, "--runs", "2"), "--judge-model", "gpt-4.1",
               "--max-cost-usd", _arg(argv, "--max-cost-usd", "10"),
               "--output", str(base / "eval" / f"{stem}.json"),
               "--output-md", str(base / "eval" / f"{stem}.md")]
        return subprocess.run(cmd, cwd=ROOT, env=env).returncode

    # multihop: run in-process with the arm's environment.
    import os

    os.environ.update(env)
    from backend.app.core.config import get_settings

    get_settings.cache_clear()
    docs = Path(_arg(argv, "--docs"))
    titles = doc_titles(docs)
    questions = json.loads(Path(glob.glob(MHRAG_GLOB)[0]).read_text())
    gold = gold_questions(questions, titles)
    random.Random(int(_arg(argv, "--seed", "7"))).shuffle(gold)
    sample = gold[: int(_arg(argv, "--n", "60"))]
    nulls = [q for q in questions if q["question_type"] == "null_query"]
    random.Random(7).shuffle(nulls)
    sample += [{**q, "gold_docs": []} for q in nulls[: int(_arg(argv, "--null", "0"))]]
    print(f"{len(gold)} gold questions inside {docs.name}; evaluating {len(sample)} in {mode} mode")
    # At most 2 in flight: 4 concurrent e2e retrievals OOM'd the 2.7 GB VM.
    rows = asyncio.run(_run_multihop(sample, titles, mode, int(_arg(argv, "--concurrency", "2"))))
    summary = summarise(rows)
    out = base / "eval" / f"multihop-{mode}-{stamp}.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
