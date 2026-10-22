#!/usr/bin/env python3
"""Companion views to retrieval_quality.py for gold sets where some questions
have no grade-2 document (the corpus does not directly answer them):

  MRR over ANSWERABLE questions only (at least one grade-2 document), and
  recall@k / precision@k over the top-k DISTINCT documents, so several chunks
  from one document do not use up the k slots;
  chunk precision@k (share of the top-k chunks whose document is gold) and
  nDCG@k, for every k in --k (default 5,10,20); and, when retrieval_quality.py
  --grade has cached passage grades for the corpus, passage precision@k and
  nDCG@k from those grades (no new grading).

    uv run python scripts/bench/retrieval_quality_extra.py benchmarks/<date> --corpus websearch3x \
        --gold eval_questions/websearch3x_gold_docs.json --doc-keys eval_questions/websearch3x_doc_keys.json \
        --arms deepinfra-ws2@eval_nogeo [--k 5,10,20] [--grader gpt-4.1-mini]
"""
from __future__ import annotations

import asyncio
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from retrieval_quality import _arg, _pkey, load_runs, mrr, ndcg, resolve


def main(argv: list[str]) -> int:
    bench = Path(argv[0])
    corpus = _arg(argv, "--corpus")
    gold_all = json.loads(Path(_arg(argv, "--gold")).read_text())
    keys = json.loads(Path(_arg(argv, "--doc-keys")).read_text())
    ks = [int(k) for k in _arg(argv, "--k", "5,10,20").split(",")]
    gfile = bench / f"{corpus}_passage_grades.{_arg(argv, '--grader', 'gpt-4.1-mini')}.json"
    grades = json.loads(gfile.read_text()) if gfile.exists() else {}
    out = {}
    for spec in _arg(argv, "--arms").split(","):
        arm, _, sub = spec.partition("@")
        runs = load_runs(bench, arm, corpus, sub or "eval")
        asyncio.run(resolve(arm, corpus, runs, keys))
        rows: dict[str, list[dict]] = {}
        for r in runs:
            gold = gold_all.get(r["set"], {}).get(r["qid"]) or {}
            if not gold:
                continue
            seen: list[str] = []
            for it in r["items"]:
                d = it.get("doc") if it.get("kind") == "chunk" else None
                if d and d not in seen:
                    seen.append(d)
            chunks = [it for it in r["items"] if it.get("kind") == "chunk"]
            cdocs = [it.get("doc") for it in chunks]
            pg = [grades.get(_pkey(r["question"], it.get("text") or ""), 0) for it in chunks] if grades else None
            m = {}
            for k in ks:
                m[f"distinct_doc_recall@{k}"] = len(set(seen[:k]) & set(gold)) / len(gold)
                m[f"distinct_doc_precision@{k}"] = (len(set(seen[:k]) & set(gold)) / len(seen[:k])) if seen else 0.0
                m[f"doc_ndcg@{k}"] = ndcg([gold.get(d, 0) for d in seen], list(gold.values()), k=k)
                m[f"chunk_precision@{k}"] = (sum(1 for d in cdocs[:k] if d in gold) / len(cdocs[:k])) if cdocs else 0.0
                if pg is not None:
                    m[f"psg_precision@{k}"] = sum(1 for g in pg[:k] if g >= 1) / max(1, len(pg[:k]))
                    m[f"psg_ndcg@{k}"] = ndcg(pg, pg, k=k)
            m["distinct_docs_retrieved"] = len(seen)
            m["chunks_retrieved"] = len(cdocs)
            if any(v == 2 for v in gold.values()):
                m["doc_mrr_answerable"] = mrr([gold.get(d, 0) == 2 for d in seen])
            rows.setdefault(f"{r['set']}/{r['mode']}", []).append(m)
        out[spec] = {}
        for key, ms in rows.items():
            agg = {"n_runs": len(ms), "n_answerable_runs": sum("doc_mrr_answerable" in m for m in ms)}
            for f in sorted({f for m in ms for f in m}):
                vals = [m[f] for m in ms if m.get(f) is not None]
                agg[f] = round(statistics.mean(vals), 3) if vals else None
            out[spec][key] = agg
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
