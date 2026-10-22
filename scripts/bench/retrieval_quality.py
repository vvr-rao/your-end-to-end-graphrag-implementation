#!/usr/bin/env python3
"""Retrieval recall + ranking quality per arm, from SAVED eval runs (no new retrieval).

    uv run python scripts/bench/retrieval_quality.py benchmarks/<date> --corpus pharma12 \
        --gold eval_questions/pharma12_gold_docs.json [--arms openai,deepinfra-b1,deepinfra-b2] \
        [--grade --grader gpt-4.1-mini --max-cost-usd 5]

Two views of the same ranked evidence lists (the chunk list is ranked 1..N;
artifacts are a separate list):

  DOCUMENT level (free): gold documents per question (grade 2 = holds the
  answer, 1 = supporting). recall@5/@10/all over the chunk list's documents
  (dedup, first occurrence), MRR of the first grade-2 document, nDCG@10 on
  graded documents, and chunk precision@10 (share of top-10 chunks whose
  document is gold).

  PASSAGE level (--grade, PAID): every retrieved chunk/artifact of every arm
  is pooled per question and graded 0/1/2 by ONE neutral model that never
  sees which arm retrieved it (TREC-style pooling). Per arm: precision@10,
  MRR of the first grade-2 passage, nDCG@10 (ideal = the best ordering of the
  passages that arm retrieved), and the count of grade-2 passages in the top 20.
  Arms index different chunk texts (their own summaries), so pooled recall
  across arms is not comparable; document recall is the recall measure.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "bench"))

GRADE_SYS = (
    "You grade search results for a question about a document collection. For EACH passage decide how useful "
    "it is for answering the question:\n"
    "  2 = directly states (part of) the answer\n"
    "  1 = on-topic background that helps, but does not state the answer\n"
    "  0 = irrelevant or about something else\n"
    "Judge each passage on its own text only. Return ONE JSON object on one line: "
    '{"grades": [{"id": <id>, "g": <0|1|2>}]} with an entry for every id.'
)


def _arg(argv, name, default=None):
    return argv[argv.index(name) + 1] if name in argv and argv.index(name) + 1 < len(argv) else default


def doc_key(file_name: str, keys: dict[str, str]) -> str | None:
    return next((k for k, fname in keys.items() if fname == file_name), None)


# --------------------------------------------------------------------------- #
# Metrics (pure)
# --------------------------------------------------------------------------- #

def ndcg(grades: list[int], ideal: list[int], k: int = 10) -> float | None:
    def dcg(gs):
        return sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(gs[:k]))
    best = dcg(sorted(ideal, reverse=True))
    return dcg(grades) / best if best else None


def mrr(flags: list[bool]) -> float:
    return next((1.0 / (i + 1) for i, f in enumerate(flags) if f), 0.0)


def doc_metrics(ranked_docs: list[str | None], gold: dict[str, int]) -> dict:
    seen: list[str] = []
    for d in ranked_docs:
        if d and d not in seen:
            seen.append(d)
    rel = set(gold)
    out = {}
    for k in (5, 10):
        top = {d for d in ranked_docs[:k] if d}
        out[f"doc_recall@{k}"] = len(top & rel) / len(rel)
    out["doc_recall_all"] = len(set(seen) & rel) / len(rel)
    out["doc_mrr"] = mrr([gold.get(d, 0) == 2 for d in seen])
    out["doc_ndcg@10"] = ndcg([gold.get(d, 0) for d in seen], list(gold.values()))
    top10 = ranked_docs[:10]
    out["chunk_precision@10"] = sum(1 for d in top10 if d in rel) / len(top10) if top10 else 0.0
    return out


def passage_metrics(grades: list[int]) -> dict:
    return {"psg_precision@10": sum(1 for g in grades[:10] if g >= 1) / max(1, len(grades[:10])),
            "psg_mrr": mrr([g == 2 for g in grades]),
            "psg_ndcg@10": ndcg(grades, grades),
            "psg_highly_relevant@20": sum(1 for g in grades[:20] if g == 2)}


# --------------------------------------------------------------------------- #
# Loading saved runs
# --------------------------------------------------------------------------- #

def load_runs(bench: Path, arm: str, corpus: str, sub: str = "eval") -> list[dict]:
    """[{set, qid, question, mode, run, items:[{kind, iri, text?, doc?}]}] in rank order."""
    ev = bench / arm / corpus / sub
    runs: list[dict] = []
    audit = ev / "audit_deep_research.jsonl"
    if audit.exists():
        for line in audit.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                runs.append({"set": "audit", "qid": str(r["id"]), "question": r["question"], "mode": "deep_research",
                             "run": 0, "items": [{"iri": i} for i in r["evidence_iris"] if i]})
    # One file per question set and mode, <set>_<mode>.json; the set name keys
    # the gold file (pharma_test, test_questions_websearch_corpus, ...).
    for mode in ("simple_qa", "deep_research"):
        for f in sorted(ev.glob(f"*_{mode}.json")):
            qset = f.name[: -len(f"_{mode}.json")]
            for q in json.loads(f.read_text())["results"]:
                for n, run in enumerate(q["runs"]):
                    items = sorted(run.get("evidence") or [], key=lambda e: int(e.get("rank") or 10 ** 6))
                    runs.append({"set": qset, "qid": q["question"], "question": q["question"], "mode": mode,
                             "run": n, "items": [{"iri": e.get("iri"), "kind": e.get("kind"), "text": e.get("text"),
                                                  "doc_title": e.get("document_title")} for e in items if e.get("iri")]})
    return runs


async def resolve(arm: str, corpus: str, runs: list[dict], keys: dict[str, str]) -> None:
    """Fill kind/doc/text for every item from the arm's database (read-only)."""
    import asyncpg
    import bench_db
    from run_arm import arm_slug

    conn = await asyncpg.connect(bench_db.dsn(f"{arm_slug(arm)}_{corpus}".replace("-", "_")[:60]))
    try:
        iris = sorted({it["iri"] for r in runs for it in r["items"]})
        chunk_rows = await conn.fetch(
            "SELECT c.chunk_identifier AS iri, c.text, d.file_name FROM graphrag.chunks c "
            "JOIN graphrag.documents d ON d.id = c.document_id WHERE c.chunk_identifier = ANY($1)", iris)
        art_rows = await conn.fetch(
            "SELECT a.artifact_identifier AS iri, a.text, array_agg(DISTINCT d.file_name) AS files "
            "FROM graphrag.intelligence_artifacts a "
            "LEFT JOIN graphrag.artifact_sources s ON s.artifact_id = a.id "
            "LEFT JOIN graphrag.chunks c ON c.id = s.chunk_id "
            "LEFT JOIN graphrag.documents d ON d.id = c.document_id "
            "WHERE a.artifact_identifier = ANY($1) GROUP BY a.artifact_identifier, a.text", iris)
    finally:
        await conn.close()
    chunks = {r["iri"]: r for r in chunk_rows}
    arts = {r["iri"]: r for r in art_rows}
    for r in runs:
        for it in r["items"]:
            if it["iri"] in chunks:
                c = chunks[it["iri"]]
                it.update(kind="chunk", text=it.get("text") or c["text"], doc=doc_key(c["file_name"], keys))
            elif it["iri"] in arts:
                a = arts[it["iri"]]
                docs = [doc_key(f, keys) for f in (a["files"] or []) if f]
                it.update(kind="artifact", text=it.get("text") or a["text"], docs=[d for d in docs if d])
            else:
                it.setdefault("kind", "relationship")


# --------------------------------------------------------------------------- #
# Pooled passage grading
# --------------------------------------------------------------------------- #

def _pkey(question: str, text: str) -> str:
    return hashlib.sha256((question + "\x00" + (text or "")[:1500]).encode()).hexdigest()[:20]


async def grade_pool(pool: dict[str, dict], model: str, max_cost: float, per_call: int = 15) -> tuple[dict, float]:
    from backend.app.core.config import get_settings
    from backend.app.services.llm_router import OpenAIProvider

    p = OpenAIProvider(api_key=get_settings().openai_api_key)
    by_q: dict[str, list[tuple[str, str]]] = {}
    for k, v in pool.items():
        by_q.setdefault(v["question"], []).append((k, v["text"]))
    batches = []
    for q, items in by_q.items():
        random.Random(7).shuffle(items)  # arm order never leaks through position
        for i in range(0, len(items), per_call):
            batches.append((q, items[i:i + per_call]))
    grades: dict[str, int] = {}
    spent = [0.0]
    sem = asyncio.Semaphore(8)

    async def one(q, items):
        if spent[0] >= max_cost:
            return
        body = "\n\n".join(f"[{n}] {(t or '')[:1500]}" for n, (_, t) in enumerate(items))
        async with sem:
            try:
                out = await p.chat(model=model, system=GRADE_SYS, user=f"QUESTION: {q}\n\nPASSAGES:\n{body}",
                                   temperature=0.0, max_tokens=1024, timeout=120, response_format="json_object")
            except Exception as exc:
                print(f"  grade call failed: {type(exc).__name__}")
                return
        spent[0] += out.cost_usd or 0
        try:
            for g in json.loads(out.text).get("grades", []):
                grades[items[int(g["id"])][0]] = max(0, min(2, int(g["g"])))
        except Exception:
            print("  unparseable grade reply")

    print(f"grading {len(pool)} pooled passages in {len(batches)} calls with {model} ...")
    await asyncio.gather(*[one(q, it) for q, it in batches])
    return grades, spent[0]


# --------------------------------------------------------------------------- #

def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    bench = Path(argv[0])
    corpus = _arg(argv, "--corpus", "pharma12")
    arms = _arg(argv, "--arms", "openai,deepinfra-b1,deepinfra-b2").split(",")
    gold_all = json.loads(Path(_arg(argv, "--gold")).read_text())
    keys = json.loads(Path(_arg(argv, "--doc-keys", str(Path(_arg(argv, "--gold")).with_name("pharma12_doc_keys.json")))).read_text())

    # "arm@subdir" scores another saved run of the same arm (same database),
    # e.g. openai-qwenemb@eval_cal vs openai-qwenemb@eval_uncal.
    data = {}
    for spec in arms:
        arm, _, sub = spec.partition("@")
        runs = load_runs(bench, arm, corpus, sub or "eval")
        asyncio.run(resolve(arm, corpus, runs, keys))
        data[spec] = runs

    grades: dict[str, int] = {}
    cost = 0.0
    if "--grade" in argv:
        pool = {}
        for runs in data.values():
            for r in runs:
                for it in r["items"]:
                    if it.get("kind") in ("chunk", "artifact") and it.get("text"):
                        pool.setdefault(_pkey(r["question"], it["text"]), {"question": r["question"], "text": it["text"]})
        # Grades are cached per grader: re-scoring with new runs only grades
        # passages not seen before, and earlier passages keep their grade.
        grader = _arg(argv, "--grader", "gpt-4.1-mini")
        cache = bench / f"{corpus}_passage_grades.{grader}.json"
        grades = json.loads(cache.read_text()) if cache.exists() else {}
        todo = {k: v for k, v in pool.items() if k not in grades}
        print(f"{len(pool)} pooled passages, {len(pool) - len(todo)} already graded")
        if todo:
            new, cost = asyncio.run(grade_pool(todo, grader, float(_arg(argv, "--max-cost-usd", "5"))))
            grades.update(new)
            cache.write_text(json.dumps(grades))

    report: dict = {"_grader": _arg(argv, "--grader", "gpt-4.1-mini") if grades else None,
                    "_grading_cost_usd": round(cost, 3), "arms": {}}
    for arm, runs in data.items():
        rows: dict[str, list[dict]] = {}
        for r in runs:
            gold = gold_all.get(r["set"], {}).get(r["qid"]) or {}
            chunks = [it for it in r["items"] if it.get("kind") == "chunk"]
            m: dict = {}
            if gold:
                m.update(doc_metrics([it.get("doc") for it in chunks], gold))
                arts = [it for it in r["items"] if it.get("kind") == "artifact"]
                adocs = {d for it in arts for d in it.get("docs", [])}
                m["artifact_doc_recall"] = len(adocs & set(gold)) / len(gold) if arts else 0.0
            if grades:
                g = [grades.get(_pkey(r["question"], it.get("text") or ""), 0) for it in chunks]
                m.update(passage_metrics(g))
                ga = [grades.get(_pkey(r["question"], it.get("text") or ""), 0)
                      for it in r["items"] if it.get("kind") == "artifact"]
                m["artifact_precision"] = sum(1 for x in ga if x >= 1) / len(ga) if ga else None
            rows.setdefault(f"{r['set']}/{r['mode']}", []).append(m)
        agg = {}
        for group, ms in rows.items():
            keys_ = sorted({k for m in ms for k in m})
            agg[group] = {"n": len(ms), **{k: round(statistics.mean(v for m in ms if (v := m.get(k)) is not None), 3)
                                         for k in keys_ if any(m.get(k) is not None for m in ms)}}
        report["arms"][arm] = agg
    out = Path(_arg(argv, "--out", str(bench / f"{corpus}_retrieval_quality.json")))
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=1))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
