#!/usr/bin/env python3
"""Map the retrieval thresholds calibrated on one embedding model onto another.

    uv run python scripts/bench/calibrate_embedding.py \
        --base-db openai_pharma12 --new-db openai_qwenemb_pharma12 \
        --new-model Qwen/Qwen3-Embedding-4B --go

Needs TWO copies of the SAME graph that differ only in their vectors (see
reembed_copy.py), so every comparison is between identical texts.

  L2 probe thresholds (qa.probe_dist_floor / probe_dist_ceiling / probe_rel_band):
    on-topic questions (the pharma audit + pharma_test sets) vs off-topic and
    nonsense queries; best-chunk L2 distance per query, embedded the way
    retrieval embeds probes (embed_query). Each threshold is mapped linearly
    between two anchors -- the on-topic median and the off-topic median -- so
    it keeps its position between "clearly on topic" and "clearly off topic".
    The relative band scales with the same gap.

  Cosine relation thresholds (qa.relationship_similarity,
  qa.relation_class_seed_similarity): relation probes parsed from the same
  questions (the retrieval's own question_relations step, base arm's model),
  embedded the way retrieval embeds them (embed), cosine to every edge.
  Each threshold maps to the same PERCENTILE of the pooled probe x edge
  distribution, and of the per-probe best-edge distribution (both reported;
  the pooled mapping is what a hop's admission test sees).

PAID: question_relations calls (gpt-4.1-mini class, ~$0.02) + embeddings.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "bench"))

OFF_TOPIC = [
    "Who won the 2018 FIFA World Cup?", "How do I bake sourdough bread at home?",
    "What is the capital of Australia?", "How does a car's transmission work?",
    "Best hiking trails in the Swiss Alps", "Explain how a Python decorator works",
    "What were the causes of the French Revolution?", "How do you tune a guitar?",
    "Which planets in the solar system have rings?", "What is the stock price history of Tesla?",
    "How to train a puppy to sit", "Who painted the Mona Lisa?",
    "What is the offside rule in soccer?", "How do solar panels generate electricity?",
    "Recommend a good science fiction novel", "What is the population of Brazil?",
    "How do airplanes stay in the air?", "Explain the rules of chess castling",
    "What does a central bank do when it raises interest rates?", "How to make a cappuccino",
]
NONSENSE = ["purple elephant quantum banana", "zxqv lorem flarb nint 42", "the of and to in is that",
            "asdf ghjk qwer tyui", "blue seven sideways tomorrow"]

BASE = {  # text-embedding-3-small values in config.yaml (qa.*)
    "qa.probe_dist_floor": 1.05, "qa.probe_dist_ceiling": 1.25, "qa.probe_rel_band": 0.10,
    "qa.relationship_similarity": 0.55, "qa.relation_class_seed_similarity": 0.40,
}


def _arg(argv, name, default=None):
    return argv[argv.index(name) + 1] if name in argv and argv.index(name) + 1 < len(argv) else default


def questions() -> list[str]:
    from backend.app.services.eval_judge import parse_question_file
    from run_audit_questions import QUESTIONS

    qs = [q for _, q, _ in QUESTIONS]
    qs += [q for q, gap in parse_question_file(ROOT / "eval_questions" / "pharma_test.txt") if not gap]
    return list(dict.fromkeys(qs))


async def load(db: str):
    import asyncpg
    import numpy as np

    import bench_db

    conn = await asyncpg.connect(bench_db.dsn(db))
    try:
        ch = await conn.fetch("SELECT embedding::text AS e FROM graphrag.chunks WHERE embedding IS NOT NULL")
        ed = await conn.fetch("SELECT embedding::text AS e FROM graphrag.graph_relationships WHERE embedding IS NOT NULL")
    finally:
        await conn.close()
    to = lambda rows: np.array([json.loads(r["e"]) for r in rows], dtype=np.float32)  # noqa: E731
    return to(ch), to(ed)


def pct_of(values, x) -> float:
    import numpy as np

    return float(np.mean(np.asarray(values) < x))


def at_pct(values, p) -> float:
    import numpy as np

    return float(np.quantile(np.asarray(values), p))


async def main_async(base_db: str, new_db: str, new_model: str, query_prefix: str | None = None) -> dict:
    import numpy as np

    from backend.app.services.embeddings import Embedder
    from backend.app.services.llm_router import LLMRouter
    from backend.app.services.retrieval import _question_relations, _relation_text

    qs = questions()
    router = LLMRouter()
    rels = await asyncio.gather(*[_question_relations(router, q) for q in qs])
    rel_texts = [_relation_text(r) for rs in rels for r in rs]
    print(f"{len(qs)} on-topic questions, {len(OFF_TOPIC)} off-topic, {len(NONSENSE)} nonsense, "
          f"{len(rel_texts)} relation probes")

    models = {
        "base": (base_db, Embedder(provider="openai", model="text-embedding-3-small", dim=1024, batch_size=100)),
        "new": (new_db, Embedder(provider="deepinfra", model=new_model, dim=1024, batch_size=100,
                                 query_prefix_override=query_prefix)),
    }
    m: dict = {}
    for tag, (db, emb) in models.items():
        C, E = await load(db)
        dist = {}
        for name, qset in (("on", qs), ("off", OFF_TOPIC), ("nonsense", NONSENSE)):
            Q = np.array(await emb.embed_query(qset), dtype=np.float32)
            d = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * (Q @ C.T)))
            dist[name] = np.sort(d, axis=1)
        R = np.array(await emb.embed(rel_texts), dtype=np.float32)
        cos = R @ E.T
        m[tag] = {
            "on_best": dist["on"][:, 0], "off_best": dist["off"][:, 0], "nonsense_best": dist["nonsense"][:, 0],
            "on_spread30": dist["on"][:, 29] - dist["on"][:, 0],
            "rel_pooled": cos.ravel(), "rel_best": cos.max(axis=1),
        }

    b, n = m["base"], m["new"]
    a_b, z_b = float(np.median(b["on_best"])), float(np.median(b["off_best"]))
    a_n, z_n = float(np.median(n["on_best"])), float(np.median(n["off_best"]))

    def lin(t):
        return a_n + (t - a_b) / (z_b - a_b) * (z_n - a_n)

    scale = (z_n - a_n) / (z_b - a_b)
    rec = {
        # floor = the same on-topic percentile as the base floor (1.05 = base p95)
        "qa.probe_dist_floor": round(at_pct(n["on_best"], pct_of(b["on_best"], BASE["qa.probe_dist_floor"])), 3),
        "qa.probe_dist_ceiling": round(lin(BASE["qa.probe_dist_ceiling"]), 3),
        "qa.probe_rel_band": round(BASE["qa.probe_rel_band"] * scale, 3),
    }
    rel_detail = {}
    for key in ("qa.relationship_similarity", "qa.relation_class_seed_similarity"):
        t = BASE[key]
        p_pool = pct_of(b["rel_pooled"], t)
        p_best = pct_of(b["rel_best"], t)
        rel_detail[key] = {"base": t, "base_pooled_percentile": round(p_pool, 4),
                           "mapped_pooled": round(at_pct(n["rel_pooled"], p_pool), 3),
                           "base_best_percentile": round(p_best, 4),
                           "mapped_best": round(at_pct(n["rel_best"], p_best), 3)}
        rec[key] = rel_detail[key]["mapped_pooled"]

    def summ(x):
        return {k: round(float(v), 3) for k, v in
                (("min", np.min(x)), ("p05", np.quantile(x, .05)), ("p50", np.median(x)),
                 ("p95", np.quantile(x, .95)), ("max", np.max(x)))}

    report = {
        "new_model": new_model,
        "distributions": {tag: {k: summ(v) for k, v in d.items() if k != "rel_pooled"} | {"rel_pooled": summ(d["rel_pooled"])}
                          for tag, d in m.items()},
        "anchors": {"base_on_median": a_b, "base_off_median": z_b, "new_on_median": a_n, "new_off_median": z_n},
        "separation": {tag: {"on_max": float(np.max(d["on_best"])), "off_min": float(np.min(d["off_best"])),
                             "overlap": bool(np.max(d["on_best"]) >= np.min(d["off_best"]))} for tag, d in m.items()},
        "relation_mapping": rel_detail,
        "recommended_profile": rec,
    }
    return report


def main(argv: list[str]) -> int:
    if "--go" not in argv:
        print(__doc__)
        return 2
    from run_arm import arm_env

    # The base arm's environment supplies the question_relations model.
    os.environ.update(arm_env(_arg(argv, "--base-arm", "openai"), _arg(argv, "--corpus", "pharma12"),
                              ROOT / "benchmarks" / "2026-10-17" / _arg(argv, "--base-arm", "openai") / "pharma12"))
    qp = _arg(argv, "--query-prefix")
    rep = asyncio.run(main_async(_arg(argv, "--base-db"), _arg(argv, "--new-db"), _arg(argv, "--new-model"),
                                 qp.replace("\\n", "\n") if qp is not None else None))
    rep["query_prefix"] = qp
    out = Path(_arg(argv, "--out", str(ROOT / "benchmarks" / "2026-10-17" / "embedding_calibration.json")))
    out.write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=1))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
