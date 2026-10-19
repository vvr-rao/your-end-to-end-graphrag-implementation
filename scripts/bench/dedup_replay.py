#!/usr/bin/env python3
"""Replay saved Stage-2 class proposals through junk filter + staging store +
nearest-neighbour candidates -- no LLM calls (embeddings only, ~cents).

    DATABASE_URL=postgresql://bench:bench@localhost:55433/ontology_build_dev \
    MODELS_CONFIG_PATH=config/models.deepinfra.example.yaml \
    uv run python scripts/bench/dedup_replay.py --audit <llm_audit.jsonl> --merge <merge folder> \
        --run-id replay-ws-attempt2 [--k 10] [--out report.json]

Reports: proposals in, junk dropped by rule, distinct folded labels, candidate
pairs per cosine band (class-class and class-existing) with examples per band
-- the input for choosing auto-merge / judge / skip thresholds -- the judge
workload those thresholds imply, and peak RSS.
"""
from __future__ import annotations

import asyncio
import json
import resource
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

BANDS = [(0.95, 1.01), (0.90, 0.95), (0.85, 0.90), (0.80, 0.85), (0.75, 0.80)]


def _arg(argv, name, default=None):
    return argv[argv.index(name) + 1] if name in argv and argv.index(name) + 1 < len(argv) else default


def band_of(c: float) -> str | None:
    for lo, hi in BANDS:
        if lo <= c < hi:
            return f"{lo:.2f}-{min(hi, 1.0):.2f}"
    return None


async def main_async(argv: list[str]) -> dict:
    from backend.app.services.class_junk_filter import filter_reply
    from backend.app.services.embeddings import Embedder
    from backend.app.services.ontology_build_store import BuildStore

    run = BuildStore(_arg(argv, "--run-id", "replay"))
    await run.clear()
    t0 = time.monotonic()
    proposals, drops = 0, Counter()
    for i, line in enumerate(Path(_arg(argv, "--audit")).open()):
        d = json.loads(line)
        if d.get("task") != "class_proposal":
            continue
        payload = d["payload"]
        proposals += len(payload.get("MATCH NOT FOUND") or [])
        payload, dropped = filter_reply(payload)
        drops.update(x["reason"] for x in dropped)
        await run.upsert("class", [{"label": x.get("LABEL"), "parent": x.get("PARENT_LABEL"),
                                    "description": x.get("DESCRIPTION")}
                                   for x in payload.get("MATCH NOT FOUND") or [] if isinstance(x, dict)],
                         chunk_ref=int(d.get("chunk_idx") or i))
    merge = json.loads((Path(_arg(argv, "--merge")) / "merged.json").read_text())
    classes = merge.get("classes_dict", merge)
    await run.upsert("existing_class", [
        {"label": (rec.get("labels") or [rec.get("name")])[0], "iri": iri,
         "description": " ".join((rec.get("comments") or []) + (rec.get("descriptions") or []))[:300]}
        for iri, rec in classes.items() if (rec.get("labels") or rec.get("name"))])
    t_load = time.monotonic() - t0
    emb = Embedder()
    n_emb = await run.embed_missing(emb)
    t_emb = time.monotonic() - t0 - t_load
    k = int(_arg(argv, "--k", "10"))
    report: dict = {"proposals_in": proposals, "junk_dropped": dict(drops), "store": await run.counts(),
                    "embedded": n_emb, "embedding_usd": round(emb.total_cost_usd, 4)}
    for to_kind in ("class", "existing_class"):
        hist, examples = Counter(), {}
        pairs_all = []
        async for batch in run.neighbours("class", to_kind, k=k, min_cosine=BANDS[-1][0]):
            pairs_all += batch
        rows = await run.rows(list({p.a_id for p in pairs_all} | {p.b_id for p in pairs_all}))
        for p in pairs_all:
            b = band_of(p.cosine)
            hist[b] += 1
            ex = examples.setdefault(b, [])
            if len(ex) < 12 and len(pairs_all) and hash((p.a_id, p.b_id)) % 7 == 0:
                ex.append(f"{rows[p.a_id]['label']} ~ {rows[p.b_id]['label']} ({p.cosine:.3f})")
        report[f"pairs_class_vs_{to_kind}"] = {b: hist[b] for b in sorted(hist, reverse=True)}
        report[f"examples_class_vs_{to_kind}"] = examples
    report["seconds"] = {"load_and_filter": round(t_load), "embed": round(t_emb),
                         "total": round(time.monotonic() - t0)}
    report["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    return report


def main(argv: list[str]) -> int:
    if "--audit" not in argv or "--merge" not in argv:
        print(__doc__)
        return 2
    rep = asyncio.run(main_async(argv))
    out = _arg(argv, "--out")
    if out:
        Path(out).write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
