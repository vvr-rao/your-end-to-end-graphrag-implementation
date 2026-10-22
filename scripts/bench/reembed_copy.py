#!/usr/bin/env python3
"""Re-embed a COPY of a benchmark database with another embedding model.

    createdb -T openai_pharma12 openai_qwenemb_pharma12        (done by the caller)
    uv run python scripts/bench/reembed_copy.py --arm openai-qwenemb --corpus pharma12 --go

Isolates the embedding model from everything else: same chunks, same graph,
same artifacts, same chat models -- only the vectors change. Each table is
re-embedded from the SAME text its builder embedded:

  chunks / intelligence_artifacts   the stored `text`
  ontology_classes                  db_ontology_import._build_embed_text(extra_metadata)
  entities                          "<name> -- <class label>"  (db_entity_extract)
  graph_relationships               db_relationship_embed.build_edge_text (embed-relationships)
  documents                         `text_summary` (the build embedded a compressed
                                    form only when a summary exceeded the input cap)

Then stamps graph_version_state.embedding_model with the new model. PAID
(embedding calls; ~3k texts on pharma12 is well under $0.01).
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "bench"))


def _json(v):
    import json

    return (json.loads(v) if isinstance(v, str) else v) or {}


def _arg(argv, name, default=None):
    return argv[argv.index(name) + 1] if name in argv and argv.index(name) + 1 < len(argv) else default


async def reembed() -> dict[str, int]:
    from sqlalchemy import text

    from backend.app.db.session import session_scope
    from backend.app.services.db_ontology_import import _build_embed_text
    from backend.app.services.db_relationship_embed import embed_relationships
    from backend.app.services.embeddings import Embedder

    emb = Embedder()
    counts: dict[str, int] = {}

    async def redo(name: str, select_sql: str, update_sql: str, build) -> None:
        async with session_scope() as s:
            rows = (await s.execute(text(select_sql))).mappings().all()
        rows = [(r["id"], build(r)) for r in rows]
        rows = [(i, t) for i, t in rows if t]
        for k in range(0, len(rows), 200):
            batch = rows[k:k + 200]
            vecs = await emb.embed([t for _, t in batch])
            async with session_scope() as s:
                for (i, _), v in zip(batch, vecs, strict=True):
                    await s.execute(text(update_sql), {"id": i, "v": "[" + ",".join(f"{x:.7f}" for x in v) + "]"})
        counts[name] = len(rows)
        print(f"  {name}: {len(rows)} re-embedded", flush=True)

    upd = "UPDATE graphrag.{t} SET embedding = CAST(:v AS vector) WHERE id = :id"
    await redo("chunks", "SELECT id, text FROM graphrag.chunks WHERE embedding IS NOT NULL",
               upd.format(t="chunks"), lambda r: r["text"])
    await redo("intelligence_artifacts", "SELECT id, text FROM graphrag.intelligence_artifacts WHERE embedding IS NOT NULL",
               upd.format(t="intelligence_artifacts"), lambda r: r["text"])
    await redo("ontology_classes", "SELECT id, extra_metadata FROM graphrag.ontology_classes WHERE embedding IS NOT NULL",
               upd.format(t="ontology_classes"), lambda r: _build_embed_text(_json(r["extra_metadata"])))
    await redo("entities",
               "SELECT e.id, e.name, c.label FROM graphrag.entities e "
               "LEFT JOIN graphrag.ontology_classes c ON c.id = e.class_id WHERE e.embedding IS NOT NULL",
               upd.format(t="entities"), lambda r: f"{r['name']} -- {r['label'] or ''}")
    await redo("documents", "SELECT id, text_summary FROM graphrag.documents WHERE embedding IS NOT NULL",
               upd.format(t="documents"), lambda r: r["text_summary"])
    async with session_scope() as s:
        await s.execute(text("UPDATE graphrag.graph_version_state SET embedding_model = :m WHERE id = 1"),
                        {"m": emb.model})
        await s.execute(text("UPDATE graphrag.graph_relationships SET embedding = NULL WHERE embedding IS NOT NULL"))
    rel = await embed_relationships(only_missing=True, embedder=emb)
    counts["graph_relationships"] = rel.embedded
    print(f"  graph_relationships: {rel.embedded} re-embedded", flush=True)
    print(f"stamped embedding_model = {emb.model}; embedding cost ${emb.total_cost_usd:.4f}")
    return counts


def main(argv: list[str]) -> int:
    arm, corpus = _arg(argv, "--arm"), _arg(argv, "--corpus")
    if not (arm and corpus):
        print(__doc__)
        return 2
    from run_arm import arm_env

    out = ROOT / "benchmarks" / _arg(argv, "--date", "2026-10-17") / arm / corpus
    env = arm_env(arm, corpus, out)
    print(f"re-embedding {env['DATABASE_URL'].rsplit('/', 1)[-1]} with models {env['MODELS_CONFIG_PATH']}")
    if "--go" not in argv:
        print("(dry run; add --go)")
        return 0
    os.environ.update(env)
    asyncio.run(reembed())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
