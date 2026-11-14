"""``extract-entities --structured-data DIR``: runs after entity extraction.

1. Metadata: read from the cache written by ``prune-expand --structured-data``
   (zero LLM calls). Only a table missing from the cache is profiled here,
   and its concepts map to EXISTING classes only (Phase 2 adds no classes).
2. Load each table into ``graphrag."<table>"`` and connect its TableMetadata
   node to ontology classes and to existing entities (``db_load``).
3. Refine (``refine``): measure each loaded table's structure, define its
   measures, test every stored query on the data (one LLM call per table;
   none when the cached metadata is already refined).
"""

from __future__ import annotations

from pathlib import Path

from backend.app.services.structured.cache import MetadataCache
from backend.app.services.structured.db_load import LoadSummary, load_and_link
from backend.app.services.structured.profiler import ClassIndex, profile_folder


async def _db_class_index() -> ClassIndex:
    from sqlalchemy import text

    from backend.app.db.session import session_scope

    async with session_scope() as s:
        rows = (await s.execute(text(
            "SELECT iri, label FROM graphrag.ontology_classes WHERE label IS NOT NULL"))).all()
    return ClassIndex({iri: {"labels": [label]} for iri, label in rows})


async def run_structured_load(
    folder: Path,
    *,
    limit: int | None = None,
    concurrency: int = 4,
    refresh: bool = False,
    refine: bool = True,
    link_entities: bool = True,
    router=None,
    embed=None,
) -> tuple[dict, LoadSummary]:
    from backend.app.services.llm_router import LLMRouter

    folder = Path(folder)
    router = router or LLMRouter()
    cache = MetadataCache()
    index = await _db_class_index()
    run = await profile_folder(folder, router, cache=cache, class_index=index,
                               concurrency=concurrency, limit=limit, refresh=refresh)
    for name, err in run.failed:
        print(f"[structured] WARN {name}: {err}")
    if embed is None:
        from backend.app.db.embedding_guard import ensure_embedding_model
        from backend.app.services.embeddings import Embedder

        embedder = Embedder()
        await ensure_embedding_model(embedder)
        embed = embedder.embed
    summary = await load_and_link(run, root_name=folder.resolve().name, cache=cache, embed=embed,
                                  link_entities=link_entities)
    refined = []
    if refine:
        from backend.app.services.structured.refine import refine_registered

        names = [n for n in summary.loaded + summary.unchanged]
        refined = await refine_registered(router, tables=names, refresh=refresh, cache=cache,
                                          embed=embed, link_entities=link_entities)
    profile = {"tables": len(run.tables), "refined": [r.get("table") for r in refined if "error" not in r], "from_cache": len(run.cached), "profiled": len(run.profiled),
               "llm_calls": run.llm_calls, "failed": run.failed}
    print(f"[structured] metadata: {profile['from_cache']} from cache, {profile['profiled']} profiled now "
          f"({run.llm_calls} LLM call(s)); tables: {len(summary.loaded)} loaded "
          f"({summary.rows} rows), {len(summary.unchanged)} unchanged, {len(summary.failed)} failed; "
          f"edges: {summary.class_edges} class, {summary.entity_edges} entity")
    return profile, summary
