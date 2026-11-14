"""``prune-expand --structured-data DIR``: runs after Stage 4.

Profiles the tables (cache first), mints the classes their columns, metrics
and filters need (through the same helper Stage 4 uses), writes the class IRIs
back into the metadata files and saves them to both cache tiers -- the run
folder copy lands in ``<run>/structured/``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.app.services.structured.cache import MetadataCache
from backend.app.services.structured.discovery import discover_tables
from backend.app.services.structured.profiler import add_classes_to_ontology, profile_folder


def check_folder(folder: Path, limit: int | None) -> int:
    """Fail fast (before any paid stage) on a missing or empty folder."""
    disc = discover_tables(Path(folder))
    n = len(disc.tables[:limit] if limit else disc.tables)
    if n == 0:
        raise SystemExit(f"--structured-data {folder}: no CSV/TSV/JSON/Parquet tables found")
    for rel, why in disc.skipped:
        print(f"[structured] skipping {rel}: {why}")
    print(f"[structured] {n} table(s) will be profiled after Stage 4 (cache first)")
    return n


def resolve_concurrency(explicit: int | None, app_cfg: dict[str, Any]) -> int:
    if explicit is not None:
        return max(1, explicit)
    return int((app_cfg.get("concurrency") or {}).get("table_profile") or 4)


async def run_structured_profile(
    *,
    folder: Path,
    ontology: dict,
    router,
    version_dir: Path,
    audit,
    base_iri: str,
    parent_iri: str | None,
    limit: int | None,
    concurrency: int,
    refresh: bool,
) -> dict:
    cache = MetadataCache(run_dir=version_dir / "structured")
    run = await profile_folder(folder, router, cache=cache, ontology=ontology,
                               concurrency=concurrency, limit=limit, refresh=refresh, audit=audit)
    created = add_classes_to_ontology(run, ontology, base_iri=base_iri,
                                      default_parent_iri=parent_iri, cache=cache)
    for name, err in run.failed:
        print(f"[structured] WARN {name}: {err}")
    summary = {
        "folder": str(folder), "tables": len(run.tables), "profiled": len(run.profiled),
        "from_cache": len(run.cached), "failed": [list(x) for x in run.failed],
        "lookups": len(run.lookups), "llm_calls": run.llm_calls,
        "created_classes": created,
        "items": {s: sum(len(m.get(s) or []) for m in run.tables)
                  for s in ("metrics", "filters", "queries")},
        "dropped_sql": sum(len(m.get("dropped") or []) for m in run.tables),
    }
    audit("table_class_added", {"created": created})
    print(f"[structured] {summary['tables']} table(s): {summary['profiled']} profiled, "
          f"{summary['from_cache']} from cache, {len(run.failed)} failed; "
          f"{len(created)} new class(es); {summary['lookups']} lookup(s); "
          f"metadata in {version_dir / 'structured'}")
    return summary
