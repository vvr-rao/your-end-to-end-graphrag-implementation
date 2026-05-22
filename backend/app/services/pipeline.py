"""Pipeline orchestration — composes the I/O, merge, LLM, and export stages.

Each public entry point corresponds to one CLI subcommand. They all share
the same return contract: produce a new version folder under output_root
and return its Path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.app.services import folder_io, ontology_export, ontology_io, ontology_merge, versioning


def run_merge(
    *,
    input_ontologies: list[Path],
    output_root: Path,
) -> Path:
    """`merge` subcommand: deterministic multi-ontology consolidation. Zero LLM."""
    output_root.mkdir(parents=True, exist_ok=True)
    version_dir = versioning.new_version_dir(output_root, "merge")

    print(f"[merge] expanding inputs: {[str(p) for p in input_ontologies]}")
    with ontology_io.enumerate_inputs(input_ontologies) as bundle:
        sources = bundle.sources
        print(f"[merge] found {len(sources)} ontology file(s)")
        loaded = ontology_merge.merge_sources(sources)

    counts = folder_io.count_entities(loaded)
    print(
        f"[merge] merged dict: {counts['classes']} classes, "
        f"{counts['object_properties']} object properties, "
        f"{counts['data_properties']} data properties, "
        f"{counts['instances']} instances"
    )

    json_path = folder_io.write_merged_json(version_dir, loaded)
    print(f"[merge] wrote {json_path.name} ({json_path.stat().st_size:,} bytes)")
    # `consume_dict=True`: drop in-memory dict entries as they're emitted to
    # the OWL. After write_merged_json the dict has already been persisted,
    # nothing further needs it, and on HP-scale merges the freed memory is
    # the difference between OOM and success.
    owl_path = ontology_export.write_owl(
        loaded, version_dir / folder_io.MERGED_OWL, consume_dict=True
    )
    print(f"[merge] wrote {owl_path.name} ({owl_path.stat().st_size:,} bytes)")

    versioning.write_manifest(
        version_dir,
        operation="merge",
        input_ontologies=list(input_ontologies),
    )
    versioning.write_stats(version_dir, {"counts": counts})
    versioning.ensure_audit_log(version_dir)
    return version_dir


# ---- LLM-using stages: implemented in a follow-up step in this same module.
# Stubs here so the CLI surface stays complete from day one.


async def run_prune(**kwargs: Any) -> Path:
    from backend.app.services.pipeline_llm import prune_only_async

    return await prune_only_async(**kwargs)


async def run_expand(**kwargs: Any) -> Path:
    from backend.app.services.pipeline_llm import expand_only_async

    return await expand_only_async(**kwargs)


async def run_prune_and_expand(**kwargs: Any) -> Path:
    from backend.app.services.pipeline_llm import prune_and_expand_async

    return await prune_and_expand_async(**kwargs)


async def run_build(**kwargs: Any) -> Path:
    from backend.app.services.pipeline_llm import build_async

    return await build_async(**kwargs)
