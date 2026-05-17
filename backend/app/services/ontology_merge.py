"""Wrap `import_ontologies` with a thin facade and provenance attached.

The actual recursive-merge-by-IRI logic lives in the helpers and is reused
unchanged. This layer just normalizes the inputs (from ontology_io.enumerate_inputs)
and tags every entity with the source labels that contributed to it.
"""

from __future__ import annotations

from typing import Any

from backend.app.services.ontology_io import OntologySource, load_ontology_files


def merge_sources(sources: list[OntologySource]) -> dict[str, Any]:
    """Deterministic merge: parse each source, union by IRI, return canonical dict.

    Provenance: every entity gets a `sources` list naming which input files
    contributed (useful for the manifest and for debugging duplicate IRIs).
    """
    # If only one source, load it directly without going through the merge path
    # (avoids the source-list overhead but keeps the same return shape).
    loaded = load_ontology_files(sources)

    # Annotate every entity with the source list. Since `import_ontologies` loses
    # which-file-came-from-where, we re-attribute by reloading each source
    # individually and recording which IRIs it contributed. Single source case
    # short-circuits to avoid double work.
    if len(sources) == 1:
        _stamp_sources(loaded, sources[0].label)
        return loaded

    _attach_provenance(loaded, sources)
    return loaded


def _stamp_sources(loaded: dict[str, Any], label: str) -> None:
    for dict_name in (
        "classes_dict",
        "object_properties_dict",
        "data_properties_dict",
        "instances_dict",
    ):
        for entity in loaded.get(dict_name, {}).values():
            existing = entity.get("sources")
            if isinstance(existing, list):
                if label not in existing:
                    existing.append(label)
            else:
                entity["sources"] = [label]


def _attach_provenance(loaded: dict[str, Any], sources: list[OntologySource]) -> None:
    """Reload each source independently and record contributed IRIs.

    Expensive (each source parsed twice) — only used when there are 2+ sources
    and we need to attribute. For very large multi-source merges this could be
    optimized by threading provenance through `import_ontologies` itself,
    but that's a helper change we'd rather avoid for Phase 1.
    """
    from backend.app.services.ontology_io import load_ontology_files

    for src in sources:
        per_source = load_ontology_files([src])
        for dict_name in (
            "classes_dict",
            "object_properties_dict",
            "data_properties_dict",
            "instances_dict",
        ):
            target = loaded.get(dict_name, {})
            for iri in per_source.get(dict_name, {}):
                entity = target.get(iri)
                if entity is None:
                    continue
                existing = entity.get("sources")
                if isinstance(existing, list):
                    if src.label not in existing:
                        existing.append(src.label)
                else:
                    entity["sources"] = [src.label]
