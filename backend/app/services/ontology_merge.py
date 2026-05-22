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
    contributed. Stamping happens inline during the per-file load via
    `load_ontology_files(..., stamp_provenance=True)` -- no second pass
    needed. The old approach reloaded each source separately just to
    attribute IRIs, which cost ~ N * (catalog parse + strip imports + load)
    and made FIBO-scale merges (222 sources) infeasible.
    """
    return load_ontology_files(sources, stamp_provenance=True)
