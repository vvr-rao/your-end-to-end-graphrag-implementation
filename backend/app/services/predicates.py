"""Canonical predicate IRIs used across Phase 2 ingestion + artifact code.

The user's rule: **no new predicate types coined during ingestion**.
Every IRI written to `graph_relationships.predicate_iri` must come
from the ontology -- either the imported `ontology_object_properties`
side table, or a small allowlist of W3C/RDFS built-ins that owlready2
treats as schema primitives + doesn't surface as user properties.

This module is the single source of truth for the IRIs the Phase 2
services use. Adding a new edge type means either:
  - It's already in the merged ontology -> reference the IRI here and
    cite the side table row.
  - It's a W3C/RDFS built-in -> add it to `_BUILTIN_ALLOWLIST` with a
    comment explaining why owlready2 doesn't surface it.

`validate_predicate(session, iri)` checks both paths.
"""
from __future__ import annotations

from typing import Final

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db.models.ontology import OntologyObjectProperty

# ---------- RDFS / OWL schema built-ins ----------
# owlready2 surfaces ontology-declared properties but NOT the schema-
# level predicates like subClassOf/equivalentClass -- it treats them
# as primitives of the RDFS layer. The ontology-edge materializer in
# db_ontology_import.py emits these; we allowlist them here so the
# validator doesn't reject them.
RDFS_SUBCLASS_OF: Final = "http://www.w3.org/2000/01/rdf-schema#subClassOf"
OWL_EQUIVALENT_CLASS: Final = "http://www.w3.org/2002/07/owl#equivalentClass"
OWL_DISJOINT_WITH: Final = "http://www.w3.org/2002/07/owl#disjointWith"
RDF_TYPE: Final = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"

# ---------- VIAO (in `ontology_object_properties`) ----------
# Confirmed present in the v20260610T103310Z-prune-expand merge.
VIAO_NS = "https://veerla-ramrao.ai/ontology/intelligence-artifact"
VIAO_CHUNK_OF: Final = f"{VIAO_NS}#chunkOf"
VIAO_HAS_CHUNK: Final = f"{VIAO_NS}#hasChunk"
VIAO_DERIVED_FROM_CHUNK: Final = f"{VIAO_NS}#derivedFromChunk"
VIAO_DERIVED_FROM_DOCUMENT: Final = f"{VIAO_NS}#derivedFromDocument"
VIAO_SUMMARIZES: Final = f"{VIAO_NS}#summarizes"
VIAO_ASSERTS_ABOUT: Final = f"{VIAO_NS}#assertsAbout"
VIAO_HAS_INTELLIGENCE_ARTIFACT: Final = f"{VIAO_NS}#hasIntelligenceArtifact"
VIAO_REFERENCES_CHUNK: Final = f"{VIAO_NS}#referencesChunk"
VIAO_REFERENCES_ARTIFACT: Final = f"{VIAO_NS}#referencesArtifact"

# ---------- W3C Time (in `ontology_object_properties`) ----------
TIME_NS = "http://www.w3.org/2006/time"
TIME_HAS_TIME: Final = f"{TIME_NS}#hasTime"
TIME_INTERVAL_DURING: Final = f"{TIME_NS}#intervalDuring"
TIME_INTERVAL_CONTAINS: Final = f"{TIME_NS}#intervalContains"
TIME_MONTH_OF_YEAR: Final = f"{TIME_NS}#monthOfYear"
TIME_BEFORE: Final = f"{TIME_NS}#before"
TIME_AFTER: Final = f"{TIME_NS}#after"

# Built-ins that don't appear in `ontology_object_properties`.
_BUILTIN_ALLOWLIST: Final = frozenset({
    RDFS_SUBCLASS_OF,
    OWL_EQUIVALENT_CLASS,
    OWL_DISJOINT_WITH,
    RDF_TYPE,
})


async def validate_predicate(session: AsyncSession, predicate_iri: str) -> bool:
    """True if the IRI is a known schema built-in OR is present in
    `ontology_object_properties`. Used as a guard before inserting
    into `graph_relationships`.
    """
    if predicate_iri in _BUILTIN_ALLOWLIST:
        return True
    result = await session.execute(
        select(OntologyObjectProperty.id).where(
            OntologyObjectProperty.iri == predicate_iri
        )
    )
    return result.scalar_one_or_none() is not None
