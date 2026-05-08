"""
Ontology helpers (relocated Phase 0).

- `ontology_parsing` (was `ontology_parsing_helpers_owlready2.py`):
  OWL → Python-dict extraction via owlready2 + rdflib.
- `ontology_pruning` (was `ontology_pruning_and_expansion_helpers.py`):
  pruning, semantic-role classification, LLM-output merging, hop-expansion.

Treated as first-class project code — refactor freely. Behavior parity with
kg_populationv5.ipynb is validated by integration tests, not line-by-line API parity.
"""
