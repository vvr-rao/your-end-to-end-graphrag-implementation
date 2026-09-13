"""A relationship's `predicate_iri` that mis-serialises a MENU predicate is
recovered; anything else is still dropped.

Measured on multihop-rag-subset: the merged ontology stores predicate IRIs
lowercased (`merged#employedby`) under camelCase labels (`employedBy`), and
gpt-4.1-mini rewrote 127 of 590 IRIs to match the label -- all dropped as
bad_predicate before this fix.
"""
from __future__ import annotations

from backend.app.services.db_entity_extract import (
    _predicate_menu_index,
    _resolve_menu_iri,
)

M = "https://veerla-ramrao.ai/ontology/merged#"
FOAF = "http://xmlns.com/foaf/0.1/"

MENU = [
    {"iri": M + "employedby", "label": "employedBy"},
    {"iri": M + "collaborateswith", "label": "collaboratesWith"},
    {"iri": M + "ceoof", "label": "ceoOf"},
]


def _resolve(raw: str, menu=MENU) -> str | None:
    return _resolve_menu_iri(raw, _predicate_menu_index(menu))


def test_exact_menu_iri_is_unchanged() -> None:
    assert _resolve(M + "ceoof") == M + "ceoof"


def test_camelcased_iri_resolves_to_the_menu_predicate() -> None:
    assert _resolve(M + "collaboratesWith") == M + "collaborateswith"


def test_label_emitted_as_iri_resolves() -> None:
    assert _resolve("employedBy") == M + "employedby"


def test_wrong_namespace_same_local_name_resolves_to_menu() -> None:
    assert _resolve(FOAF + "employedBy") == M + "employedby"


def test_predicate_not_on_menu_is_not_invented() -> None:
    assert _resolve(M + "fatherOf") is None
    assert _resolve(M + "acquires") is None
    assert _resolve("") is None


def test_local_name_shared_by_two_menu_predicates_is_not_resolved() -> None:
    menu = [
        {"iri": FOAF + "fundedBy", "label": "fundedBy"},
        {"iri": M + "fundedby", "label": "fundedBy"},
    ]
    assert _resolve("fundedBy", menu) is None
    assert _resolve(M + "FundedBy", menu) is None
    # An exact menu IRI is still accepted by the caller before recovery runs.
