"""Near-duplicate entity names share one comparison key."""
from __future__ import annotations

import pytest

from backend.app.services.db_entity_extract import (
    _dedup_key,
    _filter_entities,
    _humanize_class_name,
    _normalize_name,
)


@pytest.mark.parametrize("a,b", [
    ("SUSTAIN 6", "SUSTAIN-6"),
    ("Apnea Hypopnea Index", "ApneaHypopneaIndex"),
    ("Apnea-Hypopnea Index", "ApneaHypopneaIndex"),
    ("debt to equity ratio", "debt-to-equity ratio"),
    ("CYP 2C9", "CYP2C9"),
    ("tariff", "tariffs"),
    ("GLP-1 receptor agonist", "GLP-1 receptor agonists"),
    ("Hapag-Lloyd", "Hapag-Lloyd AG"),
    ("BioNTech", "BIONTECH"),
])
def test_same_thing_same_key(a: str, b: str) -> None:
    assert _dedup_key(a) == _dedup_key(b)


@pytest.mark.parametrize("a,b", [
    ("Phase 1 2", "Phase 12"),       # digits stay apart
    ("glass", "glas"),               # -ss is not a plural
    ("virus", "viru"),               # -us is not a plural
    ("analysis", "analysi"),         # -is is not a plural
    ("Type 1 Diabetes", "Type 2 Diabetes"),
])
def test_different_things_different_key(a: str, b: str) -> None:
    assert _dedup_key(a) != _dedup_key(b)


def test_callers_pass_either_raw_or_normalised_names() -> None:
    for name in ("SUSTAIN-6", "Apnea-Hypopnea Index", "export controls"):
        assert _dedup_key(name) == _dedup_key(_normalize_name(name))


def test_humanize_only_a_name_copied_from_a_class_label() -> None:
    labels = {"HypoxicBurden", "Biotechnology"}
    assert _humanize_class_name("HypoxicBurden", labels) == "Hypoxic Burden"
    assert _humanize_class_name("PayPal", labels) == "PayPal"          # a real name, not a label
    assert _humanize_class_name("Biotechnology", labels) == "Biotechnology"
    assert _humanize_class_name("Hypoxic Burden", labels) == "Hypoxic Burden"


def test_filter_entities_stores_the_humanized_name() -> None:
    iri = "https://x/o#HypoxicBurden"
    drops: dict[str, int] = {"no_name": 0}
    kept = _filter_entities(
        [{"canonical_name": "HypoxicBurden", "class_iri": iri, "confidence": 0.9}],
        {iri}, drops, [], cand_labels={iri: "HypoxicBurden"},
    )
    assert [e["canonical_name"] for e in kept] == ["Hypoxic Burden"]
