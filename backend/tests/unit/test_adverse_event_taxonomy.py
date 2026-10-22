"""Canonical adverse-event hierarchy (user decision 2026-10-18)."""
from __future__ import annotations

import pytest

from backend.app.services import adverse_event_taxonomy as ae


@pytest.mark.parametrize("label,want", [
    # Folded into the root (AdverseReaction folds too -- user decision).
    ("AdverseEvent", "AdverseEvent"), ("Adverse events", "AdverseEvent"), ("AdverseReaction", "AdverseEvent"),
    ("AdverseReactions", "AdverseEvent"), ("DrugSideEffect", "AdverseEvent"), ("Side effect", "AdverseEvent"),
    ("AdverseDrugReaction", "AdverseEvent"),
    # Variants of a type.
    ("GastrointestinalAdverseReaction", "GastrointestinalAdverseEvent"), ("GI adverse event", "GastrointestinalAdverseEvent"),
    ("CardiacAdverseEvent", "CardiovascularAdverseEvent"), ("DermatologicalAdverseEvent", "SkinAdverseEvent"),
    ("Injection site reaction", "AdministrationSiteAdverseEvent"), ("Neurological side effect", "NervousSystemAdverseEvent"),
    # Left alone: not marked adverse, or a specific event.
    ("CardiovascularEvent", None), ("GastrointestinalDisorder", None), ("Nausea", None), ("FetalToxicity", None),
    ("Event", None), ("MajorAdverseCardiovascularEvent", None), ("SevereGIReaction", None),
])
def test_canonical(label, want) -> None:
    assert ae.canonical(label) == want


def test_normalize_renames_rewrites_and_adds_the_hierarchy() -> None:
    result = {
        "MATCH NOT FOUND": [
            {"LABEL": "Nausea", "DESCRIPTION": "d", "PARENT_LABEL": "GastrointestinalAdverseReaction"},
            {"LABEL": "Pruritus", "DESCRIPTION": "d", "PARENT_LABEL": "DermatologicalAdverseEvent"},
            {"LABEL": "AdverseReaction", "DESCRIPTION": "d", "PARENT_LABEL": "ClinicalConcept"},
            {"LABEL": "Recession", "DESCRIPTION": "d", "PARENT_LABEL": "EconomicConcept"},
        ],
        "MATCH NOT FOUND RELATIONS": [{"LABEL": "hasAdverseEvent", "DOMAIN": "Drug", "RANGE": "DrugSideEffect"}],
        "MATCH NOT FOUND INSTANCES": [{"LABEL": "nausea at week 4", "TYPE_LABEL": "GI adverse event"}],
    }
    stats = ae.normalize(result, existing_labels={"Event", "Drug"})
    by = {x["LABEL"]: x for x in result["MATCH NOT FOUND"]}
    assert by["Nausea"]["PARENT_LABEL"] == "GastrointestinalAdverseEvent"
    assert by["Pruritus"]["PARENT_LABEL"] == "SkinAdverseEvent"
    assert by["Recession"]["PARENT_LABEL"] == "EconomicConcept"           # untouched
    assert by["AdverseEvent"]["PARENT_LABEL"] == "Event"                  # renamed + re-parented
    assert by["GastrointestinalAdverseEvent"]["PARENT_LABEL"] == "AdverseEvent"   # added
    assert by["SkinAdverseEvent"]["PARENT_LABEL"] == "AdverseEvent"
    assert "CardiovascularAdverseEvent" not in by                          # only types in use
    assert result["MATCH NOT FOUND RELATIONS"][0]["RANGE"] == "AdverseEvent"
    assert result["MATCH NOT FOUND INSTANCES"][0]["TYPE_LABEL"] == "GastrointestinalAdverseEvent"
    assert stats["renamed"] == 1 and stats["added"] == 2


def test_normalize_does_not_duplicate_existing_classes_or_touch_unrelated_results() -> None:
    result = {"MATCH NOT FOUND": [{"LABEL": "Nausea", "DESCRIPTION": "d", "PARENT_LABEL": "GI adverse event"}]}
    ae.normalize(result, existing_labels={"Event", "AdverseEvent", "GastrointestinalAdverseEvent"})
    assert [x["LABEL"] for x in result["MATCH NOT FOUND"]] == ["Nausea"]
    plain = {"MATCH NOT FOUND": [{"LABEL": "Recession", "PARENT_LABEL": "EconomicConcept"}]}
    assert ae.normalize(plain, existing_labels=set()) == {"renamed": 0, "references": 0, "added": 0}
    assert len(plain["MATCH NOT FOUND"]) == 1
