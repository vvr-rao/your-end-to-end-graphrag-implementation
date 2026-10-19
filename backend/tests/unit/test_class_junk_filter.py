"""Deterministic junk rules -- cases from the 27B web-search build (2026-10-17)
and OpenAI's proposals on the same corpus as the false-positive control."""
from __future__ import annotations

import pytest

from backend.app.services.class_junk_filter import FAMILY_KEEP, filter_reply, junk_reason, words


@pytest.mark.parametrize("label,reason", [
    ("Refinery2024", "number"), ("Refinery25bn", "number"), ("Refinery5", "number"), ("H7", "number"),
    ("2023", "number"), ("FortuneGlobal500", "number"), ("Q1Q22026", "number"),
    ("www.trulicity.com", "url"), ("info@example.org", "url"),
    ("MostNonet", "superlative"), ("LeastDuodecet", "superlative"),
    ("RefineryConvertedToFuelImportTerminalInYear", "phrase"), ("None", None), ("", "empty"),
])
def test_junk_labels_are_caught(label, reason) -> None:
    assert junk_reason(label) == reason


@pytest.mark.parametrize("label", [
    "Type2Diabetes", "MultipleEndocrineNeoplasiaType2", "Phase3ClinicalTrial", "Scope3Emission",
    "Tier1Supplier", "CO2", "PM2.5", "GLP1ReceptorAgonist", "SGLT2Inhibitor", "COVID19", "CYP3A4",
    "HER2", "5GNetwork", "Industry40", "Refinery", "FoodSecurity", "MostFavoredNationClause",
    "LowerBackPain", "HigherEducation", "Manufacturing",
])
def test_real_categories_pass(label) -> None:
    assert junk_reason(label) is None, label


def test_words_split_camel_and_acronyms() -> None:
    assert words("RareEarthElementLigandSeparation") == ["Rare", "Earth", "Element", "Ligand", "Separation"]
    assert words("HTTPServerError") == ["HTTP", "Server", "Error"]


def test_a_looping_family_keeps_its_first_members_only() -> None:
    labels = ["Refinery"] + [f"Refinery{w}" for w in (
        "Shutdown Expansion Construction Upgrade Fire Restart Sale Dispute Tender Commissioning "
        "Maintenance Closure Conversion Bid Cabinet Cracking Testing").split()] + ["FoodSecurity"]
    payload = {"MATCH NOT FOUND": [{"LABEL": label} for label in labels]}
    out, drops = filter_reply(payload)
    kept = [x["LABEL"] for x in out["MATCH NOT FOUND"]]
    assert kept[:FAMILY_KEEP] == labels[:FAMILY_KEEP] and "FoodSecurity" in kept
    assert len(kept) == FAMILY_KEEP + 1
    assert {d["reason"] for d in drops} == {"family"}


def test_a_normal_reply_is_untouched() -> None:
    payload = {"MATCH NOT FOUND": [{"LABEL": x} for x in ("Refinery", "RefineryComplex", "OilTerminal", "Tariff")]}
    out, drops = filter_reply(payload)
    assert drops == [] and len(out["MATCH NOT FOUND"]) == 4
    assert filter_reply(None) == (None, [])


def test_validity_judge_demotes_named_things_drops_junk_and_keeps_on_failure() -> None:
    import asyncio
    from types import SimpleNamespace

    from backend.app.services.class_junk_filter import run_validity_judge

    class Store:
        def __init__(self):
            self.status = {}

        async def iter_rows(self, kind, status, batch):
            yield [{"id": 1, "label": "Refinery"}, {"id": 2, "label": "Ozempic"}, {"id": 3, "label": "RefineryAbout"}]
            yield [{"id": 4, "label": "Tariff"}]

        async def set_status(self, ids, status, reason=None, merged_into=None):
            for i in ids:
                self.status[i] = status

    class Router:
        calls = 0

        async def chat(self, task, system, user):
            Router.calls += 1
            if "Tariff" in user:
                raise TimeoutError("judge timed out")
            return SimpleNamespace(text='{"verdicts": [{"id": 0, "verdict": "category"}, '
                                        '{"id": 1, "verdict": "named_thing"}, {"id": 2, "verdict": "junk"}]}')

    store = Store()
    stats = asyncio.run(run_validity_judge(store, Router(), log=lambda *_: None))
    assert store.status == {2: "instance", 3: "junk"}          # Refinery kept; Tariff kept on failure
    assert stats["failed_batches"] == 1 and stats["category"] == 1 and Router.calls == 2
