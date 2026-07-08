"""Unit tests for entity-driven probe fan-out: the trigger predicate + the
entity_probes prompt shape/registration (no LLM/DB)."""

from __future__ import annotations

from backend.app.services.prompts import PROMPTS, entity_probes
from backend.app.services.retrieval import _use_entity_probes


def test_trigger_fires_for_comparison_multi_entity() -> None:
    assert _use_entity_probes("comparison", 5, 8) is True
    assert _use_entity_probes("enumeration", 3, 8) is True


def test_trigger_skips_when_not_applicable() -> None:
    assert _use_entity_probes("factoid", 5, 8) is False          # wrong intent
    assert _use_entity_probes("comparison", 2, 8) is False       # too few entities
    assert _use_entity_probes("comparison", 5, 0) is False       # fan-out disabled
    assert _use_entity_probes("", 5, 8) is False                 # no intent


def test_entity_probes_prompt_registered_and_shaped() -> None:
    assert "entity_probes" in PROMPTS
    assert PROMPTS["entity_probes"] is entity_probes
    sys_p, user_p = entity_probes(
        "Compare the side effects of Ozempic against its competitors",
        ["Ozempic", "Mounjaro", "Trulicity"],
    )
    assert "probes" in sys_p and "SAME ORDER" in sys_p   # one-per-entity contract
    assert "Mounjaro" in user_p and "Trulicity" in user_p
    assert "side effects" in user_p.lower()
