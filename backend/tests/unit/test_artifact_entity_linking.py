"""Artifacts link to the entities their text names -- by full name OR a stored
alias -- without guessing on ambiguous short forms."""
from __future__ import annotations

from backend.app.services.db_artifact_gen import match_artifact_entities

FTX = {"entity_id": 1, "canonical_name": "FTX Trading Ltd.", "aliases": ["FTX"]}
SBF = {"entity_id": 2, "canonical_name": "Sam Bankman-Fried", "aliases": ["Bankman-Fried", "SBF"]}
APPLE = {"entity_id": 3, "canonical_name": "Apple Inc.", "aliases": ["Apple"]}
TRAVIS = {"entity_id": 4, "canonical_name": "Travis Kelce", "aliases": ["Kelce"]}
JASON = {"entity_id": 5, "canonical_name": "Jason Kelce", "aliases": ["Kelce"]}
SEC = {"entity_id": 6, "canonical_name": "Securities and Exchange Commission", "aliases": ["SEC"]}


def test_full_name_still_links():
    assert match_artifact_entities("FTX Trading Ltd. filed for bankruptcy.", [FTX]) == {1}


def test_alias_links_where_full_name_is_absent():
    # The measured misses: 56 of 59 "FTX" artifacts were unlinked.
    assert match_artifact_entities("FTX was no Ponzi scheme.", [FTX]) == {1}
    assert match_artifact_entities(
        "Bankman-Fried directed employees to conceal the flow of money.", [SBF]) == {2}
    assert match_artifact_entities("FTX\u2019s coffers", [FTX]) == {1}   # curly possessive


def test_whole_words_only():
    assert match_artifact_entities("She bought pineapple juice.", [APPLE]) == set()
    assert match_artifact_entities("AFTX is unrelated.", [FTX]) == set()


def test_short_forms_are_case_sensitive():
    assert match_artifact_entities("the sec of the quarter", [SEC]) == set()
    assert match_artifact_entities("The SEC alleges fraud.", [SEC]) == {6}


def test_shared_short_form_links_neither_but_full_names_still_do():
    both = [TRAVIS, JASON]
    assert match_artifact_entities("Kelce scored twice.", both) == set()
    assert match_artifact_entities("Travis Kelce scored twice.", both) == {4}


def test_only_candidates_in_scope_are_linked():
    assert match_artifact_entities("FTX and Apple", [APPLE]) == {3}
    assert match_artifact_entities("", [FTX]) == set()
