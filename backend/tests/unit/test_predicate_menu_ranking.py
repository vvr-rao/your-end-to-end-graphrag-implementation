"""The relationship predicate menu: ranked by fit, not cut alphabetically;
every declared domain/range counts; reversed claims fit the swapped pair.

Measured on multihop-rag-subset: the menu was cut at 40 in label order in 84
of 89 chunks, so predicates past "h" were almost never offered; 5 of 123 type
drops failed only because the first of several declared domains was checked;
26 fitted with subject and object swapped.
"""
from __future__ import annotations

from backend.app.services.db_entity_extract import _rank_predicates, _types_fit

PERSON, AGENT, ORG, CLUB, PIPE = "Person", "Agent", "Organization", "FootballClub", "PipelineOperator"
UP = {PERSON: {PERSON, AGENT}, ORG: {ORG, AGENT}}
LINES = {PERSON: {PERSON, AGENT}, ORG: {ORG, AGENT, CLUB, PIPE}}
DEPTH = {PERSON: 2, AGENT: 1, ORG: 2, CLUB: 3, PIPE: 3}


def _names(ranked):
    return [r[0] for r in ranked]


def test_exact_class_fit_ranks_before_ancestor_and_descendant_fits() -> None:
    pool = [
        ("aaa_generic", None, [AGENT], [AGENT]),       # ancestors both ends
        ("zzz_worksFor", None, [PERSON], [ORG]),       # exact both ends
        ("mmm_hasPipeline", None, [PIPE], [PERSON]),   # descendant-only domain
    ]
    ranked = _rank_predicates(pool, {PERSON, ORG}, UP, LINES, DEPTH)
    assert _names(ranked) == ["zzz_worksFor", "aaa_generic", "mmm_hasPipeline"]


def test_alphabetical_position_no_longer_decides_what_is_offered() -> None:
    pool = [(f"a{i:02d}", None, [AGENT], [AGENT]) for i in range(50)]
    pool.append(("worksFor", None, [PERSON], [ORG]))
    ranked = _rank_predicates(pool, {PERSON, ORG}, UP, LINES, DEPTH)
    assert _names(ranked)[0] == "worksFor"


def test_predicate_no_class_pair_can_use_is_not_offered() -> None:
    pool = [("playsFor", None, [CLUB], ["Stadium"])]
    assert _rank_predicates(pool, {PERSON, ORG}, UP, LINES, DEPTH) == []


def test_best_fitting_pair_is_reported_for_the_prompt() -> None:
    pool = [("created", None, [ORG, AGENT], ["Token"])]
    lines = {**LINES, "Token": {"Token"}}
    (row,) = _rank_predicates(pool, {PERSON, ORG, "Token"}, UP, lines, DEPTH)
    assert row[4] in (ORG, AGENT) and row[5] == "Token"


def test_any_declared_domain_satisfies_the_type_check() -> None:
    created = {"domain_iris": [ORG, AGENT], "range_iris": ["Token"]}
    anc = {PERSON: {PERSON, AGENT}, "Token": {"Token"}}
    # Only the first domain used to be checked, rejecting a Person creator.
    assert _types_fit(created, PERSON, "Token", anc)


def test_reversed_claim_fits_only_when_swapped() -> None:
    coached_by = {"domain_iris": [ORG], "range_iris": [PERSON]}
    anc = {PERSON: {PERSON, AGENT}, CLUB: {CLUB, ORG, AGENT}}
    assert not _types_fit(coached_by, PERSON, CLUB, anc)   # "Pochettino coachedBy Chelsea"
    assert _types_fit(coached_by, CLUB, PERSON, anc)       # swapped


def test_unknown_predicate_never_fits() -> None:
    assert not _types_fit(None, PERSON, ORG, {})
