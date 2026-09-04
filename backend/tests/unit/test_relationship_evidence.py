"""Entity->entity edges reaching the answer, and the alias guard that stopped
one bad pair from poisoning every probe.

Both defects were found by running real questions against a 60-doc corpus
whose graph held 132 verified entity->entity edges:

  - "Which teams defeated which other teams?" answered "the retrieved evidence
    contains no information" while 14 defeated/advancedOver edges sat in the
    graph. Cause: `evidence.append()` was only ever called with kind="chunk"
    and kind="artifact", so a triple had no way into the prompt. RRF is why --
    it fuses `list[list[UUID]]`, one flat id space, so entities are projected
    down to chunks BEFORE fusion and the relationship is dropped there.

  - That same question mined `teams -> Miranda Sissons` and rewrote all four
    probes as "which teams (Miranda Sissons) defeated ...".
"""
from __future__ import annotations

import pytest


# --------------------------------------------------------------------------- #
# Alias guard
# --------------------------------------------------------------------------- #


def test_short_lowercase_noun_cannot_alias_a_proper_name() -> None:
    """The exact pair that poisoned retrieval."""
    from backend.app.services.alias_mining import make_pair

    assert make_pair("Miranda Sissons", "teams", "parenthetical") is None
    assert make_pair("teams", "Miranda Sissons", "parenthetical") is None


def test_the_guard_is_symmetric_over_argument_order() -> None:
    from backend.app.services.alias_mining import make_pair

    assert make_pair("Arquimides Ordonez", "leg", "parenthetical") is None
    assert make_pair("leg", "Arquimides Ordonez", "parenthetical") is None


def test_generic_drug_names_still_pair_with_their_brand() -> None:
    """The guard must not break the case alias mining exists for: a long
    lowercase coinage is a real alias, not a common noun."""
    from backend.app.services.alias_mining import make_pair

    assert make_pair("MOUNJARO", "tirzepatide", "parenthetical") is not None
    assert make_pair("OZEMPIC", "semaglutide", "parenthetical") is not None


def test_acronym_expansions_survive() -> None:
    from backend.app.services.alias_mining import make_pair

    for a, b in [("MLS", "Major League Soccer"),
                 ("AFL", "Australian Football League"),
                 ("CSK", "Chennai Super Kings")]:
        assert make_pair(a, b, "parenthetical") is not None, (a, b)


def test_two_proper_names_still_pair() -> None:
    from backend.app.services.alias_mining import make_pair

    assert make_pair("Meta", "Facebook", "parenthetical") is not None


@pytest.mark.parametrize("surface,expected", [
    ("teams", True), ("leg", True), ("scores", True),
    ("tirzepatide", False),      # long lowercase coinage
    ("Teams", False),            # capitalized -> proper name
    ("MLS", False),              # acronym
    ("major league", False),     # multi-word handled elsewhere
])
def test_bare_common_noun_detection(surface: str, expected: bool) -> None:
    from backend.app.services.alias_mining import _is_bare_common_noun

    assert _is_bare_common_noun(surface) is expected


# --------------------------------------------------------------------------- #
# Relationship evidence
# --------------------------------------------------------------------------- #


def test_relationship_scope_supports_both_and_either() -> None:
    """`both` was too strict: it admitted 44 of 132 edges on the failing
    question and excluded every signedWith edge because the club endpoint was
    never reached. `either` recovers all three. Both forms must still be
    restricted to entity->entity edges."""
    import inspect

    from backend.app.services import retrieval_sql

    src = inspect.getsource(retrieval_sql.fetch_relationships_among_entities)
    assert " OR gr.target_node_id = ANY" in src      # either
    assert " AND gr.target_node_id = ANY" in src     # both
    assert "gr.source_node_type = 'entity'" in src
    assert "gr.target_node_type = 'entity'" in src


def test_relationship_scope_default_is_either() -> None:
    import inspect

    from backend.app.services import retrieval_sql

    sig = inspect.signature(retrieval_sql.fetch_relationships_among_entities)
    assert sig.parameters["scope"].default == "either"


def test_relationship_evidence_bypasses_rrf() -> None:
    """Triples are appended, NOT fused. Fusion ranks one flat id space, which
    is precisely what discarded them."""
    import inspect

    from backend.app.services import retrieval

    src = inspect.getsource(retrieval.retrieve_and_answer)
    rel = src.split("fetch_relationships_among_entities", 1)[1][:1200]
    assert "rrf_fuse" not in rel


def test_relationship_evidence_is_a_third_kind() -> None:
    import inspect

    from backend.app.services import retrieval

    src = inspect.getsource(retrieval.retrieve_and_answer)
    for kind in ('"kind": "chunk"', '"kind": "artifact"',
                 '"kind": "relationship"'):
        assert kind in src, kind


def test_relationship_kind_is_accepted_by_the_evidence_table() -> None:
    """retrieval_evidence has a CHECK on evidence_kind; persisting a run must
    not violate it."""
    from pathlib import Path

    sql = "\n".join(
        p.read_text() for p in Path("backend/app/db/migrations/versions").glob("*.py")
    )
    assert "'relationship'" in sql


def test_triple_renders_as_a_sentence_with_its_source_quote() -> None:
    """The synthesis prompt reads `text` for every evidence kind, so a raw
    triple would be unusable; and the quote is what makes the claim checkable."""
    import inspect

    from backend.app.services import retrieval

    src = inspect.getsource(retrieval.retrieve_and_answer)
    assert "rel['subject']} {rel['predicate']} {rel['object']}" in src
    assert "source text:" in src


def test_evidence_block_renders_a_relationship_item() -> None:
    """End of the chain: whatever we append must survive prompt rendering."""
    from backend.app.services.prompts import _format_evidence_block

    block = _format_evidence_block([{
        "kind": "relationship",
        "iri": "https://veerla-ramrao.ai/ontology/intelligence-artifact"
               "#Relationship_0001",
        "text": 'Chelsea defeated Newcastle United. (source text: "...")',
        "entities": ["Chelsea", "Newcastle United"],
    }])
    assert "viao:Relationship_0001" in block
    assert "Chelsea defeated Newcastle United" in block
    assert "about: Chelsea, Newcastle United" in block


# --------------------------------------------------------------------------- #
# Relationship RANKING and the RELATIVE probe floor
#
# Both from one failing question, "which players signed with or transferred to
# new teams", whose three `signedWith` edges were in the graph and correctly
# typed yet never reached the answer:
#
#   TRUNCATION -- relationship evidence was fetched straight to the cap ordered
#     by (support DESC, subject name ASC). Question-blind and alphabetical: the
#     signedWith edges sat at positions 8/45/68 of 132, so a cap of 25 kept one
#     and dropped two, filling the packet with marriedTo pairs instead.
#
#   ABSOLUTE FLOOR -- 4 of 5 probes were discarded at probe_dist_floor=1.05.
#     Measured bests: 1.0222, 1.0562, 1.0807, 1.0814, 1.0973 -- a 0.075-wide
#     band the fixed threshold sliced through the middle of.
# --------------------------------------------------------------------------- #


_RELS = [
    {"subject": "Amy Grant", "predicate": "marriedTo",
     "object": "Vince Gill", "support": 2},
    {"subject": "Flexport", "predicate": "acquires",
     "object": "Deliverr", "support": 2},
    {"subject": "Alex Verdugo", "predicate": "signedWith",
     "object": "Yankees", "support": 1},
    {"subject": "Kirby Yates", "predicate": "signedWith",
     "object": "Rangers", "support": 1},
]


def test_predicate_overlap_outranks_higher_support() -> None:
    """THE FIX. `signedWith` edges have support 1 and late alphabetical
    subjects; under the old ordering both lost to support-2 marriedTo pairs."""
    from backend.app.services.retrieval import _rank_relationships

    ranked = _rank_relationships(
        _RELS, "Which players signed with or transferred to new teams?")
    assert [r["predicate"] for r in ranked[:2]] == ["signedWith", "signedWith"]


def test_ranking_stems_so_signed_reaches_signedWith() -> None:
    from backend.app.services.retrieval import _rank_relationships

    ranked = _rank_relationships(_RELS, "who signed players")
    assert ranked[0]["predicate"] == "signedWith"


def test_a_different_question_promotes_a_different_predicate() -> None:
    """Ranking must be question-dependent, not a fixed re-sort."""
    from backend.app.services.retrieval import _rank_relationships

    ranked = _rank_relationships(_RELS, "which company acquired another?")
    assert ranked[0]["predicate"] == "acquires"


def test_no_lexical_hook_degrades_to_support_order_not_to_nothing() -> None:
    from backend.app.services.retrieval import _rank_relationships

    ranked = _rank_relationships(_RELS, "tell me something interesting")
    assert len(ranked) == len(_RELS)          # nothing dropped
    assert ranked[0]["support"] == 2          # support breaks the tie


def test_ranking_keeps_every_edge_capping_is_the_callers_job() -> None:
    from backend.app.services.retrieval import _rank_relationships

    assert len(_rank_relationships(_RELS, "anything")) == len(_RELS)


def test_relative_band_keeps_a_tight_cohort(monkeypatch) -> None:
    """The five real probe distances. All five must survive."""
    from backend.app.services import retrieval

    monkeypatch.setattr(retrieval, "_qa_cfg", lambda k, d=None: {
        "probe_rel_band": 0.10, "probe_dist_ceiling": 1.25,
        "probe_dist_floor": 1.05, "probe_dist_band": None,
    }.get(k, d))
    import uuid as _u
    cohort = 1.0222
    for best in (1.0222, 1.0562, 1.0807, 1.0814, 1.0973):
        kept = retrieval._trim_by_distance(
            [(_u.uuid4(), best)], cohort_best=cohort)
        assert kept, f"probe at {best} was dropped but is within the band"


def test_relative_band_still_abstains_for_a_genuinely_lost_probe(monkeypatch) -> None:
    """The behaviour the floor existed for: one probe with no match among
    probes with real hits must contribute no votes."""
    from backend.app.services import retrieval

    monkeypatch.setattr(retrieval, "_qa_cfg", lambda k, d=None: {
        "probe_rel_band": 0.10, "probe_dist_ceiling": 1.25,
        "probe_dist_floor": 1.05, "probe_dist_band": None,
    }.get(k, d))
    import uuid as _u
    assert retrieval._trim_by_distance(
        [(_u.uuid4(), 1.42)], cohort_best=0.55) == []


def test_ceiling_catches_the_case_where_every_probe_is_hopeless(monkeypatch) -> None:
    from backend.app.services import retrieval

    monkeypatch.setattr(retrieval, "_qa_cfg", lambda k, d=None: {
        "probe_rel_band": 0.10, "probe_dist_ceiling": 1.25,
        "probe_dist_floor": 1.05, "probe_dist_band": None,
    }.get(k, d))
    import uuid as _u
    # Whole cohort is terrible, so the relative test alone would keep them.
    assert retrieval._trim_by_distance(
        [(_u.uuid4(), 1.40)], cohort_best=1.38) == []


def test_probes_are_scored_before_any_is_trimmed() -> None:
    """A relative test cannot be applied one probe at a time -- trimming
    inside the scoring loop is what forced an absolute threshold."""
    import inspect

    from backend.app.services import retrieval

    src = inspect.getsource(retrieval.retrieve_and_answer)
    # The scoring loop ends where the session closes; trimming happens after.
    loop = src.split("for probe_text, pvec in zip(probes", 1)[1]
    loop = loop.split("_cohort_best = min", 1)[0]
    assert "_trim_by_distance" not in loop, (
        "trimming must not happen inside the per-probe scoring loop"
    )
    assert "_cohort_best" in src
