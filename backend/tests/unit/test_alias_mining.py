"""Deterministic synonym mining: extraction precision + pair normalization.

Pure-unit; no DB, no LLM, no embeddings, no cost.

The recall cases are drawn from the failure that motivated this module (a
query for "tirzepatide" could not reach chunks that only say "MOUNJARO"),
and the rejection cases from the kinds of parentheticals that actually
crowd financial and clinical PDFs. Precision matters more than recall
here: a wrong pair reproduces the original bug pointed at a different
drug.
"""
from __future__ import annotations

import pytest

from backend.app.services.alias_mining import (
    extract_ontology_pairs,
    is_valid_pair,
    is_valid_term,
    make_pair,
    mine_text,
    normalize_term,
)
from backend.app.services.db_entity_extract import _normalize_name


# --------------------------------------------------------------------------
# normalize_term must not drift from the ingest-time normalizer
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        "MOUNJARO",
        "MOUNJARO®",
        "Eli Lilly and Company, Inc.",
        "  BYD   Company  Ltd. ",
        "losartan potassium",
        "type-2 diabetes",
        "",
    ],
)
def test_normalize_term_matches_ingest_normalizer(raw: str) -> None:
    """`alias_mining.normalize_term` is duplicated from
    `db_entity_extract._normalize_name` to keep the retrieval path
    dependency-light. If they ever diverge, ingest and query time would
    silently disagree about what a term IS -- so pin them together."""
    assert normalize_term(raw) == _normalize_name(raw)


# --------------------------------------------------------------------------
# Parenthetical apposition -- the workhorse source
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        # The exact shape from the reported failure.
        (
            "MOUNJARO® (tirzepatide) injection, for subcutaneous use",
            ("mounjaro", "tirzepatide"),
        ),
        ("OZEMPIC (semaglutide) injection", ("ozempic", "semaglutide")),
        ("TRULICITY (dulaglutide) injection", ("dulaglutide", "trulicity")),
        # All-caps 6-letter brand: must NOT be mistaken for an acronym.
        ("COZAAR (losartan potassium) tablets", ("cozaar", "losartan potassium")),
        ("Norvasc (amlodipine besylate) tablets", ("amlodipine besylate", "norvasc")),
        # Reverse direction: generic first, brand parenthesized.
        ("dosage of tirzepatide (Mounjaro) in adults", ("mounjaro", "tirzepatide")),
        # Head must stop at the lowercase "of", not swallow it.
        (
            "The maximum dose of MOUNJARO (tirzepatide) is 15 mg",
            ("mounjaro", "tirzepatide"),
        ),
    ],
)
def test_parenthetical_pairs_extracted(text: str, expected: tuple[str, str]) -> None:
    pairs = mine_text(text)
    assert [(p.term_a, p.term_b) for p in pairs] == [expected]
    assert pairs[0].evidence_kind == "parenthetical"


def test_acronym_expansion_requires_matching_initials() -> None:
    """The classic N-words-for-N-letters rule, and the leading article
    must not break it."""
    (pair,) = mine_text("The World Health Organization (WHO) reported")
    assert (pair.term_a, pair.term_b) == ("who", "world health organization")

    # Initials don't spell the acronym -> not an expansion.
    assert mine_text("The Federal Drug Agency (WHO) said") == []


# --------------------------------------------------------------------------
# Cross-domain generality
#
# The mining CONVENTIONS (parenthetical apposition, alias phrases,
# skos:altLabel) are domain-neutral even though the guards were tuned on a
# pharma dry run. These pin the non-pharma behavior so a later
# pharma-motivated tweak cannot quietly break the finance corpus, which is
# this project's other target.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        # Initials, the easy case.
        (
            "International Business Machines (IBM) announced",
            ("ibm", "international business machines"),
        ),
        (
            "Taiwan Semiconductor Manufacturing Company (TSMC) expanded",
            ("taiwan semiconductor manufacturing company", "tsmc"),
        ),
        (
            "the Large Hadron Collider (LHC) resumed operation",
            ("large hadron collider", "lhc"),
        ),
        # Acronyms skip stopwords: SEC, not SAEC.
        (
            "The Securities and Exchange Commission (SEC) issued guidance",
            ("sec", "securities and exchange commission"),
        ),
        # ...and the expansion runs past the ordinary 4-word cap.
        (
            "earnings before interest and taxes (EBIT) declined",
            ("earnings before interest and taxes", "ebit"),
        ),
        # Company short forms are a PREFIX, not initials.
        (
            "BHP Group Limited (BHP) reported record shipments",
            ("bhp", "bhp group limited"),
        ),
        # Rename phrasing.
        (
            "Meta Platforms, formerly known as Facebook, restated",
            ("facebook", "meta platforms"),
        ),
    ],
)
def test_non_pharma_pairs(text: str, expected: tuple[str, str]) -> None:
    pairs = mine_text(text)
    assert [(p.term_a, p.term_b) for p in pairs] == [expected]


@pytest.mark.parametrize(
    "text",
    [
        "Revenue (in millions) for fiscal 2024",
        "Net loss (unaudited) for the quarter",
        "See Note 12 (Commitments and Contingencies)",
        "Total assets (restated) increased",
        "operating margin (excluding one-time charges) improved",
    ],
)
def test_financial_statement_furniture_rejected(text: str) -> None:
    assert mine_text(text) == []


@pytest.mark.parametrize(
    "text,expected",
    [
        # Verbatim from the corpus. The list marker "3.1" is invisible to
        # the word tokenizer, so the capitalised-run head ran straight
        # through it and invented the name "Ozempic Mounjaro".
        (
            "Best Alternatives to Ozempic 3.1 Mounjaro (tirzepatide) tablets",
            [("mounjaro", "tirzepatide")],
        ),
        # Same defect via a sentence/list boundary: "... in Wegovy 8.
        # Zepbound (Tirzepatide)" produced "Wegovy Zepbound".
        (
            "ingredients in Wegovy 8. Zepbound (Tirzepatide) is dual-acting",
            [("tirzepatide", "zepbound")],
        ),
    ],
)
def test_item_boundaries_do_not_merge_neighbouring_names(
    text: str, expected: list[tuple[str, str]]
) -> None:
    assert [(p.term_a, p.term_b) for p in mine_text(text)] == expected


def test_abbreviation_periods_are_not_item_boundaries() -> None:
    """A bare period inside an abbreviation must not split the head, or
    "U.S. Securities and Exchange Commission" stops resolving."""
    (pair,) = mine_text(
        "The U.S. Securities and Exchange Commission (SEC) issued guidance"
    )
    assert (pair.term_a, pair.term_b) == ("sec", "securities and exchange commission")


def test_surface_forms_are_preserved() -> None:
    """Probes embed the surface form, and "MOUNJARO" does not embed the
    same as "mounjaro"."""
    (pair,) = mine_text("MOUNJARO® (tirzepatide) injection")
    assert {pair.surface_a, pair.surface_b} == {"MOUNJARO", "tirzepatide"}


# --------------------------------------------------------------------------
# Precision guards -- everything below must yield NOTHING
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        # Cross-references
        "revenue increased (see Section 5.1) materially",
        "Results (see Table 3) were consistent",
        # Statistics / identifiers / dates
        "the study (NCT01234) enrolled patients",
        "the mean change (95% CI) was significant",
        "dose reduction (n=234) occurred",
        "A randomized trial (2024) of the drug",
        # Units and accounting furniture
        "net income (in millions) for 2024",
        "total assets (unaudited) as of December",
        "cash flow (continued) from operations",
        # Clauses, not terms
        "patients (who received placebo) improved",
        "administered in the morning (once daily)",
        "treatment was given (as needed) to patients",
        # Generic head: would mint `company <-> lilly` if unguarded.
        "Eli Lilly and Company (Lilly) announced",
        # No parenthetical at all
        "The maximum dosage of MOUNJARO is 15 mg once weekly.",
    ],
)
def test_rejects_non_synonym_parentheticals(text: str) -> None:
    assert mine_text(text) == []


@pytest.mark.parametrize(
    "text",
    [
        # A dosage FORM is a presentation, not a synonym of the product.
        # "pill" reached probe text as an alias of Wegovy before this guard.
        "Wegovy (pill) for chronic weight management",
        "Rybelsus (oral) semaglutide",
        "Ozempic (solution) for injection",
    ],
)
def test_dosage_forms_are_not_synonyms(text: str) -> None:
    assert mine_text(text) == []


def test_both_sides_lowercase_common_words_rejected() -> None:
    """At least one side must look like a proper name, or ordinary prose
    apposition mints garbage pairs."""
    assert not is_valid_pair("morning", "once daily")


def test_identical_terms_are_not_a_pair() -> None:
    assert not is_valid_pair("Mounjaro", "MOUNJARO®")
    assert make_pair("Mounjaro", "mounjaro", "parenthetical") is None


@pytest.mark.parametrize(
    "term", ["2024", "95% CI", "n=234", "mg", "the company", "see Table 3", "a"]
)
def test_invalid_terms(term: str) -> None:
    assert not is_valid_term(term)


# --------------------------------------------------------------------------
# Explicit alias phrases
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        (
            "tirzepatide, also known as Mounjaro, is a GIP receptor agonist",
            ("mounjaro", "tirzepatide"),
        ),
        # The copula must not be absorbed into the head, and the trailing
        # indication must not be absorbed into the alias.
        (
            "semaglutide is marketed as Ozempic for type 2 diabetes",
            ("ozempic", "semaglutide"),
        ),
        (
            "amlodipine, sold under the brand name Norvasc, treats hypertension",
            ("amlodipine", "norvasc"),
        ),
    ],
)
def test_phrase_pairs_extracted(text: str, expected: tuple[str, str]) -> None:
    pairs = mine_text(text)
    assert [(p.term_a, p.term_b) for p in pairs] == [expected]
    assert pairs[0].evidence_kind == "phrase"


# --------------------------------------------------------------------------
# Ontology annotations (already in the DB since ontology import)
# --------------------------------------------------------------------------


def test_ontology_synonym_annotations() -> None:
    ann = {
        "http://www.w3.org/2004/02/skos/core#altLabel": ["Mounjaro"],
        "http://www.geneontology.org/formats/oboInOwl#hasExactSynonym": ["Zepbound"],
    }
    got = {b for _, b in extract_ontology_pairs("Tirzepatide", ann)}
    assert got == {"Mounjaro", "Zepbound"}


def test_ontology_ignores_non_synonym_predicates() -> None:
    """A description is not an alias."""
    ann = {"http://purl.org/dc/terms/description": ["Mounjaro"]}
    assert extract_ontology_pairs("Tirzepatide", ann) == []


def test_ontology_handles_missing_label_or_annotations() -> None:
    assert extract_ontology_pairs(None, {"x#altLabel": ["Y"]}) == []
    assert extract_ontology_pairs("Tirzepatide", None) == []


# --------------------------------------------------------------------------
# Pair ordering -- one row must answer a lookup from either direction
# --------------------------------------------------------------------------


def test_pairs_are_stored_in_lexical_order() -> None:
    """The DB enforces `term_a < term_b` with a CHECK; both argument
    orders must produce the same row."""
    a = make_pair("MOUNJARO", "tirzepatide", "parenthetical", "chunk:1")
    b = make_pair("tirzepatide", "MOUNJARO", "parenthetical", "chunk:1")
    assert a is not None and b is not None
    assert (a.term_a, a.term_b) == (b.term_a, b.term_b) == ("mounjaro", "tirzepatide")
    assert a.term_a < a.term_b
    # Surface forms travel with their normalized side.
    assert a.surface_a == "MOUNJARO" and a.surface_b == "tirzepatide"
    assert b.surface_a == "MOUNJARO" and b.surface_b == "tirzepatide"


def test_manual_pairs_are_exempt_from_the_prune() -> None:
    """`mine-aliases` replaces the mined rows wholesale and deletes any it
    did not just produce. That refresh now runs automatically after every
    ingest, so a hand-curated pair MUST be exempt or an operator's work
    disappears with no warning. The README documents this guarantee;
    pin the SQL that provides it."""
    import inspect

    from backend.app.services import alias_mining

    src = inspect.getsource(alias_mining.mine_aliases)
    assert "DELETE FROM graphrag.term_aliases" in src, "prune removed?"
    # The exemption must sit inside the DELETE's own WHERE clause.
    delete_stmt = src[src.index("DELETE FROM graphrag.term_aliases") :]
    where_clause = delete_stmt.split('"""')[0]
    assert "evidence_kind <> 'manual'" in where_clause


def test_manual_is_an_accepted_evidence_kind() -> None:
    """The CHECK constraint on the table must admit it (migration 0007)."""
    from backend.app.db.models.entities import TermAlias

    checks = [
        str(c.sqltext)
        for c in TermAlias.__table__.constraints
        if hasattr(c, "sqltext")
    ]
    assert any("manual" in c for c in checks), checks


def test_occurrences_accumulate_across_mentions() -> None:
    from backend.app.services.alias_mining import _merge

    acc: dict = {}
    for _ in range(3):
        _merge(acc, mine_text("MOUNJARO (tirzepatide) injection"))
    (pair,) = acc.values()
    assert pair.occurrences == 3
