"""The extract-entities validate-and-review loop, and the candidate-menu filter.

Why this exists. Measured on two corpora, entity typing failed three ways:

  * 19 unrelated power plants were typed `ChollaUnit4` -- ONE plant in Arizona
    that prune-expand minted as a CLASS. Its own stored description reads
    "a specific power plant unit referenced in Arizona SIP".
  * A furniture retailer and an apparel brand, appearing as shipping
    CUSTOMERS, were typed `ContainerShippingCarrier`; canals were typed as a
    European region; cities in a bank filing were typed `LoanPortfolio`.
  * Class labels like `Organization or AppStoreOperator` -- disjunctions
    auto-minted from unresolved relation endpoints -- were used as types.

The root cause of the second family is that the extractor was handed exactly
K candidate classes with no distance floor and told "pick ONE ... if no
candidate is a sensible fit, SKIP that entity". Skipping loses the entity
entirely, so a same-topic sibling always won. Hence the abstain sentinel, the
menu filter, and the review loop tested here.

No DB and no real router: `_filter_entities`, `_sanitise_verdicts` and
`_is_menu_unfit_class` are pure, and the loop is exercised through a fake
router, matching how the existing entity tests re-implement guards in
isolation.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from backend.app.services.db_entity_extract import (
    ABSTAIN_SENTINEL,
    _dedup_key,
    _filter_entities,
    _is_menu_unfit_class,
    _resolve_canonical_forms,
    _sanitise_verdicts,
)
from backend.app.services.prompts import (
    PROMPTS,
    entity_extract,
    entity_validate,
)

_CANDIDATES = [
    {"iri": "http://ex#Organization", "label": "Organization",
     "description": "A group of people working together."},
    {"iri": "http://ex#Canal", "label": "Canal",
     "description": "An artificial waterway for navigation."},
    {"iri": "http://ex#Person", "label": "Person",
     "description": "An individual human being."},
]
_CAND_IRIS = {c["iri"] for c in _CANDIDATES}
_CLASS_META = {
    c["iri"]: {"label": c["label"], "description": c["description"]}
    for c in _CANDIDATES
}


def _norm(s: str) -> str:
    from backend.app.services.db_entity_extract import _normalize_name
    return _normalize_name(s)


def _drops() -> dict[str, int]:
    return {"off_menu_iri": 0, "no_name": 0, "abstained": 0,
            "reviewer_removed": 0, "no_candidates": 0}


# --------------------------------------------------------------------------- #
# Prompt contract
# --------------------------------------------------------------------------- #


def test_entity_validate_is_registered() -> None:
    assert PROMPTS["entity_validate"] is entity_validate


def test_reviewer_sees_the_identical_menu() -> None:
    """If the two menus ever diverged, the reviewer could propose an IRI the
    extractor never saw -- and the caller's on-menu check would silently
    discard the verdict, making the whole pass a no-op nobody noticed."""
    _, extract_user = entity_extract("text", _CANDIDATES)
    _, validate_user = entity_validate("text", _CANDIDATES, [])
    block_start = "  - http://ex#Organization"
    e_block = extract_user[extract_user.index(block_start):].split("\n\n")[0]
    v_block = validate_user[validate_user.index(block_start):].split("\n\n")[0]
    assert e_block == v_block


def test_reviewer_sees_class_label_and_description_not_just_iri() -> None:
    """Shown only `.../merged#ChollaUnit4` a reviewer cannot see the problem.
    Shown its description -- "a specific power plant unit" -- the
    individual-shaped class is self-evidently disqualifying."""
    _, user = entity_validate(
        "Suez Canal traffic fell.",
        _CANDIDATES,
        [{"index": 0, "canonical_name": "Suez Canal", "short_name": "Suez",
          "class_iri": "http://ex#Organization",
          "class_label": "Organization",
          "class_description": "A group of people working together."}],
    )
    assert "Suez Canal" in user
    assert "Organization" in user
    assert "A group of people working together." in user


def test_validate_system_prompt_states_the_load_bearing_rules() -> None:
    system, _ = entity_validate("t", _CANDIDATES, [])
    low = system.lower()
    for verdict in ("correct", "wrong_class", "not_an_entity",
                    "no_class_fits", "not_in_text"):
        assert verdict in low
    # The KIND-vs-individual test, which is the ChollaUnit4 failure.
    assert "kind" in low and "one specific" in low
    # Topic-sharing is not instantiation -- the Lovesac / canal failure.
    assert "topic" in low
    # Abstaining must read as legitimate, or the reviewer rubber-stamps.
    assert "no_class_fits\" is a normal answer" in low
    # Anti-churn: without it a temperature-0 reviewer invents "improvements",
    # and every one of them buys a re-extraction call.
    assert "just to look useful" in low


def test_feedback_block_tells_the_extractor_not_to_rubber_stamp() -> None:
    feedback = {
        "prior": [{"canonical_name": "Suez Canal",
                   "class_iri": "http://ex#Organization",
                   "class_label": "Organization"}],
        "verdicts": [{"index": 0, "verdict": "wrong_class",
                      "better_class_iri": "http://ex#Canal",
                      "reason": "a waterway, not an organization"}],
        "missing_entities": [{"canonical_name": "Ramon Fernandez",
                              "class_iri": "http://ex#Person",
                              "reason": "quoted in the passage"}],
        "note": "This passage is about shipping rates.",
    }
    _, user = entity_extract("text", _CANDIDATES, feedback=feedback)
    assert "PRIOR ATTEMPT" in user
    assert "REVIEWER FEEDBACK" in user
    assert "Suez Canal" in user and "http://ex#Canal" in user
    assert "Ramon Fernandez" in user
    assert "This passage is about shipping rates." in user
    # Wholesale replacement, not a diff: the caller swaps the list outright.
    assert "COMPLETE entity list (not a diff)" in user
    # The extractor must be allowed to disagree, or the loop just relocates
    # the error from one model to another.
    assert "NOT obliged to agree" in user


def test_abstain_is_offered_and_framed_as_correct() -> None:
    """"SKIP that entity" was indistinguishable from "no entity here", so the
    model always picked a sibling instead."""
    system, _ = entity_extract("text", _CANDIDATES)
    assert ABSTAIN_SENTINEL in system
    assert "NORMAL, CORRECT answer" in system


# --------------------------------------------------------------------------- #
# _filter_entities -- drop accounting
# --------------------------------------------------------------------------- #


def test_every_rejection_is_counted_not_silent() -> None:
    """The old code used one bare `continue` for all of these, which is how a
    corpus could lose entities steadily with nothing in the output to show."""
    drops, abstained = _drops(), []
    kept = _filter_entities(
        [
            {"canonical_name": "Acme Corp", "class_iri": "http://ex#Organization"},
            {"canonical_name": "", "class_iri": "http://ex#Organization"},
            {"canonical_name": "Ghost", "class_iri": "http://ex#NotOnMenu"},
            {"canonical_name": "Suez Canal", "class_iri": ABSTAIN_SENTINEL,
             "proposed_type": "Canal"},
        ],
        _CAND_IRIS, drops, abstained,
    )
    assert [e["canonical_name"] for e in kept] == ["Acme Corp"]
    assert drops == {"off_menu_iri": 1, "no_name": 1, "abstained": 1,
                     "reviewer_removed": 0, "no_candidates": 0}
    assert abstained == [{"canonical_name": "Suez Canal",
                          "proposed_type": "Canal"}]


def test_abstained_samples_are_capped() -> None:
    drops, abstained = _drops(), []
    _filter_entities(
        [{"canonical_name": f"E{i}", "class_iri": ABSTAIN_SENTINEL}
         for i in range(10)],
        _CAND_IRIS, drops, abstained, abstain_cap=3,
    )
    assert drops["abstained"] == 10, "the COUNT must be complete"
    assert len(abstained) == 3, "only the SAMPLE is capped"


def test_filter_survives_malformed_payloads() -> None:
    drops, abstained = _drops(), []
    assert _filter_entities(None, _CAND_IRIS, drops, abstained) == []
    assert _filter_entities("nonsense", _CAND_IRIS, drops, abstained) == []
    assert _filter_entities([None, 7, "x"], _CAND_IRIS, drops, abstained) == []


# --------------------------------------------------------------------------- #
# _sanitise_verdicts -- the reviewer is not trusted either
# --------------------------------------------------------------------------- #


def _kept(*names: str) -> list[dict]:
    return [{"canonical_name": n, "short_name": n,
             "class_iri": "http://ex#Organization", "confidence": 0.9}
            for n in names]


def test_all_correct_is_not_actionable() -> None:
    """The main cost lever: a clean chunk must cost ONE extra call (the
    review) and stop, never a re-extraction."""
    fb = _sanitise_verdicts(
        {"verdicts": [{"index": 0, "verdict": "correct"}],
         "missing_entities": []},
        _kept("Acme"), _CAND_IRIS, _CLASS_META,
    )
    assert fb["_actionable"] is False


def test_off_menu_suggestion_is_downgraded_not_applied() -> None:
    fb = _sanitise_verdicts(
        {"verdicts": [{"index": 0, "verdict": "wrong_class",
                       "better_class_iri": "http://ex#Invented"}]},
        _kept("Acme"), _CAND_IRIS, _CLASS_META,
    )
    assert fb["verdicts"][0]["verdict"] == "correct"
    assert fb["verdicts"][0]["better_class_iri"] is None
    assert fb["_actionable"] is False


def test_menu_unfit_suggestion_is_downgraded() -> None:
    """A reviewer that swaps one instance-shaped class for another has not
    helped -- it has just moved the bug."""
    meta = dict(_CLASS_META)
    meta["http://ex#ChollaUnit4"] = {
        "label": "ChollaUnit4",
        "description": "Cholla Unit 4, a specific power plant unit referenced in Arizona SIP.",
    }
    fb = _sanitise_verdicts(
        {"verdicts": [{"index": 0, "verdict": "wrong_class",
                       "better_class_iri": "http://ex#ChollaUnit4"}]},
        _kept("Craig Unit 2"), _CAND_IRIS | {"http://ex#ChollaUnit4"}, meta,
    )
    assert fb["verdicts"][0]["verdict"] == "correct"


def test_unknown_verdict_and_bad_index_fail_open() -> None:
    fb = _sanitise_verdicts(
        {"verdicts": [
            {"index": 0, "verdict": "extremely_wrong"},
            {"index": 99, "verdict": "not_in_text"},
            {"index": "abc", "verdict": "not_in_text"},
            {"index": 0, "verdict": "not_in_text"},  # duplicate index
        ]},
        _kept("Acme"), _CAND_IRIS, _CLASS_META,
    )
    assert len(fb["verdicts"]) == 1
    assert fb["verdicts"][0]["verdict"] == "correct"
    assert fb["_actionable"] is False


def test_missing_entity_with_off_menu_class_keeps_the_name() -> None:
    """The NAME is the valuable part -- it is how a genuinely missed entity
    gets recovered. Only the unusable IRI is stripped."""
    fb = _sanitise_verdicts(
        {"verdicts": [],
         "missing_entities": [
             {"canonical_name": "Ramon Fernandez", "class_iri": "http://ex#Nope"},
             {"canonical_name": "", "class_iri": "http://ex#Person"},
         ]},
        _kept("Acme"), _CAND_IRIS, _CLASS_META,
    )
    assert len(fb["missing_entities"]) == 1
    assert fb["missing_entities"][0]["canonical_name"] == "Ramon Fernandez"
    assert fb["missing_entities"][0]["class_iri"] == ""
    assert fb["_actionable"] is True


def test_wrong_class_with_valid_suggestion_is_actionable() -> None:
    fb = _sanitise_verdicts(
        {"verdicts": [{"index": 0, "verdict": "wrong_class",
                       "better_class_iri": "http://ex#Canal",
                       "reason": "a waterway"}]},
        _kept("Suez Canal"), _CAND_IRIS, _CLASS_META,
    )
    assert fb["_actionable"] is True
    assert fb["verdicts"][0]["better_class_iri"] == "http://ex#Canal"


# --------------------------------------------------------------------------- #
# Candidate-menu filter
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "label,description,reason",
    [
        ("ChollaUnit4",
         "Cholla Unit 4, a specific power plant unit referenced in Arizona SIP.",
         "individual-description"),
        ("Cholla Unit 4", "", "trailing-enumerator"),
        ("SegmentOperatingExpense or SegmentAsset",
         "Auto-created from relation 'allocatedByNumberOfEmployees'.",
         "disjunction"),
        ("Organization or AppStoreOperator", "", "disjunction"),
        ("Impairment or RestructuringCost", "", "disjunction"),
    ],
)
def test_unusable_classes_are_withheld_from_the_menu(
    label: str, description: str, reason: str
) -> None:
    unfit, got = _is_menu_unfit_class(label, description)
    assert unfit is True
    assert got == reason


@pytest.mark.parametrize(
    "label,description",
    [
        ("ElectricityGenerationFacility",
         "A facility that generates electricity for supply to the grid."),
        ("ElectricUtilityCompany", "A company that distributes electricity."),
        ("Organization", ""), ("Infrastructure", ""),
        # Pharma kinds -- a menu filter that removed these would silently make
        # a whole domain's entities untypeable.
        ("Type2Diabetes", ""), ("HER2", ""), ("CO2", ""), ("Interleukin6", ""),
        ("Phase3ClinicalTrial", ""), ("CYP3A4", ""),
        # Weak-tier shapes stay ON the menu: there is no LLM adjudication on
        # this path, so only decisive signals may withhold.
        ("ASU2019-01", "Accounting Standards Update 2019-01."),
        ("COVID19", "An infectious respiratory disease."),
    ],
)
def test_genuine_classes_stay_on_the_menu(label: str, description: str) -> None:
    unfit, reason = _is_menu_unfit_class(label, description)
    assert unfit is False, (
        f"{label!r} was withheld ({reason}); nothing adjudicates this path, so "
        "a false positive here silently removes a legitimate typing target"
    )


def test_menu_allowlist_overrides_the_filter() -> None:
    assert _is_menu_unfit_class("Cholla Unit 4", "")[0] is True
    assert _is_menu_unfit_class(
        "Cholla Unit 4", "", frozenset({"cholla unit 4"})
    )[0] is False


# --------------------------------------------------------------------------- #
# The loop, through a fake router
# --------------------------------------------------------------------------- #


class _FakeRouter:
    """Minimal stand-in: `_extract_json` only needs an object with `.text`."""

    def __init__(self, scripted: dict[str, list]) -> None:
        self.scripted = {k: list(v) for k, v in scripted.items()}
        self.calls: list[str] = []

    async def chat(self, task: str, *, system: str, user: str):
        self.calls.append(task)
        queue = self.scripted.get(task) or []
        payload = queue.pop(0) if queue else {}
        if isinstance(payload, Exception):
            raise payload
        return SimpleNamespace(text=json.dumps(payload))


async def _run_loop(router, *, rounds: int, kept: list[dict]) -> list[dict]:
    """Reproduce the loop's control flow over the fake router.

    Mirrors `_one`: validate, stop when nothing is actionable, otherwise
    re-extract; fail OPEN on a validator error; and never let a re-extraction
    empty a chunk that had entities.
    """
    from backend.app.services.db_artifact_gen import _extract_json

    stats = {"clean": 0, "revised": 0, "empty_reextract_rejected": 0,
             "validator_failed": 0}
    for _ in range(rounds):
        try:
            out = await router.chat("entity_validate", system="", user="")
            parsed = _extract_json(out.text)
        except Exception:
            stats["validator_failed"] += 1
            break
        if not isinstance(parsed, dict):
            stats["validator_failed"] += 1
            break
        fb = _sanitise_verdicts(parsed, kept, _CAND_IRIS, _CLASS_META)
        if not fb["_actionable"]:
            stats["clean"] += 1
            break
        out = await router.chat("entity_extract", system="", user="")
        revised = _filter_entities(
            _extract_json(out.text).get("entities"), _CAND_IRIS, _drops(), [],
        )
        if not revised:
            stats["empty_reextract_rejected"] += 1
            break
        stats["revised"] += 1
        kept = revised
    return kept, stats


@pytest.mark.asyncio
async def test_clean_review_costs_one_call_and_no_reextraction() -> None:
    router = _FakeRouter({"entity_validate": [
        {"verdicts": [{"index": 0, "verdict": "correct"}], "missing_entities": []},
    ]})
    kept, stats = await _run_loop(router, rounds=2, kept=_kept("Acme"))
    assert router.calls == ["entity_validate"]
    assert stats["clean"] == 1 and stats["revised"] == 0
    assert [e["canonical_name"] for e in kept] == ["Acme"]


@pytest.mark.asyncio
async def test_wrong_class_triggers_a_reextraction() -> None:
    router = _FakeRouter({
        "entity_validate": [
            {"verdicts": [{"index": 0, "verdict": "wrong_class",
                           "better_class_iri": "http://ex#Canal"}]},
            {"verdicts": [{"index": 0, "verdict": "correct"}]},
        ],
        "entity_extract": [
            {"entities": [{"canonical_name": "Suez Canal",
                           "class_iri": "http://ex#Canal"}]},
        ],
    })
    kept, stats = await _run_loop(router, rounds=2, kept=_kept("Suez Canal"))
    assert router.calls == ["entity_validate", "entity_extract", "entity_validate"]
    assert stats["revised"] == 1 and stats["clean"] == 1
    assert kept[0]["class_iri"] == "http://ex#Canal"


@pytest.mark.asyncio
async def test_rounds_cap_is_honoured() -> None:
    """A reviewer that never settles must not loop forever."""
    never_happy = {"verdicts": [{"index": 0, "verdict": "wrong_class",
                                 "better_class_iri": "http://ex#Canal"}]}
    reextract = {"entities": [{"canonical_name": "Suez Canal",
                               "class_iri": "http://ex#Canal"}]}
    router = _FakeRouter({
        "entity_validate": [never_happy] * 5,
        "entity_extract": [reextract] * 5,
    })
    _, stats = await _run_loop(router, rounds=2, kept=_kept("Suez Canal"))
    assert router.calls.count("entity_validate") == 2
    assert router.calls.count("entity_extract") == 2
    assert stats["revised"] == 2


@pytest.mark.asyncio
async def test_validator_failure_fails_open_and_keeps_pass_one() -> None:
    """Deliberately the OPPOSITE of relationship_verify's "missing verdict =
    unsupported". There the default drops one claim; here it would drop an
    entire chunk's entities, so an unreachable reviewer must change nothing."""
    router = _FakeRouter({"entity_validate": [RuntimeError("provider down")]})
    kept, stats = await _run_loop(router, rounds=2, kept=_kept("Acme", "Beta"))
    assert stats["validator_failed"] == 1
    assert [e["canonical_name"] for e in kept] == ["Acme", "Beta"]


@pytest.mark.asyncio
async def test_empty_reextraction_is_rejected() -> None:
    """A round that empties a chunk which had entities is a regression, not a
    correction -- keep pass-1 rather than losing the chunk's contribution."""
    router = _FakeRouter({
        "entity_validate": [
            {"verdicts": [{"index": 0, "verdict": "wrong_class",
                           "better_class_iri": "http://ex#Canal"}]},
        ],
        "entity_extract": [{"entities": []}],
    })
    kept, stats = await _run_loop(router, rounds=2, kept=_kept("Suez Canal"))
    assert stats["empty_reextract_rejected"] == 1
    assert [e["canonical_name"] for e in kept] == ["Suez Canal"]


# --------------------------------------------------------------------------- #
# Near-duplicate entity nodes
#
# Measured on a 3-document news corpus: 7 near-duplicate node pairs, e.g.
# `Hapag-Lloyd` / `Hapag-Lloyd AG` at trigram 0.79 -- just under the 0.85 gate
# the mint path uses. Duplicates are not cosmetic: each node carries its own
# edges, so multi-hop traversal arriving at one cannot see what hangs off the
# other. That is the same failure that motivated name-level identity.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "a,b",
    [
        ("Hapag-Lloyd AG", "Hapag-Lloyd"),
        ("CMA CGM SA", "CMA CGM"),
        ("Lovesac Co.", "Lovesac"),
        ("Under Armour Inc.", "Under Armour"),
        ("Siemens GmbH", "Siemens"),
    ],
)
def test_legal_suffixes_do_not_split_an_entity(a: str, b: str) -> None:
    assert _dedup_key(a) == _dedup_key(b)


@pytest.mark.parametrize(
    "a,b",
    [
        ("Utah", "Utah Utes"),          # a state vs a sports team
        ("Arizona", "Arizona State"),   # different institutions
        ("CMA CGM", "MSC"),
    ],
)
def test_genuinely_different_names_stay_apart(a: str, b: str) -> None:
    assert _dedup_key(a) != _dedup_key(b)


def test_a_bare_legal_form_is_not_collapsed_to_nothing() -> None:
    """'Group' and 'Co' are real short names; stripping them to an empty key
    would merge every such entity into one node."""
    assert _dedup_key("Group") == "group"
    assert _dedup_key("Co") == "co"


def test_the_models_own_short_name_pairing_merges_variants() -> None:
    """The extractor already reports canonical_name AND short_name, which is
    an assertion -- grounded in that chunk -- that the two denote one entity.
    That pairing was previously discarded for identity purposes, which is why
    `Maersk` and `A.P. Moller-Maersk A/S` became separate nodes: their
    suffix-stripped keys differ, so only the pairing can join them."""
    results = [(None, None, None, [
        {"canonical_name": "A.P. Moller-Maersk A/S", "short_name": "Maersk"},
        {"canonical_name": "Maersk", "short_name": "Maersk"},
        {"canonical_name": "Stephane Kovatchev", "short_name": "Kovatchev"},
        {"canonical_name": "Kovatchev", "short_name": "Kovatchev"},
    ], None)]
    alias, merged = _resolve_canonical_forms(results)
    assert alias[_norm("Maersk")] == _norm("A.P. Moller-Maersk A/S")
    assert alias[_norm("Kovatchev")] == _norm("Stephane Kovatchev")
    assert merged == 2


def test_the_fullest_spelling_wins() -> None:
    """The extract prompt asks for "the entity's most complete proper name",
    so the surviving node should be that one, not whichever chunk came first."""
    results = [(None, None, None, [
        {"canonical_name": "CMA CGM", "short_name": "CMA CGM"},
        {"canonical_name": "CMA CGM SA", "short_name": "CMA CGM"},
    ], None)]
    alias, _ = _resolve_canonical_forms(results)
    assert alias[_norm("CMA CGM")] == _norm("CMA CGM SA")


def test_resolution_is_a_noop_when_names_are_already_distinct() -> None:
    results = [(None, None, None, [
        {"canonical_name": "Suez Canal", "short_name": "Suez"},
        {"canonical_name": "Panama Canal", "short_name": "Panama"},
    ], None)]
    alias, _merged = _resolve_canonical_forms(results)
    # Each short form joins its own canonical's group; nothing cross-merges.
    assert alias.get(_norm("Panama Canal")) is None
    assert alias.get(_norm("Suez Canal")) is None


def test_resolution_survives_empty_and_malformed_input() -> None:
    assert _resolve_canonical_forms([]) == ({}, 0)
    assert _resolve_canonical_forms([None]) == ({}, 0)
    assert _resolve_canonical_forms([(None, None, None, [], None)]) == ({}, 0)


_P = "http://ex#Person"
_TEAM = "http://ex#CollegeFootballTeam"
_PROJ = "http://ex#OfficeProject"


def _res(rows):
    return [(None, None, None, rows, None)]


def test_a_person_referred_to_by_surname_is_one_node() -> None:
    """`Amosi` and `Guy Amosi` were two nodes with separate edges, so a
    traversal reaching one could not see what hung off the other."""
    alias, _ = _resolve_canonical_forms(_res([
        {"canonical_name": "Guy Amosi", "short_name": "Amosi", "class_iri": _P},
        {"canonical_name": "Amosi", "short_name": "Amosi", "class_iri": _P},
        {"canonical_name": "Tetairoa McMillan", "short_name": "McMillan", "class_iri": _P},
        {"canonical_name": "McMillan", "short_name": "McMillan", "class_iri": _P},
    ]), person_class_iris={_P})
    assert alias[_norm("Amosi")] == _norm("Guy Amosi")
    assert alias[_norm("McMillan")] == _norm("Tetairoa McMillan")


def test_a_place_short_form_is_not_merged_into_a_longer_name() -> None:
    """The rule that merges `Amosi` into `Guy Amosi` must not merge `Utah`
    into `Utah Utes`. A surname is a token SUFFIX of the full name; a place
    short form is a PREFIX -- and the rule is Person-scoped besides."""
    alias, _ = _resolve_canonical_forms(_res([
        {"canonical_name": "Utah", "short_name": "Utah", "class_iri": _TEAM},
        {"canonical_name": "Utah Utes", "short_name": "Utah", "class_iri": _TEAM},
        {"canonical_name": "Arizona", "short_name": "Arizona", "class_iri": _TEAM},
        {"canonical_name": "Arizona Wildcats", "short_name": "Arizona", "class_iri": _TEAM},
    ]), person_class_iris={_P})
    assert _norm("Utah") not in alias
    assert _norm("Arizona") not in alias


def test_two_different_universities_are_never_merged() -> None:
    """`Arizona State University` and `University of Arizona` share most of
    their tokens. Token-MULTISET equality is what keeps them apart -- they
    differ by "state" vs "of"."""
    alias, _ = _resolve_canonical_forms(_res([
        {"canonical_name": "Arizona State University", "short_name": "Arizona State University", "class_iri": _TEAM},
        {"canonical_name": "University of Arizona", "short_name": "University of Arizona", "class_iri": _TEAM},
    ]), person_class_iris={_P})
    assert alias == {}


def test_word_order_variants_collapse() -> None:
    alias, _ = _resolve_canonical_forms(_res([
        {"canonical_name": "Park Naimi", "short_name": "Park Naimi", "class_iri": _PROJ},
        {"canonical_name": "Naimi Park", "short_name": "Naimi Park", "class_iri": _PROJ},
    ]), person_class_iris=set())
    assert len(alias) == 1


def test_alias_chains_resolve_to_one_final_spelling() -> None:
    """a -> b -> c must land on c, or variants scatter across intermediates."""
    alias, _ = _resolve_canonical_forms(_res([
        {"canonical_name": "Or Ben Zvi Klein", "short_name": "Klein", "class_iri": _P},
        {"canonical_name": "Ben Zvi Klein", "short_name": "Klein", "class_iri": _P},
        {"canonical_name": "Klein", "short_name": "Klein", "class_iri": _P},
    ]), person_class_iris={_P})
    targets = set(alias.values())
    assert targets == {_norm("Or Ben Zvi Klein")}
    assert not (set(alias) & targets), "no variant may point at another variant"
