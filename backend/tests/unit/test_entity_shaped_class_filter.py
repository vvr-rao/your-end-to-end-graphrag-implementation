"""Stage-2 entity-shaped class filter.

Covers `_looks_like_entity_not_class` (the heuristic) and
`_filter_entity_shaped_classes` (the demoter that promotes a class
proposal to MATCH NOT FOUND INSTANCES). The point of the filter is
to keep proper nouns -- company names, country names, named reports
-- out of the ontology's class set when gpt-4.1 occasionally leaks
one through despite the prompt's HARD RULE.
"""

from __future__ import annotations

import pytest

from backend.app.services.pipeline_llm import (
    _build_known_places_from_ontology,
    _compile_extra_suffix_regex,
    _compile_extra_word_regex,
    _filter_entity_shaped_classes,
    _looks_like_entity_not_class,
    _looks_like_individual_weak,
    _split_camel,
)


_DEFAULT_KNOWN_PLACES = frozenset({"myanmar", "vietnam", "asia", "tokyo",
                                    "united states", "hong kong"})


@pytest.mark.parametrize(
    "label,reason_prefix",
    [
        # Corporate suffix patterns
        ("BYD Company Ltd.", "corporate-suffix"),
        ("OCI N.V.", "corporate-suffix"),
        ("Apple Inc", "corporate-suffix"),
        ("Saudi Aramco Total Refinery & Petrochemicals Co.", "corporate-suffix"),
        ("Samsung Electronics Co., Ltd.", "corporate-suffix"),
        ("ThyssenKrupp AG", "corporate-suffix"),
        # Known proper-noun places (sourced from the dynamic set)
        ("Myanmar", "known-place"),
        ("Vietnam", "known-place"),
        ("Asia", "known-place"),
        ("Tokyo", "known-place"),
        ("United States", "known-place"),
        ("Hong Kong", "known-place"),
        # Document/report titles (whitespace-separated)
        ("Sovereign Risk Tracker", "document-title"),
        ("Vietnam's Manufacturing & Supply Chain Industry Report", "document-title"),
        ("Fertilizer Market Dashboard", "document-title"),
        ("Energy Outlook", "document-title"),
        # CamelCase document/tool titles (v2 catches these because the
        # CamelCase splitter exposes the tail word).
        ("FertilizerMarketDashboard", "document-title"),
        ("SovereignRiskTracker", "document-title"),
        ("EarlyWarningHub", "document-title"),
        ("GlobalDebtDatabase", "document-title"),
        ("FoodSecurityPortal", "document-title"),
        ("FuelShortageTracker", "document-title"),
        ("MonetaryPolicyTracker", "document-title"),
        ("CompanyDatabase", "document-title"),
        ("ReleaseCalendar", "document-title"),
        ("InflationDataSource", "document-title"),
        ("InvestorDirectory", "document-title"),
        ("AgricultureNewsPlatform", "document-title"),
        ("PublicOpinionSurvey", "document-title"),
        # Year-prefixed labels with document-title tail words are caught
        # by the document-title heuristic first (more specific reason).
        ("2025 Factbook", "document-title"),
        ("Q1 2024 Outlook", "document-title"),
        # Year-prefix-only fallback (no document-title tail word).
        ("2025 Strategic Review Conference", "year-prefix"),
        ("Q3 2024 Annual Summary", "year-prefix"),
    ],
)
def test_entity_shaped_labels_are_flagged(label: str, reason_prefix: str) -> None:
    is_entity, reason = _looks_like_entity_not_class(
        label, known_places=_DEFAULT_KNOWN_PLACES,
    )
    assert is_entity is True, f"expected {label!r} to be flagged as entity"
    assert reason == reason_prefix, (
        f"expected reason {reason_prefix!r} for {label!r}, got {reason!r}"
    )


@pytest.mark.parametrize(
    "label",
    [
        # Abstract category names -- these are LEGITIMATE class proposals
        #
        # The "Index"/"Site"/"Service" group was RECLASSIFIED (2026-08-28).
        # These three tail words were previously enough on their own to demote
        # a label, and a 30-doc run showed the cost: HealthcareService,
        # InjectionSite, BodyMassIndex and CardiologyCareService were all
        # demoted out of the class hierarchy. A demoted class cannot be a
        # class_iri target in extract-entities, so every entity that belonged
        # to it is then silently dropped by the `class_iri not in cand_iris`
        # check -- invisible, and biased against one whole shape of class name.
        #
        # ConsumerPriceIndex moved here from the flagged list for the same
        # reason. CPI is a KIND of indicator with many instances (US CPI, EU
        # HICP), so it is a defensible class, and the error is asymmetric: a
        # wrongly-demoted class loses data silently, whereas a wrongly-kept
        # class is still caught downstream by dedup and the Layer-H
        # classification audit. These three now demote only alongside a second
        # document signal (year, quarter, FY, possessive, title punctuation).
        "ConsumerPriceIndex",
        "BodyMassIndex",
        "InjectionSite",
        "HealthcareService",
        "ManufacturingSite",
        "WebService",
        "CarManufacturer",
        "FertilizerProducer",
        "TradeAgreement",
        "SoutheastAsianCountry",
        "ChemicalElement",
        "Regulation",
        "SupplyChainRisk",
        "Person",                    # a class, not a person's name
        "Organization",
        "Country",
        # Bare tail-words alone are legitimate categories -- DO NOT flag.
        # ("Forecast" by itself = a kind of prediction; "Alert" by itself
        # = a category of warning.)
        "Forecast",
        "Alert",
        "Dashboard",
        "Tracker",
        "Monitor",
        "Report",
        # Edge cases that should pass
        "",
        "   ",
    ],
)
def test_category_labels_are_not_flagged(label: str) -> None:
    is_entity, reason = _looks_like_entity_not_class(
        label, known_places=_DEFAULT_KNOWN_PLACES,
    )
    assert is_entity is False, (
        f"expected {label!r} to be kept as a class (got reason {reason!r})"
    )


def test_filter_promotes_entity_shaped_to_instances() -> None:
    stage2_result = {
        "MATCHES FOUND": [{"IRI": "ex:Country", "TEXT_SNIPPET": "Vietnam"}],
        "MATCH NOT FOUND": [
            {
                "LABEL": "BYD Company Ltd.",
                "DESCRIPTION": "A Chinese EV manufacturer.",
                "PARENT_LABEL": "Organization",
            },
            {
                "LABEL": "CarManufacturer",
                "DESCRIPTION": "A company that makes cars.",
                "PARENT_LABEL": "Organization",
            },
            {
                "LABEL": "Myanmar",
                "DESCRIPTION": "A Southeast Asian country.",
                "PARENT_LABEL": "Country",
            },
        ],
        "MATCH NOT FOUND INSTANCES": [
            {
                "LABEL": "Jan 2024",
                "CANONICAL_FORM": "January 2024",
                "TYPE_LABEL": "Month",
                "DESCRIPTION": "",
            }
        ],
    }
    updated, demotions = _filter_entity_shaped_classes(
        stage2_result, known_places=_DEFAULT_KNOWN_PLACES,
    )
    assert updated is not None
    # Only the legitimate category survives in MATCH NOT FOUND
    surviving_labels = [c["LABEL"] for c in updated["MATCH NOT FOUND"]]
    assert surviving_labels == ["CarManufacturer"]
    # Both entity-shaped proposals were promoted to INSTANCES, preserving
    # the original temporal instance that was already there.
    instance_labels = [i["LABEL"] for i in updated["MATCH NOT FOUND INSTANCES"]]
    assert "Jan 2024" in instance_labels
    assert "BYD Company Ltd." in instance_labels
    assert "Myanmar" in instance_labels
    # Demotion records expose the reason for each move.
    demoted_labels = {d["label"]: d["reason"] for d in demotions}
    assert demoted_labels == {
        "BYD Company Ltd.": "corporate-suffix",
        "Myanmar": "known-place",
    }
    # Promoted instances inherit the PARENT_LABEL as their TYPE_LABEL.
    byd_instance = next(
        i for i in updated["MATCH NOT FOUND INSTANCES"] if i["LABEL"] == "BYD Company Ltd."
    )
    assert byd_instance["TYPE_LABEL"] == "Organization"
    assert byd_instance["CANONICAL_FORM"] == "BYD Company Ltd."
    assert byd_instance["DESCRIPTION"] == "A Chinese EV manufacturer."


def test_filter_is_safe_on_missing_keys() -> None:
    # No MATCH NOT FOUND at all -- returns the same dict, no demotions.
    result = {"MATCHES FOUND": []}
    updated, demotions = _filter_entity_shaped_classes(result)
    assert updated == {"MATCHES FOUND": []}
    assert demotions == []


def test_filter_is_safe_on_none() -> None:
    updated, demotions = _filter_entity_shaped_classes(None)
    assert updated is None
    assert demotions == []


def test_filter_is_idempotent_when_no_entities_to_promote() -> None:
    # All entries are legitimate categories -- the result is unchanged
    # and no demotions are produced.
    result = {
        "MATCH NOT FOUND": [
            {"LABEL": "TradeAgreement", "DESCRIPTION": "", "PARENT_LABEL": "NONE"},
        ],
    }
    updated, demotions = _filter_entity_shaped_classes(result)
    assert updated["MATCH NOT FOUND"] == [
        {"LABEL": "TradeAgreement", "DESCRIPTION": "", "PARENT_LABEL": "NONE"}
    ]
    assert demotions == []
    # MATCH NOT FOUND INSTANCES was not touched (and was absent).
    assert "MATCH NOT FOUND INSTANCES" not in updated


def test_filter_preserves_non_dict_entries() -> None:
    # Stray malformed entries (strings, None) pass through untouched.
    result = {
        "MATCH NOT FOUND": [
            "stray-string",
            None,
            {"LABEL": "Apple Inc", "DESCRIPTION": "", "PARENT_LABEL": "Organization"},
        ],
    }
    updated, demotions = _filter_entity_shaped_classes(result)
    assert updated["MATCH NOT FOUND"] == ["stray-string", None]
    assert len(demotions) == 1
    assert demotions[0]["label"] == "Apple Inc"


# ---------------------------------------------------------------------------
# v2 helpers: CamelCase split, dynamic known-places, config extensions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "label,expected",
    [
        ("FertilizerMarketDashboard", "Fertilizer Market Dashboard"),
        ("EarlyWarningHub", "Early Warning Hub"),
        ("GlobalDebtDatabase", "Global Debt Database"),
        ("WEO", "WEO"),                 # all-caps acronym preserved
        ("EUTrade", "EU Trade"),        # acronym + CamelCase
        ("Fertilizer Market Dashboard", "Fertilizer Market Dashboard"),  # already spaced
        ("2025Factbook", "2025 Factbook"),
        ("", ""),
        ("Person", "Person"),           # single-word category
    ],
)
def test_split_camel(label: str, expected: str) -> None:
    assert _split_camel(label) == expected


def _make_ontology_fixture() -> dict:
    """Tiny ontology mimicking the geography subtree pattern. Three layers:
    Continent / Region / Country at the top; specific continents (Asia),
    regions (SoutheastAsia), countries (Vietnam) below."""
    return {
        "classes_dict": {
            "ex:Continent": {"labels": ["Continent"], "superclasses": ["ex:Place"]},
            "ex:Region":    {"labels": ["Region"],    "superclasses": ["ex:Place"]},
            "ex:Country":   {"labels": ["Country"],   "superclasses": ["ex:Place"]},
            "ex:Place":     {"labels": ["Place"],     "superclasses": []},
            "ex:Asia":      {"labels": ["Asia"],      "superclasses": ["ex:Continent"]},
            "ex:Africa":    {"labels": ["Africa"],    "superclasses": ["ex:Continent"]},
            "ex:SoutheastAsia": {"labels": ["Southeast Asia"], "superclasses": ["ex:Region", "ex:Asia"]},
            "ex:Vietnam":   {"labels": ["Vietnam"],   "superclasses": ["ex:Country", "ex:SoutheastAsia"]},
            "ex:Myanmar":   {"labels": ["Myanmar"],   "superclasses": ["ex:Country"]},
            # Not a place
            "ex:CarManufacturer": {"labels": ["Car Manufacturer"], "superclasses": ["ex:Organization"]},
            "ex:Organization":    {"labels": ["Organization"],     "superclasses": []},
        }
    }


def test_build_known_places_from_ontology_discovers_geography_descendants() -> None:
    places = _build_known_places_from_ontology(_make_ontology_fixture())
    # All ancestors-of-Place classes contribute their labels (Place itself
    # is NOT a place-kind label, but Continent / Region / Country are).
    assert "continent" in places
    assert "country" in places
    assert "region" in places
    assert "asia" in places
    assert "vietnam" in places
    assert "myanmar" in places
    assert "southeast asia" in places
    # Non-geography classes are not included.
    assert "car manufacturer" not in places
    assert "organization" not in places


def test_build_known_places_returns_empty_when_no_geography_present() -> None:
    minimal = {"classes_dict": {"ex:Foo": {"labels": ["Foo"], "superclasses": []}}}
    places = _build_known_places_from_ontology(minimal)
    assert places == frozenset()


def test_build_known_places_unions_extras() -> None:
    minimal = {"classes_dict": {"ex:Foo": {"labels": ["Foo"], "superclasses": []}}}
    places = _build_known_places_from_ontology(
        minimal, extra_labels=["Atlantis", "Wakanda"]
    )
    assert places == frozenset({"atlantis", "wakanda"})


def test_build_known_places_safe_on_none() -> None:
    assert _build_known_places_from_ontology(None) == frozenset()
    assert _build_known_places_from_ontology({}) == frozenset()
    assert _build_known_places_from_ontology({"classes_dict": None}) == frozenset()


def test_known_places_default_skips_place_check() -> None:
    # When known_places is None or empty, "Myanmar" passes the heuristic
    # (the place check is the only branch that would catch it; corporate
    # suffix + doc tail + year prefix do not).
    is_entity, _ = _looks_like_entity_not_class("Myanmar", known_places=None)
    assert is_entity is False
    is_entity, _ = _looks_like_entity_not_class("Myanmar", known_places=frozenset())
    assert is_entity is False


def test_extra_corporate_suffix_regex_extends_builtins() -> None:
    extra = _compile_extra_suffix_regex(["KK", "OAO"])
    # A label using ONLY the extra suffix is now caught.
    is_entity, reason = _looks_like_entity_not_class(
        "Toyota KK", extra_corporate_suffix_re=extra,
    )
    assert is_entity is True
    assert reason == "corporate-suffix"
    # And built-ins still fire too.
    is_entity, reason = _looks_like_entity_not_class(
        "Toyota Inc", extra_corporate_suffix_re=extra,
    )
    assert is_entity is True
    assert reason == "corporate-suffix"


def test_extra_tail_word_regex_extends_builtins() -> None:
    extra = _compile_extra_word_regex(["Compendium", "Almanac"])
    # CamelCase + extra tail word now caught.
    is_entity, reason = _looks_like_entity_not_class(
        "FoodSecurityCompendium", extra_tail_word_re=extra,
    )
    assert is_entity is True
    assert reason == "document-title"
    # Built-in tail words still work.
    is_entity, _ = _looks_like_entity_not_class(
        "SovereignRiskTracker", extra_tail_word_re=extra,
    )
    assert is_entity is True


def test_compile_extras_return_none_on_empty_or_garbage() -> None:
    assert _compile_extra_suffix_regex(None) is None
    assert _compile_extra_suffix_regex([]) is None
    assert _compile_extra_suffix_regex([""]) is None
    assert _compile_extra_word_regex(None) is None
    assert _compile_extra_word_regex(["", "   "]) is None


# --------------------------------------------------------------------------- #
# Cross-domain matrix (individual-vs-class, heuristics 5-7)
#
# The failure that motivated these: `ChollaUnit4` -- ONE power plant unit --
# was minted as a class and then used to type 18 unrelated plants in four
# other states, making it the 5th most-used class in the graph.
#
# These tests exist to enforce GENERALIZATION rather than assert it. The
# corpora span finance, energy, pharma, legal and news, and the danger is
# asymmetric: a rule tuned on `ChollaUnit4` also describes `Interleukin6`, and
# demoting a real class deletes it AND silently drops every entity that would
# have instantiated it. So the negative table below is the load-bearing one.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "label,description,reason",
    [
        # Energy -- spaced enumerated unit names.
        ("Cholla Unit 4", "", "trailing-enumerator"),
        ("Craig Unit 2", "", "trailing-enumerator"),
        ("Hayden Units 1 and 2", "", "trailing-enumerator"),
        ("Naughton Unit 3", "", "trailing-enumerator"),
        # Legal / regulatory.
        ("SEC Rule 10b-5", "", "trailing-enumerator"),
        ("Annex IV", "", "trailing-enumerator"),
        # Pharma / research.
        ("Study 301", "", "trailing-enumerator"),
        ("Protocol 12B", "", "trailing-enumerator"),
        # News -- CamelCase year prefix, which the pre-fix heuristic 4 missed
        # because it was gated on the label already containing a space.
        ("2025Factbook", "", "document-title"),
        # The description signal: the strongest and most domain-neutral one,
        # and the ONLY thing that catches the real `ChollaUnit4` label
        # deterministically. This description is verbatim from the live DB.
        (
            "ChollaUnit4",
            "Cholla Unit 4, a specific power plant unit referenced in Arizona SIP.",
            "individual-description",
        ),
        (
            "Gadsby Peakers, a specific peaking plant operated in Utah",
            "Gadsby Peakers, a specific peaking plant operated in Utah.",
            "individual-description",
        ),
    ],
)
def test_individual_shaped_labels_are_demoted(
    label: str, description: str, reason: str
) -> None:
    is_entity, got = _looks_like_entity_not_class(
        label, description=description, known_places=_DEFAULT_KNOWN_PLACES,
    )
    assert is_entity is True, f"expected {label!r} to be demoted to an instance"
    assert got == reason


@pytest.mark.parametrize(
    "label,description",
    [
        # Pharma / chemistry: digits are INTERNAL, a kind-noun follows. These
        # are the cases a naive "label contains a digit" rule destroys.
        ("Type2Diabetes", ""),
        ("Phase3ClinicalTrial", ""),
        ("Interleukin6", "A cytokine involved in inflammation."),
        ("HER2", "A receptor tyrosine kinase."),
        ("CYP3A4", "A cytochrome P450 enzyme."),
        ("PM2.5", "Fine particulate matter."),
        ("CO2", "Carbon dioxide."),
        ("COVID19", "An infectious respiratory disease."),
        # Finance / supply chain: same shape, genuine categories.
        ("Scope3Emission", ""),
        ("Tier1Supplier", ""),
        ("B2BTransaction", ""),
        # Energy: the KIND, as opposed to the named unit.
        ("CoalFiredGeneratingUnit", ""),
        ("PowerPlantUnit", ""),
        ("GeneratingUnit", ""),
        ("Unit", ""),
        # The series CLASS, as opposed to one item in the series.
        ("AccountingStandardUpdate", ""),
        ("ConsumerPriceIndex", ""),
        # A normal class description must never trip heuristic 7.
        (
            "ElectricityGenerationFacility",
            "A facility that generates electricity for supply to the grid.",
        ),
    ],
)
def test_genuine_kinds_survive_the_individual_heuristics(
    label: str, description: str
) -> None:
    is_entity, reason = _looks_like_entity_not_class(
        label, description=description, known_places=_DEFAULT_KNOWN_PLACES,
    )
    assert is_entity is False, (
        f"expected {label!r} to stay a CLASS (got reason {reason!r}). "
        "Demoting a genuine class also drops every entity that would have "
        "instantiated it, so a false positive here is silent data loss."
    )


@pytest.mark.parametrize(
    "label,reason",
    [
        ("ASU2019-01", "series-designator"),
        ("ASU2018-07", "series-designator"),
        ("NCT02345678", "series-designator"),
        ("IFRS16", "series-designator"),
        # `COVID19` has the SAME shape as `ASU2019-01`. It is nominated too,
        # and that is the point: no regex can separate them, so both go to the
        # Layer-H LLM audit rather than being deleted on a guess.
        ("COVID19", "series-designator"),
        ("ChollaUnit4", "trailing-enumerator-camel"),
        ("DaveJohnstonUnit4", "trailing-enumerator-camel"),
    ],
)
def test_weak_shapes_are_nominated_for_llm_audit_not_demoted(
    label: str, reason: str
) -> None:
    """The weak tier NOMINATES, it does not decide.

    These labels must reach `classification_audit` (which owns the
    CONVERT_TO_INSTANCE verdict) and must NOT be demoted deterministically.
    """
    weak, got = _looks_like_individual_weak(label)
    assert weak is True, f"expected {label!r} to be nominated for the audit"
    assert got == reason
    strong, _ = _looks_like_entity_not_class(
        label, known_places=_DEFAULT_KNOWN_PLACES,
    )
    assert strong is False, (
        f"{label!r} must NOT be demoted by the deterministic tier -- it is "
        "shape-ambiguous and only an LLM can settle it"
    )


@pytest.mark.parametrize(
    "label",
    ["HER2", "CO2", "PM2.5", "CYP3A4", "Interleukin6", "IL6"],
)
def test_single_stem_symbols_are_not_even_nominated(label: str) -> None:
    """Cost control: without the multi-word requirement, every gene, chemical
    and biomarker symbol in a pharma corpus gets nominated for the PAID audit
    and all of them come back "keep"."""
    weak, reason = _looks_like_individual_weak(label)
    assert weak is False, f"{label!r} should not reach the paid audit ({reason})"


def test_domain_allowlist_overrides_every_heuristic() -> None:
    """The per-domain escape hatch, checked before anything else."""
    label, descr = "Cholla Unit 4", "a specific power plant unit"
    assert _looks_like_entity_not_class(label, description=descr)[0] is True
    assert _looks_like_entity_not_class(
        label, description=descr, allowlist=frozenset({"cholla unit 4"}),
    )[0] is False


def test_filter_reads_the_description_from_the_proposal() -> None:
    """Heuristic 7 only works if `_filter_entity_shaped_classes` actually
    passes DESCRIPTION through -- it is the sole signal that catches the real
    `ChollaUnit4` label."""
    stage2_result = {
        "MATCH NOT FOUND": [
            {
                "LABEL": "ChollaUnit4",
                "DESCRIPTION":
                    "Cholla Unit 4, a specific power plant unit referenced "
                    "in Arizona SIP.",
                "PARENT_LABEL": "Infrastructure",
            },
            {
                "LABEL": "CoalFiredGeneratingUnit",
                "DESCRIPTION": "A generating unit fuelled by coal.",
                "PARENT_LABEL": "Infrastructure",
            },
        ],
    }
    updated, demotions = _filter_entity_shaped_classes(stage2_result)
    kept = [e["LABEL"] for e in updated["MATCH NOT FOUND"]]
    promoted = [e["LABEL"] for e in updated["MATCH NOT FOUND INSTANCES"]]
    assert kept == ["CoalFiredGeneratingUnit"]
    assert promoted == ["ChollaUnit4"]
    assert demotions[0]["reason"] == "individual-description"
    # The demoted class keeps its parent as its TYPE_LABEL, so the instance
    # lands under the right kind rather than at owl:Thing.
    assert updated["MATCH NOT FOUND INSTANCES"][0]["TYPE_LABEL"] == "Infrastructure"


@pytest.mark.parametrize(
    "label,description",
    [
        # Measured against the live ontology: the SAME sentence opening, with
        # opposite correct answers. This is why a bare "a specific ..." only
        # nominates for the LLM audit and never demotes on its own.
        ("AffordableCleanEnergyRule",
         "A specific EPA regulation addressing emissions from power plants."),
        ("ComplianceDeadline",
         "A specific date by which regulated entities must comply."),
        ("TermLoanB", "A specific type of term loan."),
        ("GovernmentAgency",
         "A government body responsible for oversight in a specific domain."),
        ("SegmentOperatingExpense",
         "Operating expenses allocated to a specific business segment."),
        ("moratorium",
         "A temporary suspension of a particular activity, such as rate changes."),
    ],
)
def test_bare_specific_in_a_description_never_demotes_on_its_own(
    label: str, description: str
) -> None:
    """`AffordableCleanEnergyRule` IS one specific rule; `ComplianceDeadline`
    is a category of date. Both descriptions open "A specific ...", so no
    regex can separate them -- the weak tier hands them to the audit instead.
    """
    strong, reason = _looks_like_entity_not_class(label, description=description)
    assert strong is False, f"{label!r} demoted on an ambiguous signal ({reason})"
    weak, wreason = _looks_like_individual_weak(label, description=description)
    assert weak is True, f"{label!r} should still reach the LLM audit"
    assert wreason == "individual-description-weak"


def test_finance_instrument_notes_survive_the_document_tail_rule() -> None:
    """"Note" and "Paper" head real debt-instrument classes. They were moved
    from the STRONG to the WEAK document tails after withholding
    `promissory note`, `floating rate note` and `ExchangeableNote` from a
    finance ontology."""
    for label in ("promissory note", "floating rate note", "medium term note",
                  "ExchangeableNote", "JuniorSubordinatedNote",
                  "commercial paper"):
        assert _looks_like_entity_not_class(label)[0] is False, label
    # A genuine named document still demotes, because the second signal fires.
    assert _looks_like_entity_not_class("2025 Outlook Note")[0] is True


@pytest.mark.parametrize(
    "label",
    ["MISO", "AESO", "PSALM", "CPUC", "IPUC", "UPSC", "WUTC", "GEMA"],
)
def test_bare_acronyms_are_nominated_for_audit(label: str) -> None:
    """Measured on a utility 10-K: 20 of 393 minted classes were bare
    acronyms, and every one named ONE organization. Extraction then typed each
    organization to its own eponymous class -- `AESO` typed `AESO` and
    `Alberta Electric System Operator`, and nothing else, ever."""
    weak, reason = _looks_like_individual_weak(label)
    assert weak is True
    assert reason == "bare-acronym"
    # Never demoted deterministically: GDP / ESG / EBITDA are the same shape
    # and are genuine concepts. Only the LLM can tell them apart.
    assert _looks_like_entity_not_class(label)[0] is False


@pytest.mark.parametrize("label", ["GDP", "ESG", "EBITDA", "NAAQS"])
def test_concept_acronyms_are_never_hard_demoted(label: str) -> None:
    assert _looks_like_entity_not_class(label)[0] is False
