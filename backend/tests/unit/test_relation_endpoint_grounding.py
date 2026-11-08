from backend.app.services import relation_endpoint_grounding as reg

TEXT = (
    "The group reported cost of sales of $4.2bn. Greenbushes mine output rose. "
    "Semiconductor revenue grew while prices for electric vehicles fell."
)


def _rel(label, domain, rng):
    return {"LABEL": label, "DESCRIPTION": "", "DOMAIN": domain, "RANGE": rng}


def test_name_words_splits_camel_case_and_drops_short_words():
    assert reg.name_words("ElectricVehiclePricePremium") == ["electric", "vehicle", "price", "premium"]
    assert reg.name_words("CostOfSales") == ["cost", "sales"]
    assert reg.name_words("ESGAlignment") == ["esg", "alignment"]


def test_grounded_allows_word_endings_and_nearby_words():
    words = reg.text_words(TEXT)
    assert reg.is_grounded("CostOfSales", words)
    assert reg.is_grounded("MineSite", reg.text_words("the mine site at Greenbushes"))
    assert reg.is_grounded("ElectricVehiclePrice", words)  # "prices for electric vehicles"
    assert not reg.is_grounded("SemiconductorHighEthics", words)
    assert not reg.is_grounded("ElectricVehiclePriceAmbling", words)


def test_ungrounded_new_endpoint_drops_the_relation():
    parsed = {"MATCH NOT FOUND RELATIONS": [
        _rel("reports", "Organization", "CostOfSales"),
        _rel("highEthics", "TechnologyConcept", "SemiconductorHighEthics"),
    ]}
    out, stats = reg.ground_relations(parsed, TEXT, {"organization", "technologyconcept"})
    assert [r["LABEL"] for r in out["MATCH NOT FOUND RELATIONS"]] == ["reports"]
    assert stats == {"ungrounded": 1, "over_cap": 0, "kept_new": 1}


def test_known_classes_iris_and_open_ends_are_never_checked():
    parsed = {
        "MATCH NOT FOUND": [{"LABEL": "Quantum Widget"}],
        "MATCH NOT FOUND RELATIONS": [
            _rel("a", "http://xmlns.com/foaf/0.1/Organization", "Revenue Stream"),
            _rel("b", "Organization", "NONE"),
            _rel("c", "Organization", "QuantumWidget"),
        ],
    }
    out, stats = reg.ground_relations(parsed, "unrelated text", {"organization", "revenuestream"})
    assert len(out["MATCH NOT FOUND RELATIONS"]) == 3
    assert stats["ungrounded"] == 0 and stats["kept_new"] == 0


def test_cap_limits_new_endpoints_per_chunk():
    text = "alpha beta gamma delta"
    parsed = {"MATCH NOT FOUND RELATIONS": [
        _rel("r1", "Known", "Alpha"), _rel("r2", "Known", "Beta"),
        _rel("r3", "Known", "Gamma"), _rel("r4", "Known", "Alpha"),
    ]}
    out, stats = reg.ground_relations(parsed, text, {"known"}, max_new=2)
    assert [r["LABEL"] for r in out["MATCH NOT FOUND RELATIONS"]] == ["r1", "r2", "r4"]
    assert stats == {"ungrounded": 0, "over_cap": 1, "kept_new": 2}


def test_cap_counts_both_ends_of_a_relation_together():
    parsed = {"MATCH NOT FOUND RELATIONS": [_rel("r", "Alpha", "Beta")]}
    out, stats = reg.ground_relations(parsed, "alpha beta", set(), max_new=1)
    assert out["MATCH NOT FOUND RELATIONS"] == []
    assert stats["over_cap"] == 1 and stats["kept_new"] == 0


def test_known_labels_reads_names_labels_and_iri_tails():
    onto = {"classes_dict": {
        "http://x.org/a#CostOfSales": {"labels": ["cost of sales"]},
        "http://x.org/a#B": {"name": "Mine Site", "labels": [{"value": "mine site"}]},
    }}
    labels = reg.known_labels(onto)
    assert {"costofsales", "minesite", "b"} <= labels


def test_markdown_is_ignored_in_the_text():
    parsed = {"MATCH NOT FOUND RELATIONS": [_rel("r", "Known", "CostOfSales")]}
    out, _ = reg.ground_relations(parsed, "**cost** of _sales_", {"known"})
    assert len(out["MATCH NOT FOUND RELATIONS"]) == 1


def test_empty_or_missing_relations_pass_through():
    assert reg.ground_relations(None, TEXT, set())[0] is None
    parsed = {"MATCH NOT FOUND": []}
    assert reg.ground_relations(parsed, TEXT, set())[0] is parsed
