"""Stage 4's incremental class indexes must equal full rebuilds (2026-10-19:
rebuilding them per item made Stage 4 quadratic -- 29.5 min -> 3.8 s on the
13.8k-class web-search build, identical output)."""
from __future__ import annotations

import random

import backend.app.helpers.ontology_pruning as op


def _classes(n: int, rng: random.Random) -> dict:
    words = "oil refinery crude market price supply gas plant trade tariff export import".split()
    out = {}
    for i in range(n):
        label = "".join(w.capitalize() for w in rng.sample(words, rng.randint(1, 3)))
        out[f"x#c{i}"] = {"name": f"c{i}", "labels": [label], "superclasses": [{"iri": rng.choice(["x#A", "x#B"])}]}
    return out


def _scan(tokens, parent, classes, token_index):
    """The original step-3 containment scan."""
    for existing_tokens, existing_iri in token_index.items():
        if existing_tokens and (tokens < existing_tokens or existing_tokens < tokens):
            if op._first_super_iri(classes.get(existing_iri) or {}) == parent:
                return existing_iri
    return None


def test_incremental_indexes_equal_full_rebuilds() -> None:
    rng = random.Random(3)
    classes = _classes(40, rng)
    idx = op._ClassIndexes(classes)
    for i in range(40, 140):
        assert idx.label == op._build_label_to_iri_index(classes)
        assert idx.token == op._build_token_index(classes)
        position, _ = idx.postings
        assert list(position) == list(idx.token)                   # insertion order kept
        classes[f"x#n{i}"] = _classes(1, rng)["x#c0"]          # a class minted mid-loop
        idx.added(f"x#n{i}")
    # a change it was not told about triggers a rebuild, never a stale answer
    classes["x#other"] = {"name": "other", "labels": ["Refinery"]}
    assert idx.label == op._build_label_to_iri_index(classes)


def test_indexed_containment_returns_what_the_scan_returns() -> None:
    rng = random.Random(4)
    classes = _classes(300, rng)
    idx = op._ClassIndexes(classes)
    queries = ["OilRefinery", "Refinery", "CrudeOilMarketPrice", "GasPlant", "Tariff", "TradeExportImport"]
    for q in queries:
        tokens = op._tokenize_label(q)
        for parent in ("x#A", "x#B", None):
            want = _scan(tokens, parent, classes, idx.token)
            got = op._containment_lookup(tokens, parent, classes, idx.token, *idx.postings)
            assert got == want, (q, parent)


def test_stem_relations_pairs_unchanged() -> None:
    classes = {f"x#{n}": {"name": n, "labels": [n]} for n in
               ("helium", "helium_market", "helium_supply", "helium_market_price", "neon")}
    props: dict = {}
    op.infer_stem_relations(classes, props, "x#")
    pairs = sorted((p["domain"][0]["iri"], p["range"][0]["iri"]) for p in props.values())
    assert ("x#helium", "x#helium_market") in pairs and ("x#helium", "x#helium_supply") in pairs
    assert ("x#helium_market", "x#helium_market_price") in pairs
    assert not any("neon" in a or "neon" in b for a, b in pairs)
