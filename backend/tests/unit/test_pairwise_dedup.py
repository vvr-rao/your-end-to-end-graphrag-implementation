"""Pairwise dedup: band logic, union-find caps, canonical choice, and the
decision flow over a fake store (no database)."""
from __future__ import annotations

import asyncio

from backend.app.services.ontology_build_store import Pair
from backend.app.services.pairwise_dedup import DedupConfig, UnionFind, band, canonical, run_pairwise_dedup


def test_bands() -> None:
    cfg = DedupConfig(t_hi=0.95, t_lo=0.80)
    assert [band(c, cfg) for c in (0.99, 0.95, 0.9, 0.80, 0.79)] == ["auto", "auto", "judge", "judge", "skip"]
    assert band(0.999, DedupConfig()) == "judge"         # no auto-merge band by default
    assert band(0.84, DedupConfig()) == "skip"           # t_lo 0.85 (calibrated 2026-10-19)


def test_union_find_caps_group_size() -> None:
    uf = UnionFind(max_group=3)
    assert uf.union(1, 2) and uf.union(2, 3)
    assert not uf.union(3, 4)                      # would make 4 members
    assert sorted(next(iter(uf.groups().values()))) == [1, 2, 3]


def test_canonical_prefers_existing_then_frequency_then_short() -> None:
    meta = {1: (False, 5, "OilRefinery"), 2: (False, 9, "Refineries"), 3: (True, 1, "Refinery Plant")}
    assert canonical([1, 2, 3], meta) == 3
    assert canonical([1, 2], meta) == 2
    meta2 = {1: (False, 3, "RefineryComplex"), 2: (False, 3, "Refinery")}
    assert canonical([1, 2], meta2) == 2


class FakeStore:
    def __init__(self, rows, pairs_by_kind):
        self._rows, self._pairs, self.status, self.recorded = rows, pairs_by_kind, {}, []
        self._exact = []

    async def exact_existing_matches(self):
        return self._exact

    async def neighbours(self, from_kind, to_kind, k, min_cosine):
        yield [p for p in self._pairs.get(to_kind, []) if p.cosine >= min_cosine]

    async def rows(self, ids):
        return {i: self._rows[i] for i in ids}

    async def record(self, decisions):
        self.recorded += decisions

    async def set_status(self, ids, status, reason=None, merged_into=None):
        for i in ids:
            self.status[i] = (status, merged_into)

    async def set_merged(self, pairs, reason=None):
        for i, canon in pairs:
            self.status[i] = ("merged", canon)


def _row(i, label, kind="class", occ=1):
    return {"id": i, "kind": kind, "label": label, "parent": None, "description": None, "occurrences": occ}


def test_flow_merges_auto_and_judged_and_routes_to_existing() -> None:
    rows = {1: _row(1, "Refineries", occ=4), 2: _row(2, "Refinery Plant", "existing_class"),
            3: _row(3, "Hypertension"), 4: _row(4, "HighBloodPressure"), 5: _row(5, "Import"), 6: _row(6, "Export")}
    pairs = {"existing_class": [Pair(1, 2, 0.98)],
             "class": [Pair(3, 4, 0.88), Pair(5, 6, 0.86), Pair(3, 6, 0.70)]}
    store = FakeStore(rows, pairs)

    async def judge(batch):
        return [{"id": p["id"], "keep": "A"} for p in batch if p["a_label"] == "Hypertension"]

    st = asyncio.run(run_pairwise_dedup(store, judge, DedupConfig(t_hi=0.97), log=lambda *_: None))
    assert store.status[1] == ("merged", 2)                     # routed onto the existing class
    assert 2 not in store.status                                # existing classes are never retired
    assert set(store.status) == {1, 3} or set(store.status) == {1, 4}
    assert (5 not in store.status) and (6 not in store.status)  # judged distinct
    assert st.auto_merge == 1 and st.judged == 2 and st.judged_same == 1 and st.to_existing == 1
    assert {d[3] for d in store.recorded} == {"auto_merge", "judged_same", "judged_distinct"}


def test_no_judge_counts_the_band_without_merging() -> None:
    rows = {3: _row(3, "Hypertension"), 4: _row(4, "HighBloodPressure")}
    store = FakeStore(rows, {"class": [Pair(3, 4, 0.88)]})
    st = asyncio.run(run_pairwise_dedup(store, None, DedupConfig(), log=lambda *_: None))
    assert st.by_band == {"judge": 1} and store.status == {}


def test_exact_match_to_existing_and_already_joined_pairs_cost_no_call() -> None:
    rows = {1: _row(1, "Concept"), 2: _row(2, "Concept", "existing_class"), 3: _row(3, "Concepts")}
    store = FakeStore(rows, {"class": [Pair(1, 3, 0.9)], "existing_class": [Pair(1, 2, 0.99)]})
    store._exact = [(1, 2)]
    calls = []

    async def judge(batch):
        calls.append(batch)
        return [{"id": p["id"], "keep": "B"} for p in batch]

    st = asyncio.run(run_pairwise_dedup(store, judge, DedupConfig(), log=lambda *_: None))
    assert st.exact_to_existing == 1 and st.already_joined == 1      # 1~2 skipped after the exact merge
    assert store.status[1] == ("merged", 2) and store.status[3] == ("merged", 2)
    assert len(calls) == 1
