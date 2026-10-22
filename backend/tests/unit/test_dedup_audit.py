"""dedup-audit estimates over a fake store and judge."""
from __future__ import annotations

import asyncio

import pytest

from backend.app.services.dedup_audit import audit, band_name, export_csv, wilson


def test_wilson_interval() -> None:
    lo, hi = wilson(0, 40)
    assert lo == 0.0 and 0.05 < hi < 0.12
    lo, hi = wilson(20, 40)
    assert lo < 0.5 < hi
    assert wilson(0, 0) == (0.0, 1.0)


def test_band_names() -> None:
    assert band_name(0.97) == "0.95-1.00" and band_name(0.86) == "0.85-0.90" and band_name(0.7) is None


class Store:
    def __init__(self):
        self.r = {i: {"id": i, "kind": "class", "label": f"L{i}", "description": None, "parent": None,
                      "status": "kept", "status_reason": None, "merged_into": None} for i in range(1, 11)}
        self.r[20] = {**self.r[1], "id": 20, "label": "L20", "status": "merged", "merged_into": 1}

    async def count(self, kind, status):
        return sum(1 for r in self.r.values() if r["status"] == status)

    async def sample(self, kind, status, n, seed=0.0, reason=None):
        return [r for r in self.r.values() if r["status"] == status][:n]

    async def knn_for(self, aid, to_kind, k):
        if to_kind != "class":
            return []
        return [(aid % 10 + 1, 0.97), ((aid + 3) % 10 + 1, 0.83)]   # one high, one low-band neighbour each

    async def rows(self, ids):
        return {i: self.r[i] for i in ids if i in self.r}


def test_audit_estimates_leftovers_and_over_merges(tmp_path) -> None:
    async def judge(batch):                    # high-band pairs are duplicates, low-band are not
        return [{"id": p["id"]} for p in batch if int(p["a_label"][1:]) % 10 + 1 == int(p["b_label"][1:])
                or int(p["b_label"][1:]) % 10 + 1 == int(p["a_label"][1:])]

    rep = asyncio.run(audit(Store(), judge, label_sample=10, per_band=40))
    assert rep.bands["0.95-1.00"]["same_rate"] == 1.0
    assert rep.bands["0.80-0.85"]["same_rate"] == 0.0
    assert rep.est_leftover_duplicate_pairs == pytest.approx(rep.bands["0.95-1.00"]["est_pairs"], abs=1)
    assert rep.over_merge["sampled"] == 1
    out = export_csv(rep, tmp_path / "audit.csv")
    assert out.read_text().splitlines()[0] == "sample,a_or_label,b_or_reason,judge,human"
