"""Relationship-aware walk: later hops must not re-take edges an earlier hop
already matched (D15). Measured on fingen, 91-100% of hop-2 hits were hop-1
edges -- the hop-2 frontier touches the edge that reached it, and that edge
matched the relation once, so it scores at the top again."""
from __future__ import annotations

import asyncio
import contextlib
import uuid

from backend.app.services import retrieval as rt
from backend.app.services import retrieval_sql as rsql

SEED, A, B = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
E1, E2 = uuid.uuid4(), uuid.uuid4()
# SEED -E1(0.9)- A -E2(0.8)- B. E1 outranks E2 for any frontier holding A.
EDGES = [(E1, SEED, A, 0.9), (E2, A, B, 0.8)]


def _patch(monkeypatch, *, limit: int):
    calls: list[dict] = []

    @contextlib.asynccontextmanager
    async def scope():
        yield None

    async def hop(session, frontier, probes, *, threshold, limit, exclude_edge_ids=None):
        seen = set(exclude_edge_ids or ())
        calls.append({"frontier": list(frontier), "seen": seen})
        rows = [{"id": e, "source_node_id": s, "target_node_id": t, "sim": sim}
                for e, s, t, sim in EDGES
                if (s in frontier or t in frontier) and e not in seen]
        return sorted(rows, key=lambda r: -r["sim"])[:limit]

    monkeypatch.setattr(rt, "session_scope", scope)
    monkeypatch.setattr(rsql, "relationship_hop", hop)
    cfg = {"relationship_similarity": 0.5, "relationship_hop_limit": limit,
           "relationship_hop_decay": 0.7}
    monkeypatch.setattr(rt, "_qa_cfg", lambda key, default: cfg.get(key, default))
    return calls


def test_later_hops_exclude_edges_already_taken(monkeypatch) -> None:
    calls = _patch(monkeypatch, limit=50)
    reached, edges, stage = asyncio.run(rt._relationship_walk([SEED], [[0.0]], max_hops=3))
    assert stage == "anchored"
    assert edges == [E1, E2]                       # each edge once
    assert calls[1]["frontier"] == [A] and calls[1]["seen"] == {E1}
    assert set(reached) == {SEED, A, B}


def test_a_full_hop_is_not_crowded_out_by_the_previous_edge(monkeypatch) -> None:
    # limit 1: without the exclusion hop 2 spends its only slot re-taking E1
    # and never reaches B.
    _patch(monkeypatch, limit=1)
    reached, edges, _ = asyncio.run(rt._relationship_walk([SEED], [[0.0]], max_hops=2))
    assert edges == [E1, E2] and B in reached


def test_hop_sql_excludes_seen_edges_before_the_limit() -> None:
    sql = rsql._REL_HOP_SQL.text
    assert sql.index("NOT (g.id = ANY(CAST(:seen AS uuid[])))") < sql.index("LIMIT :limit")


def test_relationship_hop_binds_the_seen_list(monkeypatch) -> None:
    bound: list[dict] = []

    class _Rows:
        def all(self):
            return []

    class _Session:
        async def execute(self, stmt, params=None):
            if params is not None:
                bound.append(params)
            return _Rows()

    asyncio.run(rsql.relationship_hop(_Session(), [SEED], [[0.0]], exclude_edge_ids=[E1]))
    assert bound and bound[-1]["seen"] == [str(E1)]
    asyncio.run(rsql.relationship_hop(_Session(), [SEED], [[0.0]]))
    assert bound[-1]["seen"] == []
