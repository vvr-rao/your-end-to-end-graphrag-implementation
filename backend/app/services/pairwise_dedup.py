"""Pairwise class dedup at scale: nearest-neighbour candidates + a pair judge.

Replaces the LLM "echo" dedup, which sees ~200 proposals per call: on 20,174
web-search proposals it returned 13,844 classes from 13,828 distinct labels --
nothing merged beyond identical labels -- after 19 GPU-minutes. Here:

  1. candidates   each kept class's top-k neighbours (HNSW, staging store)
                  among the run's classes AND the existing ontology classes;
  2. bands        cosine >= t_hi -> merge; [t_lo, t_hi) -> pair judge;
                  < t_lo -> keep apart (not recorded);
  3. judge        `class_pair_judge` (Qwen, its own prompt), batched;
  4. union-find   over label ids, a group size cap so a chain cannot fuse
                  unrelated concepts; canonical = an existing class, else the
                  most frequent label, else the shortest;
  5. apply        status=merged / merged_into=canonical in the store; every
                  decision lands in pair_decisions (the measurement input).

Work and memory are per candidate batch; the union-find holds label ids only.
"""
from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from backend.app.services.ontology_build_store import BuildStore, Pair

TASK = "class_pair_judge"


@dataclass
class DedupConfig:
    # 0.95 still auto-merged siblings (ExportRestriction~ExportBan, 0.956) in
    # the 2026-10-18 replay; 0.97 until the labelled calibration set sets it.
    t_hi: float = 0.97
    t_lo: float = 0.80
    k: int = 10
    judge_batch: int = 60
    judge_concurrency: int = 8
    max_group: int = 25


@dataclass
class DedupStats:
    candidates: int = 0
    auto_merge: int = 0
    judged: int = 0
    judged_same: int = 0
    merged_labels: int = 0
    group_cap_refusals: int = 0
    judge_failures: int = 0
    to_existing: int = 0
    already_joined: int = 0
    exact_to_existing: int = 0
    by_band: dict[str, int] = field(default_factory=dict)


class UnionFind:
    """Groups over label ids with a size cap. `meta[id]` = (is_existing,
    occurrences, label) decides the canonical of each group."""

    def __init__(self, max_group: int) -> None:
        self.parent: dict[int, int] = {}
        self.size: dict[int, int] = {}
        self.max_group = max_group

    def find(self, x: int) -> int:
        self.parent.setdefault(x, x)
        self.size.setdefault(x, 1)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return True
        if self.size[ra] + self.size[rb] > self.max_group:
            return False
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        return True

    def groups(self) -> dict[int, list[int]]:
        out: dict[int, list[int]] = {}
        for x in list(self.parent):
            out.setdefault(self.find(x), []).append(x)
        return {r: m for r, m in out.items() if len(m) > 1}


def canonical(members: list[int], meta: dict[int, tuple[bool, int, str]]) -> int:
    """An existing class wins; then most occurrences; then the shortest label;
    then the lowest id (deterministic)."""
    return min(members, key=lambda i: (not meta[i][0], -meta[i][1], len(meta[i][2]), i))


def band(cos: float, cfg: DedupConfig) -> str:
    if cos >= cfg.t_hi:
        return "auto"
    if cos >= cfg.t_lo:
        return "judge"
    return "skip"


Judge = Callable[[list[dict[str, Any]]], Awaitable[list[dict[str, Any]]]]


def router_judge(router: Any) -> Judge:
    """The pair judge on the `class_pair_judge` task (Qwen prompt)."""
    from backend.app.services.prompts_qwen import qwen_class_pair_judge

    async def judge(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
        system, user = qwen_class_pair_judge(batch)
        out = await router.chat(TASK, system=system, user=user)
        from backend.app.helpers.ontology_pruning import extract_json_from_output

        data = extract_json_from_output(out.text) or {}
        return [d for d in (data.get("decisions") or []) if isinstance(d, dict)]

    return judge


async def run_pairwise_dedup(store: BuildStore, judge: Judge | None, cfg: DedupConfig | None = None,
                             log=print) -> DedupStats:
    """Dedup the run's kept classes in place in the store. `judge=None` skips
    the judge band (counts it), e.g. for a free replay."""
    cfg = cfg or DedupConfig()
    st = DedupStats()
    uf = UnionFind(cfg.max_group)
    meta: dict[int, tuple[bool, int, str]] = {}
    sem = asyncio.Semaphore(cfg.judge_concurrency)

    cache: dict[int, dict[str, Any]] = {}

    async def remember(ids: list[int]) -> dict[int, dict[str, Any]]:
        """Fetch rows not yet seen (labels/descriptions for the judge, meta
        for the canonical choice). Holds one dict per label touched."""
        need = list({i for i in ids if i not in cache})
        for i, r in (await store.rows(need)).items():
            cache[i] = r
            meta[i] = (r["kind"] == "existing_class", int(r["occurrences"] or 1), str(r["label"]))
        return cache

    async def judge_batch(batch: list[tuple[Pair, dict, dict]]) -> list[tuple[Pair, bool, str]]:
        payload = [{"id": n, "a_label": a["label"], "a_desc": a.get("description"), "a_parent": a.get("parent"),
                    "b_label": b["label"], "b_desc": b.get("description"), "b_parent": b.get("parent"),
                    "b_existing": b["kind"] == "existing_class"} for n, (_p, a, b) in enumerate(batch)]
        async with sem:
            try:
                decisions = await judge(payload)  # type: ignore[misc]
            except Exception as exc:          # a failed batch keeps its pairs apart
                log(f"[pairwise-dedup] judge batch failed: {type(exc).__name__}: {exc}")
                st.judge_failures += len(batch)
                return []
        # The judge lists only the SAME pairs; every other id is distinct.
        same = {str(d.get("id")): d for d in decisions if d.get("same", True) is not False}
        return [(p, str(n) in same, str(same.get(str(n), {}).get("reason") or "")[:200])
                for n, (p, _a, _b) in enumerate(batch)]

    # Exact folded-label match to an existing class: merge with no lookup.
    exact = await store.exact_existing_matches()
    await remember([i for pair in exact for i in pair])
    for cls_id, ex_id in exact:
        if uf.union(cls_id, ex_id):
            st.exact_to_existing += 1
    for to_kind in ("existing_class", "class"):
        async for pairs in store.neighbours("class", to_kind, k=cfg.k, min_cosine=cfg.t_lo):
            if not pairs:
                continue
            rows = await remember([i for p in pairs for i in (p.a_id, p.b_id)])
            st.candidates += len(pairs)
            decisions: list[tuple[int, int, float, str, str | None]] = []
            to_judge: list[tuple[Pair, dict, dict]] = []
            for p in pairs:
                if uf.find(p.a_id) == uf.find(p.b_id):     # already one group: no call
                    st.already_joined += 1
                    continue
                b = band(p.cosine, cfg)
                st.by_band[b] = st.by_band.get(b, 0) + 1
                if b == "auto":
                    st.auto_merge += 1
                    if not uf.union(p.a_id, p.b_id):
                        st.group_cap_refusals += 1
                    decisions.append((p.a_id, p.b_id, p.cosine, "auto_merge", None))
                elif judge is not None:
                    to_judge.append((p, rows[p.a_id], rows[p.b_id]))
            if to_judge:
                results = await asyncio.gather(*[judge_batch(to_judge[i:i + cfg.judge_batch])
                                                 for i in range(0, len(to_judge), cfg.judge_batch)])
                for res in results:
                    for p, same, why in res:
                        st.judged += 1
                        if same:
                            st.judged_same += 1
                            if not uf.union(p.a_id, p.b_id):
                                st.group_cap_refusals += 1
                        decisions.append((p.a_id, p.b_id, p.cosine,
                                          "judged_same" if same else "judged_distinct", why))
            await store.record(decisions)
    for members in uf.groups().values():
        canon = canonical(members, meta)
        others = [m for m in members if m != canon and not meta[m][0]]   # never retire an existing class
        if meta[canon][0]:
            st.to_existing += len(others)
        await store.set_status(others, "merged", "pairwise", merged_into=canon)
        st.merged_labels += len(others)
    log("[pairwise-dedup] " + json.dumps(st.__dict__))
    return st
