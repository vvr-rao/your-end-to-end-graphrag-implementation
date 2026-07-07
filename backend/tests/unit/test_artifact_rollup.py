"""Unit tests for the artifact-rollup building blocks: union-find clustering,
the artifact_merge prompt shape + registry, and task/model wiring."""

from __future__ import annotations

import uuid

from backend.app.services.db_artifact_rollup import (
    ALL_ROLLUP_TYPES,
    _level_predicate,
    _UnionFind,
)
from backend.app.services.prompts import PROMPTS, artifact_merge


def test_union_find_disjoint_and_chained() -> None:
    a, b, c, d, e = [uuid.uuid4() for _ in range(5)]
    uf = _UnionFind([a, b, c, d, e])
    # chain a-b-c into one component; d-e into another; (leave nothing else)
    uf.union(a, b)
    uf.union(b, c)
    uf.union(d, e)
    sizes = sorted(len(v) for v in uf.components().values())
    assert sizes == [2, 3]
    assert uf.find(a) == uf.find(c)  # transitive closure
    assert uf.find(a) != uf.find(d)


def test_union_find_singletons_are_their_own_component() -> None:
    ids = [uuid.uuid4() for _ in range(4)]
    uf = _UnionFind(ids)
    comps = uf.components()
    assert len(comps) == 4
    assert all(len(v) == 1 for v in comps.values())


def test_union_find_idempotent_union() -> None:
    a, b = uuid.uuid4(), uuid.uuid4()
    uf = _UnionFind([a, b])
    uf.union(a, b)
    uf.union(a, b)  # repeat must not corrupt sizes
    comps = uf.components()
    assert len(comps) == 1
    assert next(iter(comps.values())).__len__() == 2


def test_artifact_merge_prompt_shape() -> None:
    system, user = artifact_merge(
        "Claim",
        ["A owns 30% of B.", "A owns 30% of B.", "B is headquartered in Ohio."],
    )
    # Lossless / dedup framing must be present.
    assert "LOSSLESS" in system.upper()
    assert "duplicate" in system.lower()
    assert "text" in system and "confidence" in system  # JSON contract
    # Every input item is enumerated into the user message.
    assert "A owns 30% of B." in user
    assert "Ohio" in user
    assert "CLAIM ITEMS TO MERGE" in user


def test_artifact_merge_registered() -> None:
    assert "artifact_merge" in PROMPTS
    assert PROMPTS["artifact_merge"] is artifact_merge


def test_merge_eval_revise_prompts_registered() -> None:
    for k in ("artifact_merge_evaluate", "artifact_merge_revise"):
        assert k in PROMPTS
    # evaluate: returns the {complete, missing_items} contract + includes merged text
    esys, euser = PROMPTS["artifact_merge_evaluate"](
        "Claim", ["A owns 30% of B.", "B is in Ohio."], "A owns 30% of B.")
    assert "complete" in esys and "missing_items" in esys
    assert "B is in Ohio." in euser and "A owns 30% of B." in euser
    # revise: includes the missing items + the {text, confidence} contract
    rsys, ruser = PROMPTS["artifact_merge_revise"](
        "Claim", ["A owns 30% of B.", "B is in Ohio."], "A owns 30% of B.", ["B is in Ohio."])
    assert "text" in rsys and "confidence" in rsys
    assert "B is in Ohio." in ruser


def test_level_predicate_leaves_vs_rollups() -> None:
    leaf = _level_predicate(0)
    assert "IS DISTINCT FROM 'true'" in leaf
    lvl1 = _level_predicate(1)
    assert "'rollup') = 'true'" in lvl1
    assert "'layer') = '1'" in lvl1


def test_all_rollup_types_default_set() -> None:
    # StructuredTable excluded (tabular JSON-LD). Summary IS included: --rollup
    # additively adds layers on top of its automatic rollup.
    assert "StructuredTable" not in ALL_ROLLUP_TYPES
    assert "Summary" in ALL_ROLLUP_TYPES
    assert "Claim" in ALL_ROLLUP_TYPES
    assert "Insight" in ALL_ROLLUP_TYPES


def test_additive_level_math() -> None:
    # generate_rollups adds `layers` NEW levels on top of a type's base level:
    # src_level = base + step - 1, new_level = base + step.
    def new_levels(base: int, layers: int) -> list[int]:
        return [base + step for step in range(1, layers + 1)]

    assert new_levels(0, 2) == [1, 2]   # fresh leaves -> layers 1,2
    assert new_levels(2, 2) == [3, 4]   # already at layer 2 -> adds 3,4
    assert new_levels(4, 1) == [5]
