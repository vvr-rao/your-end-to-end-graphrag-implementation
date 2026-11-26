"""Wide-table profiling and id-driven continuation (mocked model replies).

Synthetic fixtures only: forced truncation with a low output cap, related
columns straddling a batch boundary, more than six continuation calls, and
resume from a checkpoint with no missing or duplicate ids.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from pathlib import Path

from backend.app.services.structured import cache as tcache
from backend.app.services.structured.continuation import Budget, StageStats, parse_reply, run_by_ids
from backend.app.services.structured.discovery import discover_tables
from backend.app.services.structured.profiling import profile_table
from backend.app.services.structured.refine import _refine_batched
from backend.app.services.structured.wide_profile import column_families, profile_wide, structural_evidence


@dataclass
class Res:
    text: str
    finish_reason: str = "stop"
    completion_tokens: int = 100
    reasoning_tokens: int = 0
    requested_max_tokens: int | None = 1000
    effective_max_tokens: int | None = 1000
    attempts: int = 1


def _cut(obj: dict, key: str, keep: int) -> str:
    """A reply cut off by the output cap after ``keep`` complete records
    (the next record is half-written)."""
    recs = obj[key]
    head = json.dumps({key: recs[:keep]})[:-2]
    nxt = json.dumps(recs[keep]) if keep < len(recs) else '{"x": 1}'
    return head + (", " if keep else "") + nxt[: len(nxt) // 2]


# ---------------------------------------------------------------- engine


def test_truncated_reply_keeps_complete_records_and_retries_the_rest():
    seen: list[list[str]] = []

    async def call(batch, _n):
        seen.append(list(batch))
        recs = [{"id": i, "v": i.upper()} for i in batch]
        if len(batch) > 2:          # a low cap: only 2 complete records fit
            return parse_reply(_cut({"items": recs}, "items", 2))[0].get("items", []), \
                Res("", finish_reason="length")
        return recs, Res("")

    out, failed, st = asyncio.run(run_by_ids("t", [f"c{i}" for i in range(7)], call,
                                             validate=lambda _i, r: None if r.get("v") else "no v",
                                             batch_size=5, budget=Budget(max_calls=50)))
    assert sorted(out) == [f"c{i}" for i in range(7)] and not failed
    assert st.truncated_calls >= 2 and min(st.batch_sizes) <= 2       # shrank after truncation
    assert all(r["v"] == r["id"].upper() for r in out.values())         # no partial record kept


def test_partial_record_is_never_published():
    text = '{"items": [{"id": "a", "v": "A"}, {"id": "b", "v": "B'
    obj, salvaged = parse_reply(text)
    assert salvaged and [r["id"] for r in obj["items"]] == ["a"]


def test_budget_and_no_progress_give_explicit_statuses():
    async def never(batch, _n):
        return [], Res("{}")

    _out, failed, st = asyncio.run(run_by_ids("t", ["a", "b"], never, validate=lambda *_: None,
                                              batch_size=2, budget=Budget(no_progress_calls=2)))
    assert set(failed.values()) == {"invalid_reply"} and st.stop_reason == "no_progress"

    async def slow(batch, _n):
        return [{"id": batch[0]}], Res("")

    _out, failed, st = asyncio.run(run_by_ids("t", list("abcd"), slow, validate=lambda *_: None,
                                              batch_size=1, budget=Budget(max_calls=2)))
    assert failed == {"c": "incomplete_due_to_budget", "d": "incomplete_due_to_budget"}
    assert st.remaining == 2 and st.stop_reason == "budget"


def test_stats_do_not_double_count_reasoning_tokens():
    st = StageStats(stage="s")
    st.add_call(Res("", completion_tokens=900, reasoning_tokens=600, attempts=2), 3)
    st.add_call(Res("", completion_tokens=100, reasoning_tokens=0, finish_reason="length"), 3)
    assert st.output_tokens == 1000 and st.reasoning_tokens == 600      # reasoning is a subset
    assert st.retries == 1 and st.finish_reasons == {"stop": 1, "length": 1}
    assert st.requested_cap == 1000 and st.effective_cap == 1000


# ---------------------------------------------------------------- wide profiling


def _wide_csv(root: Path, ncols: int = 90) -> Path:
    # 'metric_2021'..'metric_2023' straddle the 40-column boundary (columns 39-41)
    names = ["row_id"] + [f"attr_{i:02d}" for i in range(1, 38)] + ["metric_2021", "metric_2022", "metric_2023"]
    names += [f"extra_{i:02d}" for i in range(ncols - len(names))]
    rows = [",".join(names)]
    for r in range(30):
        rows.append(",".join([str(r)] + [str((r * 7 + j) % 11) for j in range(1, len(names))]))
    d = root / "wide"
    d.mkdir()
    (d / "wide.csv").write_bytes(("\n".join(rows) + "\n").encode())
    return d


class WideRouter:
    """Answers the three wide-profiling prompts; column calls are cut off after
    ``cap_records`` complete records (a low output cap)."""

    def __init__(self, cap_records: int = 1000, fail_after: int | None = None):
        self.cap = cap_records
        self.calls: list[tuple[str, str]] = []
        self.fail_after = fail_after

    def task_spec(self, task):
        return {"provider": "fake", "model": "m1"}

    async def chat(self, task, *, system, user, **kw):
        if self.fail_after is not None and len(self.calls) >= self.fail_after:
            raise RuntimeError("simulated outage")
        self.calls.append((task, user))
        if "ASSIGNED COLUMNS:" in user:
            batch = json.loads(user.split("ASSIGNED COLUMNS:\n", 1)[1].split("\n\n", 1)[0])
            recs = [{"name": n, "description": f"column {n}", "role": "measure" if n.startswith("metric") else "attribute",
                     "roles": ["quantity"] if n.startswith("metric") else ["descriptive"],
                     **({"family": "metric"} if n.startswith("metric") else {})} for n in batch]
            if len(recs) > self.cap:
                return Res(_cut({"columns": recs}, "columns", self.cap), finish_reason="length")
            return Res(json.dumps({"columns": recs}))
        if "ALREADY LISTED:" in user:
            return Res(json.dumps({"description": "synthetic", "grain": "one row per row_id",
                                   "candidates": [{"id": "metric", "label": "Metric",
                                                   "source_columns": ["metric_2021", "metric_2022", "metric_2023"]},
                                                  {"id": "attr_total", "label": "Attribute total",
                                                   "source_columns": ["attr_01"]}],
                                   "more_candidates": False}))
        if "CANDIDATES:\n" in user:
            cands = json.loads(user.split("CANDIDATES:\n", 1)[1].split("\n\nReturn JSON", 1)[0])
            return Res(json.dumps({"capabilities": [
                {"id": c["id"], "metrics": [{"id": f"sum_{c['id']}", "label": "Total",
                                             "query": {"select": [{"agg": "count", "as": "n"}]}}]}
                for c in cands]}))
        raise AssertionError("unexpected prompt")


def _src(tmp_path: Path):
    folder = _wide_csv(tmp_path)
    src = discover_tables(folder).tables[0]
    return src, profile_table(src)


def test_related_columns_across_a_batch_boundary_share_schema_and_evidence(tmp_path):
    src, prof = _src(tmp_path)
    assert column_families([c.name for c in prof.columns])["metric"] == ["metric_2021", "metric_2022", "metric_2023"]
    router = WideRouter()
    reply, rep = asyncio.run(profile_wide(src, prof, router, cands=[], cache=tcache.MetadataCache(user_dir=tmp_path / "c"),
                                          column_batch=40, budget=Budget(max_batch=40), log=lambda *_: None))
    col_calls = [u for t, u in router.calls if "ASSIGNED COLUMNS:" in u]
    assert len(col_calls) == 3
    first = json.loads(col_calls[0].split("ASSIGNED COLUMNS:\n", 1)[1].split("\n\n", 1)[0])
    second = json.loads(col_calls[1].split("ASSIGNED COLUMNS:\n", 1)[1].split("\n\n", 1)[0])
    assert "metric_2021" in first and "metric_2023" in second            # the family straddles the boundary
    for u in col_calls:                                                    # ...but every batch sees all of it
        assert "metric_2023 [" in u and '"metric": ["metric_2021", "metric_2022", "metric_2023"]' in u
    assert rep["complete"] and len(reply["columns"]) == 90
    assert len({c["name"] for c in reply["columns"]}) == 90                # one record per column id
    assert {c["family"] for c in reply["columns"] if c["name"].startswith("metric")} == {"metric"}
    assert [m["capability_ids"] for m in reply["metrics"]] == [["cand_metric"], ["cand_attr_total"]]
    assert structural_evidence(prof)["single_column_keys"] == ["row_id"]


def test_low_cap_forces_truncation_and_still_covers_every_column(tmp_path):
    src, prof = _src(tmp_path)
    router = WideRouter(cap_records=7)
    reply, rep = asyncio.run(profile_wide(src, prof, router, cands=[], cache=tcache.MetadataCache(user_dir=tmp_path / "c"),
                                          column_batch=40, budget=Budget(max_calls=200), log=lambda *_: None))
    st = next(s for s in rep["stages"] if s["stage"] == "profile_columns")
    assert st["truncated_calls"] > 0 and st["calls"] > 6                  # more than six continuation calls
    assert st["finish_reasons"].get("length") and min(st["batch_sizes"]) < 40
    assert rep["complete"] and sorted(c["name"] for c in reply["columns"]) == sorted(c.name for c in prof.columns)


def test_resume_from_checkpoint_has_no_missing_or_duplicate_ids(tmp_path):
    src, prof = _src(tmp_path)
    cache = tcache.MetadataCache(user_dir=tmp_path / "c")
    r1 = WideRouter(cap_records=7)
    _reply, rep1 = asyncio.run(profile_wide(src, prof, r1, cands=[], cache=cache, column_batch=40,
                                            budget=Budget(max_calls=3), log=lambda *_: None))
    assert not rep1["complete"]
    assert "incomplete_due_to_budget" in set(rep1["coverage"]["columns"].values())
    done1 = {n for n, v in rep1["coverage"]["columns"].items() if v == "done"}
    assert Path(rep1["checkpoint"]).exists() and done1

    r2 = WideRouter(cap_records=7)
    reply, rep2 = asyncio.run(profile_wide(src, prof, r2, cands=[], cache=cache, column_batch=40,
                                           budget=Budget(max_calls=200), log=lambda *_: None))
    asked = [n for _t, u in r2.calls if "ASSIGNED COLUMNS:" in u
             for n in json.loads(u.split("ASSIGNED COLUMNS:\n", 1)[1].split("\n\n", 1)[0])]
    assert not (set(asked) & done1)                                        # finished work not redone
    names = [c["name"] for c in reply["columns"]]
    assert rep2["complete"] and len(names) == len(set(names)) == len(prof.columns)


def test_profile_folder_uses_the_wide_path_and_records_coverage(tmp_path):
    from backend.app.services.structured.profiler import profile_folder

    folder = _wide_csv(tmp_path)
    router = WideRouter()
    run = asyncio.run(profile_folder(folder, router, cache=tcache.MetadataCache(user_dir=tmp_path / "c"),
                                     log=lambda *_: None))
    meta = run.tables[0]
    assert meta["profile_run"]["complete"] and meta["inventory"]["coverage"]["synthesis"] == "done"
    assert all(c["description"] for c in meta["columns"])
    assert [c["candidate_id"] if "candidate_id" in c else c["id"] for c in meta["inventory"]["candidates"]] \
        == ["cand_metric", "cand_attr_total"]


# ---------------------------------------------------------------- refine continuation


class RefineRouter:
    def __init__(self, spec):
        self.spec = spec
        self.calls: list[list[str]] = []

    def task_spec(self, task):
        return self.spec

    async def chat(self, task, *, system, user, **kw):
        m = re.search(r"Cover exactly these candidate_ids now:\n(\{.*?\n\})", user, re.S)
        batch = json.loads(m.group(1))["candidate_ids"]
        self.calls.append(batch)
        inv = [{"candidate_id": c, "candidate": c, "status": "materialized", "measure": f"m_{c}"} for c in batch]
        meas = [{"id": f"m_{c}", "value_column": "v"} for c in batch]
        return Res(json.dumps({"columns": [], "measures": meas, "inventory": inv}))


def test_refine_continues_by_candidate_id_past_six_calls():
    cands = [{"id": f"q{i}", "label": f"Quantity {i}"} for i in range(13)]
    ptable = {"table": "t", "candidate_inventory": {"candidates": cands}}
    router = RefineRouter({"provider": "fake", "model": "m", "continuation": {"batch": 2, "max_batch": 2}})
    stats: list = []
    acc, calls = asyncio.run(_refine_batched(router, ptable, log=lambda *_: None, table="t", stats=stats))
    assert calls == 7 > 6
    ids = [e["candidate_id"] for e in acc["inventory"]]
    assert len(ids) == len(set(ids)) == 13 and all(e["status"] == "materialized" for e in acc["inventory"])
    assert len({m["id"] for m in acc["measures"]}) == len(acc["measures"]) == 13
    assert stats[0]["ids_done"] == 13 and stats[0]["remaining"] == 0


def test_refine_budget_leaves_explicit_incomplete_entries():
    cands = [{"id": f"q{i}", "label": f"Quantity {i}"} for i in range(5)]
    ptable = {"table": "t", "candidate_inventory": {"candidates": cands}}
    router = RefineRouter({"provider": "fake", "model": "m", "continuation": {"batch": 2, "max_calls": 1}})
    acc, calls = asyncio.run(_refine_batched(router, ptable, log=lambda *_: None, table="t"))
    st = {e["candidate_id"]: e["status"] for e in acc["inventory"]}
    assert calls == 1 and len(st) == 5
    assert sorted(st.values()) == ["incomplete_due_to_budget"] * 3 + ["materialized"] * 2
