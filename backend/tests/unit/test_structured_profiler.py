"""structured.profiler with a fake router: parsing, repair, lookups, classes, cache reuse."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from backend.app.services.structured import cache as tc
from backend.app.services.structured.profiler import add_classes_to_ontology, profile_folder


@dataclass
class _Res:
    text: str
    cost_usd: float = 0.0


class FakeRouter:
    """Replies by task; records every call."""

    def __init__(self, replies: dict[str, list[dict]], spec: dict | None = None):
        self.replies = {k: list(v) for k, v in replies.items()}
        self.calls: list[tuple[str, str]] = []
        self.spec = spec or {"provider": "fake", "model": "m1"}

    def task_spec(self, task):
        return self.spec

    async def chat(self, task, *, system, user, **kw):
        self.calls.append((task, user))
        queue = self.replies.get(task) or []
        if not queue:
            raise AssertionError(f"unexpected call to {task}")
        r = queue.pop(0)
        return _Res(r if isinstance(r, str) else json.dumps(r))


def _write(root: Path, rel: str, body: bytes) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(body)


PRICES = b"commodity,period,value\nCrude oil,2024-01-01,80.5\nCopper,2024-01-01,8500\nCrude oil,2024-02-01,82\n"
COMPANIES = b"company,cik,commodity\nAcme Corp,0000320193,Copper\n"


def _profile_reply(table: str, cols: list[str], *, bad_sql: bool = False) -> dict:
    sql_ok = f"SELECT {cols[0]}, COUNT(*) AS n FROM {table} GROUP BY {cols[0]}"
    return {
        "description": f"{table} table", "grain": "one row per thing",
        "columns": [{"name": c, "description": f"the {c}", "role": "dimension",
                     "concept": {"iri": None, "label": f"{c.title()}Thing", "parent_label": "Dimension"}}
                    for c in cols],
        "metrics": [{"id": "count by first", "label": "Count", "description": "rows per value",
                     "columns": [cols[0]], "concept": {"iri": None, "label": "RowCount", "parent_label": "Metric"},
                     "params": [], "sql": sql_ok}],
        "filters": [{"id": "by_first", "label": "By first", "description": "filter",
                     "columns": [cols[0]], "concept": None,
                     "params": [{"name": "v", "type": "text", "column": cols[0], "description": "", "example": "x"}],
                     "sql": f"SELECT * FROM {table} WHERE {cols[0]} = :v"}],
        "queries": [{"id": "bad" if bad_sql else "all_distinct", "label": "q", "description": "",
                     "params": [],
                     "sql": f"DELETE FROM {table}" if bad_sql else f"SELECT DISTINCT {cols[0]} FROM {table}"}],
    }


@pytest.fixture
def folder(tmp_path):
    root = tmp_path / "data"
    _write(root, "worldbank/pink-sheet/s1/manifest.json", b"{}")
    _write(root, "worldbank/pink-sheet/s1/normalized/prices.csv", PRICES)
    _write(root, "crosswalk/companies.csv", COMPANIES)
    return root


def _router(bad_sql=False, lookup=True, repair_ok=True):
    lookups = {"lookups": [{
        "id": "company_prices", "label": "Prices of a company's commodity", "description": "",
        "table": "companies", "other_table": "prices",
        "join": [{"column": "commodity", "other_column": "commodity"}],
        "params": [{"name": "company", "type": "text", "column": "company", "description": "", "example": "Acme"}],
        "sql": "SELECT c.company, p.period, p.value FROM companies c JOIN prices p ON p.commodity = c.commodity "
               "WHERE c.company = :company"}]} if lookup else {"lookups": []}
    repair = {"items": [{"id": "bad", "params": [], "sql": "SELECT COUNT(*) FROM prices"}]} if repair_ok \
        else {"items": [{"id": "bad", "params": [], "sql": "DROP TABLE prices"}]}
    # profile calls run concurrently -> answer by table name in the prompt
    return FakeRouter({"table_lookups": [lookups], "table_profile": []}), \
        _profile_reply("companies", ["company", "cik", "commodity"]), \
        _profile_reply("prices", ["commodity", "period", "value"], bad_sql=bad_sql), repair


class ByTableRouter(FakeRouter):
    def __init__(self, base: FakeRouter, companies: dict, prices: dict, repair: dict):
        super().__init__(base.replies)
        self.companies, self.prices, self.repair = companies, prices, repair

    async def chat(self, task, *, system, user, **kw):
        if task == "table_profile":
            self.calls.append((task, user))
            if "REJECTED STATEMENTS" in user:
                return _Res(json.dumps(self.repair))
            return _Res(json.dumps(self.companies if "TABLE companies" in user else self.prices))
        return await super().chat(task, system=system, user=user)


def _make(bad_sql=False, repair_ok=True):
    base, c, p, r = _router(bad_sql=bad_sql, repair_ok=repair_ok)
    return ByTableRouter(base, c, p, r)


async def test_profile_builds_metadata_and_lookups(folder, tmp_path):
    router = _make()
    cache = tc.MetadataCache(user_dir=tmp_path / "cache", run_dir=tmp_path / "run" / "structured")
    run = await profile_folder(folder, router, cache=cache, log=lambda *_: None)
    assert sorted(run.profiled) == ["companies", "prices"] and run.failed == []
    m = run.by_name()["prices"]
    assert m["row_count"] == 3 and [c["type"] for c in m["columns"]] == ["text", "date", "numeric"]
    assert m["metrics"][0]["id"] == "count_by_first"
    assert m["source"]["dataset"] == "worldbank/pink-sheet"
    # lookup stored in BOTH tables, flipped for the other side
    lk_c = run.by_name()["companies"]["lookups"][0]
    lk_p = m["lookups"][0]
    assert lk_c["other_table"] == "prices" and lk_p["other_table"] == "companies"
    assert lk_p["join"] == [{"column": "commodity", "other_column": "commodity"}]
    assert (tmp_path / "run" / "structured" / "prices.metadata.json").exists()
    assert (tmp_path / "run" / "structured" / "lookups.json").exists()
    assert [t for t, _ in router.calls].count("table_profile") == 2


async def test_second_run_reads_cache_with_zero_llm_calls(folder, tmp_path):
    cache_dir = tmp_path / "cache"
    first = await profile_folder(folder, _make(), cache=tc.MetadataCache(user_dir=cache_dir), log=lambda *_: None)
    silent = FakeRouter({})          # any call raises
    run = await profile_folder(folder, silent, cache=tc.MetadataCache(user_dir=cache_dir), log=lambda *_: None)
    assert silent.calls == [] and run.llm_calls == 0
    assert sorted(run.cached) == ["companies", "prices"] and run.lookups_cached
    assert run.by_name()["prices"]["metrics"] == first.by_name()["prices"]["metrics"]


async def test_cache_survives_llm_mode_switch(folder, tmp_path):
    cache_dir = tmp_path / "cache"
    await profile_folder(folder, _make(), cache=tc.MetadataCache(user_dir=cache_dir), log=lambda *_: None)
    other_mode = FakeRouter({}, spec={"provider": "anthropic", "model": "claude-sonnet-4-6"})
    run = await profile_folder(folder, other_mode, cache=tc.MetadataCache(user_dir=cache_dir), log=lambda *_: None)
    assert other_mode.calls == [] and len(run.cached) == 2


async def test_bad_sql_is_repaired_once(folder, tmp_path):
    router = _make(bad_sql=True)
    run = await profile_folder(folder, router, cache=tc.MetadataCache(user_dir=tmp_path / "c"), log=lambda *_: None)
    q = run.by_name()["prices"]["queries"]
    assert q[0]["sql"] == "SELECT COUNT(*) FROM prices" and q[0]["repaired"]
    assert sum("REJECTED STATEMENTS" in u for _t, u in router.calls) == 1


async def test_unrepairable_sql_is_dropped_and_recorded(folder, tmp_path):
    run = await profile_folder(folder, _make(bad_sql=True, repair_ok=False),
                               cache=tc.MetadataCache(user_dir=tmp_path / "c"), log=lambda *_: None)
    meta = run.by_name()["prices"]
    assert meta["queries"] == [] and meta["dropped"][0]["id"] == "bad"
    assert "only SELECT" in meta["dropped"][0]["error"]


async def test_llm_disabled_reports_miss(folder, tmp_path):
    run = await profile_folder(folder, None, cache=tc.MetadataCache(user_dir=tmp_path / "c"),
                               allow_llm=False, log=lambda *_: None)
    assert run.tables == [] and {t for t, _ in run.failed} == {"companies", "prices"}


async def test_limit_keeps_names_of_full_folder(folder, tmp_path):
    run = await profile_folder(folder, _make(), cache=tc.MetadataCache(user_dir=tmp_path / "c"),
                               limit=1, log=lambda *_: None)
    assert [m["table_name"] for m in run.tables] == ["companies"] and run.lookups == []


async def test_classes_added_and_iris_written_back(folder, tmp_path):
    run = await profile_folder(folder, _make(), cache=tc.MetadataCache(user_dir=tmp_path / "c"), log=lambda *_: None)
    onto = {"classes_dict": {}, "object_properties_dict": {}, "data_properties_dict": {}, "instances_dict": {}}
    created = add_classes_to_ontology(run, onto, base_iri="https://ex.org/o#", default_parent_iri=None)
    labels = {e["labels"][0] for e in onto["classes_dict"].values()}
    assert {"CommodityThing", "ValueThing"} <= labels and len(created) == len(labels)
    assert "RowCount" not in labels                    # aggregation, not a concept
    col = next(c for c in run.by_name()["prices"]["columns"] if c["name"] == "commodity")
    assert col["class_iri"] and col["class_iri"] in onto["classes_dict"]


async def test_prune_expand_hook_writes_run_folder_and_classes(folder, tmp_path, monkeypatch):
    from backend.app.services.structured import prune_expand_hook as hook

    monkeypatch.setenv("GRAPHRAG_CACHE_ROOT", str(tmp_path / "cache root"))
    audits: list[tuple[str, dict]] = []
    onto = {"classes_dict": {}, "object_properties_dict": {}, "data_properties_dict": {}, "instances_dict": {}}
    version_dir = tmp_path / "v1-prune-expand"
    assert hook.check_folder(folder, None) == 2
    summary = await hook.run_structured_profile(
        folder=folder, ontology=onto, router=_make(), version_dir=version_dir,
        audit=lambda t, p: audits.append((t, p)), base_iri="https://ex.org/o#", parent_iri=None,
        limit=None, concurrency=2, refresh=False)
    assert summary["tables"] == 2 and summary["profiled"] == 2 and summary["created_classes"]
    meta = json.loads((version_dir / "structured" / "prices.metadata.json").read_text(encoding="utf-8"))
    assert any(c["class_iri"] for c in meta["columns"]) and meta["lookups"]
    assert {t for t, _ in audits} >= {"table_profile", "table_lookups", "table_class_added"}
    # extract-entities later: same files, any LLM mode -> cache only
    silent = FakeRouter({})
    run = await profile_folder(folder, silent, cache=tc.MetadataCache(), log=lambda *_: None)
    assert silent.calls == [] and len(run.cached) == 2


def test_check_folder_rejects_empty(tmp_path):
    from backend.app.services.structured.prune_expand_hook import check_folder

    with pytest.raises(SystemExit, match="no CSV"):
        check_folder(tmp_path, None)


def test_no_new_class_for_identifier_columns_or_aggregation_labels():
    from backend.app.services.structured.profiler import new_class_blocked

    assert new_class_blocked({"role": "identifier"}, "RegistrationNumber")
    for bad in ("TotalPrice", "AveragePrice", "FactCount", "RowCount", "NumberOfFacts", "MaxValue"):
        assert new_class_blocked({"role": "measure"}, bad), bad
    for ok in ("CommodityPrice", "Country", "CountryCode", "Revenue", "Jurisdiction", "TotalityIndex"):
        assert not new_class_blocked({"role": "dimension"}, ok), ok
