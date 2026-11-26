"""Total rows, chained calls and the identifier guard (structured retrieval).

Generic fixtures only: a sales table that ships an 'All' row per group next to
its parts, and query results with made-up ids.
"""

from __future__ import annotations

import pytest

from backend.app.services.structured.query_builder import build_sql
from backend.app.services.structured.retrieval_step import (
    MAX_FANOUT,
    entity_id_columns,
    expand_refs,
    identifier_guard,
    name_columns,
)
from backend.app.services.structured.sql_exec import SqlResult
from backend.tests.unit.test_structured_db import _SAFE, DSN

TYPES = {"sales": {"seller": "text", "product": "text", "year": "integer", "amount": "numeric"}}
TOTALS = {"sales": {"seller": "All"}}


def _sql(spec: dict) -> str:
    return build_sql(spec, "sales", TYPES, totals=TOTALS).sql


# ---------------------------------------------------------------- builder


def test_query_not_using_the_column_reads_only_total_rows():
    sql = _sql({"select": ["product", {"agg": "sum", "column": "amount", "as": "t"}]})
    assert "\"seller\" = 'All'" in sql


def test_query_using_the_column_reads_only_the_parts():
    for spec in (
        {"select": ["seller", {"agg": "sum", "column": "amount", "as": "t"}]},          # grouped
        {"select": ["product", {"agg": "count_distinct", "column": "seller", "as": "n"}]},  # counted
        {"select": ["product", "amount"], "order_by": [{"column": "seller"}]},           # ordered
    ):
        assert "\"seller\" IS DISTINCT FROM 'All'" in _sql(spec), spec


def test_explicit_filter_on_the_column_is_left_alone():
    sql = _sql({"select": ["amount"], "where": [{"column": "seller", "op": "=", "param": "seller"}]})
    assert "'All'" not in sql


def test_optional_param_on_the_column_defaults_to_the_total():
    sql = _sql({"select": ["product", "amount"],
                "where": [{"column": "seller", "op": "=", "param": "seller", "optional": True}]})
    assert "COALESCE(CAST(:seller AS text), 'All')" in sql
    sql = _sql({"select": ["amount"],
                "where": [{"column": "seller", "op": "in", "param": "sellers", "optional": True}]})
    assert "COALESCE(CAST(:sellers AS text[]), ARRAY['All'])" in sql


def test_no_totals_changes_nothing():
    spec = {"select": ["product", {"agg": "sum", "column": "amount", "as": "t"}]}
    assert build_sql(spec, "sales", TYPES).sql == build_sql(spec, "sales", TYPES, totals={}).sql
    assert "All" not in build_sql(spec, "sales", TYPES).sql


# ---------------------------------------------------------------- chaining


def _res(columns, rows) -> SqlResult:
    return SqlResult(sql="SELECT 1", params={}, columns=columns, rows=rows, row_count=len(rows),
                     truncated=False, total_rows=None, seconds=0.0, error=None)


def test_chained_param_fans_out_one_run_per_value():
    done = {1: [_res(["item_id", "name"], [["X1", "a"], ["X2", "b"], ["X1", "a"]])]}
    call = {"table": "t2", "query_id": "q", "params": {"item": {"from_call": 1, "column": "item_id"}, "k": 3}}
    out, err = expand_refs(call, done)
    assert err is None
    assert [c["params"] for c in out] == [{"item": "X1", "k": 3}, {"item": "X2", "k": 3}]
    assert all(c["_from_calls"] == {"item": {"from_call": 1, "column": "item_id"}} for c in out)


def test_chained_param_as_list_and_fanout_cap():
    rows = [[f"X{i}"] for i in range(MAX_FANOUT + 3)]
    done = {1: [_res(["ITEM_ID"], rows)]}
    out, err = expand_refs({"params": {"ids": {"from_call": 1, "column": "item_id", "as_list": True}}}, done)
    assert err is None and len(out) == 1 and len(out[0]["params"]["ids"]) == MAX_FANOUT + 3
    out, err = expand_refs({"params": {"id": {"from_call": 1, "column": "item_id"}}}, done)
    assert err is None and len(out) == MAX_FANOUT and "_fanout_cut" in out[0]


def test_chained_param_errors_are_reported_not_guessed():
    done = {1: [_res(["item_id"], [["X1"]])]}
    for params, needle in (({"id": {"from_call": 2, "column": "item_id"}}, "no result"),
                           ({"id": {"from_call": 1, "column": "nope"}}, "no column"),
                           ({"id": {"from_call": "first", "column": "item_id"}}, "number")):
        out, err = expand_refs({"params": params}, done)
        assert out == [] and needle in err


def test_call_without_refs_is_unchanged():
    call = {"table": "t", "query_id": "q", "params": {"a": 1}}
    assert expand_refs(call, {}) == ([call], None)


# ---------------------------------------------------------------- identifier guard


META = {
    "table_name": "items",
    "columns": [{"name": "item_id", "type": "text", "role": "identifier"},
                {"name": "source_file", "type": "text", "role": "identifier"},
                {"name": "item_name", "type": "text", "role": "dimension"},
                {"name": "kind", "type": "text", "role": "dimension"},
                {"name": "n", "type": "integer", "role": "measure"}],
    "column_stats": {"item_id": {"distinct": 40}, "source_file": {"distinct": 2},
                     "item_name": {"distinct": 40}, "kind": {"distinct": 3}},
    "analysis": {"dependencies": {"item_id": ["item_name", "kind"]}},
}
ITYPES = {"items": {"item_id": "text", "source_file": "text", "item_name": "text", "kind": "text", "n": "integer"}}


async def test_guard_allows_a_name_whose_words_the_question_states_in_any_order():
    """An identifier that is a multi-word NAME is stated by the question when
    all its significant words are, whatever their order."""
    sql = "SELECT n FROM items WHERE item_id = CAST(:id AS text)"
    kw = {"types": ITYPES, "metas": {"items": META}, "seen_values": set()}
    assert await identifier_guard(sql, {"id": "Crude oil, Brent"},
                                  question="What was the annual Brent crude oil price?", **kw) == []
    # (a name with a word the question lacks goes on to the row check -- the
    # DB-backed tests below cover what that check refuses)


def test_entity_id_columns_skip_few_valued_identifiers():
    assert entity_id_columns(META) == {"item_id"}
    assert name_columns(META, "item_id") == ["item_name"]     # 'kind' has 3 values: not a name


async def test_guard_allows_ids_from_the_question_or_an_earlier_result():
    sql = "SELECT n FROM items WHERE item_id = CAST(:id AS text)"
    kw = {"types": ITYPES, "metas": {"items": META}}
    assert await identifier_guard(sql, {"id": "AB-12"}, question="What is n for ab-12?",
                                  seen_values=set(), **kw) == []
    assert await identifier_guard(sql, {"id": "AB-12"}, question="What is n for widget?",
                                  seen_values={"ab12"}, **kw) == []
    # a few-valued identifier (a source file) is a category, never checked
    sql2 = "SELECT n FROM items WHERE source_file = CAST(:f AS text)"
    assert await identifier_guard(sql2, {"f": "b.csv"}, question="n?", seen_values=set(), **kw) == []


# ---------------------------------------------------------------- against Postgres

db_only = pytest.mark.skipif(not _SAFE, reason="set ONTOLOGY_BUILD_TEST_DSN to a migrated database named *_test")


@pytest.fixture
async def pg(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", DSN)
    from sqlalchemy import text

    from backend.app.db.engine import dispose_engine, reset_engine_cache
    from backend.app.db.session import session_scope

    reset_engine_cache()
    async with session_scope() as s:
        await s.execute(text("DROP TABLE IF EXISTS graphrag.t_sales, graphrag.t_items"))
        await s.execute(text("CREATE TABLE graphrag.t_sales (seller text, product text, year integer, "
                             "amount numeric, price numeric)"))
        rows = []
        for p, parts in (("p1", [10, 20, 30]), ("p2", [5]), ("p3", [7, 8]), ("p4", [1, 1, 1, 1])):
            for y in (2023, 2024, 2025):
                for i, a in enumerate(parts):
                    rows.append((f"s{i}", p, y, a * (y - 2020), 2.5 + i))
                rows.append(("All", p, y, sum(parts) * (y - 2020), 3.0))      # total row; price not additive
        for r in rows:
            await s.execute(text("INSERT INTO graphrag.t_sales VALUES (:a, :b, :c, :d, :e)"),
                            dict(zip("abcde", r, strict=True)))
        await s.execute(text("CREATE TABLE graphrag.t_items (item_id text, item_name text, n integer)"))
        words = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota",
                 "kappa", "lambda", "omicron"]
        for i, w in enumerate(words):
            await s.execute(text("INSERT INTO graphrag.t_items VALUES (:i, :n, :k)"),
                            {"i": f"ID{i:03d}", "n": f"widget {w}", "k": i})
    yield session_scope
    async with session_scope() as s:
        await s.execute(text("DROP TABLE IF EXISTS graphrag.t_sales, graphrag.t_items"))
    await dispose_engine()


SALES_META = {"table_name": "t_sales", "columns": [
    {"name": "seller", "type": "text", "role": "dimension"},
    {"name": "product", "type": "text", "role": "dimension"},
    {"name": "year", "type": "integer", "role": "time"},
    {"name": "amount", "type": "numeric", "role": "measure"},
    {"name": "price", "type": "numeric", "role": "measure"}]}


@db_only
async def test_total_rows_are_detected_and_not_counted_twice(pg):
    from backend.app.services.structured.analysis import analyze_table, total_markers
    from backend.app.services.structured.sql_exec import run_sql

    async with pg() as s:
        a = await analyze_table(s, SALES_META)
    assert [(t["column"], t["value"]) for t in a["total_rows"]] == [("seller", "All")]
    totals = {"t_sales": total_markers({"analysis": a})}
    types = {"t_sales": {c["name"]: c["type"] for c in SALES_META["columns"]}}
    spec = {"select": [{"agg": "sum", "column": "amount", "as": "total"}],
            "where": [{"column": "year", "op": "=", "value": 2024}]}
    built = build_sql(spec, "t_sales", types, totals=totals)
    res = await run_sql(built.sql, {}, {"t_sales"}, max_rows=5, timeout_ms=5000)
    assert float(res.rows[0][0]) == (60 + 5 + 15 + 4) * 4             # each part once
    per_seller = build_sql({"select": [{"agg": "count_distinct", "column": "seller", "as": "n"}]},
                           "t_sales", types, totals=totals)
    res = await run_sql(per_seller.sql, {}, {"t_sales"}, max_rows=5, timeout_ms=5000)
    assert int(res.rows[0][0]) == 4                                     # s0..s3, not 'All'


@db_only
async def test_table_without_totals_has_none(pg):
    from sqlalchemy import text

    from backend.app.services.structured.analysis import analyze_table
    async with pg() as s:
        await s.execute(text("DELETE FROM graphrag.t_sales WHERE seller = 'All'"))
        a = await analyze_table(s, SALES_META)
    assert a["total_rows"] == []


@db_only
async def test_guard_refuses_a_guessed_id_and_says_what_it_is(pg):
    meta = {"table_name": "t_items",
            "columns": [{"name": "item_id", "type": "text", "role": "identifier"},
                        {"name": "item_name", "type": "text", "role": "dimension"},
                        {"name": "n", "type": "integer", "role": "measure"}],
            "column_stats": {"item_id": {"distinct": 12}, "item_name": {"distinct": 12}}}
    types = {"t_items": {"item_id": "text", "item_name": "text", "n": "integer"}}
    sql = "SELECT n FROM t_items WHERE item_id = CAST(:id AS text)"
    # 'widget' is in every name: sharing it proves nothing
    out = await identifier_guard(sql, {"id": "ID004"}, question="What is n for widget gamma?",
                                 seen_values=set(), types=types, metas={"t_items": meta})
    assert out and out[0]["value"] == "ID004" and out[0]["row_values"] == ["widget epsilon"]
    # the id's own row names what the question asks about -> allowed
    assert await identifier_guard(sql, {"id": "ID002"}, question="What is n for widget gamma?",
                                  seen_values=set(), types=types, metas={"t_items": meta}) == []
    # an ordinary question word ("have") in the id's row vouches for nothing
    from sqlalchemy import text
    async with pg() as s:
        await s.execute(text("INSERT INTO graphrag.t_items VALUES ('ID099', 'people who have widgets', 0)"))
    out = await identifier_guard(sql, {"id": "ID099"}, question="Which sites did SURVEY-2 have?",
                                 seen_values=set(), types=types, metas={"t_items": meta})
    assert out and out[0]["value"] == "ID099"
    # a NAME-like id carries its own distinctive word: 'Brent' names one row
    # even though the question omits 'crude'; 'oil' alone (shared) would not
    async with pg() as s:
        await s.execute(text("INSERT INTO graphrag.t_items VALUES ('Crude oil, Brent', 'misc a', 0), "
                             "('Crude oil, WTI', 'misc b', 0), ('Palm oil', 'misc c', 0)"))
    kw = {"seen_values": set(), "types": types, "metas": {"t_items": meta}}
    assert await identifier_guard(sql, {"id": "Crude oil, Brent"},
                                  question="How did the monthly Brent oil price move?", **kw) == []
    out = await identifier_guard(sql, {"id": "Crude oil, WTI"},
                                 question="How did the monthly Brent oil price move?", **kw)
    assert out and out[0]["value"] == "Crude oil, WTI" and out[0]["row_values"] == ["misc b"]


def test_function_words_never_count_as_shared_words():
    from backend.app.services.structured.retrieval_step import _sig_words
    assert _sig_words("In which countries did the SURVEY-2 study have sites?") == {
        "countries", "survey-2", "study", "sites"}


# ---------------------------------------------------------------- latest-row queries


def test_latest_row_query_gets_a_count_of_rows_with_an_empty_period():
    from backend.app.services.structured.retrieval_step import unranked_count_sql
    sql = ('SELECT DISTINCT ON ("item") "item", "v" FROM "t" WHERE "item" = CAST(:i AS text) '
           'ORDER BY "item" ASC, "period" DESC NULLS LAST')
    count_sql, col = unranked_count_sql(sql, {"period"})
    assert col == "period"
    low = count_sql.lower()
    assert "count(*)" in low and '"period" is null' in low and "order by" not in low
    assert "distinct" not in low and ":i" in low
    limited = 'SELECT "v" FROM "t" ORDER BY "period" DESC NULLS LAST LIMIT 1'
    assert unranked_count_sql(limited, {"period"})[1] == "period"


def test_other_queries_get_no_empty_period_check():
    from backend.app.services.structured.retrieval_step import unranked_count_sql
    for sql in ('SELECT "v" FROM "t" ORDER BY "period" DESC NULLS LAST',                 # not cut
                'SELECT "v" FROM "t" ORDER BY "period" ASC LIMIT 1',                     # earliest
                'SELECT "v" FROM "t" ORDER BY "v" DESC NULLS LAST LIMIT 1',              # not a period
                'SELECT "item", MAX("v") FROM "t" GROUP BY "item" ORDER BY "item" DESC LIMIT 1'):
        assert unranked_count_sql(sql, {"period", "item"}) is None, sql


@db_only
async def test_latest_row_query_reports_rows_with_an_empty_period(pg):
    from sqlalchemy import text

    from backend.app.services.structured.retrieval_step import _unranked_rows
    from backend.app.services.structured.sql_exec import run_sql
    async with pg() as s:
        await s.execute(text("CREATE TABLE IF NOT EXISTS graphrag.t_items_p (item text, period text, v numeric)"))
        await s.execute(text("DELETE FROM graphrag.t_items_p"))
        await s.execute(text("INSERT INTO graphrag.t_items_p VALUES ('a','2025',1),('a','2026',2),('a',NULL,9)"))
    try:
        sql = ('SELECT "v" FROM "t_items_p" WHERE "item" = CAST(:i AS text) '
               'ORDER BY "period" DESC NULLS LAST LIMIT 1')
        res = await run_sql(sql, {"i": "a"}, {"t_items_p"}, max_rows=5, timeout_ms=5000)
        meta = {"columns": [{"name": "period", "type": "text", "role": "time"}]}
        rec = {"table": "t_items_p", "sql": res.sql, "params": {"i": "a"}}
        assert await _unranked_rows(rec, res, meta) == {"column": "period", "rows": 1}
    finally:
        async with pg() as s:
            await s.execute(text("DROP TABLE IF EXISTS graphrag.t_items_p"))


# ---------------------------------------------------------------- values checked before the query runs

SITES_TYPES = {"t_sites": {"site": "text", "city": "text", "notes": "text"}}


@pytest.fixture
async def sites(pg):
    from sqlalchemy import text
    async with pg() as s:
        await s.execute(text("DROP TABLE IF EXISTS graphrag.t_sites"))
        await s.execute(text("CREATE TABLE graphrag.t_sites (site text, city text, notes text)"))
        for i, c in enumerate(["Springfield", "Rivertown", "Lakeside", "Hill Valley", "Port Royal"] * 4):
            await s.execute(text("INSERT INTO graphrag.t_sites VALUES (:s, :c, :n)"),
                            {"s": f"S{i}", "c": c, "n": "long free text " * 20})
    yield pg
    async with pg() as s:
        await s.execute(text("DROP TABLE IF EXISTS graphrag.t_sites"))


@db_only
async def test_value_not_in_its_column_is_refused_before_the_query_runs(sites):
    from backend.app.services.structured.retrieval_step import prevalidate_values
    sql = "SELECT COUNT(*) FROM t_sites WHERE city = CAST(:city AS text)"
    params, missing, subs = await prevalidate_values(sql, {"city": "Acme Widgets Inc."}, col_types=SITES_TYPES)
    assert subs == [] and missing and missing[0]["column"] == "city"
    params, missing, subs = await prevalidate_values(sql, {"city": "Springfeld"}, col_types=SITES_TYPES)
    assert missing and "Springfield" in missing[0]["existing_similar"]      # a near miss: suggested, not swapped
    params, missing, subs = await prevalidate_values(sql, {"city": "hill valley"}, col_types=SITES_TYPES)
    assert missing == [] and params == {"city": "Hill Valley"}               # case only: swapped
    assert subs == [{"param": "city", "from": "hill valley", "to": "Hill Valley"}]
    params, missing, subs = await prevalidate_values(sql, {"city": "Lakeside"}, col_types=SITES_TYPES)
    assert (missing, subs) == ([], [])


@db_only
async def test_short_text_columns_get_a_trigram_index(sites):
    from sqlalchemy import text

    from backend.app.services.structured.db_load import ensure_value_indexes
    meta = {"columns": [{"name": "site", "type": "text", "role": "identifier"},
                        {"name": "city", "type": "text", "role": "dimension"},
                        {"name": "notes", "type": "text", "role": "attribute"}]}
    async with sites() as s:
        assert await ensure_value_indexes(s, "t_sites", meta) == ["site", "city"]     # notes: prose
        assert await ensure_value_indexes(s, "t_sites", meta) == ["site", "city"]     # idempotent
        idx = (await s.execute(text("SELECT indexdef FROM pg_indexes WHERE tablename = 't_sites'"))).scalars().all()
    assert sum("gin_trgm_ops" in d for d in idx) == 2


# ---------------------------------------------------------------- document evidence in the table step


def test_doc_context_takes_the_best_ranked_text_evidence():
    from backend.app.services.retrieval import structured_doc_context
    ev = [{"kind": "artifact", "rank": 1, "iri": "a", "text": "artifact"},
          {"kind": "chunk", "rank": 3, "iri": "c2", "text": "second  chunk"},
          {"kind": "chunk", "rank": 2, "iri": "c1", "text": "first chunk " * 10},
          {"kind": "relationship", "rank": 4, "iri": "r", "text": "X locatedIn Y."}]
    out = structured_doc_context(ev, items=3, chars=20)
    assert [d["id"] for d in out] == ["c1", "c2", "r"] and len(out[0]["text"]) == 20
    assert out[1]["text"] == "second chunk"
    assert structured_doc_context(ev, items=0) == []


def test_selection_prompt_shows_document_evidence():
    from backend.app.services.prompts import PROMPTS
    _s, user = PROMPTS["table_query_select"]("q?", [], doc_context=[{"id": "viao:Chunk_1",
                                                                     "text": "Made in Rivertown."}])
    assert "DOCUMENT EVIDENCE" in user and "[viao:Chunk_1] Made in Rivertown." in user
    _s, user = PROMPTS["table_query_select"]("q?", [])
    assert "DOCUMENT EVIDENCE" not in user


async def test_guard_accepts_an_identifier_stated_in_the_document_evidence():
    sql = "SELECT n FROM items WHERE item_id = CAST(:id AS text)"
    kw = {"types": ITYPES, "metas": {"items": META}, "seen_values": set(), "question": "n for the trial?"}
    assert await identifier_guard(sql, {"id": "AB-1234"}, context_norm="theyranab1234in2020", **kw) == []


# ---------------------------------------------------------------- empty values are not a category

@db_only
async def test_distinct_listing_and_group_by_do_not_count_empty_values(pg):
    from sqlalchemy import text

    from backend.app.services.structured.sql_exec import run_sql
    async with pg() as s:
        await s.execute(text("CREATE TABLE IF NOT EXISTS graphrag.t_places (site text, country text)"))
        await s.execute(text("DELETE FROM graphrag.t_places"))
        await s.execute(text("INSERT INTO graphrag.t_places VALUES ('a','France'),('b','Peru'),"
                             "('c',NULL),('d',''),('e','Peru')"))
    try:
        types = {"t_places": {"site": "text", "country": "text"}}
        listing = build_sql({"select": ["country"], "distinct": True}, "t_places", types)
        res = await run_sql(listing.sql, {}, {"t_places"}, max_rows=50, timeout_ms=5000)
        assert sorted(r[0] for r in res.rows) == ["France", "Peru"]
        counted = build_sql({"select": [{"agg": "count_distinct", "column": "country", "as": "n"}]},
                            "t_places", types)
        res = await run_sql(counted.sql, {}, {"t_places"}, max_rows=50, timeout_ms=5000)
        assert int(res.rows[0][0]) == 2
        per = build_sql({"select": ["country", {"agg": "count", "as": "sites"}]}, "t_places", types)
        res = await run_sql(per.sql, {}, {"t_places"}, max_rows=50, timeout_ms=5000)
        assert {r[0]: int(r[1]) for r in res.rows} == {"France": 1, "Peru": 2}
        assert counted.empty_excluded == [] and per.empty_excluded == ["country"]
    finally:
        async with pg() as s:
            await s.execute(text("DROP TABLE IF EXISTS graphrag.t_places"))


def test_evidence_says_empty_values_were_left_out_and_keeps_the_other_notes():
    from backend.app.services.structured.retrieval_step import (
        EMPTY_EXCLUDED_NOTE,
        UNRANKED_NOTE,
        _evidence_text,
    )
    res = SqlResult(sql="SELECT 1", params={}, columns=["country"], rows=[["France"], ["Peru"]],
                    row_count=2, truncated=True, total_rows=9)
    rec = {"table": "t_places", "empty_excluded": ["country"],
           "unranked_rows": {"column": "period", "rows": 1}}
    out = _evidence_text(rec, res, {}, 4000)
    assert EMPTY_EXCLUDED_NOTE.format(columns="country") in out
    assert UNRANKED_NOTE.format(column="period", rows=1) in out      # a capped result kept it
    assert "PARTIAL RESULT: 2 row(s) shown here; the query matched 9 rows in total." in out


@db_only
async def test_one_row_per_key_prefers_a_row_that_has_the_values(pg):
    from sqlalchemy import text

    from backend.app.services.structured.sql_exec import run_sql
    async with pg() as s:
        await s.execute(text("CREATE TABLE IF NOT EXISTS graphrag.t_wide (item text, period text, "
                             "v_y1 numeric, v_y2 numeric)"))
        await s.execute(text("DELETE FROM graphrag.t_wide"))
        # each item has a row without the yearly values (another period) and one with them
        await s.execute(text("INSERT INTO graphrag.t_wide VALUES ('a','P2',NULL,NULL),('a',NULL,1,2),"
                             "('b',NULL,3,4),('b','P2',NULL,NULL),('c','P2',NULL,NULL)"))
    try:
        b = build_sql({"select": ["item", "v_y1", "v_y2"], "distinct_on": ["item"]}, "t_wide",
                      {"t_wide": {"item": "text", "period": "text", "v_y1": "numeric", "v_y2": "numeric"}})
        res = await run_sql(b.sql, {}, {"t_wide"}, max_rows=10, timeout_ms=5000)
        got = {r[0]: (r[1], r[2]) for r in res.rows}
        assert got["a"] == ("1", "2") and got["b"] == ("3", "4")
        assert got["c"] == (None, None)          # no row with values: still listed
    finally:
        async with pg() as s:
            await s.execute(text("DROP TABLE IF EXISTS graphrag.t_wide"))


# ------------------------------------------------- typography in pattern values

def test_fold_text_keeps_one_character_per_input_character():
    from backend.app.services.structured.retrieval_step import fold_text
    for s in ("C\u00f4te d\u2019Ivoire", "2024\u201325", "A\u00a0B", "\u0130stanbul", "Stra\u00dfe"):
        assert len(fold_text(s)) == len(s)
    assert fold_text("C\u00f4te d\u2019Ivoire") == fold_text("cote d'ivoire") == "cote d'ivoire"


def test_pattern_values_finds_like_params_but_not_user_wildcards():
    from backend.app.services.structured.retrieval_step import pattern_values
    sql = ("SELECT item_id FROM t_items WHERE item_name ILIKE '%' || CAST(:name AS text) || '%' "
           "AND item_id = CAST(:id AS text)")
    assert pattern_values(sql, {"name": "x", "id": "ID001"}, {"t_items": ITYPES["items"]}) == [
        {"table": "t_items", "column": "item_name", "param": "name", "value": "x"}]
    assert pattern_values(sql, {"name": "x%", "id": "ID001"}, {"t_items": ITYPES["items"]}) == []


@db_only
async def test_pattern_value_is_rewritten_to_the_tables_spelling(pg):
    """A name copied from a document with a curly apostrophe (or without its
    accent) finds the table's straight-apostrophe spelling; a value the table
    does not hold is left alone."""
    from sqlalchemy import text

    from backend.app.services.structured.retrieval_step import prevalidate_values
    async with pg() as s:
        await s.execute(text("INSERT INTO graphrag.t_items VALUES ('ID900', 'C\u00f4te d''Ivoire', 1)"))
    types = {"t_items": {"item_id": "text", "item_name": "text", "n": "integer"}}
    sql = "SELECT item_id FROM t_items WHERE item_name ILIKE '%' || CAST(:name AS text) || '%'"
    for asked in ("C\u00f4te d\u2019Ivoire", "cote d\u2019ivoire", "C\u00f4te d'Ivoire"):
        params, missing, subs = await prevalidate_values(sql, {"name": asked}, col_types=types)
        assert not missing
        async with pg() as s:
            n = (await s.execute(text("SELECT count(*) FROM graphrag.t_items WHERE item_name ILIKE "
                                    "'%' || :n || '%'"), {"n": params["name"]})).scalar()
        assert n == 1, (asked, params, subs)
    params, missing, subs = await prevalidate_values(sql, {"name": "Atlantis"}, col_types=types)
    assert params == {"name": "Atlantis"} and subs == []
