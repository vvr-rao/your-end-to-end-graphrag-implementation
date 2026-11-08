"""Source-level behaviour of structured_download.py, driven through the real
CLI parser with a fake HTTP session (no network)."""

from __future__ import annotations

import csv
import io
import json
import zipfile
from pathlib import Path

import pytest

from backend.tests.unit.structured_fakes import (
    FIXTURES,
    FakeResponse,
    FakeSession,
    load_manifest,
    parse,
    run_cli,
)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def only_root(out: Path) -> Path:
    roots = [p.parent for p in out.rglob("manifest.json")]
    assert len(roots) == 1
    return roots[0]


# ------------------------------------------------------------- CLI surface


def test_list_and_info(capsys):
    import structured_download
    assert structured_download.main(["list"]) == 0
    out = capsys.readouterr().out
    for name in ("sec", "fred", "gleif", "ffiec", "clinicaltrials", "drugsfda", "orangebook",
                 "chembl", "opentargets"):
        assert name in out
    assert structured_download.main(["info", "ffiec"]) == 0
    info = capsys.readouterr().out
    assert "FFIEC_TOKEN" in info and "CAPTCHA" in info


def test_unsupported_mode_rejected():
    with pytest.raises(SystemExit):
        parse(["fred", "bulk", "--series-id", "GDP"])


@pytest.mark.parametrize("argv,needle", [
    (["fred", "api", "--series-id", "GDP", "--native-type", "csv", "--format", "csv"],
     "normalized CSV is built from FRED's JSON"),
    (["sec", "bulk", "--dataset", "companyfacts-zip", "--format", "csv"], "not supported"),
    (["drugsfda", "bulk", "--dataset", "openfda-json", "--format", "csv"], "provider JSON"),
    (["clinicaltrials", "api"], "give --condition"),
    (["chembl", "api", "--dataset", "activity", "--chembl-id", "CHEMBL1"], "not available"),
    (["gleif", "api", "--lei", "NOTALEI"], "invalid LEI"),
])
def test_unsupported_combinations_rejected(argv, needle, capsys, monkeypatch, tmp_path):
    monkeypatch.setenv("FRED_API_KEY", "k" * 32)
    monkeypatch.setenv("SEC_USER_AGENT", "Test Org test@org.test")
    sess = FakeSession(lambda *a: pytest.fail("no request expected"))
    assert run_cli([*argv, "-o", str(tmp_path)], sess) == 2
    assert needle in capsys.readouterr().err
    assert sess.calls == []


@pytest.mark.parametrize("argv,var", [
    (["fred", "api", "--series-id", "GDP"], "FRED_API_KEY"),
    (["sec", "api", "--dataset", "companyfacts", "--cik", "320193"], "SEC_USER_AGENT"),
    (["ffiec", "api", "--dataset", "reporting-periods"], "FFIEC_USER_ID"),
])
def test_missing_credentials_name_the_variable(argv, var, capsys, monkeypatch, tmp_path):
    for v in ("FRED_API_KEY", "SEC_USER_AGENT", "FFIEC_USER_ID", "FFIEC_TOKEN"):
        monkeypatch.delenv(v, raising=False)
    sess = FakeSession(lambda *a: pytest.fail("no request expected"))
    assert run_cli([*argv, "-o", str(tmp_path)], sess) == 2
    assert var in capsys.readouterr().err


def test_sec_rejects_placeholder_user_agent(capsys, monkeypatch, tmp_path):
    monkeypatch.setenv("SEC_USER_AGENT", "me (contact@example.com)")
    sess = FakeSession(lambda *a: pytest.fail("no request expected"))
    rc = run_cli(["sec", "api", "--dataset", "companyfacts", "--cik", "1", "-o", str(tmp_path)],
                 sess)
    assert rc == 2 and "real contact email" in capsys.readouterr().err


def test_dry_run_downloads_nothing(capsys, tmp_path, monkeypatch):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    sess = FakeSession(lambda *a: pytest.fail("no request expected"))
    rc = run_cli(["fred", "api", "--series-id", "GDP", "--dry-run", "-o", str(tmp_path)], sess)
    assert rc == 0 and "[plan] fred api" in capsys.readouterr().out
    assert list(tmp_path.iterdir()) == []


# ----------------------------------------------------- ClinicalTrials.gov


def test_clinicaltrials_related_tables_without_cartesian_products(tmp_path):
    page = json.loads((FIXTURES / "ct_studies_page.json").read_text())

    def handler(method, url, params, headers, body):
        if url.endswith("/version"):
            return FakeResponse(200, {"dataTimestamp": "2026-11-06T10:00:05"})
        return FakeResponse(200, page)

    rc = run_cli(["clinicaltrials", "api", "--condition", "obesity", "--format", "csv",
                  "-o", str(tmp_path)], FakeSession(handler))
    assert rc == 0
    root = only_root(tmp_path)
    n = root / "normalized"
    studies = read_csv(n / "studies.csv")
    assert [s["nct_id"] for s in studies] == ["NCT00000001", "NCT00000002"]
    assert studies[0]["org_study_id"] == "0042"
    assert len(read_csv(n / "conditions.csv")) == 3          # 2 + 1, not multiplied
    locs = read_csv(n / "locations.csv")
    assert len(locs) == 3 and locs[0]["zip"] == "02115"        # leading zero kept
    assert {r["role"] for r in read_csv(n / "sponsors.csv")} == {"lead", "collaborator"}
    assert [o["kind"] for o in read_csv(n / "outcomes.csv")] == ["primary", "secondary"]
    m = load_manifest(tmp_path)
    assert m["status"] == "complete" and not m["truncated"]
    assert m["source_release"] == "2026-11-06T10:00:05"
    studies_entry = next(f for f in m["files"] if f["path"] == "normalized/studies.csv")
    assert "SELECTED-FIELD" in studies_entry["description"]
    assert studies_entry["derived_from"] == ["native/studies-0001.json"]
    native = json.loads((root / "native" / "studies-0001.json").read_text())
    assert "resultsSection" in native["studies"][0]              # complete record kept


def test_clinicaltrials_page_token_pagination_and_limit(tmp_path):
    calls = []

    def handler(method, url, params, headers, body):
        if url.endswith("/version"):
            return FakeResponse(200, {"dataTimestamp": "x"})
        calls.append(dict(params))
        n = len(calls)
        studies = [{"protocolSection": {"identificationModule": {"nctId": f"NCT{n:04d}{i:04d}"}}}
                   for i in range(params["pageSize"])]
        return FakeResponse(200, {"totalCount": 99, "studies": studies,
                                  "nextPageToken": f"tok{n}"})

    rc = run_cli(["clinicaltrials", "api", "--term", "x", "--limit", "5", "-o", str(tmp_path)],
                 FakeSession(handler))
    assert rc == 0
    assert calls[0]["pageSize"] == 5 and "pageToken" not in calls[0]
    assert len(calls) == 1
    m = load_manifest(tmp_path)
    assert m["truncated"] and m["status"] == "sample"
    assert m["pagination"] == {"pages_fetched": 1, "records": 5, "total_reported": 99,
                               "complete": False}


def test_max_pages_follows_tokens_then_stops(tmp_path):
    seen = []

    def handler(method, url, params, headers, body):
        if url.endswith("/version"):
            return FakeResponse(200, {})
        seen.append(params.get("pageToken"))
        return FakeResponse(200, {"totalCount": 10, "studies": [{"protocolSection": {}}],
                                  "nextPageToken": f"t{len(seen)}"})

    run_cli(["clinicaltrials", "api", "--term", "x", "--max-pages", "3", "-o", str(tmp_path)],
            FakeSession(handler))
    assert seen == [None, "t1", "t2"]
    assert load_manifest(tmp_path)["truncation_reasons"] == ["--max-pages 3 reached"]


# ----------------------------------------------------------------- openFDA


def test_openfda_follows_link_header_and_preserves_ids(tmp_path):
    def rec(n):
        return {"application_number": f"ANDA{n:06d}", "sponsor_name": "S",
                "products": [{"product_number": "001", "active_ingredients": [
                    {"name": "A", "strength": "1MG"}, {"name": "B", "strength": "2MG"}]}]}

    def handler(method, url, params, headers, body):
        if "page2" in url:
            assert params == {}                       # next link carries every parameter
            return FakeResponse(200, {"meta": {"results": {"total": 2}}, "results": [rec(7)]})
        assert params["limit"] == 1000 and params["search"] == "x:y"
        return FakeResponse(200, {"meta": {"last_updated": "2026-11-06",
                                           "results": {"total": 2}}, "results": [rec(42)]},
                            {"Link": '<https://api.fda.gov/drug/drugsfda.json?page2>; rel="next"'})

    rc = run_cli(["drugsfda", "api", "--search", "x:y", "--format", "csv", "-o", str(tmp_path)],
                 FakeSession(handler))
    assert rc == 0
    n = only_root(tmp_path) / "normalized"
    apps = read_csv(n / "drugsfda.csv")
    assert [a["application_number"] for a in apps] == ["ANDA000042", "ANDA000007"]
    ings = read_csv(n / "drugsfda__products__active_ingredients.csv")
    assert len(ings) == 4 and ings[0]["application_number"] == "ANDA000042"
    assert read_csv(n / "drugsfda__products.csv")[0]["product_number"] == "001"
    m = load_manifest(tmp_path)
    assert m["status"] == "complete" and m["pagination"]["pages_fetched"] == 2
    assert m["source_release"] == "2026-11-06"


def test_openfda_api_key_sent_but_never_recorded(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENFDA_API_KEY", "KEY-SECRET-1")
    sess = FakeSession(lambda *a: FakeResponse(200, {"meta": {}, "results": []}))
    run_cli(["orangebook", "api", "--search", "a:b", "-o", str(tmp_path)], sess)
    assert sess.calls[0]["params"]["api_key"] == "KEY-SECRET-1"
    for p in tmp_path.rglob("*"):
        if p.is_file():
            assert "KEY-SECRET-1" not in p.read_text(encoding="utf-8", errors="ignore")


def _zip_bytes(members: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for k, v in members.items():
            zf.writestr(k, v)
    return buf.getvalue()


def test_orange_book_flatfiles_tilde_to_csv(tmp_path):
    products = ("Ingredient~DF;Route~Trade_Name~Appl_Type~Appl_No~Product_No\n"
                "SEMAGLUTIDE~SOLUTION;SUBCUTANEOUS~WEGOVY~N~215256~003\n"
                "PHENTERMINE~CAPSULE;ORAL~ADIPEX~A~040083~001\n")
    payload = _zip_bytes({"products.txt": products, "patent.txt": "Appl_No~Patent_No\n040083~123\n"})

    def handler(method, url, params, headers, body):
        if method == "HEAD":
            return FakeResponse(200, b"", {"Content-Length": str(len(payload)),
                                           "Last-Modified": "Tue, 13 Oct 2026 16:09:02 GMT"})
        return FakeResponse(200, payload, {"Content-Length": str(len(payload))})

    rc = run_cli(["orangebook", "bulk", "--dataset", "flatfiles", "--format", "csv",
                  "-o", str(tmp_path)], FakeSession(handler))
    assert rc == 0
    root = only_root(tmp_path)
    assert root.name == "20261013"                    # snapshot named by provider date
    rows = read_csv(root / "normalized" / "products.csv")
    assert rows[1]["Appl_No"] == "040083" and rows[1]["Product_No"] == "001"
    assert rows[0]["DF;Route"] == "SOLUTION;SUBCUTANEOUS"
    m = load_manifest(tmp_path)
    native = next(f for f in m["files"] if f["path"].endswith("products.txt"))
    assert native["delimiter"] == "tilde" and native["rows"] == 2
    assert m["snapshot_kind"] == "full" and m["source_release"] == "2026-10-13"
    assert any(f["path"] == "native/orangebook_flatfiles.zip" for f in m["files"])


def test_bulk_snapshot_not_overwritten_silently_and_resume_skips(tmp_path, capsys):
    payload = _zip_bytes({"Products.txt": "a\tb\n1\t2\n"})
    gets = []

    def handler(method, url, params, headers, body):
        if method == "HEAD":
            return FakeResponse(200, b"", {"Last-Modified": "Tue, 13 Oct 2026 16:09:02 GMT"})
        gets.append(url)
        return FakeResponse(200, payload)

    argv = ["drugsfda", "bulk", "--dataset", "flatfiles", "-o", str(tmp_path)]
    assert run_cli(argv, FakeSession(handler)) == 0
    assert run_cli(argv, FakeSession(handler)) == 2
    assert "already exists" in capsys.readouterr().err
    assert run_cli([*argv, "--resume"], FakeSession(handler)) == 0
    assert len(gets) == 1                                     # verified file skipped
    assert run_cli([*argv, "--overwrite"], FakeSession(handler)) == 0
    assert len(gets) == 2


def test_failed_bulk_download_writes_failed_manifest(tmp_path):
    html = b"<!doctype html><html>Access Denied</html>"

    def handler(method, url, params, headers, body):
        if method == "HEAD":
            return FakeResponse(200, b"", {})
        return FakeResponse(200, html, {"Content-Type": "text/html"})

    rc = run_cli(["drugsfda", "bulk", "--dataset", "flatfiles", "-o", str(tmp_path)],
                 FakeSession(handler))
    assert rc == 1
    m = load_manifest(tmp_path)
    assert m["status"] == "failed" and "HTML page" in m["errors"][0]
    assert not any(p.suffix == ".zip" for p in tmp_path.rglob("*"))


# --------------------------------------------------------------------- SEC


def test_sec_companyfacts_keeps_every_fact_and_marks_cap(tmp_path, monkeypatch):
    monkeypatch.setenv("SEC_USER_AGENT", "Test Org test@org.test")
    facts = json.loads((FIXTURES / "sec_companyfacts.json").read_text())
    sess = FakeSession(lambda *a: FakeResponse(200, facts))
    rc = run_cli(["sec", "api", "--dataset", "companyfacts", "--cik", "320193", "--format",
                  "csv", "-o", str(tmp_path)], sess)
    assert rc == 0
    assert sess.calls[0]["url"].endswith("/CIK0000320193.json")
    root = only_root(tmp_path)
    rows = read_csv(root / "normalized" / "facts.csv")
    rev = [r for r in rows if r["tag"] == "Revenues"]
    assert len(rev) == 4                                       # amendment + repeat kept
    assert {r["form"] for r in rev} == {"10-Q", "10-Q/A"}
    ytd = next(r for r in rev if r["end"] == "2024-03-30")
    assert ytd["val"] == "190" and ytd["duration_days"] == "182"   # YTD, not de-cumulated
    assets = next(r for r in rows if r["tag"] == "Assets")
    assert assets["period_type"] == "instant" and assets["cik"] == "0000320193"
    assert load_manifest(tmp_path)["status"] == "complete"

    out2 = tmp_path / "capped"
    run_cli(["sec", "api", "--dataset", "companyfacts", "--cik", "320193", "--format", "csv",
             "--limit", "2", "-o", str(out2)], FakeSession(lambda *a: FakeResponse(200, facts)))
    m = load_manifest(out2)
    assert m["status"] == "sample" and "native JSON complete" in m["truncation_reasons"][0]


def test_sec_user_agent_header_is_used(tmp_path, monkeypatch):
    monkeypatch.setenv("SEC_USER_AGENT", "Test Org test@org.test")
    sess = FakeSession(lambda *a: FakeResponse(200, {"cik": 1, "facts": {}}))
    run_cli(["sec", "api", "--dataset", "companyfacts", "--cik", "1", "-o", str(tmp_path)], sess)
    assert sess.headers["User-Agent"] == "Test Org test@org.test"


# -------------------------------------------------------------------- FRED


def test_fred_observations_paging_and_missing_marker(tmp_path, monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "abc123")

    def handler(method, url, params, headers, body):
        assert params["api_key"] == "abc123"
        if url.endswith("/series"):
            return FakeResponse(200, {"seriess": [{"id": "GDP", "units": "Billions",
                                                   "frequency": "Quarterly",
                                                   "seasonal_adjustment": "SAAR"}]})
        obs = [{"realtime_start": "2026-01-01", "realtime_end": "2026-01-01",
                "date": f"2020-0{i + 1}-01", "value": "." if i == 1 else str(i)}
               for i in range(params["limit"])]
        return FakeResponse(200, {"count": 3, "observations": obs[: 3 - params["offset"]]})

    rc = run_cli(["fred", "api", "--series-id", "gdp", "--format", "csv", "-o", str(tmp_path)],
                 FakeSession(handler))
    assert rc == 0
    root = only_root(tmp_path)
    obs = read_csv(root / "normalized" / "observations.csv")
    assert [o["value"] for o in obs] == ["0", "", "2"]
    native = json.loads(next((root / "native" / "GDP").glob("observations-*.json")).read_text())
    assert native["observations"][1]["value"] == "."           # native untouched
    assert read_csv(root / "normalized" / "series.csv")[0]["seasonal_adjustment"] == "SAAR"
    m = load_manifest(tmp_path)
    assert m["params"].get("api_key") is None
    assert "abc123" not in json.dumps(m)


def test_fred_vintage_columns_unpivoted(tmp_path, monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "abc123")

    def handler(method, url, params, headers, body):
        if url.endswith("/series"):
            return FakeResponse(200, {"seriess": [{"id": "GDP"}]})
        assert params["vintage_dates"] == "2024-01-01,2025-01-01"
        assert params["output_type"] == 2
        return FakeResponse(200, {"count": 2, "output_type": 2, "observations": [
            {"date": "2022-01-01", "GDP_20240101": "25029.116", "GDP_20250101": "25215.491"},
            {"date": "2022-04-01", "GDP_20240101": ".", "GDP_20250101": "25805.791"}]})

    rc = run_cli(["fred", "api", "--series-id", "GDP", "--vintage-dates",
                  "2024-01-01,2025-01-01", "--format", "csv", "-o", str(tmp_path)],
                 FakeSession(handler))
    assert rc == 0
    obs = read_csv(only_root(tmp_path) / "normalized" / "observations.csv")
    assert [(o["vintage_date"], o["date"], o["value"]) for o in obs] == [
        ("2024-01-01", "2022-01-01", "25029.116"), ("2025-01-01", "2022-01-01", "25215.491"),
        ("2024-01-01", "2022-04-01", ""), ("2025-01-01", "2022-04-01", "25805.791")]


def test_fred_csv_native_zip_is_extracted(tmp_path, monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "abc123")
    payload = _zip_bytes({"GDP.csv": "observation_date,GDP\n2020-01-01,1\n",
                          "README_SERIES_ID_SORT.txt": "readme"})

    def handler(method, url, params, headers, body):
        if url.endswith("/series"):
            return FakeResponse(200, {"seriess": []})
        return FakeResponse(200, payload)

    rc = run_cli(["fred", "api", "--series-id", "GDP", "--native-type", "csv", "-o",
                  str(tmp_path)], FakeSession(handler))
    assert rc == 0
    m = load_manifest(tmp_path)
    csvs = [f for f in m["files"] if f["path"].endswith("GDP.csv")]
    assert csvs and csvs[0]["rows"] == 1 and csvs[0]["role"] == "extracted"


# ------------------------------------------------------------------- GLEIF


def test_gleif_relationships_and_exceptions_kept_apart(tmp_path):
    lei_parent, lei_child = "HWUPKR0MPOU8FGXBT394", "549300G81RQKP7XW2N18"

    def entity(lei, name):
        return {"type": "lei-records", "id": lei,
                "attributes": {"lei": lei, "entity": {"legalName": {"name": name},
                                                      "otherNames": [{"name": "Old"}]}}}

    rel = {"type": "relationship-records", "id": "R1",
           "attributes": {"relationship": {"startNode": {"id": lei_child},
                                           "endNode": {"id": lei_parent},
                                           "type": "IS_DIRECTLY_CONSOLIDATED_BY",
                                           "status": "ACTIVE",
                                           "periods": [{"type": "ACCOUNTING_PERIOD"},
                                                       {"type": "RELATIONSHIP_PERIOD"}]},
                          "registration": {"status": "PUBLISHED"}}}
    exc = {"type": "reporting-exceptions", "id": "E1",
           "attributes": {"lei": lei_parent, "category": "DIRECT_ACCOUNTING_CONSOLIDATION_PARENT",
                          "reason": "NATURAL_PERSONS"}}

    def handler(method, url, params, headers, body):
        if url.endswith("/lei-records"):
            return FakeResponse(200, {"meta": {"pagination": {"total": 2, "lastPage": 1}},
                                      "links": {}, "data": [entity(lei_parent, "Parent"),
                                                            entity(lei_child, "Child")]})
        if url.endswith(f"{lei_child}/direct-parent-relationship"):
            return FakeResponse(200, {"data": rel})
        if url.endswith(f"{lei_parent}/direct-child-relationships"):
            return FakeResponse(200, {"data": [rel], "meta": {"pagination": {"lastPage": 1}}})
        if url.endswith(f"{lei_parent}/direct-parent-reporting-exception"):
            return FakeResponse(200, {"data": exc})
        return FakeResponse(404, {"errors": [{"status": "404"}]})

    rc = run_cli(["gleif", "api", "--name", "x", "--relationships", "--format", "csv", "-o",
                  str(tmp_path)], FakeSession(handler))
    assert rc == 0
    n = only_root(tmp_path) / "normalized"
    rels = read_csv(n / "relationships.csv")
    assert len(rels) == 1                                    # same record seen twice, kept once
    assert rels[0]["start_lei"] == lei_child and rels[0]["end_lei"] == lei_parent
    assert len(read_csv(n / "relationship_periods.csv")) == 2
    excs = read_csv(n / "reporting_exceptions.csv")
    assert [e["reason"] for e in excs] == ["NATURAL_PERSONS"]
    ents = read_csv(n / "entities.csv")
    assert {e["lei"] for e in ents} == {lei_parent, lei_child}
    assert len(read_csv(n / "entities__entity__otherNames.csv")) == 2


# ------------------------------------------------------------------ ChEMBL


def test_chembl_page_meta_paging_keeps_measurement_columns(tmp_path):
    def act(i):
        return {"activity_id": 100 + i, "molecule_chembl_id": f"CHEMBL{i}",
                "standard_type": "IC50", "standard_relation": ">", "standard_value": "100000.0",
                "standard_units": "nM", "type": "IC50", "relation": ">", "value": "100.0",
                "units": "uM", "pchembl_value": None,
                "activity_properties": [{"type": "TIME", "value": "24"}]}

    def handler(method, url, params, headers, body):
        if url.endswith("status.json"):
            return FakeResponse(200, {"chembl_db_version": "ChEMBL_37"})
        if "offset=2" in url:
            return FakeResponse(200, {"page_meta": {"next": None, "total_count": 3},
                                      "activities": [act(3)]})
        assert params["target_chembl_id"] == "CHEMBL231"
        return FakeResponse(200, {"page_meta": {"next": "/chembl/api/data/activity.json?offset=2",
                                                "total_count": 3},
                                  "activities": [act(1), act(2)]})

    rc = run_cli(["chembl", "api", "--dataset", "activity", "--filter",
                  "target_chembl_id=CHEMBL231", "--format", "csv", "-o", str(tmp_path)],
                 FakeSession(handler))
    assert rc == 0
    n = only_root(tmp_path) / "normalized"
    rows = read_csv(n / "activities.csv")
    assert len(rows) == 3
    r = rows[0]
    assert (r["standard_value"], r["standard_units"], r["value"], r["units"]) == \
        ("100000.0", "nM", "100.0", "uM")
    assert r["standard_relation"] == ">" and r["relation"] == ">"
    assert len(read_csv(n / "activities__activity_properties.csv")) == 3
    m = load_manifest(tmp_path)
    assert m["status"] == "complete" and m["source_release"] == "ChEMBL_37"


def test_chembl_bulk_checksum_mismatch_fails(tmp_path):
    good = b"\x1f\x8b" + b"data"
    listing = '<a href="chembl_37_chemreps.txt.gz">x</a><a href="checksums.txt">c</a>'

    def handler(method, url, params, headers, body):
        if method == "HEAD":
            return FakeResponse(200, b"", {})
        if url.endswith("/releases/chembl_37/"):
            return FakeResponse(200, listing)
        if url.endswith("checksums.txt"):
            return FakeResponse(200, "SHA256\tfile\n" + "0" * 64 + "\tchembl_37_chemreps.txt.gz\n")
        return FakeResponse(200, good)

    rc = run_cli(["chembl", "bulk", "--release", "37", "--asset", "chemreps", "-o",
                  str(tmp_path)], FakeSession(handler))
    assert rc == 1
    assert "SHA-256 does not match" in load_manifest(tmp_path)["errors"][0]


# ------------------------------------------------------------ Open Targets


def test_opentargets_graphql_paging_and_release(tmp_path):
    posts = []

    def handler(method, url, params, headers, body):
        posts.append(body)
        if "meta" in body["query"] and "variables" not in body:
            return FakeResponse(200, {"data": {"meta": {"dataVersion": {
                "year": "26", "month": "09", "iteration": None}}}})
        idx = body["variables"]["index"]
        size = body["variables"]["size"]
        rows = [{"score": 0.5, "disease": {"id": f"D{idx}{i}", "name": "d"},
                 "datatypeScores": [{"id": "literature", "score": 0.1}]} for i in range(size)]
        return FakeResponse(200, {"data": {"target": {
            "id": "ENSG1", "approvedSymbol": "AR",
            "associatedDiseases": {"count": 250, "rows": rows}}}})

    rc = run_cli(["opentargets", "api", "--dataset", "target", "--id", "ENSG1", "--limit", "150",
                  "--format", "csv", "-o", str(tmp_path)], FakeSession(handler))
    assert rc == 0
    sizes = [(p["variables"]["index"], p["variables"]["size"]) for p in posts
             if "variables" in p]
    assert sizes == [(0, 100), (1, 50)]
    n = only_root(tmp_path) / "normalized"
    assert len(read_csv(n / "target.csv")) == 2                      # one row per page
    assert len(read_csv(n / "target__associatedDiseases__rows.csv")) == 150
    m = load_manifest(tmp_path)
    assert m["source_release"] == "26.09" and m["truncated"]
    assert any(f["path"] == "native/query.graphql" for f in m["files"])


def test_opentargets_parquet_to_csv_streams(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    tbl = pa.table({"id": ["ENSG00000000003", "ENSG00000000005"],
                    "approvedSymbol": ["TSPAN6", "TNMD"],
                    "synonyms": [[{"label": "T245"}], []]})
    buf = io.BytesIO()
    pq.write_table(tbl, buf)
    data = buf.getvalue()
    listing = '<a href="part-00000-x.zstd.parquet">p</a><a href="_SUCCESS">s</a>'

    def handler(method, url, params, headers, body):
        if method == "HEAD":
            return FakeResponse(200, b"", {"Content-Length": str(len(data))})
        if url.endswith("/target/"):
            return FakeResponse(200, listing)
        return FakeResponse(200, data)

    rc = run_cli(["opentargets", "bulk", "--dataset", "target", "--release", "26.09",
                  "--format", "csv", "-o", str(tmp_path)], FakeSession(handler))
    assert rc == 0
    root = only_root(tmp_path)
    assert root.name == "26.09"
    rows = read_csv(root / "normalized" / "target.csv")
    assert rows[0]["id"] == "ENSG00000000003"
    assert json.loads(rows[0]["synonyms"]) == [{"label": "T245"}]
    m = load_manifest(tmp_path)
    assert any("synonyms" in t for t in m["transformations"])
    assert any(f["path"].endswith(".parquet") and f["role"] == "native" for f in m["files"])


# ------------------------------------------------------------------- FFIEC


def test_ffiec_rest_headers_and_panel_csv(tmp_path, monkeypatch):
    monkeypatch.setenv("FFIEC_USER_ID", "user1")
    monkeypatch.setenv("FFIEC_TOKEN", "tok-SECRET")
    panel = [{"ID_RSSD": 724904, "Name": "SHOREHAM BANK", "ZIP": 2886, "FilingType": 51,
              "HasFiledForReportingPeriod": True}]
    sess = FakeSession(lambda *a: FakeResponse(200, panel))
    rc = run_cli(["ffiec", "api", "--dataset", "panel", "--period", "2025-03-31", "--format",
                  "csv", "-o", str(tmp_path)], sess)
    assert rc == 0
    h = sess.calls[0]["headers"]
    assert h["UserID"] == "user1" and h["Authentication"] == "Bearer tok-SECRET"
    assert h["reportingPeriodEndDate"] == "03/31/2025" and h["dataSeries"] == "Call"
    assert sess.calls[0]["url"] == "https://ffieccdr.azure-api.us/public/RetrievePanelOfReporters"
    row = read_csv(only_root(tmp_path) / "normalized" / "institutions.csv")[0]
    assert row["ZIP"] == "02886" and row["FilingType"] == "051" and row["ID_RSSD"] == "724904"
    for p in tmp_path.rglob("*"):
        if p.is_file():
            assert "tok-SECRET" not in p.read_text(encoding="utf-8", errors="ignore")


def test_ffiec_call_bulk_archive_dictionary(tmp_path):
    sched = ('"IDRSSD"\t"RCON2170"\t"RCON2948"\t\n'
             '\t"TOTAL ASSETS"\t"TOTAL LIABILITIES"\t\n'
             '"37"\t"123"\t"100"\t\n')
    arc = tmp_path / "FFIEC CDR Call Bulk All Schedules 03312025.zip"
    arc.write_bytes(_zip_bytes({"FFIEC CDR Call Schedule RC 03312025.txt": sched,
                                "Readme.txt": "notes"}))
    out = tmp_path / "out"
    rc = run_cli(["ffiec", "archive", "--dataset", "cdr-call-bulk", "--from-archive", str(arc),
                  "--format", "csv", "-o", str(out)], FakeSession(lambda *a: pytest.fail()))
    assert rc == 0
    n = only_root(out) / "normalized"
    data = read_csv(n / "FFIEC CDR Call Schedule RC 03312025.csv")
    assert data == [{"IDRSSD": "37", "RCON2170": "123", "RCON2948": "100"}]
    dic = read_csv(n / "field_dictionary.csv")
    assert dic[1]["mdrm_code"] == "RCON2170" and dic[1]["description"] == "TOTAL ASSETS"
    m = load_manifest(out)
    assert m["mode"] == "archive" and arc.exists()            # user's file left in place


# ------------------------------------------------------------------- AACT


def test_aact_flatfiles_pipe_delimited(tmp_path):
    arc = tmp_path / "20261108_pipe-delimited-export.zip"
    arc.write_bytes(_zip_bytes({"studies.txt": "nct_id|brief_title\nNCT00000001|A | B\n",
                                "conditions.txt": "id|nct_id|name\n1|NCT00000001|Obesity\n"}))
    out = tmp_path / "out"
    rc = run_cli(["clinicaltrials", "archive", "--dataset", "aact-flatfiles", "--from-archive",
                  str(arc), "-o", str(out)], FakeSession(lambda *a: pytest.fail()))
    # "A | B" contains the delimiter inside an unquoted field -> row has 3 fields;
    # counting still works, conversion is not requested here.
    assert rc == 0
    m = load_manifest(out)
    cond = next(f for f in m["files"] if f["path"].endswith("conditions.txt"))
    assert cond["delimiter"] == "pipe" and cond["rows"] == 1
    assert only_root(out).name == "20261108_pipe-delimited-export"
