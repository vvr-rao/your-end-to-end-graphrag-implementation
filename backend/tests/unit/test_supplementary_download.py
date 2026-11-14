"""source_documents/supplementary_download.py: parsers + CLI, offline (fake fetcher)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import supplementary_download as sd


class FakeFetcher(sd.Fetcher):
    """Serves canned bodies by URL substring; records every request."""

    def __init__(self, routes: dict[str, bytes | None], html: tuple[str, ...] = ()):
        super().__init__()
        self.routes, self.html, self.calls = routes, html, []

    def get(self, url, params=None, dest=None, allow_404=False):
        self.calls.append((url, dict(params or {})))
        self.requests += 1
        key = url + "?" + "&".join(f"{k}={v}" for k, v in (params or {}).items())
        body = next((b for k, b in self.routes.items() if k in key), b"")
        if body is None:
            return None if allow_404 else (_ for _ in ()).throw(AssertionError(f"404 {url}"))
        if dest is not None:
            if any(h in url for h in self.html) and dest.suffix != ".html":
                raise sd.UsageError(f"{url} returned an HTML page, not {dest.suffix} data")
            dest.write_bytes(body)
        return body


def read(p: Path) -> list[dict]:
    with p.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def snapshot(root: Path) -> Path:
    (m,) = root.rglob("manifest.json")
    return m.parent


# ------------------------------------------------------------- World Bank


def test_pinksheet_unpivots_and_strips_footnote_markers(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    monthly, annual = tmp_path / "CMO-Historical-Data-Monthly.xlsx", tmp_path / "CMO-Historical-Data-Annual.xlsx"
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Monthly Prices"
    for r in (["World Bank Commodity Price Data"], ["Updated on November 03, 2026"], [],
              [None, "Crude oil, average", "Beef **", "Potassium chloride **"],
              [None, "($/bbl)", "($/kg)", "($/mt)"],
              ["2025M01", 78.5, 6.1, "…"], ["2025M02", 75.0, None, 340.25], ["notes", 1, 2, 3]):
        ws.append(r)
    wb.save(monthly)
    wb = openpyxl.Workbook()
    for i, title in enumerate(("Annual Prices (Nominal)", "Annual Prices (Real)")):
        ws = wb.active if i == 0 else wb.create_sheet()
        ws.title = title
        for r in (["x"], ["Updated on September 02, 2026"], [], [None, "Crude oil, average", "Beef"],
                  [None, "($/bbl)", "($/kg)"], ["2025", 80.0, 6.0]):
            ws.append(r)
    wb.save(annual)
    prices, series, updated = sd.parse_pinksheet({monthly.name: monthly, annual.name: annual})
    monthly_rows = [p for p in prices if p["frequency"] == "monthly"]
    assert len(monthly_rows) == 4                       # '…' and empty cells dropped, footer row ignored
    kcl = next(p for p in monthly_rows if p["commodity"] == "Potassium chloride")
    assert kcl["footnote_marker"] == "**" and kcl["value"] == "340.25" and kcl["unit"] == "($/mt)"
    beef = {(p["frequency"], p["price_basis"]) for p in prices if p["commodity"] == "Beef"}
    assert beef == {("monthly", "nominal_usd"), ("annual", "nominal_usd"), ("annual", "real_2010_usd")}
    assert len(series) == 7                     # 3 monthly + 2 annual nominal + 2 real
    assert updated[f"{monthly.name}:Monthly Prices"] == "November 03, 2026"


def test_worldbank_discovery_requires_both_links():
    page = (b'<a href="https://thedocs.worldbank.org/en/doc/abc-0050012026/related/'
            b'CMO-Historical-Data-Monthly.xlsx">m</a>')
    with pytest.raises(sd.UsageError, match="Annual"):
        sd.discover_worldbank(FakeFetcher({"commodity-markets": page}))


# ----------------------------------------------------------- OWID / USGS


def test_owid_run_stacks_charts_and_writes_manifest(tmp_path):
    meta = json.dumps({"columns": {"ev": {"titleLong": "Electric cars sold", "unit": "cars",
                                          "citationLong": "IEA. Global EV Outlook 2025. \u2013 processed"}}}).encode()
    body = b"entity,code,year,ev_sales\nWorld,OWID_WRL,2024,17500000\nChina,CHN,2024,\n"
    fake = FakeFetcher({"electric-car-sales.csv": body, "electric-car-sales.metadata.json": meta})
    rc = sd.main(["owid-iea-ev", "-o", str(tmp_path), "--charts", "electric-car-sales", "--snapshot", "s1"],
                 fetcher=fake)
    assert rc == 0
    snap = tmp_path / "iea-ev" / "s1"
    obs = read(snap / "normalized" / "observations.csv")
    assert obs == [{"indicator": "electric-car-sales", "entity": "World", "code": "OWID_WRL",
                    "year": "2024", "value": "17500000", "unit": "cars"}]   # empty value dropped
    m = json.loads((snap / "manifest.json").read_text())
    assert m["source"] == "owid" and m["dataset"] == "iea-ev" and m["status"] == "complete"
    assert {f["path"] for f in m["files"]} >= {"native/electric-car-sales.csv", "normalized/observations.csv"}
    assert all(len(f["sha256"]) == 64 for f in m["files"])


def test_usgs_cp1252_is_reencoded_and_numeric_column_added(tmp_path):
    src = tmp_path / "mcs.csv"
    src.write_bytes("Commodity,Country,Value\nPotash,Israel,\"2,000\"\nBromine,Israel,Large\n"
                    "Copper,World\x97total,W\n".replace("\x97", "—").encode("cp1252"))
    enc = sd._sniff_encoding(src)
    assert enc == "cp1252"
    rows, fields, text_vals = sd.normalise_usgs(src, enc)
    assert fields[-1] == "value_numeric"
    assert [r["value_numeric"] for r in rows] == ["2000", "", ""]
    assert rows[2]["Country"] == "World—total" and text_vals == 2


def test_usgs_discovery_picks_matching_item():
    j = json.dumps({"items": [
        {"title": "Mineral Commodity Summaries 2026 Data Release - Mineral Industry Trends", "files": []},
        {"title": "Mineral Commodity Summaries 2026 Data Release - Commodity Salient U.S. and World Statistics",
         "link": {"url": "https://sb/item/1"},
         "files": [{"name": "MCS2026_Commodities_Data.csv", "url": "https://sb/csv"},
                   {"name": "MCS2026_Commodities_Data.xml", "url": "https://sb/xml"},
                   {"name": "other.zip", "url": "https://sb/zip"}]}]}).encode()
    _title, item, want = sd.discover_usgs(FakeFetcher({"sciencebase": j}), 2026)
    assert item == "https://sb/item/1" and set(want) == {"MCS2026_Commodities_Data.csv",
                                                         "MCS2026_Commodities_Data.xml"}


# ------------------------------------------------------ CMS / RxNorm / FAERS


def test_cms_cut_matches_generic_substrings(tmp_path):
    p = tmp_path / "annual.csv"
    p.write_text("Brnd_Name,Gnrc_Name,Tot_Spndng_2024\nOzempic,Semaglutide,12970296347\n"
                 "Exforge,Amlodipine Besylate/Valsartan,1\nLipitor,Atorvastatin Calcium,2\n", encoding="utf-8")
    rows = sd.cut_cms({"annual.csv": p}, ["semaglutide", "valsartan"])
    assert [r["Brnd_Name"] for r in rows] == ["Ozempic", "Exforge"] and rows[0]["file"] == "annual.csv"


def test_rxnorm_parse():
    dj = {"drugGroup": {"conceptGroup": [{"tty": "BPCK"}, {"tty": "SBD", "conceptProperties": [
        {"rxcui": "1", "name": "semaglutide 2 MG/ML [Ozempic]", "tty": "SBD", "synonym": "Ozempic"}]}]}}
    ing, prods = sd.parse_rxnorm("semaglutide", dj, {"idGroup": {"rxnormId": ["1991302"]}})
    assert ing == {"ingredient": "semaglutide", "rxcui": "1991302"}
    assert prods == [{"ingredient_query": "semaglutide", "tty": "SBD", "rxcui": "1",
                      "name": "semaglutide 2 MG/ML [Ozempic]", "synonym": "Ozempic"}]


def test_faers_run_truncated_manifest_and_key_never_written(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENFDA_API_KEY", "secret-key-123")
    monkeypatch.setattr(sd.time, "sleep", lambda *_: None)
    total = json.dumps({"meta": {"results": {"total": 78023}}}).encode()
    counts = json.dumps({"results": [{"term": "NAUSEA", "count": 9000}, {"term": "VOMITING", "count": 5000}]}).encode()
    fake = FakeFetcher({"limit=1": total, "count=": counts, "zzz": None})
    assert sd.main(["faers", "-o", str(tmp_path), "--drugs", "Semaglutide", "--top", "2"], fetcher=fake) == 0
    snap = snapshot(tmp_path)
    assert read(snap / "normalized" / "report_totals.csv") == [{"drug": "semaglutide", "reports": "78023"}]
    assert [r["reaction_meddra_pt"] for r in read(snap / "normalized" / "reaction_counts.csv")] == ["NAUSEA", "VOMITING"]
    m = json.loads((snap / "manifest.json").read_text())
    assert m["truncated"] is True and m["snapshot_kind"] == "selected"
    assert all(c[1].get("api_key") == "secret-key-123" for c in fake.calls)      # sent as a param ...
    assert not any("secret-key-123" in p.read_text(errors="ignore")             # ... never written
                   for p in snap.rglob("*") if p.is_file())


def test_faers_unknown_drug_gives_zero_not_crash(tmp_path, monkeypatch):
    monkeypatch.setattr(sd.time, "sleep", lambda *_: None)
    monkeypatch.delenv("OPENFDA_API_KEY", raising=False)
    assert sd.main(["faers", "-o", str(tmp_path), "--drugs", "nosuchdrug"],
                   fetcher=FakeFetcher({"api.fda.gov": None})) == 0
    snap = snapshot(tmp_path)
    assert read(snap / "normalized" / "report_totals.csv") == [{"drug": "nosuchdrug", "reports": "0"}]


# ------------------------------------------------------------------- CLI


def test_dry_run_downloads_nothing(tmp_path, capsys):
    fake = FakeFetcher({})
    assert sd.main(["owid-iea-ev", "-o", str(tmp_path), "--dry-run"], fetcher=fake) == 0
    assert fake.calls == [] and not any(tmp_path.iterdir())
    assert "[plan] GET https://ourworldindata.org/grapher/electric-car-sales.csv" in capsys.readouterr().out


def test_existing_snapshot_needs_overwrite(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(sd.time, "sleep", lambda *_: None)
    snap = tmp_path / "drug-crosswalk" / "s1"
    snap.mkdir(parents=True)
    (snap / "keep.txt").write_text("x")
    assert sd.main(["rxnorm", "-o", str(tmp_path), "--snapshot", "s1", "--drugs", "x"],
                   fetcher=FakeFetcher({})) == 2
    assert "--overwrite" in capsys.readouterr().err and (snap / "keep.txt").exists()


def test_html_served_as_data_is_rejected(tmp_path, capsys):
    fake = FakeFetcher({"grapher": b"<html>blocked</html>"}, html=("grapher",))
    assert sd.main(["owid-iea-ev", "-o", str(tmp_path), "--charts", "c"], fetcher=fake) == 2
    assert "HTML page" in capsys.readouterr().err


def test_prefer_ipv4_restores_getaddrinfo():
    import socket
    orig = socket.getaddrinfo
    with sd.prefer_ipv4(True):
        assert socket.getaddrinfo is not orig
    assert socket.getaddrinfo is orig
