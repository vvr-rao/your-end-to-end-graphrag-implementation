"""Supplementary public datasets that structured_download.py does not cover.

Six small sources used to enrich the finance and pharma test datasets. Each
run writes one snapshot in the same layout as structured_download.py, so
hf_dataset.py can upload it (default output: source_documents/structured_<source>/):

    <output>/<dataset>/<UTC timestamp>/
        native/         files exactly as served
        normalized/     tidy UTF-8 CSVs (all values kept as text)
        manifest.json   source, dataset, status, files + sha256, terms, warnings

Sources (no API keys needed):

    worldbank-pinksheet  World Bank commodity prices (Pink Sheet): monthly +
                         annual workbooks -> long prices.csv
    owid-iea-ev          IEA Global EV Outlook car series as republished by
                         Our World in Data (iea.org refuses scripted downloads)
    usgs-mcs             USGS Mineral Commodity Summaries data release:
                         US + world production / reserves by commodity
    cms-partd            Medicare Part D spending by drug (annual + quarterly
                         national tables, plus a cut for --drugs)
    rxnorm               NLM RxNorm: ingredient -> branded / clinical products
    faers                openFDA FAERS: top reactions + report totals per drug
    owid-energy          Our World in Data energy dataset (one wide row per
                         country-year, ~130 columns; codebook kept as native)
    faostat-fertilizer   FAOSTAT fertilizers by product (bulk, normalized form)
    worldbank-wdi        World Bank World Development Indicators API, long
                         (indicator, economy, year) for --indicators x --countries
    imf-weo              IMF World Economic Outlook via the DataMapper API, long
                         (indicator, economy, year) incl. projection years
    comtrade             UN Comtrade public preview (no key; <= 500 rows per
                         call): annual trade by partner for --hs x --reporters x --years

Examples:

    uv run --extra structured python source_documents/supplementary_download.py \\
        worldbank-pinksheet -o source_documents/my-bundle/worldbank
    uv run python source_documents/supplementary_download.py owid-iea-ev
    uv run python source_documents/supplementary_download.py usgs-mcs --year 2026
    uv run python source_documents/supplementary_download.py cms-partd --drugs semaglutide,tirzepatide
    uv run python source_documents/supplementary_download.py faers --drugs semaglutide --top 50

Release URLs are discovered at run time (World Bank landing page, ScienceBase
catalogue, data.cms.gov catalogue); pass --url to pin one instead.
worldbank-pinksheet needs openpyxl (`uv sync --extra structured`).
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import os
import re
import socket
import sys
import time
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
UA = "yegi-supplementary-download/1.0 (research dataset build)"

PHARMA_DRUGS = ("semaglutide tirzepatide dulaglutide liraglutide exenatide lixisenatide amlodipine "
                "atorvastatin valsartan enalapril hydrochlorothiazide lisinopril losartan metformin "
                "sitagliptin").split()
OWID_CHARTS = ["electric-car-sales", "electric-car-stocks", "electric-car-sales-share",
               "share-car-stocks-electric"]


# Default scopes of the supply-chain bundle (overridable on the command line).
WDI_INDICATORS = ("NY.GDP.MKTP.KD.ZG NY.GDP.MKTP.CD NY.GDP.PCAP.CD FP.CPI.TOTL.ZG NV.IND.MANF.ZS "
                  "NV.IND.MANF.CD NE.EXP.GNFS.ZS NE.IMP.GNFS.ZS TX.VAL.TECH.MF.ZS TM.VAL.FUEL.ZS.UN "
                  "TX.VAL.FUEL.ZS.UN AG.CON.FERT.ZS EG.IMP.CONS.ZS EG.ELC.RNEW.ZS BX.KLT.DINV.WD.GD.ZS").split()
WDI_COUNTRIES = ("CHN JPN KOR IND IDN VNM THA MYS PHL SGP BGD PAK LKA KHM MMR LAO MNG NPL AUS NZL HKG "
                 "SAU ARE IRN IRQ KWT QAT OMN ISR TUR EGY USA CAN MEX BRA ARG DEU FRA GBR ITA NLD ESP "
                 "RUS UKR POL NOR ZAF NGA KEN MAR WLD EAS EAP SAS MEA ECS NAC LCN SSF HIC LMY").split()
IMF_INDICATORS = ("NGDP_RPCH NGDPD NGDPDPC PCPIPCH PCPIEPCH LUR BCA_NGDPD GGXWDG_NGDP "
                  "GGXCNL_NGDP PPPSH").split()
# UN M49 reporter codes (490 = "Other Asia, nes", Comtrade's code for Taiwan)
COMTRADE_REPORTERS = "156 392 410 699 490 704 458 702 764 842 276 528 682 784 36".split()
COMTRADE_HS = ["8542", "31", "2709"]          # integrated circuits, fertilizers, crude oil
COMTRADE_YEARS = ["2022", "2023", "2024"]


class UsageError(Exception):
    pass


# ----------------------------------------------------------------- plumbing


@contextlib.contextmanager
def prefer_ipv4(enabled: bool) -> Iterator[None]:
    """Try IPv4 addresses first (for hosts whose IPv6 route hangs)."""
    if not enabled:
        yield
        return
    orig = socket.getaddrinfo

    def ordered(*a, **k):
        res = orig(*a, **k)
        return sorted(res, key=lambda r: r[0] != socket.AF_INET)

    socket.getaddrinfo = ordered
    try:
        yield
    finally:
        socket.getaddrinfo = orig


class Fetcher:
    """GET with retries + backoff; optional atomic write via a .part file."""

    def __init__(self, session: requests.Session | None = None, retries: int = 4, timeout: int = 180):
        self.s = session or requests.Session()
        self.s.headers.setdefault("User-Agent", UA)
        self.retries, self.timeout = retries, timeout
        self.requests = 0

    def get(self, url: str, params: dict | None = None, dest: Path | None = None,
            allow_404: bool = False) -> bytes | None:
        for attempt in range(self.retries + 1):
            self.requests += 1
            try:
                r = self.s.get(url, params=params, timeout=self.timeout)
            except requests.RequestException:
                if attempt == self.retries:
                    raise
                time.sleep(2 * (attempt + 1))
                continue
            if r.status_code == 404 and allow_404:
                return None
            if r.status_code in (429, 500, 502, 503, 504) and attempt < self.retries:
                wait = r.headers.get("Retry-After")
                time.sleep(float(wait) if wait and wait.isdigit()
                           else (15 if r.status_code == 429 else 2) * (attempt + 1))
                continue
            r.raise_for_status()
            if dest is not None:
                ctype = r.headers.get("Content-Type", "")
                if "text/html" in ctype and dest.suffix not in (".html", ".htm"):
                    raise UsageError(f"{url} returned an HTML page, not {dest.suffix} data")
                tmp = dest.with_name(dest.name + ".part")
                tmp.write_bytes(r.content)
                os.replace(tmp, dest)
            return r.content
        return None  # unreachable


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


class Snapshot:
    def __init__(self, root: Path, source: str, dataset: str, ts: str, overwrite: bool = False):
        self.dir = root / dataset / ts
        if self.dir.exists() and any(self.dir.iterdir()) and not overwrite:
            raise UsageError(f"{self.dir} already exists (pass --overwrite to replace it)")
        self.native = self.dir / "native"
        self.norm = self.dir / "normalized"
        self.native.mkdir(parents=True, exist_ok=True)
        self.norm.mkdir(exist_ok=True)
        self.source, self.dataset = source, dataset
        self.files: list[dict] = []

    def add(self, p: Path, role: str, rows: int | None = None, derived: list[str] | None = None,
            **extra) -> str:
        rel = p.relative_to(self.dir).as_posix()
        d = {"path": rel, "role": role, "format": p.suffix.lstrip(".") or "bin",
             "bytes": p.stat().st_size, "sha256": sha256(p), **extra}
        if rows is not None:
            d["rows"] = rows
        if derived:
            d["derived_from"] = derived
        self.files.append(d)
        return rel

    def write_csv(self, name: str, rows: list[dict], fields: list[str] | None = None,
                  derived: list[str] | None = None) -> int:
        p = self.norm / name
        if fields is None:
            fields = []
            for r in rows:
                fields.extend(k for k in r if k not in fields)
        with p.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        self.add(p, "normalized", len(rows), derived)
        return len(rows)

    def finish(self, **meta) -> Path:
        m = {"source": self.source, "dataset": self.dataset, "status": "complete",
             "retrieved_at": datetime.now(UTC).isoformat(timespec="seconds"),
             **meta, "files": self.files}
        (self.dir / "manifest.json").write_text(json.dumps(m, indent=2, ensure_ascii=False) + "\n",
                                                encoding="utf-8")
        return self.dir


# ------------------------------------------------------------- World Bank


WB_LANDING = "https://www.worldbank.org/en/research/commodity-markets"
WB_SHEETS = [("CMO-Historical-Data-Monthly.xlsx", "Monthly Prices", "monthly", "nominal_usd"),
             ("CMO-Historical-Data-Annual.xlsx", "Annual Prices (Nominal)", "annual", "nominal_usd"),
             ("CMO-Historical-Data-Annual.xlsx", "Annual Prices (Real)", "annual", "real_2010_usd")]
_NA = {"", "…", "..", "...", "n.a.", "na"}


def discover_worldbank(f: Fetcher) -> dict[str, str]:
    html = f.get(WB_LANDING).decode("utf-8", "replace")
    urls = {}
    for name in ("CMO-Historical-Data-Monthly.xlsx", "CMO-Historical-Data-Annual.xlsx"):
        m = re.search(r"https://thedocs\.worldbank\.org/[^\"'\s]+/" + re.escape(name), html)
        if not m:
            raise UsageError(f"{name} link not found on {WB_LANDING}; pass --url for each file")
        urls[name] = m.group(0)
    return urls


def parse_pinksheet(paths: dict[str, Path]) -> tuple[list[dict], list[dict], dict[str, str]]:
    """Unpivot the wide price sheets. Returns (prices, series, updated-on per sheet)."""
    try:
        import openpyxl
    except ImportError as e:  # pragma: no cover - environment dependent
        raise UsageError("worldbank-pinksheet needs openpyxl: uv sync --extra structured") from e
    prices: list[dict] = []
    series: dict[tuple, str] = {}
    updated: dict[str, str] = {}
    for fname, sheet, freq, basis in WB_SHEETS:
        wb = openpyxl.load_workbook(paths[fname], read_only=True, data_only=True)
        rows = list(wb[sheet].iter_rows(values_only=True))
        wb.close()
        for r in rows[:8]:
            if r and isinstance(r[0], str) and r[0].startswith("Updated on"):
                updated[f"{fname}:{sheet}"] = r[0].removeprefix("Updated on").strip()
        try:
            h = next(i for i, r in enumerate(rows)
                     if r and len(r) > 1 and r[1] and str(r[1]).startswith("Crude oil"))
        except StopIteration:
            raise UsageError(f"{fname}/{sheet}: header row not found (layout changed?)") from None
        names, units = rows[h], rows[h + 1]
        for r in rows[h + 2:]:
            if not r or r[0] is None or not re.fullmatch(r"\d{4}(M\d{2})?", str(r[0]).strip()):
                continue
            period = str(r[0]).strip()
            for j in range(1, len(names)):
                if not names[j]:
                    continue
                raw = re.sub(r"\s+", " ", str(names[j])).strip()
                name = raw.rstrip("* ").strip()
                note = raw[len(name):].strip()
                unit = str(units[j]).strip() if j < len(units) and units[j] else ""
                v = r[j] if j < len(r) else None
                if v is None or (isinstance(v, str) and v.strip().lower() in _NA):
                    continue
                prices.append({"frequency": freq, "price_basis": basis, "commodity": name,
                               "footnote_marker": note, "unit": unit, "period": period,
                               "value": repr(v) if isinstance(v, float) else str(v)})
                key = (freq, basis, name)
                if unit or key not in series:
                    series[key] = unit
    srows = [{"frequency": k[0], "price_basis": k[1], "commodity": k[2], "unit": u}
             for k, u in sorted(series.items())]
    return prices, srows, updated


def run_worldbank(a, f: Fetcher, snap: Snapshot) -> Path:
    names = ("CMO-Historical-Data-Monthly.xlsx", "CMO-Historical-Data-Annual.xlsx")
    urls = dict(zip(names, _pinned(a, 2), strict=True)) if a.url else discover_worldbank(f)
    if a.dry_run:
        return _plan(urls.values())
    paths = {}
    for name, u in urls.items():
        paths[name] = snap.native / name
        f.get(u, dest=paths[name])
    nat = [snap.add(p, "native") for p in paths.values()]
    prices, series, updated = parse_pinksheet(paths)
    n = snap.write_csv("prices.csv", prices, derived=nat)
    snap.write_csv("series.csv", series, derived=nat)
    print(f"  prices.csv {n:,} rows; {len(series)} series")
    return snap.finish(
        mode="bulk", snapshot_kind="full", source_release=updated,
        params={"urls": list(urls.values()), "landing_page": WB_LANDING},
        transformations=[
            "prices.csv: Monthly Prices + Annual Prices (Nominal, Real) unpivoted to one row per "
            "commodity x period; not-available cells dropped; values kept as in the workbook",
            "footnote markers ('*', '**') moved from the commodity name to footnote_marker so names "
            "match across sheets; the notes are in the native 'Description' sheet",
            "index sheets and index weights are kept only in the native xlsx"],
        attribution="World Bank Commodity Price Data (The Pink Sheet)",
        terms_urls=["https://www.worldbank.org/en/about/legal/terms-of-use-for-datasets"],
        data_dictionary_urls=[WB_LANDING])


# --------------------------------------------------- Our World in Data / IEA


def stack_owid(snap_native: Path, charts: list[str]) -> tuple[list[dict], list[dict]]:
    rows, inds = [], []
    for c in charts:
        meta = json.loads((snap_native / f"{c}.metadata.json").read_text(encoding="utf-8"))
        col = next(iter(meta["columns"].values()))
        with (snap_native / f"{c}.csv").open(newline="", encoding="utf-8") as fh:
            rd = csv.DictReader(fh)
            vcols = [k for k in rd.fieldnames or [] if k not in ("entity", "code", "year")]
            if len(vcols) != 1:
                raise UsageError(f"{c}.csv: expected one value column, got {vcols}")
            for r in rd:
                if r[vcols[0]] != "":
                    rows.append({"indicator": c, "entity": r["entity"], "code": r["code"],
                                 "year": r["year"], "value": r[vcols[0]], "unit": col.get("unit", "")})
        inds.append({"indicator": c, "title": col.get("titleLong") or col.get("titleShort", ""),
                     "unit": col.get("unit", ""), "timespan": col.get("timespan", ""),
                     "last_updated": col.get("lastUpdated", ""), "citation": col.get("citationLong", "")})
    return rows, inds


def run_owid(a, f: Fetcher, snap: Snapshot) -> Path:
    charts = a.charts.split(",") if a.charts else OWID_CHARTS
    base = "https://ourworldindata.org/grapher/"
    if a.dry_run:
        return _plan(f"{base}{c}.csv (+ .metadata.json)" for c in charts)
    for c in charts:
        f.get(f"{base}{c}.csv", params={"v": "1", "csvType": "full", "useColumnShortNames": "true"},
              dest=snap.native / f"{c}.csv")
        f.get(f"{base}{c}.metadata.json", dest=snap.native / f"{c}.metadata.json")
    nat = [snap.add(p, "native") for p in sorted(snap.native.iterdir())]
    rows, inds = stack_owid(snap.native, charts)
    snap.write_csv("observations.csv", rows, derived=[n for n in nat if n.endswith(".csv")])
    snap.write_csv("indicators.csv", inds)
    print(f"  observations.csv {len(rows):,} rows; {len(inds)} indicators")
    release = sorted({i["citation"].split("\u2013")[0].strip() for i in inds if i["citation"]})
    return snap.finish(
        mode="api", snapshot_kind="full", source_release="; ".join(release)[:300],
        params={"charts": charts, "url_template": base + "<chart>.csv?v=1&csvType=full&useColumnShortNames=true"},
        transformations=["observations.csv: chart CSVs stacked long (indicator, entity, code, year, value, unit)"],
        attribution="International Energy Agency, Global EV Outlook - processed by Our World in Data",
        terms_urls=["https://www.iea.org/terms", "https://ourworldindata.org/faqs#can-i-use-or-redistribute-your-data"],
        warnings=["Secondary source: iea.org refuses scripted downloads (HTTP 403), so the official "
                  "'EV data by country' XLSX is not fetched; OWID republishes a subset",
                  "Entities include regions/aggregates (World, Europe, Rest of the world)"])


# -------------------------------------------------------------------- USGS


SCIENCEBASE = "https://www.sciencebase.gov/catalog/items"


def discover_usgs(f: Fetcher, year: int) -> tuple[str, str, dict[str, str]]:
    title = f"Mineral Commodity Summaries {year} Data Release - Commodity Salient"
    j = json.loads(f.get(SCIENCEBASE, params={"q": title, "format": "json", "fields": "title,files", "max": "10"}))
    for it in j.get("items", []):
        if it["title"].startswith(title):
            files = {x["name"]: x["url"] for x in it.get("files", [])}
            want = {n: u for n, u in files.items() if n == f"MCS{year}_Commodities_Data.csv"
                    or n == f"MCS{year}_Commodities_Data.xml"}
            if f"MCS{year}_Commodities_Data.csv" in want:
                return it["title"], it.get("link", {}).get("url", ""), want
    raise UsageError(f"ScienceBase has no '{title}…' item with MCS{year}_Commodities_Data.csv "
                     "(try another --year)")


def normalise_usgs(src: Path, encoding: str) -> tuple[list[dict], list[str], int]:
    with src.open(encoding=encoding, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise UsageError(f"{src.name} is empty")
    fields = [*rows[0], "value_numeric"]
    text_vals = 0
    for r in rows:
        v = (r.get("Value") or "").replace(",", "").strip()
        r["value_numeric"] = v if re.fullmatch(r"-?\d+(\.\d+)?", v) else ""
        text_vals += bool((r.get("Value") or "").strip()) and not r["value_numeric"]
    return rows, fields, text_vals


def _sniff_encoding(p: Path) -> str:
    try:
        p.read_bytes().decode("utf-8")
        return "utf-8-sig"
    except UnicodeDecodeError:
        return "cp1252"


def run_usgs(a, f: Fetcher, snap: Snapshot) -> Path:
    year = a.year or datetime.now(UTC).year
    if a.url:
        title, item, want = f"user-supplied ({year})", "", {Path(a.url[0].split("?")[0]).name or "commodities.csv": a.url[0]}
    else:
        title, item, want = discover_usgs(f, year)
    if a.dry_run:
        return _plan(want.values())
    for n, u in want.items():
        f.get(u, dest=snap.native / n)
    csv_src = next(snap.native / n for n in want if n.endswith(".csv"))
    enc = _sniff_encoding(csv_src)
    for p in sorted(snap.native.iterdir()):
        snap.add(p, "native", **({"encoding": enc.replace("-sig", "")} if p == csv_src else {}))
    rows, fields, text_vals = normalise_usgs(csv_src, enc)
    snap.write_csv("commodities.csv", rows, fields, derived=[f"native/{csv_src.name}"])
    print(f"  commodities.csv {len(rows):,} rows ({text_vals:,} non-numeric values)")
    return snap.finish(
        mode="bulk", snapshot_kind="full", source_release=title,
        params={"sciencebase_item": item, "year": year, "urls": list(want.values())},
        transformations=[f"re-encoded {enc} -> UTF-8",
                         "value_numeric: Value without thousands separators; empty when Value is text "
                         "(W = withheld, NA, ranges, '>50', 'Large')"],
        attribution=f"U.S. Geological Survey, Mineral Commodity Summaries {year}",
        terms_urls=["https://www.usgs.gov/information-policies-and-instructions/copyrights-and-credits"],
        data_dictionary_urls=[f"native/{n}" for n in want if n.endswith(".xml")],
        warnings=[f"{text_vals} rows have a non-numeric Value; kept as text in Value"])


# --------------------------------------------------------------- CMS Part D


CMS_CATALOG = "https://data.cms.gov/data.json"
CMS_TITLES = {"annual": "Medicare Part D Spending by Drug", "quarterly": "Medicare Quarterly Part D Spending by Drug"}


def discover_cms(f: Fetcher) -> dict[str, tuple[str, str]]:
    cat = json.loads(f.get(CMS_CATALOG))
    out = {}
    for kind, title in CMS_TITLES.items():
        for ds in cat.get("dataset", []):
            if ds.get("title", "").split(" : ")[0].strip() != title:
                continue
            for dist in ds.get("distribution", []):
                url = dist.get("downloadURL") or dist.get("accessURL") or ""
                if (dist.get("format") or dist.get("mediaType") or "").upper().endswith("CSV") or url.endswith(".csv"):
                    out[kind] = (url, dist.get("title") or ds["title"])
                    break
            if kind in out:
                break
        if kind not in out:
            raise UsageError(f"'{title}' CSV not found in {CMS_CATALOG}")
    return out


def cut_cms(paths: dict[str, Path], drugs: list[str]) -> list[dict]:
    sub = []
    for name, p in paths.items():
        with p.open(newline="", encoding="utf-8-sig") as fh:
            rd = csv.DictReader(fh)
            gcol = next((c for c in rd.fieldnames or [] if c.lower().startswith("gnrc_name")), None)
            if gcol is None:
                raise UsageError(f"{p.name}: no Gnrc_Name column")
            for r in rd:
                g = r[gcol].lower()
                if any(d in g for d in drugs):
                    sub.append({"file": name, **r})
    return sub


def run_cms(a, f: Fetcher, snap: Snapshot) -> Path:
    drugs = _drugs(a)
    found = ({k: (u, k) for k, u in zip(("annual", "quarterly"), _pinned(a, 2), strict=True)}
             if a.url else discover_cms(f))
    if a.dry_run:
        return _plan(u for u, _ in found.values())
    paths = {}
    for kind, (url, _title) in found.items():
        name = f"{kind}_{Path(url.split('?')[0]).name}"
        paths[name] = snap.native / name
        f.get(url, dest=paths[name])
    nat = [snap.add(p, "native") for p in paths.values()]
    sub = cut_cms(paths, drugs)
    if sub:
        snap.write_csv("study_drugs.csv", sub, derived=nat)
    print(f"  study_drugs.csv {len(sub):,} rows for {len(drugs)} drugs")
    return snap.finish(
        mode="bulk", snapshot_kind="full",
        source_release={k: t for k, (_u, t) in found.items()},
        params={"urls": [u for u, _ in found.values()], "catalog": CMS_CATALOG, "drugs": drugs},
        transformations=["native files are the complete national tables (all drugs)",
                         "study_drugs.csv: rows whose generic name contains one of --drugs (substring "
                         "match, so combinations are included); 'file' names the source table"],
        attribution="Centers for Medicare & Medicaid Services, Medicare Part D Spending by Drug",
        terms_urls=["https://data.cms.gov/about"],
        data_dictionary_urls=["https://data.cms.gov/resources/medicare-part-d-spending-by-drug-data-dictionary"])


# ------------------------------------------------------------------- RxNorm


RXNAV = "https://rxnav.nlm.nih.gov/REST"


def parse_rxnorm(drug: str, drugs_json: dict, rxcui_json: dict) -> tuple[dict, list[dict]]:
    ing = {"ingredient": drug, "rxcui": ";".join(rxcui_json.get("idGroup", {}).get("rxnormId", []) or [])}
    prods = []
    for g in drugs_json.get("drugGroup", {}).get("conceptGroup", []) or []:
        for c in g.get("conceptProperties", []) or []:
            prods.append({"ingredient_query": drug, "tty": c.get("tty", ""), "rxcui": c.get("rxcui", ""),
                          "name": c.get("name", ""), "synonym": c.get("synonym", "")})
    return ing, prods


def run_rxnorm(a, f: Fetcher, snap: Snapshot) -> Path:
    drugs = _drugs(a)
    if a.dry_run:
        return _plan(f"{RXNAV}/drugs.json?name={d} (+ rxcui.json)" for d in drugs)
    ings, prods = [], []
    for d in drugs:
        dj = json.loads(f.get(f"{RXNAV}/drugs.json", params={"name": d}, dest=snap.native / f"drugs_{_slug(d)}.json"))
        rj = json.loads(f.get(f"{RXNAV}/rxcui.json", params={"name": d, "search": "0"},
                              dest=snap.native / f"rxcui_{_slug(d)}.json"))
        i, p = parse_rxnorm(d, dj, rj)
        ings.append(i)
        prods.extend(p)
        time.sleep(0.1)  # RxNav asks for <= 20 requests/second
    for p in sorted(snap.native.iterdir()):
        snap.add(p, "native")
    snap.write_csv("ingredients.csv", ings, ["ingredient", "rxcui"])
    snap.write_csv("products.csv", prods, ["ingredient_query", "tty", "rxcui", "name", "synonym"])
    print(f"  {len(ings)} ingredients, {len(prods):,} products")
    return snap.finish(
        mode="api", snapshot_kind="selected",
        params={"endpoints": [f"{RXNAV}/drugs.json?name=<drug>", f"{RXNAV}/rxcui.json?name=<drug>"], "drugs": drugs},
        transformations=["products.csv: every concept in drugs.json conceptGroups (SBD branded, SCD clinical, "
                         "packs); a combination product appears under each of its ingredients"],
        attribution="U.S. National Library of Medicine, RxNorm (via RxNav)",
        terms_urls=["https://lhncbc.nlm.nih.gov/RxNav/TermsofService.html"])


# -------------------------------------------------------------------- FAERS


OPENFDA_EVENT = "https://api.fda.gov/drug/event.json"


def run_faers(a, f: Fetcher, snap: Snapshot) -> Path:
    drugs = _drugs(a)
    if a.dry_run:
        return _plan(f'{OPENFDA_EVENT}?search=patient.drug.openfda.generic_name:"{d}"' for d in drugs)
    key = os.environ.get("OPENFDA_API_KEY")  # optional; never written to disk
    rows, totals = [], []
    for d in drugs:
        q = f'patient.drug.openfda.generic_name:"{d}"'
        base = {"search": q, **({"api_key": key} if key else {})}
        tot = f.get(OPENFDA_EVENT, params={**base, "limit": "1"}, dest=snap.native / f"total_{_slug(d)}.json",
                    allow_404=True)
        totals.append({"drug": d, "reports": json.loads(tot)["meta"]["results"]["total"] if tot else 0})
        data = f.get(OPENFDA_EVENT, params={**base, "count": "patient.reaction.reactionmeddrapt.exact",
                                            "limit": str(a.top)},
                     dest=snap.native / f"reactions_{_slug(d)}.json", allow_404=True)
        for i, r in enumerate(json.loads(data)["results"] if data else [], 1):
            rows.append({"drug": d, "rank": i, "reaction_meddra_pt": r["term"], "report_count": r["count"]})
        time.sleep(0.3)
    for p in sorted(snap.native.iterdir()):
        snap.add(p, "native")
    snap.write_csv("reaction_counts.csv", rows, ["drug", "rank", "reaction_meddra_pt", "report_count"])
    snap.write_csv("report_totals.csv", totals, ["drug", "reports"])
    print(f"  {len(rows):,} reaction rows for {len(drugs)} drugs")
    return snap.finish(
        mode="api", snapshot_kind="selected", truncated=True,
        params={"endpoint": OPENFDA_EVENT, "search": 'patient.drug.openfda.generic_name:"<drug>"',
                "count": "patient.reaction.reactionmeddrapt.exact", "limit": a.top, "drugs": drugs},
        transformations=[f"top {a.top} MedDRA preferred terms per drug by report count; report_totals = "
                         "all FAERS reports listing the drug"],
        attribution="U.S. Food and Drug Administration, FAERS via openFDA",
        terms_urls=["https://open.fda.gov/terms/", "https://open.fda.gov/license/"],
        warnings=["FAERS reports are spontaneous and unverified: counts do not show that a drug caused a "
                  "reaction or how often it happens",
                  "A report can list several drugs; counts include every report that lists the drug",
                  f"Only the {a.top} most frequent reactions per drug are kept (truncated by design)"])


# ---------------------------------------------------------------------- CLI


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def _drugs(a) -> list[str]:
    return [d.strip().lower() for d in (a.drugs or ",".join(PHARMA_DRUGS)).split(",") if d.strip()]


def _pinned(a, n: int) -> list[str]:
    if len(a.url) != n:
        raise UsageError(f"{a.source} needs exactly {n} --url values (got {len(a.url)})")
    return a.url


# ------------------------------------------------------- supply-chain bundle


def _csv_list(v: str | None, default) -> list[str]:
    return [x.strip() for x in v.split(",") if x.strip()] if v else list(default)


def _years(v: str | None, default: tuple[int, int]) -> tuple[int, int]:
    if not v:
        return default
    m = re.fullmatch(r"(\d{4})(?::(\d{4}))?", v.strip())
    if not m:
        raise UsageError(f"--years must be YYYY or YYYY:YYYY, got {v!r}")
    lo = int(m.group(1))
    return lo, int(m.group(2) or lo)


def _num(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def _cached_get(f: Fetcher, url: str, params: dict | None, dest: Path) -> bytes:
    """Reuse a file an interrupted run already saved (resume with --snapshot <ts> --overwrite)."""
    if dest.exists() and dest.stat().st_size:
        return dest.read_bytes()
    return f.get(url, params=params, dest=dest)


def _quota_stop(e: requests.HTTPError) -> str | None:
    """A rate/quota refusal that retrying now cannot fix: the message to record, else None."""
    r = e.response
    if r is None or r.status_code not in (403, 429):
        return None
    wait = r.headers.get("Retry-After")
    return f"HTTP {r.status_code} {r.reason}" + (f" (Retry-After {wait}s)" if wait else "")


OWID_ENERGY = "https://raw.githubusercontent.com/owid/energy-data/master/"


def run_owid_energy(a, f: Fetcher, snap: Snapshot) -> Path:
    urls = {"owid-energy-data.csv": OWID_ENERGY + "owid-energy-data.csv",
            "owid-energy-codebook.csv": OWID_ENERGY + "owid-energy-codebook.csv"}
    if a.url:
        urls["owid-energy-data.csv"] = a.url[0]
    if a.dry_run:
        return _plan(urls.values())
    for n, u in urls.items():
        f.get(u, dest=snap.native / n)
    nat = [snap.add(snap.native / n, "native") for n in urls]
    with (snap.native / "owid-energy-data.csv").open(newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise UsageError("owid-energy-data.csv is empty")
    snap.write_csv("energy.csv", rows, list(rows[0]), derived=[nat[0]])
    print(f"  energy.csv {len(rows):,} rows x {len(rows[0])} columns")
    return snap.finish(
        mode="bulk", snapshot_kind="full", source_release="owid/energy-data (GitHub master)",
        params={"urls": list(urls.values())},
        transformations=["energy.csv: owid-energy-data.csv unchanged (UTF-8, values as text)"],
        attribution="Our World in Data energy dataset (Energy Institute Statistical Review, Ember, EIA and others)",
        terms_urls=["https://github.com/owid/energy-data#license"],
        data_dictionary_urls=[nat[1]],
        warnings=["Entities include regions and aggregates (World, Asia, Non-OECD, ...): iso_code is empty for them",
                  "Column units and sources are in native/owid-energy-codebook.csv"])


FAOSTAT_BULK = "https://bulks-faostat.fao.org/production/"
FAOSTAT_DATASETS = {"products": ("Inputs_FertilizersProduct_E_All_Data_(Normalized)", "fertilizer_products.csv"),
                    "nutrients": ("Inputs_FertilizersNutrient_E_All_Data_(Normalized)", "fertilizer_nutrients.csv")}


def run_faostat(a, f: Fetcher, snap: Snapshot) -> Path:
    import io
    import zipfile

    stem, out_name = FAOSTAT_DATASETS[a.fao_dataset]
    url = a.url[0] if a.url else f"{FAOSTAT_BULK}{stem}.zip"
    if a.dry_run:
        return _plan([url])
    dest = snap.native / f"{stem}.zip"
    f.get(url, dest=dest)
    nat = snap.add(dest, "native")
    zf = zipfile.ZipFile(dest)
    members = [n for n in zf.namelist() if n.endswith(".csv") and "All_Data" in n]
    if not members:
        raise UsageError(f"{dest.name} holds no *All_Data*.csv")
    raw = zf.read(members[0])
    try:
        text, enc = raw.decode("utf-8-sig"), "utf-8"
    except UnicodeDecodeError:
        text, enc = raw.decode("latin-1"), "latin-1"
    rows = list(csv.DictReader(io.StringIO(text, newline="")))
    if not rows:
        raise UsageError(f"{members[0]} is empty")
    snap.write_csv(out_name, rows, list(rows[0]), derived=[nat])
    print(f"  {out_name} {len(rows):,} rows")
    return snap.finish(
        mode="bulk", snapshot_kind="full", source_release=f"FAOSTAT bulk {stem}",
        params={"url": url, "member": members[0]},
        transformations=[f"{members[0]} re-encoded {enc} -> UTF-8, rows unchanged"],
        attribution="FAO. FAOSTAT. Food and Agriculture Organization of the United Nations",
        terms_urls=["https://www.fao.org/contact-us/terms/db-terms-of-use/en/"],
        warnings=["Area includes regional and world aggregates (Area Code >= 5000)",
                  "Flag marks estimated/imputed values (see FAOSTAT flag list)"])


WDI_API = "https://api.worldbank.org/v2/country/{countries}/indicator/{indicator}"


def run_wdi(a, f: Fetcher, snap: Snapshot) -> Path:
    inds = _csv_list(a.indicators, WDI_INDICATORS)
    ctry = _csv_list(a.countries, WDI_COUNTRIES)
    lo, hi = _years(a.years, (2000, 2025))
    params = {"format": "json", "date": f"{lo}:{hi}", "per_page": "20000"}
    urls = [WDI_API.format(countries=";".join(ctry), indicator=i) for i in inds]
    if a.dry_run:
        return _plan(urls)
    rows, missing, stopped, left = [], [], None, []
    for k, (ind, u) in enumerate(zip(inds, urls, strict=True)):
        dest = snap.native / f"wdi_{ind}.json"
        fresh = not dest.exists()
        try:
            body = _cached_get(f, u, params, dest)
        except requests.HTTPError as e:
            stopped = _quota_stop(e)
            if not stopped:
                raise
            left = inds[k:]
            break
        if fresh:
            time.sleep(a.pause)
        j = json.loads(body)
        if not isinstance(j, list) or len(j) < 2 or not j[1]:
            missing.append(ind)
            continue
        for r in j[1]:
            rows.append({"indicator_id": r["indicator"]["id"], "indicator": r["indicator"]["value"],
                         "economy_iso3": r.get("countryiso3code") or r["country"]["id"],
                         "economy": r["country"]["value"], "year": r.get("date", ""),
                         "value": _num(r.get("value")), "unit": r.get("unit", ""),
                         "obs_status": r.get("obs_status", "")})
    nat = [snap.add(p, "native") for p in sorted(snap.native.iterdir())]
    snap.write_csv("development_indicators.csv", rows, derived=nat)
    print(f"  development_indicators.csv {len(rows):,} rows ({len(inds) - len(missing)} indicators)")
    if stopped:
        print(f"  stopped: {stopped}; {len(left)} indicator(s) left -- rerun with --snapshot "
              f"{snap.dir.name} --overwrite to resume")
    return snap.finish(
        mode="api", snapshot_kind="partial" if stopped else "full",
        source_release="World Development Indicators (API v2, source 2)",
        params={"indicators": inds, "countries": ctry, "years": [lo, hi]},
        transformations=["one row per indicator x economy x year; value empty when the API has none"],
        attribution="World Bank, World Development Indicators",
        terms_urls=["https://www.worldbank.org/en/about/legal/terms-of-use-for-datasets"],
        warnings=[*(["indicators with no data: " + ", ".join(missing)] if missing else []),
                  *([f"stopped by {stopped}; not fetched: {', '.join(left)}"] if stopped else []),
                  "Economies include World Bank aggregates (WLD, EAS, SAS, HIC, ...)",
                  "Taiwan is not covered by WDI"])


IMF_API = "https://www.imf.org/external/datamapper/api/v1/"


def run_imf(a, f: Fetcher, snap: Snapshot) -> Path:
    inds = _csv_list(a.indicators, IMF_INDICATORS)
    lo, hi = _years(a.years, (2000, 2030))
    meta_urls = {n: IMF_API + n for n in ("indicators", "countries", "regions", "groups")}
    if a.dry_run:
        return _plan([*meta_urls.values(), *(IMF_API + i for i in inds)])
    meta = {n: json.loads(f.get(u, dest=snap.native / f"{n}.json")) for n, u in meta_urls.items()}
    ind_meta = meta["indicators"].get("indicators", {})
    names = {}
    for n, kind in (("countries", "country"), ("regions", "region"), ("groups", "group")):
        for code, v in (meta[n].get(n) or {}).items():
            names.setdefault(code, ((v or {}).get("label") or "", kind))
    rows, missing = [], []
    for ind in inds:
        body = json.loads(f.get(IMF_API + ind, dest=snap.native / f"weo_{ind}.json"))
        series = (body.get("values") or {}).get(ind) or {}
        if not series:
            missing.append(ind)
            continue
        im = ind_meta.get(ind, {})
        for code, by_year in series.items():
            label, kind = names.get(code, ("", ""))
            for y, v in sorted(by_year.items()):
                if y.isdigit() and lo <= int(y) <= hi:
                    rows.append({"indicator_id": ind, "indicator": im.get("label", ""), "unit": im.get("unit", ""),
                                 "economy_code": code, "economy": label, "economy_kind": kind,
                                 "year": y, "value": _num(v), "edition": im.get("source", "")})
    nat = [snap.add(p, "native") for p in sorted(snap.native.iterdir())]
    snap.write_csv("weo_indicators.csv", rows, derived=nat)
    editions = sorted({r["edition"] for r in rows if r["edition"]})
    print(f"  weo_indicators.csv {len(rows):,} rows; edition {', '.join(editions)}")
    return snap.finish(
        mode="api", snapshot_kind="full", source_release="; ".join(editions),
        params={"indicators": inds, "years": [lo, hi]},
        transformations=["one row per indicator x economy x year within the year range"],
        attribution="International Monetary Fund, World Economic Outlook database (DataMapper)",
        terms_urls=["https://www.imf.org/en/About/copyright-and-terms"],
        warnings=[*(["indicators with no data: " + ", ".join(missing)] if missing else []),
                  "Years after the latest actual are IMF projections; the API does not mark the "
                  "per-economy start of projections",
                  "economy_kind separates countries from regions and analytical groups"])


COMTRADE_API = "https://comtradeapi.un.org/public/v1/preview/C/A/HS"
COMTRADE_FIELDS = ["refYear", "reporterCode", "reporterISO", "reporterDesc", "flowCode", "flowDesc",
                   "partnerCode", "partnerISO", "partnerDesc", "classificationCode", "cmdCode", "cmdDesc",
                   "aggrLevel", "qtyUnitAbbr", "qty", "netWgt", "cifvalue", "fobvalue", "primaryValue",
                   "isReported", "isAggregate"]
COMTRADE_CAP = 500


def run_comtrade(a, f: Fetcher, snap: Snapshot) -> Path:
    reps = _csv_list(a.reporters, COMTRADE_REPORTERS)
    hs = _csv_list(a.hs, COMTRADE_HS)
    years = _csv_list(a.comtrade_years, COMTRADE_YEARS)
    calls = [(r, y, c) for c in hs for r in reps for y in years]
    if a.dry_run:
        return _plan(f"{COMTRADE_API}?reporterCode={r}&period={y}&cmdCode={c}&flowCode=M,X" for r, y, c in calls)
    rows, capped, empty, errors, stopped, left = [], [], [], [], None, []
    for k, (r, y, c) in enumerate(calls):
        params = {"reporterCode": r, "period": y, "cmdCode": c, "flowCode": "M,X", "includeDesc": "true"}
        dest = snap.native / f"comtrade_{c}_{r}_{y}.json"
        fresh = not dest.exists()
        try:
            body = _cached_get(f, COMTRADE_API, params, dest)
        except requests.HTTPError as e:
            stopped = _quota_stop(e)
            if not stopped:
                raise
            left = [f"{c2}/{r2}/{y2}" for r2, y2, c2 in calls[k:]]
            break
        j = json.loads(body or b"{}")
        if j.get("error"):
            errors.append(f"{c}/{r}/{y}: {j['error']}")
        data = j.get("data") or []
        if not data:
            empty.append(f"{c}/{r}/{y}")
        if len(data) >= COMTRADE_CAP:
            capped.append(f"{c}/{r}/{y}")
        rows.extend({col: _num(d.get(col)) for col in COMTRADE_FIELDS} for d in data)
        if fresh:
            time.sleep(a.pause)
    nat = [snap.add(p, "native") for p in sorted(snap.native.iterdir())]
    snap.write_csv("trade_flows.csv", rows, COMTRADE_FIELDS, derived=nat)
    print(f"  trade_flows.csv {len(rows):,} rows from {len(calls) - len(left)}/{len(calls)} calls "
          f"({len(empty)} empty, {len(capped)} capped)")
    if stopped:
        print(f"  stopped: {stopped}; {len(left)} call(s) left -- rerun with --snapshot "
              f"{snap.dir.name} --overwrite to resume")
    return snap.finish(
        mode="api", snapshot_kind="partial" if capped or stopped else "full",
        source_release="UN Comtrade public preview API (HS, annual, as reported classification)",
        params={"reporters": reps, "hs": hs, "years": years, "flows": ["M", "X"]},
        transformations=["one row per reporter x year x flow x partner x commodity; selected columns, values as text"],
        attribution="United Nations Comtrade Database",
        terms_urls=["https://comtradeplus.un.org/TradeFlow/Terms"],
        warnings=[*([f"calls at the {COMTRADE_CAP}-row preview cap (rows may be missing): {', '.join(capped)}"]
                    if capped else []),
                  *([f"calls with no data: {', '.join(empty)}"] if empty else []),
                  *([f"stopped by {stopped}; not fetched: {', '.join(left)}"] if stopped else []),
                  *(errors[:20]),
                  "partnerCode 0 / partnerDesc 'World' is the reporter's total across partners",
                  "reporterCode 490 'Other Asia, nes' is Comtrade's code for Taiwan"])


def _plan(urls) -> Path:
    for u in urls:
        print(f"  [plan] GET {u}")
    print("  dry run: nothing downloaded")
    return Path()


SOURCES: dict[str, tuple[str, str, Callable]] = {
    "worldbank-pinksheet": ("worldbank", "pink-sheet", run_worldbank),
    "owid-iea-ev": ("owid", "iea-ev", run_owid),
    "usgs-mcs": ("usgs", "mcs", run_usgs),
    "cms-partd": ("cms", "partd-spending", run_cms),
    "rxnorm": ("rxnorm", "drug-crosswalk", run_rxnorm),
    "faers": ("openfda", "faers-reaction-counts", run_faers),
    "owid-energy": ("owid", "energy", run_owid_energy),
    "faostat-fertilizer": ("faostat", "fertilizers", run_faostat),
    "worldbank-wdi": ("worldbank", "wdi", run_wdi),
    "imf-weo": ("imf", "weo", run_imf),
    "comtrade": ("comtrade", "trade-flows", run_comtrade),
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("source", choices=list(SOURCES))
    p.add_argument("-o", "--output", type=Path,
                   help="output root (default source_documents/structured_<source>/); the snapshot lands in "
                        "<output>/<dataset>/<UTC timestamp>/")
    p.add_argument("--url", action="append", help="pin release URL(s) instead of discovering them "
                   "(worldbank: monthly then annual xlsx; cms: annual then quarterly csv; usgs: the csv)")
    p.add_argument("--drugs", help="comma-separated generic names (cms-partd, rxnorm, faers); "
                   "default: the pharma-30 study drugs")
    p.add_argument("--top", type=int, default=100, help="faers: reactions kept per drug (max 1000)")
    p.add_argument("--year", type=int, help="usgs-mcs: MCS edition (default: current year)")
    p.add_argument("--charts", help="owid-iea-ev: comma-separated OWID chart slugs")
    p.add_argument("--indicators", help="worldbank-wdi / imf-weo: comma-separated indicator ids")
    p.add_argument("--countries", help="worldbank-wdi: comma-separated ISO3 economy codes (aggregates allowed)")
    p.add_argument("--years", help="worldbank-wdi / imf-weo: YYYY or YYYY:YYYY")
    p.add_argument("--reporters", help="comtrade: comma-separated UN M49 reporter codes")
    p.add_argument("--hs", help="comtrade: comma-separated HS codes (2-6 digits)")
    p.add_argument("--comtrade-years", help="comtrade: comma-separated years")
    p.add_argument("--pause", type=float, default=1.0, help="comtrade: seconds between calls")
    p.add_argument("--fao-dataset", choices=list(FAOSTAT_DATASETS), default="products",
                   help="faostat-fertilizer: which bulk table")
    p.add_argument("--snapshot", help="snapshot folder name (default: UTC timestamp)")
    p.add_argument("--overwrite", action="store_true", help="replace an existing snapshot folder")
    p.add_argument("--dry-run", action="store_true", help="print the URLs to fetch; download nothing")
    p.add_argument("--ipv4", action="store_true", help="try IPv4 addresses first (hosts whose IPv6 route hangs)")
    return p


def main(argv: list[str] | None = None, fetcher: Fetcher | None = None) -> int:
    a = build_parser().parse_args(argv)
    try:
        from dotenv import load_dotenv  # optional OPENFDA_API_KEY
        load_dotenv(HERE.parent / ".env")
    except ImportError:  # pragma: no cover
        pass
    source, dataset, fn = SOURCES[a.source]
    if a.source == "usgs-mcs":
        dataset = f"mcs-{a.year or datetime.now(UTC).year}"
    root = a.output or HERE / f"structured_{source}"
    ts = a.snapshot or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    f = fetcher or Fetcher()
    try:
        with prefer_ipv4(a.ipv4):
            snap = None if a.dry_run else Snapshot(root, source, dataset, ts, a.overwrite)
            print(f"[{a.source}] -> {root / dataset / ts}")
            out = fn(a, f, snap)
    except UsageError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except requests.HTTPError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    if not a.dry_run:
        print(f"[{a.source}] status=complete requests={f.requests} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
