"""FRED / ALFRED (Federal Reserve Bank of St. Louis) time series.

Verified 2026-11-08 against https://fred.stlouisfed.org/docs/api/fred/
(series, series/observations, series/vintagedates, errors pages):
observations limit 1..100000 with offset; file_type json|xml|csv|xlsx where
csv arrives as a zip (CSV + README); vintage_dates per request <= 2000 for
JSON; 120 requests/minute. FRED marks missing observations with "." -- that
marker is converted to an empty cell only in the normalized CSV.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from .._archive import safe_extract
from .._base import (
    Context,
    EnvVar,
    ModeSpec,
    Plan,
    Source,
    UsageError,
    iter_json_records,
    register,
    require_env,
)
from .._tabular import CsvTable

API = "https://api.stlouisfed.org/fred"
PAGE_MAX = 100_000
VINTAGE_CHUNK_JSON = 2000
VINTAGE_CHUNK_CSV = 400          # under every documented csv limit except USRECD (110)
SERIES_COLUMNS = ["id", "title", "units", "units_short", "frequency", "frequency_short",
                  "seasonal_adjustment", "seasonal_adjustment_short", "observation_start",
                  "observation_end", "last_updated", "popularity", "notes", "realtime_start",
                  "realtime_end"]
OBS_COLUMNS = ["series_id", "realtime_start", "realtime_end", "vintage_date", "date", "value"]
# output_type 2/3 return one column per vintage, named <SERIES>_<YYYYMMDD>.
_VINTAGE_COL = re.compile(r"(.+)_(\d{4})(\d{2})(\d{2})")
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


@register
class FredSource(Source):
    name = "fred"
    title = "FRED / ALFRED"
    domain = "finance"
    summary = ("Economic time series from FRED: series metadata and observations for a date "
               "range, optionally with real-time periods / historical vintages (ALFRED).")
    modes = {
        "api": ModeSpec(
            "FRED API v1 (no bulk mode: FRED publishes no bulk archive)",
            {"series": "metadata + observations for selected series ids"},
            formats=("native", "csv")),
    }
    env = [EnvVar("FRED_API_KEY", "api mode",
                  "free key from https://fredaccount.stlouisfed.org/apikeys")]
    docs_urls = ["https://fred.stlouisfed.org/docs/api/fred/"]
    terms_urls = ["https://fred.stlouisfed.org/docs/api/terms_of_use.html"]
    attribution = ("Federal Reserve Bank of St. Louis, FRED; series may carry their own "
                   "copyright -- see each series' notes")
    rate_per_second = 1.5
    notes = ["Observations and series metadata are stored separately (observations.csv / "
             "series.csv). '.' (missing) becomes an empty cell only in observations.csv."]

    def add_arguments(self, mode: str, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--series-id", action="append", required=True,
                            help="FRED series id (repeat or comma-separate), e.g. GDP,UNRATE")
        parser.add_argument("--start", help="observation_start YYYY-MM-DD")
        parser.add_argument("--end", help="observation_end YYYY-MM-DD")
        parser.add_argument("--realtime-start", help="realtime_start YYYY-MM-DD (ALFRED)")
        parser.add_argument("--realtime-end", help="realtime_end YYYY-MM-DD (ALFRED)")
        parser.add_argument("--vintage-dates",
                            help="comma-separated YYYY-MM-DD vintage dates (ALFRED)")
        parser.add_argument("--all-vintages", action="store_true",
                            help="fetch every vintage date of each series (ALFRED; can be large)")
        parser.add_argument("--output-type", type=int, choices=(1, 2, 3, 4), default=None,
                            help="1 real-time period (default), 2 vintage all obs, 3 vintage "
                                 "new/revised only, 4 initial release only")
        parser.add_argument("--units", help="lin chg ch1 pch pc1 pca cch cca log")
        parser.add_argument("--frequency", help="aggregate to lower frequency (json only)")
        parser.add_argument("--aggregation-method", choices=("avg", "sum", "eop"))
        parser.add_argument("--native-type", choices=("json", "csv"), default="json",
                            help="provider format to save: json (default) or csv "
                                 "(FRED returns it zipped; extracted safely)")

    def credentials(self, mode, args):
        key = require_env("FRED_API_KEY", "the FRED API",
                          docs="https://fredaccount.stlouisfed.org/apikeys")
        return {"params": {"api_key": key}}

    def validate(self, mode, args):
        ids = []
        for v in args.series_id:
            ids += [s.strip().upper() for s in v.split(",") if s.strip()]
        if not ids or any(not re.fullmatch(r"[A-Z0-9_.]+", s) for s in ids):
            raise UsageError(f"invalid --series-id {args.series_id!r}")
        args.series_id = ids
        for flag in ("start", "end", "realtime_start", "realtime_end"):
            v = getattr(args, flag)
            if v and not _DATE.fullmatch(v):
                raise UsageError(f"--{flag.replace('_', '-')} must be YYYY-MM-DD")
        if args.vintage_dates:
            vds = [d.strip() for d in args.vintage_dates.split(",") if d.strip()]
            if any(not _DATE.fullmatch(d) for d in vds):
                raise UsageError("--vintage-dates must be comma-separated YYYY-MM-DD")
            args.vintage_dates = vds
        if args.vintage_dates and args.all_vintages:
            raise UsageError("use either --vintage-dates or --all-vintages, not both")
        if (args.vintage_dates or args.all_vintages) and (args.realtime_start or args.realtime_end):
            raise UsageError("vintage dates and realtime_start/end are alternative ways to ask "
                             "for history; pass one or the other")
        if args.native_type == "csv":
            if args.format == "csv":
                raise UsageError("normalized CSV is built from FRED's JSON; use --native-type "
                                 "json with --format csv (or --native-type csv alone to keep "
                                 "FRED's own zipped CSV)")
            if args.frequency:
                raise UsageError("FRED only aggregates (--frequency) for json/xml output")

    # --------------------------------------------------------------- plan

    def _obs_params(self, args, series_id: str, vintages: list[str] | None) -> dict:
        p = {"series_id": series_id, "file_type": args.native_type}
        for flag, name in (("start", "observation_start"), ("end", "observation_end"),
                           ("realtime_start", "realtime_start"), ("realtime_end", "realtime_end"),
                           ("units", "units"), ("frequency", "frequency"),
                           ("aggregation_method", "aggregation_method"),
                           ("output_type", "output_type")):
            v = getattr(args, flag)
            if v not in (None, ""):
                p[name] = v
        if vintages:
            p["vintage_dates"] = ",".join(vintages)
            p.setdefault("output_type", 2)
        return p

    def plan(self, ctx: Context) -> Plan:
        a = ctx.args
        p = Plan(self.name, a.dataset, "api", ctx.root.name)
        for sid in a.series_id:
            p.requests.append(f"{API}/series?series_id={sid}&file_type=json")
            p.requests.append(f"{API}/series/observations "
                              + " ".join(f"{k}={v}" for k, v in
                                         self._obs_params(a, sid, a.vintage_dates).items()))
            if a.all_vintages:
                p.requests.append(f"{API}/series/vintagedates?series_id={sid} (then observations "
                                  f"in chunks of vintage dates)")
        return p

    # ---------------------------------------------------------------- run

    def run(self, ctx: Context, plan: Plan) -> None:
        a = ctx.args
        ctx.manifest.snapshot_kind = "query"
        meta_files: list[Path] = []
        obs_files: list[Path] = []
        records = 0
        pages = 0
        for sid in a.series_id:
            url = f"{API}/series"
            meta = ctx.client.get_json(url, params={"series_id": sid, "file_type": "json",
                                                    **_rt(a)})
            meta_files.append(ctx.save_json(f"native/{sid}/series.json", meta, url=url,
                                            records=len(meta.get("seriess") or [])))
            vintages = a.vintage_dates
            if a.all_vintages:
                vintages = self._vintage_dates(ctx, sid)
            chunk = VINTAGE_CHUNK_JSON if a.native_type == "json" else (
                100 if sid == "USRECD" else VINTAGE_CHUNK_CSV)
            groups = [vintages[i:i + chunk] for i in range(0, len(vintages), chunk)] \
                if vintages else [None]
            for gi, group in enumerate(groups, start=1):
                offset = 0
                while True:
                    reason = ctx.cap(records, pages)
                    if reason:
                        ctx.manifest.truncate(reason)
                        break
                    size = PAGE_MAX
                    if a.limit:
                        size = max(1, min(PAGE_MAX, a.limit - records))
                    params = {**self._obs_params(a, sid, group), "limit": size, "offset": offset}
                    stem = f"native/{sid}/observations-v{gi:03d}-o{offset:07d}"
                    if a.native_type == "json":
                        data = ctx.client.get_json(f"{API}/series/observations", params=params)
                        got = len(data.get("observations") or [])
                        obs_files.append(ctx.save_json(stem + ".json", data,
                                                       url=f"{API}/series/observations",
                                                       records=got))
                        total = int(data.get("count") or 0)
                    else:
                        got, total = self._csv_page(ctx, stem, params)
                    pages += 1
                    records += got
                    offset += got
                    if got == 0 or offset >= total:
                        break
                    if a.limit and records >= a.limit and offset < total:
                        ctx.manifest.truncate(f"--limit {a.limit} reached for {sid}")
                        break
                if ctx.manifest.truncated:
                    break
        ctx.manifest.pagination = {"pages_fetched": pages, "records": records,
                                   "complete": not ctx.manifest.truncated}
        if ctx.want_csv:
            self._normalize(ctx, meta_files, obs_files)

    def _vintage_dates(self, ctx: Context, sid: str) -> list[str]:
        out: list[str] = []
        offset = 0
        while True:
            url = f"{API}/series/vintagedates"
            data = ctx.client.get_json(url, params={"series_id": sid, "file_type": "json",
                                                    "limit": 10000, "offset": offset})
            got = data.get("vintage_dates") or []
            ctx.save_json(f"native/{sid}/vintagedates-o{offset:06d}.json", data, url=url,
                          records=len(got))
            out += got
            offset += len(got)
            if not got or offset >= int(data.get("count") or 0):
                return out

    def _csv_page(self, ctx: Context, stem: str, params: dict) -> tuple[int, int]:
        """FRED csv arrives as a zip; save it, extract safely, count rows."""
        from .._base import PlannedFile
        from .._tabular import count_csv_rows
        dest = ctx.fetch(PlannedFile(f"{API}/series/observations", stem + ".zip", expect="zip",
                                     params=params, description="FRED zipped CSV"),
                         format="zip")
        files = safe_extract(dest, ctx.path(stem + "_extracted"))
        rows = 0
        for f in files:
            is_csv = f.suffix.lower() == ".csv"
            n = count_csv_rows(f) if is_csv else None
            rows += n or 0
            ctx.manifest.add_file(ctx.root, f, role="extracted", format=f.suffix.lstrip("."),
                                  delimiter="comma" if is_csv else None, rows=n,
                                  derived_from=[dest])
        # A short page means the end; FRED's csv has no total count field.
        total = params["offset"] + rows + (1 if rows >= params["limit"] else 0)
        return rows, total

    def _normalize(self, ctx: Context, meta_files: list[Path], obs_files: list[Path]) -> None:
        series = CsvTable(ctx.path("normalized/series.csv"), SERIES_COLUMNS)
        for rec in iter_json_records(meta_files, "seriess"):
            series.write({c: rec.get(c) for c in SERIES_COLUMNS})
        series.close()
        obs = CsvTable(ctx.path("normalized/observations.csv"), OBS_COLUMNS)
        nulls = 0
        for path in obs_files:
            sid = path.parent.name
            for rec in iter_json_records([path], "observations"):
                if "value" in rec:                      # output_type 1 / 4: long already
                    cells = [(None, rec.get("value"))]
                else:                                   # output_type 2 / 3: wide vintages
                    cells = []
                    for col, val in rec.items():
                        m = _VINTAGE_COL.fullmatch(col)
                        if m and m.group(1).upper() == sid:
                            cells.append((f"{m.group(2)}-{m.group(3)}-{m.group(4)}", val))
                for vintage, value in cells:
                    if value == ".":
                        value = None
                        nulls += 1
                    obs.write({"series_id": sid, "realtime_start": rec.get("realtime_start"),
                               "realtime_end": rec.get("realtime_end"), "vintage_date": vintage,
                               "date": rec.get("date"), "value": value})
        obs.close()
        ctx.register_tables({"series": series}, meta_files, description="FRED series metadata")
        ctx.register_tables({"observations": obs}, obs_files,
                            description="FRED observations (long format)")
        ctx.manifest.transformations.append(
            f"observations JSON -> observations.csv; FRED missing marker '.' -> empty "
            f"({nulls} cells); values otherwise copied as text; vintage responses "
            f"(output_type 2/3, one <SERIES>_<YYYYMMDD> column per vintage) unpivoted to one "
            f"row per (vintage_date, date)")


def _rt(a) -> dict:
    return {k: v for k, v in (("realtime_start", a.realtime_start),
                              ("realtime_end", a.realtime_end)) if v}
