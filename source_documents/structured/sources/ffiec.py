"""FFIEC Central Data Repository (Call Reports / UBPR) and Federal Reserve NIC.

Verified 2026-11-08:
* CDR REST API -- spec CDR-PDD-SIS-611 v1.10
  (https://cdr.ffiec.gov/public/Files/SIS611_-_Retrieve_Public_Data_via_Web_Service.pdf):
  GET https://ffieccdr.azure-api.us/public/<Method>, headers ``UserID`` +
  ``Authentication: Bearer <token>`` (token from a CDR PWS account, renewed
  every 90 days) plus per-method headers (dataSeries, reportingPeriodEndDate
  MM/DD/YYYY, fiIdType, fiId, facsimileFormat). The legacy SOAP service was
  retired on 2026-02-28; it is not used here.
* CDR bulk page (https://cdr.ffiec.gov/public/pws/downloadbulkdata.aspx) is an
  ASP.NET postback form with no documented file URLs, and the NIC download
  page (https://www.ffiec.gov/npw/FinancialReport/FinancialDataDownload)
  answers automated clients with a CAPTCHA. Both are therefore supported as
  ``archive`` mode: the user downloads the zip in a browser and this tool
  verifies, extracts, documents and (optionally) converts it.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import re
from datetime import datetime
from pathlib import Path

from .._base import (
    Context,
    CredentialError,
    EnvVar,
    ModeSpec,
    Plan,
    PlannedFile,
    Source,
    UsageError,
    ingest_archive,
    register,
    require_env,
)
from .._manifest import utc_stamp
from .._tabular import CsvTable, iter_rows

API = "https://ffieccdr.azure-api.us/public"
FACSIMILE_EXT = {"XBRL": "xml", "SDF": "sdf.txt", "PDF": "pdf"}
PANEL_COLUMNS = ["ID_RSSD", "FDICCertNumber", "OCCChartNumber", "OTSDockNumber",
                 "PrimaryABARoutNumber", "Name", "State", "City", "Address", "ZIP", "FilingType",
                 "HasFiledForReportingPeriod"]


def _period(value: str) -> str:
    """Accept YYYY-MM-DD or MM/DD/YYYY; return the API's MM/DD/YYYY."""
    for fmt in ("%Y-%m-%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(value, fmt).strftime("%m/%d/%Y")
        except ValueError:
            continue
    raise UsageError(f"--period {value!r}: use YYYY-MM-DD or MM/DD/YYYY (quarter end)")


@register
class FfiecSource(Source):
    name = "ffiec"
    title = "FFIEC CDR / Federal Reserve NIC"
    domain = "finance"
    summary = ("Bank Call Report data from the FFIEC Central Data Repository (authenticated "
               "REST API), and user-downloaded CDR bulk / NIC archives.")
    modes = {
        "api": ModeSpec("CDR REST API (needs a free CDR PWS account)",
                        {"reporting-periods": "Call (or --series UBPR) reporting period list",
                         "panel": "panel of reporters for --period (institution ids)",
                         "filers-since": "institutions that filed/amended since --since",
                         "facsimile": "one report per --rssd for --period (XBRL / SDF / PDF)"},
                        formats=("native", "csv")),
        "archive": ModeSpec("process a zip you downloaded in a browser",
                            {"cdr-call-bulk": "CDR 'Call Reports -- Single Period' tab-delimited "
                                              "zip (one file per schedule)",
                             "nic": "Federal Reserve NIC download zip (attributes / "
                                    "relationships / transformations, CSV)"},
                            formats=("native", "csv"), bounded_by_limit=False),
    }
    env = [EnvVar("FFIEC_USER_ID", "api mode", "CDR PWS account user name"),
           EnvVar("FFIEC_TOKEN", "api mode",
                  "CDR PWS security token (regenerate every 90 days at "
                  "https://cdr.ffiec.gov/public/PWS/PWSPage.aspx)")]
    docs_urls = ["https://cdr.ffiec.gov/public/Files/SIS611_-_Retrieve_Public_Data_via_Web_Service.pdf",
                 "https://cdr.ffiec.gov/public/pws/downloadbulkdata.aspx",
                 "https://www.ffiec.gov/npw/FinancialReport/FinancialDataDownload"]
    terms_urls = ["https://cdr.ffiec.gov/public/HelpFileContainers/Disclaimer.aspx",
                  "https://www.ffiec.gov/npw/"]
    attribution = ("Federal Financial Institutions Examination Council (CDR); Board of Governors "
                   "of the Federal Reserve System (NIC)")
    data_dictionaries = ["https://www.federalreserve.gov/apps/mdrm/",
                         "https://www.ffiec.gov/npw/Help/Glossary"]
    rate_per_second = 0.5          # PWS allows roughly 2,500 downloads per hour
    notes = [
        "CDR Call Reports (bank regulatory filings) and NIC (institution structure: attributes, "
        "relationships, transformations) are different datasets and are kept apart.",
        "CDR bulk: open https://cdr.ffiec.gov/public/pws/downloadbulkdata.aspx, choose "
        "'Call Reports -- Single Period', a date and 'Tab Delimited', download, then run "
        "ffiec archive --dataset cdr-call-bulk --from-archive <zip>.",
        "NIC: download from https://www.ffiec.gov/npw/FinancialReport/FinancialDataDownload "
        "in a browser (the page uses a CAPTCHA), then ffiec archive --dataset nic "
        "--from-archive <zip>.",
        "Live status: the REST path follows the published spec but was NOT exercised during "
        "development (no PWS account available).",
    ]

    def add_arguments(self, mode: str, parser: argparse.ArgumentParser) -> None:
        if mode == "api":
            parser.add_argument("--series", choices=("Call", "UBPR"), default="Call",
                                help="data series for reporting-periods (default Call)")
            parser.add_argument("--period", help="reporting period end, YYYY-MM-DD or MM/DD/YYYY")
            parser.add_argument("--since", help="filers-since: date YYYY-MM-DD or MM/DD/YYYY")
            parser.add_argument("--rssd", action="append",
                                help="facsimile: ID_RSSD values (repeat or comma-separate)")
            parser.add_argument("--facsimile-format", choices=tuple(FACSIMILE_EXT),
                                default="XBRL")
        else:
            parser.add_argument("--from-archive", type=Path, action="append", required=True,
                                help="zip you downloaded (repeat for several NIC files)")

    def credentials(self, mode, args):
        if mode != "api":
            return {}
        user = require_env("FFIEC_USER_ID", "the FFIEC CDR REST API",
                           docs="https://cdr.ffiec.gov/public/PWS/PWSPage.aspx")
        token = require_env("FFIEC_TOKEN", "the FFIEC CDR REST API",
                            docs="https://cdr.ffiec.gov/public/PWS/PWSPage.aspx")
        if token.lower().startswith("bearer "):
            raise CredentialError("FFIEC_TOKEN should be the bare token (without 'Bearer ')")
        return {"headers": {"UserID": user, "Authentication": f"Bearer {token}"}}

    def validate(self, mode, args):
        if mode == "archive":
            for p in args.from_archive:
                if not p.exists():
                    raise UsageError(f"--from-archive not found: {p}")
            if args.dataset == "nic" and args.format == "csv":
                raise UsageError("NIC downloads already are CSV; use --format native (files are "
                                 "extracted and documented)")
            return
        ds = args.dataset
        if ds in ("panel", "filers-since", "facsimile"):
            if not args.period:
                raise UsageError(f"ffiec api --dataset {ds} needs --period")
            args.period = _period(args.period)
        if ds == "filers-since":
            if not args.since:
                raise UsageError("filers-since needs --since")
            args.since = _period(args.since)
        if ds == "facsimile":
            ids = []
            for v in args.rssd or []:
                ids += [x.strip() for x in v.split(",") if x.strip()]
            if not ids or any(not x.isdigit() for x in ids):
                raise UsageError("facsimile needs --rssd with numeric ID_RSSD values")
            args.rssd = ids
            if args.format == "csv":
                raise UsageError("facsimiles are saved in the provider format (XBRL/SDF/PDF); "
                                 "--format csv is not supported for them")
        if ds != "reporting-periods" and args.series != "Call":
            raise UsageError("--series UBPR is only supported for reporting-periods")

    def snapshot_label(self, mode, args):
        if mode == "archive":
            stem = args.from_archive[0].name.split(".")[0]
            return re.sub(r"[^A-Za-z0-9._-]+", "_", stem)[:80] or utc_stamp()
        if getattr(args, "period", None):
            return datetime.strptime(args.period, "%m/%d/%Y").strftime("%Y%m%d") + "-" + utc_stamp()
        return utc_stamp()

    def _headers(self, a, extra: dict | None = None) -> dict:
        h = {"dataSeries": a.series}
        if a.period:
            h["reportingPeriodEndDate"] = a.period
        return {**h, **(extra or {})}

    def plan(self, ctx: Context) -> Plan:
        a = ctx.args
        p = Plan(self.name, a.dataset, ctx.manifest.mode, ctx.root.name)
        if ctx.manifest.mode == "archive":
            for f in a.from_archive:
                p.files.append(PlannedFile(str(f), f"native/{f.name}", size=f.stat().st_size,
                                           description="user-downloaded archive (local copy)"))
            return p
        method = {"reporting-periods": "RetrieveReportingPeriods" if a.series == "Call"
                  else "RetrieveUBPRReportingPeriods",
                  "panel": "RetrievePanelOfReporters",
                  "filers-since": "RetrieveFilersSinceDate",
                  "facsimile": "RetrieveFacsimile"}[a.dataset]
        n = len(a.rssd or []) or 1
        if a.limit and a.dataset == "facsimile":
            n = min(n, a.limit)
        p.requests.append(f"GET {API}/{method} x{n} (headers: UserID, Authentication=***, "
                          f"dataSeries={a.series}"
                          + (f", reportingPeriodEndDate={a.period}" if a.period else "") + ")")
        return p

    def run(self, ctx: Context, plan: Plan) -> None:
        a = ctx.args
        if ctx.manifest.mode == "archive":
            ctx.manifest.snapshot_kind = "full"
            for f in a.from_archive:
                if a.dataset == "cdr-call-bulk":
                    files = ingest_archive(ctx, f, delimiter="\t", quoting=csv.QUOTE_MINIMAL,
                                           description="CDR Call Reports single-period bulk zip")
                    if ctx.want_csv:
                        self._call_bulk_csv(ctx, files)
                else:
                    ingest_archive(ctx, f, delimiter=",", quoting=csv.QUOTE_MINIMAL,
                                   description="Federal Reserve NIC download")
            return
        ctx.manifest.snapshot_kind = "query"
        if a.dataset == "facsimile":
            return self._facsimiles(ctx)
        method = plan.requests[0].split()[1].rsplit("/", 1)[1]
        extra = {"lastUpdateDateTime": a.since} if a.dataset == "filers-since" else None
        resp = ctx.client.request("GET", f"{API}/{method}", headers=self._headers(a, extra))
        data = resp.json()
        native = ctx.save_json(f"native/{method}.json", data, url=f"{API}/{method}",
                               records=len(data) if isinstance(data, list) else None)
        if ctx.want_csv and isinstance(data, list):
            if data and isinstance(data[0], dict):
                cols = list(dict.fromkeys(PANEL_COLUMNS + [k for r in data for k in r]))
                t = CsvTable(ctx.path("normalized/institutions.csv"), cols)
                padded = 0
                for r in data:
                    row = {k: r.get(k) for k in cols}
                    for k, width in (("ZIP", 5), ("FilingType", 3)):
                        if isinstance(row.get(k), int):
                            row[k] = str(row[k]).zfill(width)
                            padded += 1
                    row = {k: (str(v) if isinstance(v, int) and not isinstance(v, bool) else v)
                           for k, v in row.items()}
                    t.write(row)
            else:
                t = CsvTable(ctx.path("normalized/values.csv"), ["value"])
                padded = 0
                for v in data:
                    t.write({"value": v})
            t.close()
            ctx.register_tables({t.path.stem: t}, [native])
            ctx.manifest.transformations.append(
                "JSON list -> CSV; identifiers written as text" +
                (f"; {padded} integer ZIP/FilingType values left-padded (ZIP 5, FilingType 3 "
                 f"digits) because the API returns them as numbers" if padded else ""))

    def _facsimiles(self, ctx: Context) -> None:
        a = ctx.args
        ext = FACSIMILE_EXT[a.facsimile_format]
        done = 0
        for rssd in a.rssd:
            if a.limit and done >= a.limit:
                ctx.manifest.truncate(f"--limit {a.limit} facsimiles of {len(a.rssd)}")
                break
            resp = ctx.client.request("GET", f"{API}/RetrieveFacsimile", headers=self._headers(
                a, {"fiIdType": "ID_RSSD", "fiId": rssd, "facsimileFormat": a.facsimile_format}))
            raw = resp.content
            body = _decode_facsimile(raw)
            if body is None:
                path = ctx.path(f"native/{rssd}.response.json")
                ctx.manifest.warn(f"facsimile for {rssd}: response shape not recognised; saved "
                                  f"raw response")
            else:
                path = ctx.path(f"native/{rssd}.{ext}")
                raw = body
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.name + ".part")
            tmp.write_bytes(raw)
            os.replace(tmp, path)
            ctx.manifest.add_file(ctx.root, path, role="native", format=ext.split(".")[-1],
                                  source_url=f"{API}/RetrieveFacsimile",
                                  description=f"Call Report {a.period} ID_RSSD {rssd}")
            done += 1

    def _call_bulk_csv(self, ctx: Context, files: list[Path]) -> None:
        """Each schedule file: row 1 = MDRM codes, row 2 = descriptions, then data."""
        dictionary = CsvTable(ctx.path("normalized/field_dictionary.csv"),
                              ["file", "column_index", "mdrm_code", "description"])
        sources = []
        for f in files:
            if not f.name.lower().endswith(".txt") or "readme" in f.name.lower():
                continue
            rows = iter_rows(f, "\t", quoting=csv.QUOTE_MINIMAL)
            header = next(rows, None)
            if not header:
                continue
            header = [h.strip() for h in header]
            if header and header[-1] == "":
                header = header[:-1]
            second = next(rows, None)
            desc_row = second if second and not (second[0].strip().isdigit()) else None
            for i, code in enumerate(header):
                dictionary.write({"file": f.name, "column_index": i + 1, "mdrm_code": code,
                                  "description": (desc_row[i].strip() if desc_row and
                                                  i < len(desc_row) else None)})
            out = CsvTable(ctx.path(f"normalized/{f.stem}.csv"), header)
            pending = [] if desc_row else ([second] if second else [])
            for r in pending + list(_rest(rows)):
                if len(r) == len(header) + 1 and r[-1] == "":
                    r = r[:-1]
                if not r or r == [""]:
                    continue
                if len(r) != len(header):
                    out.abort()
                    raise ValueError(f"{f.name}: row with {len(r)} fields, header has "
                                     f"{len(header)}; refusing a misaligned CSV")
                out.write(dict(zip(header, r, strict=True)))
            out.close()
            ctx.manifest.add_file(ctx.root, out.path, role="normalized", format="csv",
                                  delimiter="comma", rows=out.rows, derived_from=[f])
            sources.append(f)
        dictionary.close()
        ctx.manifest.add_file(ctx.root, dictionary.path, role="dictionary", format="csv",
                              delimiter="comma", rows=dictionary.rows, derived_from=sources)
        ctx.manifest.transformations.append(
            "CDR schedule files: MDRM code row -> CSV header, description row -> "
            "field_dictionary.csv; values (incl. IDRSSD) copied as text")


def _rest(it):
    yield from it


def _decode_facsimile(raw: bytes) -> bytes | None:
    """The spec says the JSON body is 'a byte stream array'; accept a base64
    string, a list of byte values, or a one-element list holding either."""
    try:
        data = json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        return raw if raw else None          # already the file itself
    if isinstance(data, list) and len(data) == 1 and isinstance(data[0], (str, list)):
        data = data[0]
    if isinstance(data, str):
        try:
            return base64.b64decode(data, validate=True)
        except ValueError:
            return data.encode("utf-8")
    if isinstance(data, list) and all(isinstance(x, int) and 0 <= x < 256 for x in data):
        return bytes(data)
    return None
