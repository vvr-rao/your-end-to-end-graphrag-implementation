"""SEC EDGAR: XBRL Company Facts / submissions (API) and the Financial
Statement (and Notes) Data Sets (bulk).

Verified 2026-11-08 against
https://www.sec.gov/search-filings/edgar-application-programming-interfaces,
https://www.sec.gov/os/accessing-edgar-data and
https://www.sec.gov/dera/data/financial-statement-data-sets.html
(2024q1.zip inspected by range request: sub.txt, num.txt, tag.txt, pre.txt,
readme.htm).
"""

from __future__ import annotations

import argparse
import csv
import re
from datetime import date
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
from .._tabular import CsvTable

DATA = "https://data.sec.gov"
FILES = "https://www.sec.gov/files/dera/data"
FSDS_URL = FILES + "/financial-statement-data-sets/{period}.zip"
NOTES_URL = FILES + "/financial-statement-notes-data-sets/{period}_notes.zip"
NIGHTLY = {
    "companyfacts-zip": "https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip",
    "submissions-zip": "https://www.sec.gov/Archives/edgar/daily-index/bulkdata/submissions.zip",
}
FACT_COLUMNS = ["cik", "entity_name", "taxonomy", "tag", "unit", "period_type", "start", "end",
                "duration_days", "val", "accn", "fy", "fp", "form", "filed", "frame"]


def _cik(value: str) -> str:
    digits = value.strip().upper().removeprefix("CIK")
    if not digits.isdigit() or len(digits) > 10:
        raise UsageError(f"invalid CIK {value!r}: expected up to 10 digits")
    return digits.zfill(10)


def _split(values: list[str] | None) -> list[str]:
    out: list[str] = []
    for v in values or []:
        out.extend(x for x in (s.strip() for s in v.split(",")) if x)
    return out


@register
class SecSource(Source):
    name = "sec"
    title = "SEC EDGAR"
    domain = "finance"
    summary = ("XBRL Company Facts and submissions for selected CIKs (API), and the quarterly "
               "Financial Statement Data Sets / monthly Notes data sets (bulk).")
    modes = {
        "api": ModeSpec(
            "data.sec.gov JSON for selected CIKs",
            {"companyfacts": "all XBRL facts per company (original JSON; csv = one row per "
                             "reported fact, nothing collapsed)",
             "submissions": "filing history per company (original JSON; csv = filings table)"},
            formats=("native", "csv")),
        "bulk": ModeSpec(
            "official bulk archives (whole files; --limit does not apply)",
            {"fsds": "Financial Statement Data Sets, one quarter (e.g. 2024q1): SUB/NUM/TAG/PRE "
                     "tab-delimited tables, kept as-is",
             "notes": "Financial Statement and Notes data sets, one month (2026_10) or quarter "
                      "(2025q3) -- large (60-450 MB)",
             "companyfacts-zip": "nightly companyfacts.zip, every company (multi-GB)",
             "submissions-zip": "nightly submissions.zip, every company (multi-GB)"},
            formats=("native", "csv"), bounded_by_limit=False),
    }
    env = [EnvVar("SEC_USER_AGENT", "all modes",
                  'identifying User-Agent, e.g. "Your Org yourname@your-domain.com" '
                  "(SEC fair-access rules)")]
    docs_urls = ["https://www.sec.gov/search-filings/edgar-application-programming-interfaces",
                 "https://www.sec.gov/dera/data/financial-statement-data-sets.html"]
    terms_urls = ["https://www.sec.gov/os/accessing-edgar-data",
                  "https://www.sec.gov/privacy#dissemination"]
    attribution = "U.S. Securities and Exchange Commission, EDGAR"
    data_dictionaries = ["https://www.sec.gov/files/financial-statement-data-sets.pdf",
                         "https://www.sec.gov/files/aqfsn_1.pdf"]
    rate_per_second = 8.0          # SEC maximum is 10 requests/second
    notes = [
        "companyfacts CSV keeps every fact entry, including amendments and the same value "
        "reported in several filings; year-to-date durations are NOT converted to quarters "
        "(use start/end/duration_days and fp to interpret).",
    ]

    def add_arguments(self, mode: str, parser: argparse.ArgumentParser) -> None:
        if mode == "api":
            parser.add_argument("--cik", action="append", required=True,
                                help="CIK (repeat or comma-separate), e.g. 320193")
        else:
            parser.add_argument("--period",
                                help="fsds: YYYYqN (2009q1+); notes: YYYY_MM or YYYYqN")

    def credentials(self, mode, args):
        ua = require_env("SEC_USER_AGENT", "SEC EDGAR (identifying User-Agent)")
        if "example.com" in ua.lower() or "@" not in ua:
            raise CredentialError(
                "SEC_USER_AGENT must name your organisation and a real contact email, "
                'e.g. "Acme Research jane@acme.org" (SEC blocks generic agents).')
        return {"user_agent": ua}

    def validate(self, mode, args):
        if mode == "api":
            args.cik = [_cik(c) for c in _split(args.cik)]
            return
        ds = args.dataset
        if ds in ("fsds", "notes") and not args.period:
            raise UsageError(f"sec bulk --dataset {ds} needs --period")
        if ds == "fsds":
            m = re.fullmatch(r"(\d{4})q([1-4])", args.period or "")
            if not m or int(m.group(1)) < 2009:
                raise UsageError("--period for fsds must look like 2024q1 (2009q1 or later)")
        if ds == "notes" and not re.fullmatch(r"\d{4}(q[1-4]|_\d{2})", args.period or ""):
            raise UsageError("--period for notes must look like 2026_10 or 2025q3")
        if ds in NIGHTLY and args.period:
            raise UsageError(f"{ds} is the latest nightly file; --period does not apply")
        if ds in NIGHTLY and args.format == "csv":
            raise UsageError(f"{ds} contains per-company JSON; --format csv is not supported "
                             f"for it (use api mode for normalized facts)")

    def snapshot_label(self, mode, args):
        if mode == "bulk":
            return args.period or date.today().strftime("%Y%m%d")
        return utc_stamp()

    # --------------------------------------------------------------- plan

    def plan(self, ctx: Context) -> Plan:
        a = ctx.args
        p = Plan(self.name, a.dataset, ctx.manifest.mode, ctx.root.name)
        if ctx.manifest.mode == "api":
            for cik in a.cik:
                p.requests.append(self._api_url(a.dataset, cik))
            if a.limit:
                p.notes.append("--limit caps rows written to the normalized CSV per company; "
                               "the original JSON is always complete.")
            return p
        url = (FSDS_URL.format(period=a.period) if a.dataset == "fsds"
               else NOTES_URL.format(period=a.period) if a.dataset == "notes"
               else NIGHTLY[a.dataset])
        head = ctx.client.head(url)
        size = int(head.headers["Content-Length"]) if head is not None and \
            head.headers.get("Content-Length", "").isdigit() else None
        p.files.append(PlannedFile(url, f"native/{url.rsplit('/', 1)[1]}", expect="zip",
                                   size=size, description=f"SEC {a.dataset} archive"))
        p.release = a.period or (head.headers.get("Last-Modified") if head is not None else None)
        p.notes.append("bulk archives download whole; there is no row limit for them.")
        return p

    @staticmethod
    def _api_url(dataset: str, cik: str) -> str:
        if dataset == "companyfacts":
            return f"{DATA}/api/xbrl/companyfacts/CIK{cik}.json"
        return f"{DATA}/submissions/CIK{cik}.json"

    # ---------------------------------------------------------------- run

    def run(self, ctx: Context, plan: Plan) -> None:
        if ctx.manifest.mode == "bulk":
            return self._run_bulk(ctx, plan)
        ctx.manifest.snapshot_kind = "query"
        natives: list[Path] = []
        for cik in ctx.args.cik:
            url = self._api_url(ctx.args.dataset, cik)
            payload = ctx.client.get_json(url)
            natives.append(ctx.save_json(f"native/CIK{cik}.json", payload, url=url))
            if ctx.args.dataset == "submissions":
                for extra in (payload.get("filings") or {}).get("files") or []:
                    name = extra.get("name")
                    if name:
                        u = f"{DATA}/submissions/{name}"
                        natives.append(ctx.save_json(f"native/{name}", ctx.client.get_json(u),
                                                     url=u, description="older filings page"))
        if ctx.want_csv:
            if ctx.args.dataset == "companyfacts":
                self._facts_csv(ctx, natives)
            else:
                self._filings_csv(ctx, natives)

    def _facts_csv(self, ctx: Context, natives: list[Path]) -> None:
        import json
        facts = CsvTable(ctx.path("normalized/facts.csv"), FACT_COLUMNS)
        concepts = CsvTable(ctx.path("normalized/concepts.csv"),
                            ["taxonomy", "tag", "label", "description"])
        seen_concepts: set[tuple[str, str]] = set()
        limit = ctx.args.limit
        try:
            for path in natives:
                data = json.loads(path.read_text(encoding="utf-8"))
                cik = str(data.get("cik", "")).zfill(10)
                name = data.get("entityName", "")
                written = 0
                capped = False
                for taxonomy, tags in (data.get("facts") or {}).items():
                    for tag, concept in tags.items():
                        if (taxonomy, tag) not in seen_concepts:
                            seen_concepts.add((taxonomy, tag))
                            concepts.write({"taxonomy": taxonomy, "tag": tag,
                                            "label": concept.get("label"),
                                            "description": concept.get("description")})
                        for unit, entries in (concept.get("units") or {}).items():
                            for e in entries:
                                if limit is not None and written >= limit:
                                    capped = True
                                    break
                                facts.write(_fact_row(cik, name, taxonomy, tag, unit, e))
                                written += 1
                if capped:
                    ctx.manifest.truncate(f"facts.csv capped at --limit {limit} rows for CIK "
                                          f"{cik} (native JSON complete)")
        except BaseException:
            facts.abort()
            concepts.abort()
            raise
        facts.close()
        concepts.close()
        ctx.register_tables({"facts": facts, "concepts": concepts}, natives,
                            description="SEC XBRL facts, one row per reported fact")
        ctx.manifest.transformations += [
            "companyfacts JSON -> facts.csv: one row per (taxonomy, tag, unit, fact entry); "
            "no deduplication -- amended and repeated observations are all kept",
            "period_type = instant when no start date, else duration; duration_days = end - "
            "start + 1. Year-to-date values are not converted into quarterly values.",
            "CIK zero-padded to 10 digits; accession numbers kept verbatim",
        ]

    def _filings_csv(self, ctx: Context, natives: list[Path]) -> None:
        import json
        rows_out: list[dict] = []
        columns: dict[str, None] = {"cik": None}
        for path in natives:
            data = json.loads(path.read_text(encoding="utf-8"))
            block = (data.get("filings") or {}).get("recent") if "filings" in data else data
            cik = str(data.get("cik") or path.stem.split("-")[0].removeprefix("CIK")).zfill(10)
            if not isinstance(block, dict):
                continue
            keys = [k for k, v in block.items() if isinstance(v, list)]
            n = max((len(block[k]) for k in keys), default=0)
            for k in keys:
                columns.setdefault(k, None)
            for i in range(n):
                rows_out.append({"cik": cik, **{k: (block[k][i] if i < len(block[k]) else None)
                                                for k in keys}})
        limit = ctx.args.limit
        if limit is not None and len(rows_out) > limit:
            rows_out = rows_out[:limit]
            ctx.manifest.truncate(f"filings.csv capped at --limit {limit} rows")
        t = CsvTable(ctx.path("normalized/filings.csv"), list(columns))
        for r in rows_out:
            t.write(r)
        t.close()
        ctx.register_tables({"filings": t}, natives, description="one row per filing")
        ctx.manifest.transformations.append(
            "submissions filings (parallel arrays) -> filings.csv, one row per accession")

    def _run_bulk(self, ctx: Context, plan: Plan) -> None:
        ctx.manifest.snapshot_kind = "full"
        pf = plan.files[0]
        archive = ctx.fetch(pf, format="zip")
        if ctx.args.dataset in NIGHTLY:
            ctx.manifest.warn("nightly SEC file is overwritten daily by the SEC; the snapshot "
                              "folder is named by download date")
            return
        # The SEC readme says only "tab-delimited", but values that contain a tab
        # (e.g. NUM.segments "InvestmentIdentifier=...\t...") are double-quoted, so
        # parse with standard quoting: QUOTE_NONE would split them into extra fields.
        # Checked on 2024q1: every SUB/NUM/TAG/PRE row parses, no embedded newlines.
        ingest_archive(ctx, archive, delimiter="\t", table_suffixes=(".txt", ".tsv"),
                       quoting=csv.QUOTE_MINIMAL, convert=ctx.want_csv,
                       description=f"SEC {ctx.args.dataset} archive")
        ctx.manifest.transformations.append(
            "SUB/NUM/TAG/PRE (and notes) tables kept in their original tab-delimited form; "
            "parsed with standard double-quote rules (SEC quotes values containing a tab)")


def _fact_row(cik: str, name: str, taxonomy: str, tag: str, unit: str, e: dict) -> dict:
    start, end = e.get("start"), e.get("end")
    duration = None
    if start and end:
        try:
            duration = (date.fromisoformat(end) - date.fromisoformat(start)).days + 1
        except ValueError:
            duration = None
    return {"cik": cik, "entity_name": name, "taxonomy": taxonomy, "tag": tag, "unit": unit,
            "period_type": "duration" if start else "instant", "start": start, "end": end,
            "duration_days": duration, "val": e.get("val"), "accn": e.get("accn"),
            "fy": e.get("fy"), "fp": e.get("fp"), "form": e.get("form"),
            "filed": e.get("filed"), "frame": e.get("frame")}
