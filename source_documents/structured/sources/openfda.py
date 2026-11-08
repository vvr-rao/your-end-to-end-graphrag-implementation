"""Shared openFDA + FDA flat-file logic for Drugs@FDA and the Orange Book.

Verified 2026-11-08: https://open.fda.gov/apis/query-parameters/ (limit max
1000), https://open.fda.gov/apis/paging/ (``Link: <...>; rel="next"`` with
search_after, observed on drug/drugsfda.json and drug/orangebook.json),
https://open.fda.gov/apis/authentication/ (api_key; 240 req/min, 1,000/day
without a key), https://api.fda.gov/download.json (results.drug.drugsfda /
results.drug.orangebook: export_date, partitions[{file, size_mb, records}]).
FDA flat files: Drugs@FDA zip (tab-delimited, 12 tables) and Orange Book zip
(tilde-delimited Products/Patent/Exclusivity).
"""

from __future__ import annotations

import argparse
import os
import re
from datetime import date
from email.utils import parsedate_to_datetime
from pathlib import Path

from .._base import (
    Context,
    EnvVar,
    ModeSpec,
    Plan,
    PlannedFile,
    Source,
    UsageError,
    ingest_archive,
    iter_json_records,
)
from .._manifest import utc_stamp
from .._tabular import JsonFlattener

API = "https://api.fda.gov"
MANIFEST_URL = f"{API}/download.json"
PAGE_MAX = 1000


def _next_link(link_header: str | None) -> str | None:
    if not link_header:
        return None
    for part in link_header.split(","):
        m = re.match(r'\s*<([^>]+)>\s*;\s*rel="?next"?', part.strip(), re.IGNORECASE)
        if m:
            return m.group(1)
    return None


class OpenFdaSource(Source):
    """Base for the two FDA drug sources; subclasses fill in the constants."""

    endpoint = ""                  # e.g. "drug/drugsfda"
    flatfile_url = ""
    flatfile_delimiter = "\t"
    record_key: tuple[str, ...] = ()  # fields copied into every child row
    rate_per_second = 3.0           # 240/min documented
    env = [EnvVar("OPENFDA_API_KEY", "optional",
                  "raises the daily limit from 1,000 to 120,000 requests "
                  "(https://open.fda.gov/apis/authentication/)", required=False)]
    terms_urls = ["https://open.fda.gov/terms/", "https://open.fda.gov/license/"]

    @staticmethod
    def build_modes(endpoint: str, flat_desc: str) -> dict[str, ModeSpec]:
        return {
            "api": ModeSpec(f"openFDA {endpoint}.json (search / paging)",
                            {"records": "records matching --search (original JSON pages; "
                                        "csv = related tables, identifiers kept as text)"},
                            formats=("native", "csv")),
            "bulk": ModeSpec("official snapshots (whole files; --limit does not apply)",
                             {"flatfiles": flat_desc,
                              "openfda-json": "openFDA bulk JSON partitions listed in "
                                              "download.json"},
                             formats=("native", "csv"), bounded_by_limit=False),
        }

    def add_arguments(self, mode: str, parser: argparse.ArgumentParser) -> None:
        if mode == "api":
            parser.add_argument("--search",
                                help='openFDA search expression, e.g. '
                                     '\'openfda.brand_name:"OZEMPIC"\'')
            parser.add_argument("--sort", help="e.g. application_number:asc")

    def credentials(self, mode, args):
        key = os.environ.get("OPENFDA_API_KEY", "").strip()
        return {"params": {"api_key": key}} if key and mode == "api" else {}

    def validate(self, mode, args):
        if mode == "bulk" and args.dataset == "openfda-json" and args.format == "csv":
            raise UsageError("openfda-json bulk partitions are kept as provider JSON; use api "
                             "mode for normalized CSV, or --dataset flatfiles --format csv")

    def snapshot_label(self, mode, args):
        return utc_stamp() if mode == "api" else "pending"

    # ---------------------------------------------------------------- plan

    def plan(self, ctx: Context) -> Plan:
        a = ctx.args
        mode = ctx.manifest.mode
        p = Plan(self.name, a.dataset, mode, ctx.root.name)
        if mode == "api":
            size = ctx.page_size(PAGE_MAX)
            q = f"{API}/{self.endpoint}.json?limit={size}"
            if a.search:
                q += f"&search={a.search}"
            if a.sort:
                q += f"&sort={a.sort}"
            p.requests.append(q + "  (then follows Link rel=next)")
            if not os.environ.get("OPENFDA_API_KEY"):
                p.notes.append("no OPENFDA_API_KEY: limited to 1,000 requests/day per IP")
            return p
        if a.dataset == "openfda-json":
            listing = ctx.client.get_json(MANIFEST_URL)
            cat, ds = self.endpoint.split("/")
            entry = ((listing.get("results") or {}).get(cat) or {}).get(ds) or {}
            p.release = entry.get("export_date")
            for part in entry.get("partitions") or []:
                url = part["file"]
                size = float(part.get("size_mb") or 0) * 1e6 or None
                p.files.append(PlannedFile(url, f"native/{url.rsplit('/', 1)[1]}", expect="zip",
                                           size=int(size) if size else None,
                                           description=f"{part.get('records')} records"))
            if not p.files:
                raise UsageError(f"download.json lists no partitions for {self.endpoint}")
        else:
            head = ctx.client.head(self.flatfile_url)
            size = None
            if head is not None:
                cl = head.headers.get("Content-Length", "")
                size = int(cl) if cl.isdigit() else None
                lm = head.headers.get("Last-Modified")
                if lm:
                    try:
                        p.release = parsedate_to_datetime(lm).date().isoformat()
                    except (TypeError, ValueError):
                        pass
            p.files.append(PlannedFile(self.flatfile_url, f"native/{self.name}_flatfiles.zip",
                                       expect="zip", size=size,
                                       description="FDA flat-file archive"))
        p.snapshot = (p.release or date.today().isoformat()).replace("-", "")
        p.notes.append("bulk files are full snapshots (no incremental updates); "
                       "--limit does not shrink them.")
        return p

    # ----------------------------------------------------------------- run

    def run(self, ctx: Context, plan: Plan) -> None:
        if ctx.manifest.mode == "api":
            return self._run_api(ctx)
        ctx.manifest.snapshot_kind = "full"
        if plan.release:
            ctx.manifest.source_release = plan.release
        for pf in plan.files:
            archive = ctx.fetch(pf, format="zip")
            if ctx.args.dataset == "flatfiles":
                ingest_archive(ctx, archive, delimiter=self.flatfile_delimiter,
                               convert=ctx.want_csv, description=pf.description)
                ctx.manifest.transformations.append(
                    "identifiers (application / product numbers) copied as text, so leading "
                    "zeros are preserved")
            else:
                ingest_archive(ctx, archive, delimiter=None, description=pf.description)

    def _run_api(self, ctx: Context) -> None:
        a = ctx.args
        ctx.manifest.snapshot_kind = "query"
        url: str | None = f"{API}/{self.endpoint}.json"
        params: dict | None = {"limit": ctx.page_size(PAGE_MAX)}
        if a.search:
            params["search"] = a.search
        if a.sort:
            params["sort"] = a.sort
        pages: list[Path] = []
        records = 0
        total = None
        while url:
            reason = ctx.cap(records, len(pages))
            if reason:
                ctx.manifest.truncate(reason)
                break
            if a.limit:
                params = {**(params or {}), "limit": max(1, min(PAGE_MAX, a.limit - records))} \
                    if params is not None else None
            resp = ctx.client.request("GET", url, params=params,
                                      headers={"Accept": "application/json"},
                                      ok_statuses=(200, 404))
            if resp.status_code == 404:          # openFDA: "No matches found!"
                ctx.manifest.warn("openFDA returned no matches for this search")
                break
            data = resp.json()
            results = data.get("results") or []
            meta = (data.get("meta") or {})
            total = (meta.get("results") or {}).get("total", total)
            if meta.get("last_updated"):
                ctx.manifest.source_release = meta["last_updated"]
            pages.append(ctx.save_json(f"native/page-{len(pages) + 1:04d}.json", data,
                                       url=url, records=len(results)))
            records += len(results)
            url = _next_link(resp.headers.get("Link"))
            params = None                         # the next link carries every parameter
            if url and a.limit and records >= a.limit:
                ctx.manifest.truncate(f"--limit {a.limit} reached ({records} of {total})")
                break
        ctx.manifest.pagination = {"pages_fetched": len(pages), "records": records,
                                   "total_reported": total,
                                   "complete": not ctx.manifest.truncated}
        if ctx.want_csv and pages:
            keyf = lambda r, i: {"record_ordinal": i, **{k: r.get(k) for k in self.record_key}}  # noqa: E731
            flat = JsonFlattener(self.name, keyf)
            tables = flat.write(lambda: iter_json_records(pages, "results"),
                                ctx.path("normalized"))
            ctx.register_tables(tables, pages)
            ctx.manifest.transformations.append(
                f"openFDA JSON -> related CSV tables (child tables keyed by "
                f"{', '.join(self.record_key) or 'record'} + ordinals; nested arrays never "
                f"cross-joined; all values text)")
            if flat.json_columns:
                ctx.manifest.transformations.append(
                    "kept as JSON text (nesting deeper than 4 levels): "
                    + ", ".join(sorted(flat.json_columns)))
