"""GLEIF: LEI records, relationship records, reporting exceptions (API) and
the Golden Copy / delta files (bulk).

Verified 2026-11-08 by live requests (the Postman docs page renders client-
side only): https://api.gleif.org/api/v1/lei-records with filter[...] and
page[size]/page[number] (meta.pagination.lastPage/total, links.next);
per-record links /direct-parent-relationship, /ultimate-parent-relationship,
/direct-child-relationships, /ultimate-child-relationships,
/direct-parent-reporting-exception, /ultimate-parent-reporting-exception.
Golden Copy: https://goldencopy.gleif.org/api/v2/golden-copies/publishes
(data[].{lei2,rr,repex}.full_file.{csv,json,xml} and
.delta_files.{IntraDay,LastDay,LastWeek,LastMonth}.{csv,json,xml} with url,
size, record_count, publish_date).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from .._base import (
    Context,
    ModeSpec,
    Plan,
    PlannedFile,
    Source,
    UsageError,
    ingest_archive,
    iter_json_records,
    register,
)
from .._http import DownloadError
from .._manifest import utc_stamp
from .._tabular import CsvTable, JsonFlattener

API = "https://api.gleif.org/api/v1"
GC_API = "https://goldencopy.gleif.org/api/v2/golden-copies/publishes"
PAGE_MAX = 200
DELTAS = ("IntraDay", "LastDay", "LastWeek", "LastMonth")
REL_COLUMNS = ["relationship_id", "query", "start_lei", "end_lei", "relationship_type",
               "relationship_status", "valid_from", "valid_to", "registration_status",
               "corroboration_level", "initial_registration_date", "last_update_date",
               "managing_lou"]
PERIOD_COLUMNS = ["relationship_id", "period_ordinal", "period_type", "start_date", "end_date"]
EXC_COLUMNS = ["exception_id", "lei", "category", "reason", "reference", "valid_from",
               "valid_to"]
_LEI = re.compile(r"[A-Z0-9]{18}[0-9]{2}")


@register
class GleifSource(Source):
    name = "gleif"
    title = "GLEIF (Legal Entity Identifiers)"
    domain = "finance"
    summary = ("Legal entities (LEI records), parent/child relationship records and reporting "
               "exceptions via the GLEIF API; full Golden Copy or delta files in bulk.")
    modes = {
        "api": ModeSpec("GLEIF API v1 (JSON:API)",
                        {"lei-records": "entities by --lei / --name / --country / --fulltext, "
                                        "optionally with --relationships"},
                        formats=("native", "csv")),
        "bulk": ModeSpec("Golden Copy files (latest publish; whole files)",
                         {"lei2": "Level 1 LEI records (who is who), ~500 MB csv zip",
                          "rr": "Level 2 relationship records (who owns whom)",
                          "repex": "Level 2 reporting exceptions"},
                         formats=("native",), bounded_by_limit=False),
    }
    docs_urls = ["https://api.gleif.org/docs",
                 "https://www.gleif.org/en/lei-data/gleif-golden-copy/download-the-golden-copy"]
    terms_urls = ["https://www.gleif.org/en/meta/lei-data-terms-of-use"]
    attribution = "Global Legal Entity Identifier Foundation (GLEIF), CC0 data"
    data_dictionaries = [
        "https://www.gleif.org/en/about-lei/common-data-file-format",
        "https://www.gleif.org/lei-data/gleif-golden-copy/"
        "2022-02-23_gleif-golden-copy-and-delta-files_v2.2-final.pdf"]
    rate_per_second = 1.0          # GLEIF asks for moderate use (60 requests/minute)
    notes = ["Relationship records and reporting exceptions are written to separate tables; "
             "an exception (e.g. NATURAL_PERSONS) is not a relationship.",
             "Bulk: --delta full (default) is a complete snapshot; IntraDay/LastDay/LastWeek/"
             "LastMonth are change sets relative to an earlier publish."]

    def add_arguments(self, mode: str, parser: argparse.ArgumentParser) -> None:
        if mode == "api":
            parser.add_argument("--lei", action="append", help="LEI (repeat or comma-separate)")
            parser.add_argument("--name", help="filter[entity.legalName]")
            parser.add_argument("--country", help="filter[entity.legalAddress.country] (ISO 2)")
            parser.add_argument("--fulltext", help="filter[fulltext]")
            parser.add_argument("--relationships", action="store_true",
                                help="also fetch direct/ultimate parent relationship records, "
                                     "direct children and reporting exceptions per entity")
        else:
            parser.add_argument("--delta", choices=("full", *DELTAS), default="full")
            parser.add_argument("--native-type", choices=("csv", "json", "xml"), default="csv",
                                help="Golden Copy file format (default csv)")

    def validate(self, mode, args):
        if mode != "api":
            return
        leis = []
        for v in args.lei or []:
            leis += [x.strip().upper() for x in v.split(",") if x.strip()]
        bad = [x for x in leis if not _LEI.fullmatch(x)]
        if bad:
            raise UsageError(f"invalid LEI(s): {', '.join(bad)}")
        args.lei = leis
        if not (leis or args.name or args.country or args.fulltext):
            raise UsageError("give --lei, --name, --country or --fulltext (unfiltered "
                             "listing of all LEIs is a bulk job: use `gleif bulk`)")

    def snapshot_label(self, mode, args):
        return utc_stamp() if mode == "api" else "pending"

    # ---------------------------------------------------------------- plan

    def _filters(self, a) -> dict:
        f = {}
        if a.lei:
            f["filter[lei]"] = ",".join(a.lei)
        if a.name:
            f["filter[entity.legalName]"] = a.name
        if a.country:
            f["filter[entity.legalAddress.country]"] = a.country.upper()
        if a.fulltext:
            f["filter[fulltext]"] = a.fulltext
        return f

    def plan(self, ctx: Context) -> Plan:
        a = ctx.args
        p = Plan(self.name, a.dataset, ctx.manifest.mode, ctx.root.name)
        if ctx.manifest.mode == "api":
            p.requests.append(f"{API}/lei-records " + " ".join(
                f"{k}={v}" for k, v in self._filters(a).items())
                + f" page[size]={ctx.page_size(PAGE_MAX)}")
            if a.relationships:
                p.requests.append("per entity: direct/ultimate parent relationship, direct "
                                  "children (paged), parent reporting exceptions "
                                  "(~5+ requests per entity, 1 request/second)")
            return p
        latest = ctx.client.get_json(GC_API, params={"per_page": 1, "page": 1})
        pub = (latest.get("data") or [{}])[0]
        block = pub.get(a.dataset) or {}
        if a.delta == "full":
            f = (block.get("full_file") or {}).get(a.native_type)
        else:
            f = ((block.get("delta_files") or {}).get(a.delta) or {}).get(a.native_type)
        if not f or not f.get("url"):
            raise UsageError(f"latest Golden Copy publish has no {a.dataset} {a.delta} "
                             f"{a.native_type} file")
        url = f["url"]
        p.release = f.get("publish_date") or pub.get("publish_date")
        stamp = re.sub(r"[^0-9]", "", p.release or "")[:12]
        p.snapshot = f"{stamp or utc_stamp()}-{a.delta}-{a.native_type}"
        p.files.append(PlannedFile(url, f"native/{url.rsplit('/', 1)[1]}", expect="zip",
                                   size=f.get("size"),
                                   description=f"{a.dataset} {f.get('delta_type')} "
                                               f"({f.get('record_count')} records, "
                                               f"{f.get('cdf_version')})"))
        p.notes.append(f"{f.get('record_count')} records; whole file is downloaded "
                       f"(--limit does not apply)")
        return p

    # ----------------------------------------------------------------- run

    def run(self, ctx: Context, plan: Plan) -> None:
        if ctx.manifest.mode == "bulk":
            ctx.manifest.snapshot_kind = "full" if ctx.args.delta == "full" else "delta"
            archive = ctx.fetch(plan.files[0], format="zip")
            delim = "," if ctx.args.native_type == "csv" else None
            import csv
            ingest_archive(ctx, archive, delimiter=delim, quoting=csv.QUOTE_MINIMAL,
                           description=plan.files[0].description)
            return
        a = ctx.args
        ctx.manifest.snapshot_kind = "query"
        pages: list[Path] = []
        records = 0
        params: dict | None = {**self._filters(a), "page[size]": ctx.page_size(PAGE_MAX),
                               "page[number]": 1}
        url: str | None = f"{API}/lei-records"
        total = None
        while url:
            reason = ctx.cap(records, len(pages))
            if reason:
                ctx.manifest.truncate(reason)
                break
            data = ctx.client.get_json(url, params=params)
            items = data.get("data") or []
            pg = (data.get("meta") or {}).get("pagination") or {}
            total = pg.get("total", total)
            gc = ((data.get("meta") or {}).get("goldenCopy") or {}).get("publishDate")
            if gc:
                ctx.manifest.source_release = gc
            if a.limit and records + len(items) > a.limit:
                items = items[: a.limit - records]
                data = {**data, "data": items}
                ctx.manifest.truncate(f"--limit {a.limit} reached (of {total})")
                url = None
            pages.append(ctx.save_json(f"native/lei-records-{len(pages) + 1:04d}.json", data,
                                       url=f"{API}/lei-records", records=len(items)))
            records += len(items)
            if url:
                url = (data.get("links") or {}).get("next")
                params = None
                if url and a.limit and records >= a.limit:
                    ctx.manifest.truncate(f"--limit {a.limit} reached (of {total})")
                    url = None
        ctx.manifest.pagination = {"pages_fetched": len(pages), "records": records,
                                   "total_reported": total,
                                   "complete": not ctx.manifest.truncated}
        leis = [r["id"] for r in iter_json_records(pages, "data")]
        rel_files: list[Path] = []
        if a.relationships:
            rel_files = self._relationships(ctx, leis)
        if ctx.want_csv:
            flat = JsonFlattener("entities", lambda r, i: {"lei": r.get("lei")},
                                 skip_fields=("lei",))
            tables = flat.write(
                lambda: ({**(rec.get("attributes") or {}), "lei": rec.get("id")}
                         for rec in iter_json_records(pages, "data")),
                ctx.path("normalized"))
            ctx.register_tables(tables, pages, description="GLEIF LEI records")
            ctx.manifest.transformations.append(
                "lei-records attributes -> entities.csv (+ child tables for repeated names / "
                "address lines), keyed by LEI")
            if rel_files:
                self._relationship_tables(ctx, rel_files)

    def _get_optional(self, ctx: Context, url: str, params=None):
        try:
            resp = ctx.client.request("GET", url, params=params,
                                      headers={"Accept": "application/vnd.api+json"},
                                      ok_statuses=(200, 404))
        except DownloadError as exc:
            ctx.manifest.warn(str(exc))
            return None
        if resp.status_code == 404:
            return None
        return resp.json()

    def _relationships(self, ctx: Context, leis: list[str]) -> list[Path]:
        files: list[Path] = []
        for lei in leis:
            base = f"{API}/lei-records/{lei}"
            for kind in ("direct-parent-relationship", "ultimate-parent-relationship",
                         "direct-parent-reporting-exception",
                         "ultimate-parent-reporting-exception"):
                data = self._get_optional(ctx, f"{base}/{kind}")
                if data and data.get("data"):
                    files.append(ctx.save_json(f"native/relationships/{lei}/{kind}.json", data,
                                               url=f"{base}/{kind}", records=1))
            page = 1
            while True:
                url = f"{base}/direct-child-relationships"
                data = self._get_optional(ctx, url, {"page[size]": PAGE_MAX,
                                                     "page[number]": page})
                if not data or not data.get("data"):
                    break
                files.append(ctx.save_json(
                    f"native/relationships/{lei}/direct-child-relationships-{page:04d}.json",
                    data, url=url, records=len(data["data"])))
                last = ((data.get("meta") or {}).get("pagination") or {}).get("lastPage", page)
                if page >= last:
                    break
                page += 1
        ctx.manifest.transformations.append(
            "relationship records fetched per entity: direct + ultimate parent, direct "
            "children; reporting exceptions fetched separately")
        return files

    def _relationship_tables(self, ctx: Context, files: list[Path]) -> None:
        import json
        rels = CsvTable(ctx.path("normalized/relationships.csv"), REL_COLUMNS)
        periods = CsvTable(ctx.path("normalized/relationship_periods.csv"), PERIOD_COLUMNS)
        excs = CsvTable(ctx.path("normalized/reporting_exceptions.csv"), EXC_COLUMNS)
        seen: set[str] = set()
        for path in files:
            payload = json.loads(path.read_text(encoding="utf-8"))
            items = payload["data"] if isinstance(payload["data"], list) else [payload["data"]]
            query = path.stem.rsplit("-", 1)[0] if path.stem[-4:].isdigit() else path.stem
            for item in items:
                attrs = item.get("attributes") or {}
                if item.get("type") == "reporting-exceptions":
                    excs.write({"exception_id": item.get("id"), "lei": attrs.get("lei"),
                                "category": attrs.get("category"), "reason": attrs.get("reason"),
                                "reference": attrs.get("reference"),
                                "valid_from": attrs.get("validFrom"),
                                "valid_to": attrs.get("validTo")})
                    continue
                rid = item.get("id")
                if rid in seen:          # same record reached from both ends of a query
                    continue
                seen.add(rid)
                rel = attrs.get("relationship") or {}
                reg = attrs.get("registration") or {}
                rels.write({"relationship_id": rid, "query": query,
                            "start_lei": (rel.get("startNode") or {}).get("id"),
                            "end_lei": (rel.get("endNode") or {}).get("id"),
                            "relationship_type": rel.get("type"),
                            "relationship_status": rel.get("status"),
                            "valid_from": attrs.get("validFrom"),
                            "valid_to": attrs.get("validTo"),
                            "registration_status": reg.get("status"),
                            "corroboration_level": reg.get("corroborationLevel"),
                            "initial_registration_date": reg.get("initialRegistrationDate"),
                            "last_update_date": reg.get("lastUpdateDate"),
                            "managing_lou": reg.get("managingLou")})
                for i, per in enumerate(rel.get("periods") or [], start=1):
                    periods.write({"relationship_id": rid, "period_ordinal": i,
                                   "period_type": per.get("type"),
                                   "start_date": per.get("startDate"),
                                   "end_date": per.get("endDate")})
        for t in (rels, periods, excs):
            t.close()
        ctx.register_tables({"relationships": rels, "relationship_periods": periods,
                             "reporting_exceptions": excs}, files)
        ctx.manifest.transformations.append(
            "relationship records -> relationships.csv (start_lei IS_..._BY end_lei, "
            "deduplicated by record id) + relationship_periods.csv; reporting exceptions -> "
            "reporting_exceptions.csv (kept apart from relationships)")
