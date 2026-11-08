"""ClinicalTrials.gov API v2 (studies) and AACT archives.

Verified 2026-11-08 by live requests (the API docs page renders client-
side): GET https://clinicaltrials.gov/api/v2/studies with query.cond /
query.term / query.intr / filter.overallStatus / filter.ids, pageSize
(capped at 1000), pageToken -> nextPageToken, countTotal=true -> totalCount;
format=csv returns CT.gov's own selected-field CSV with the next token in the
``x-next-page-token`` header. Study JSON: protocolSection.{identification,
status, sponsorCollaborators, conditions, design, armsInterventions,
outcomes, contactsLocations}Module, hasResults, resultsSection.

AACT (https://aact.ctti-clinicaltrials.org/) requires a signed-in account to
download snapshots, so the archive mode processes a file the user downloaded.
"""

from __future__ import annotations

import argparse
import csv
import json
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
from .._manifest import utc_stamp
from .._tabular import TableSet, count_csv_rows

API = "https://clinicaltrials.gov/api/v2"
PAGE_MAX = 1000
_NCT = re.compile(r"NCT\d{8}")

STUDY_COLUMNS = [
    "nct_id", "org_study_id", "brief_title", "official_title", "acronym", "overall_status",
    "start_date", "primary_completion_date", "completion_date", "study_first_submit_date",
    "last_update_post_date", "study_type", "phases", "allocation", "intervention_model",
    "primary_purpose", "masking", "enrollment_count", "enrollment_type", "has_results",
    "lead_sponsor_name", "lead_sponsor_class", "brief_summary"]
CHILD_COLUMNS = {
    "conditions": ["nct_id", "ordinal", "condition"],
    "keywords": ["nct_id", "ordinal", "keyword"],
    "sponsors": ["nct_id", "ordinal", "role", "name", "class"],
    "arm_groups": ["nct_id", "ordinal", "label", "type", "description", "intervention_names"],
    "interventions": ["nct_id", "ordinal", "type", "name", "description", "arm_group_labels",
                      "other_names"],
    "outcomes": ["nct_id", "ordinal", "kind", "measure", "description", "time_frame"],
    "locations": ["nct_id", "ordinal", "facility", "status", "city", "state", "zip", "country",
                  "lat", "lon"],
}


def _g(d: dict | None, *path, default=None):
    for p in path:
        if not isinstance(d, dict):
            return default
        d = d.get(p)
    return d if d is not None else default


def study_rows(study: dict) -> list[tuple[str, dict]]:
    """Split one study into a selected-field ``studies`` row and one row per
    repeated element in each related table (no cross products)."""
    ps = study.get("protocolSection") or {}
    nct = _g(ps, "identificationModule", "nctId")
    ident = ps.get("identificationModule") or {}
    status = ps.get("statusModule") or {}
    design = ps.get("designModule") or {}
    spons = ps.get("sponsorCollaboratorsModule") or {}
    out: list[tuple[str, dict]] = [("studies", {
        "nct_id": nct,
        "org_study_id": _g(ident, "orgStudyIdInfo", "id"),
        "brief_title": ident.get("briefTitle"),
        "official_title": ident.get("officialTitle"),
        "acronym": ident.get("acronym"),
        "overall_status": status.get("overallStatus"),
        "start_date": _g(status, "startDateStruct", "date"),
        "primary_completion_date": _g(status, "primaryCompletionDateStruct", "date"),
        "completion_date": _g(status, "completionDateStruct", "date"),
        "study_first_submit_date": status.get("studyFirstSubmitDate"),
        "last_update_post_date": _g(status, "lastUpdatePostDateStruct", "date"),
        "study_type": design.get("studyType"),
        "phases": json.dumps(design.get("phases")) if design.get("phases") else None,
        "allocation": _g(design, "designInfo", "allocation"),
        "intervention_model": _g(design, "designInfo", "interventionModel"),
        "primary_purpose": _g(design, "designInfo", "primaryPurpose"),
        "masking": _g(design, "designInfo", "maskingInfo", "masking"),
        "enrollment_count": _g(design, "enrollmentInfo", "count"),
        "enrollment_type": _g(design, "enrollmentInfo", "type"),
        "has_results": study.get("hasResults"),
        "lead_sponsor_name": _g(spons, "leadSponsor", "name"),
        "lead_sponsor_class": _g(spons, "leadSponsor", "class"),
        "brief_summary": _g(ps, "descriptionModule", "briefSummary"),
    })]
    cond = ps.get("conditionsModule") or {}
    for i, c in enumerate(cond.get("conditions") or [], 1):
        out.append(("conditions", {"nct_id": nct, "ordinal": i, "condition": c}))
    for i, k in enumerate(cond.get("keywords") or [], 1):
        out.append(("keywords", {"nct_id": nct, "ordinal": i, "keyword": k}))
    sponsors = ([("lead", spons["leadSponsor"])] if spons.get("leadSponsor") else []) + \
        [("collaborator", c) for c in spons.get("collaborators") or []]
    for i, (role, s) in enumerate(sponsors, 1):
        out.append(("sponsors", {"nct_id": nct, "ordinal": i, "role": role,
                                 "name": s.get("name"), "class": s.get("class")}))
    arms = ps.get("armsInterventionsModule") or {}
    for i, a in enumerate(arms.get("armGroups") or [], 1):
        out.append(("arm_groups", {"nct_id": nct, "ordinal": i, "label": a.get("label"),
                                   "type": a.get("type"), "description": a.get("description"),
                                   "intervention_names": json.dumps(a.get("interventionNames"))
                                   if a.get("interventionNames") else None}))
    for i, iv in enumerate(arms.get("interventions") or [], 1):
        out.append(("interventions", {
            "nct_id": nct, "ordinal": i, "type": iv.get("type"), "name": iv.get("name"),
            "description": iv.get("description"),
            "arm_group_labels": json.dumps(iv.get("armGroupLabels"))
            if iv.get("armGroupLabels") else None,
            "other_names": json.dumps(iv.get("otherNames")) if iv.get("otherNames") else None}))
    outcomes = ps.get("outcomesModule") or {}
    n = 0
    for kind, key in (("primary", "primaryOutcomes"), ("secondary", "secondaryOutcomes"),
                      ("other", "otherOutcomes")):
        for o in outcomes.get(key) or []:
            n += 1
            out.append(("outcomes", {"nct_id": nct, "ordinal": n, "kind": kind,
                                     "measure": o.get("measure"),
                                     "description": o.get("description"),
                                     "time_frame": o.get("timeFrame")}))
    for i, loc in enumerate(_g(ps, "contactsLocationsModule", "locations", default=[]), 1):
        out.append(("locations", {"nct_id": nct, "ordinal": i, "facility": loc.get("facility"),
                                  "status": loc.get("status"), "city": loc.get("city"),
                                  "state": loc.get("state"), "zip": loc.get("zip"),
                                  "country": loc.get("country"),
                                  "lat": _g(loc, "geoPoint", "lat"),
                                  "lon": _g(loc, "geoPoint", "lon")}))
    return out


@register
class ClinicalTrialsSource(Source):
    name = "clinicaltrials"
    title = "ClinicalTrials.gov / AACT"
    domain = "pharma"
    summary = ("Registered clinical studies via the ClinicalTrials.gov API v2, and AACT "
               "snapshot / flat-file archives downloaded by the user.")
    modes = {
        "api": ModeSpec("ClinicalTrials.gov API v2 /studies",
                        {"studies": "studies matching --condition/--term/--intervention/"
                                    "--status/--nct-id (complete study JSON; csv = selected "
                                    "fields + related tables)"},
                        formats=("native", "csv")),
        "archive": ModeSpec("process an AACT file you downloaded (AACT requires sign-in)",
                            {"aact-flatfiles": "AACT pipe-delimited flat-file zip",
                             "aact-pgdump": "AACT PostgreSQL dump zip (recorded + checksummed; "
                                            "restore with pg_restore)"},
                            formats=("native", "csv"), bounded_by_limit=False),
    }
    docs_urls = ["https://clinicaltrials.gov/data-api/api", "https://aact.ctti-clinicaltrials.org/"]
    terms_urls = ["https://clinicaltrials.gov/about-site/terms-conditions",
                  "https://aact.ctti-clinicaltrials.org/"]
    attribution = ("ClinicalTrials.gov, U.S. National Library of Medicine; AACT by the Clinical "
                   "Trials Transformation Initiative (CTTI)")
    rate_per_second = 0.8          # CT.gov asks for ~50 requests/minute
    data_dictionaries = ["https://clinicaltrials.gov/data-api/about-api/study-data-structure",
                         "https://aact.ctti-clinicaltrials.org/"]
    notes = [
        "Results sections hold AGGREGATE outcome data reported by sponsors -- never "
        "patient-level data.",
        "studies.csv is a SELECTED-FIELD export; the native JSON pages are the complete record.",
        "AACT: sign in at https://aact.ctti-clinicaltrials.org/, download a flat-file or "
        "pg_dump zip, then: clinicaltrials archive --dataset aact-flatfiles --from-archive <zip>",
    ]

    def add_arguments(self, mode: str, parser: argparse.ArgumentParser) -> None:
        if mode == "api":
            parser.add_argument("--condition", help="query.cond, e.g. obesity")
            parser.add_argument("--term", help="query.term (any field)")
            parser.add_argument("--intervention", help="query.intr, e.g. semaglutide")
            parser.add_argument("--status", help="filter.overallStatus, e.g. RECRUITING,COMPLETED")
            parser.add_argument("--nct-id", action="append",
                                help="NCT ids (repeat or comma-separate)")
            parser.add_argument("--native-type", choices=("json", "csv"), default="json",
                                help="json = complete study records (default); csv = "
                                     "CT.gov's own selected-field CSV export")
        else:
            parser.add_argument("--from-archive", type=Path, required=True,
                                help="path to the AACT zip you downloaded")

    def validate(self, mode, args):
        if mode == "archive":
            if args.dataset == "aact-pgdump" and args.format == "csv":
                raise UsageError("aact-pgdump is a database dump; --format csv is not supported "
                                 "(restore it with pg_restore, or use aact-flatfiles)")
            return
        ids = []
        for v in args.nct_id or []:
            ids += [x.strip().upper() for x in v.split(",") if x.strip()]
        bad = [x for x in ids if not _NCT.fullmatch(x)]
        if bad:
            raise UsageError(f"invalid NCT id(s): {', '.join(bad)}")
        args.nct_id = ids
        if not (ids or args.condition or args.term or args.intervention):
            raise UsageError("give --condition, --term, --intervention or --nct-id "
                             "(the whole registry is a bulk job: use AACT)")
        if args.native_type == "csv" and args.format == "csv":
            raise UsageError("--native-type csv already is CT.gov's CSV export; normalized "
                             "related tables are built from JSON (--native-type json "
                             "--format csv)")

    def snapshot_label(self, mode, args):
        if mode == "archive":
            stem = args.from_archive.name.split(".")[0]
            return re.sub(r"[^A-Za-z0-9._-]+", "_", stem)[:80] or utc_stamp()
        return utc_stamp()

    def _params(self, a) -> dict:
        p = {"countTotal": "true", "format": a.native_type}
        for flag, name in (("condition", "query.cond"), ("term", "query.term"),
                           ("intervention", "query.intr"), ("status", "filter.overallStatus")):
            v = getattr(a, flag)
            if v:
                p[name] = v
        if a.nct_id:
            p["filter.ids"] = ",".join(a.nct_id)
        return p

    def plan(self, ctx: Context) -> Plan:
        a = ctx.args
        p = Plan(self.name, a.dataset, ctx.manifest.mode, ctx.root.name)
        if ctx.manifest.mode == "archive":
            p.files.append(PlannedFile(str(a.from_archive), f"native/{a.from_archive.name}",
                                       size=a.from_archive.stat().st_size
                                       if a.from_archive.exists() else None,
                                       description="user-supplied AACT archive (local copy)"))
            return p
        params = self._params(a)
        p.requests.append(f"{API}/studies " + " ".join(f"{k}={v}" for k, v in params.items())
                          + f" pageSize={ctx.page_size(PAGE_MAX)} (then pageToken)")
        return p

    def run(self, ctx: Context, plan: Plan) -> None:
        a = ctx.args
        if ctx.manifest.mode == "archive":
            ctx.manifest.mode = "archive"
            ctx.manifest.snapshot_kind = "full"
            ctx.manifest.warn("AACT archive supplied by the user; its provenance is the file "
                              "they downloaded (name recorded as the snapshot label)")
            if a.dataset == "aact-pgdump":
                ingest_archive(ctx, a.from_archive, delimiter=None,
                               description="AACT PostgreSQL dump (restore with pg_restore)")
            else:
                ingest_archive(ctx, a.from_archive, delimiter="|", quoting=csv.QUOTE_MINIMAL,
                               convert=ctx.want_csv, description="AACT pipe-delimited flat files")
            return
        ctx.manifest.snapshot_kind = "query"
        params = {**self._params(a), "pageSize": ctx.page_size(PAGE_MAX)}
        pages: list[Path] = []
        records = 0
        total = None
        token = None
        while True:
            reason = ctx.cap(records, len(pages))
            if reason:
                ctx.manifest.truncate(reason)
                break
            if a.limit:
                params["pageSize"] = max(1, min(PAGE_MAX, a.limit - records))
            if token:
                params["pageToken"] = token
            resp = ctx.client.request("GET", f"{API}/studies", params=params,
                                      headers={"Accept": "application/json, text/csv"})
            n = len(pages) + 1
            if a.native_type == "json":
                data = resp.json()
                studies = data.get("studies") or []
                total = data.get("totalCount", total)
                pages.append(ctx.save_json(f"native/studies-{n:04d}.json", data,
                                           url=f"{API}/studies", records=len(studies)))
                got = len(studies)
                token = data.get("nextPageToken")
            else:
                path = ctx.path(f"native/studies-{n:04d}.csv")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(resp.content)
                got = count_csv_rows(path)
                ctx.manifest.add_file(ctx.root, path, role="native", format="csv",
                                      delimiter="comma", rows=got, source_url=f"{API}/studies",
                                      description="CT.gov selected-field CSV export (not the "
                                                  "complete study record)")
                pages.append(path)
                token = resp.headers.get("x-next-page-token")
            records += got
            if not token or got == 0:
                break
            if a.limit and records >= a.limit:
                ctx.manifest.truncate(f"--limit {a.limit} reached (of {total})")
                break
        ctx.manifest.pagination = {"pages_fetched": len(pages), "records": records,
                                   "total_reported": total,
                                   "complete": not ctx.manifest.truncated}
        try:
            v = ctx.client.get_json(f"{API}/version")
            ctx.manifest.source_release = v.get("dataTimestamp")
        except Exception:
            pass
        if ctx.want_csv and pages:
            self._normalize(ctx, pages)

    def _normalize(self, ctx: Context, pages: list[Path]) -> None:
        ts = TableSet(ctx.path("normalized"))
        try:
            ts.table("studies", STUDY_COLUMNS)
            for name, cols in CHILD_COLUMNS.items():
                ts.table(name, cols)
            for study in iter_json_records(pages, "studies"):
                for table, row in study_rows(study):
                    ts.write(table, CHILD_COLUMNS.get(table, STUDY_COLUMNS), row)
        except BaseException:
            ts.abort()
            raise
        tables = ts.close()
        ctx.register_tables({"studies": tables["studies"]}, pages,
                            description="SELECTED-FIELD study table (native JSON is complete)")
        ctx.register_tables({k: v for k, v in tables.items() if k != "studies"}, pages,
                            description="related table keyed by nct_id + ordinal")
        ctx.manifest.transformations += [
            "studies.csv = selected protocol fields only (not a complete study export)",
            "conditions, keywords, sponsors (lead + collaborators), arm_groups, interventions, "
            "outcomes (primary/secondary/other), locations -> one table each, keyed by nct_id + "
            "ordinal; independent arrays never cross-joined",
            "short lists inside one element (phases, intervention names, other names) kept as "
            "JSON text",
            "results sections (aggregate data) are left in the native JSON only",
        ]
