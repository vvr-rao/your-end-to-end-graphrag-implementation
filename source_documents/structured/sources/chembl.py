"""ChEMBL: filtered REST resources (API) and release assets (bulk).

Verified 2026-11-08: https://chembl.gitbook.io/chembl-interface-documentation/
web-services/chembl-data-web-services (base https://www.ebi.ac.uk/chembl/api/
data/, .json suffix, [field]__[filter]=value, page_meta.next) -- limit capped
at 1000 per page (checked live); release folders
https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_NN/ with
checksums.txt (SHA-256, "SHA256<TAB>file"); latest/README names the release.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from urllib.parse import urljoin

from .._base import (
    Context,
    ModeSpec,
    Plan,
    PlannedFile,
    Source,
    UsageError,
    iter_json_records,
    register,
)
from .._http import DownloadError
from .._manifest import utc_stamp
from .._tabular import JsonFlattener

API = "https://www.ebi.ac.uk/chembl/api/data"
FTP = "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb"
PAGE_MAX = 1000
RESOURCES = {
    "molecule": ("molecules", "molecule_chembl_id"),
    "target": ("targets", "target_chembl_id"),
    "assay": ("assays", "assay_chembl_id"),
    "activity": ("activities", "activity_id"),
    "document": ("documents", "document_chembl_id"),
}
# asset key -> filename suffix after "chembl_NN"
ASSETS = {
    "sqlite": "_sqlite.tar.gz", "postgresql": "_postgresql.tar.gz", "mysql": "_mysql.tar.gz",
    "sdf": ".sdf.gz", "chemreps": "_chemreps.txt.gz", "fasta": ".fa.gz",
    "fps": ".fps.gz", "h5": ".h5", "release-notes": "_release_notes.txt",
    "schema-docs": "schema_documentation.txt", "uniprot-mapping": "chembl_uniprot_mapping.txt",
}
_FILTER = re.compile(r"[a-z_]+(__[a-z]+)?")


@register
class ChemblSource(Source):
    name = "chembl"
    title = "ChEMBL"
    domain = "pharma"
    summary = ("Bioactive molecules, targets, assays, activities and documents via the ChEMBL "
               "REST API (filtered), and pinned release assets for bulk use.")
    modes = {
        "api": ModeSpec("ChEMBL web services (JSON)",
                        {k: f"{v[0]} (key {v[1]})" for k, v in RESOURCES.items()},
                        formats=("native", "csv")),
        "bulk": ModeSpec("release assets from the EBI FTP (whole files, SHA-256 verified)",
                         {"release": "selected assets of one ChEMBL release"},
                         formats=("native",), bounded_by_limit=False),
    }
    docs_urls = ["https://chembl.gitbook.io/chembl-interface-documentation/web-services/"
                 "chembl-data-web-services",
                 "https://chembl.gitbook.io/chembl-interface-documentation/downloads"]
    terms_urls = ["https://chembl.gitbook.io/chembl-interface-documentation/about#data-licensing"]
    attribution = "ChEMBL, EMBL-EBI (CC BY-SA 3.0); cite the release DOI"
    data_dictionaries = [FTP + "/latest/schema_documentation.html"]
    rate_per_second = 3.0
    notes = ["Activities keep standard_type / standard_relation / standard_value / "
             "standard_units AND the originally reported type / relation / value / units in "
             "separate columns; values of different measurement types are never merged.",
             "Bulk databases are 2-5 GB each; pick assets with --asset."]

    def add_arguments(self, mode: str, parser: argparse.ArgumentParser) -> None:
        if mode == "api":
            parser.add_argument("--chembl-id", action="append",
                                help="ids of the selected resource (repeat or comma-separate)")
            parser.add_argument("--filter", action="append", default=[],
                                help="ChEMBL filter field[__op]=value, e.g. "
                                     "target_chembl_id=CHEMBL231, standard_type=IC50, "
                                     "pchembl_value__gte=6 (repeatable)")
            parser.add_argument("--only", help="comma-separated fields to return")
            parser.add_argument("--order-by", help="field (prefix - for descending)")
        else:
            parser.add_argument("--release", required=True,
                                help="release number (e.g. 37) or 'latest' (resolved and "
                                     "recorded)")
            parser.add_argument("--asset", action="append", required=True,
                                choices=sorted(ASSETS), help="asset(s) to download (repeatable)")

    def validate(self, mode, args):
        if mode == "bulk":
            if args.release != "latest" and not re.fullmatch(r"\d{1,3}(_\d)?", args.release):
                raise UsageError("--release must be a number like 37 or 'latest'")
            return
        filters = {}
        for f in args.filter:
            if "=" not in f:
                raise UsageError(f"--filter {f!r} must be field=value")
            k, v = f.split("=", 1)
            if not _FILTER.fullmatch(k):
                raise UsageError(f"--filter field {k!r} is not a ChEMBL field[__op] name")
            filters[k] = v
        ids = []
        for v in args.chembl_id or []:
            ids += [x.strip().upper() for x in v.split(",") if x.strip()]
        if ids:
            key = RESOURCES[args.dataset][1]
            if not key.endswith("chembl_id"):
                raise UsageError(f"--chembl-id is not available for {args.dataset}; use "
                                 f"--filter {key}=... or another filter")
            if any(not re.fullmatch(r"CHEMBL\d+", i) for i in ids):
                raise UsageError("--chembl-id values look like CHEMBL25")
            filters[f"{key}__in"] = ",".join(ids)
        if not filters:
            raise UsageError("give --chembl-id or at least one --filter (an unfiltered "
                             f"{args.dataset} listing is a bulk job: use `chembl bulk`)")
        args.filters = filters

    def snapshot_label(self, mode, args):
        return utc_stamp() if mode == "api" else "pending"

    def _release(self, ctx: Context) -> str:
        rel = ctx.args.release
        if rel != "latest":
            return rel
        resp = ctx.client.request("GET", f"{FTP}/latest/README")
        m = re.search(r"Release:\s*chembl_(\d+(?:_\d)?)", resp.text)
        if not m:
            raise DownloadError("could not read the release number from latest/README; pass "
                                "--release explicitly")
        return m.group(1)

    def plan(self, ctx: Context) -> Plan:
        a = ctx.args
        p = Plan(self.name, a.dataset, ctx.manifest.mode, ctx.root.name)
        if ctx.manifest.mode == "api":
            q = " ".join(f"{k}={v}" for k, v in a.filters.items())
            p.requests.append(f"{API}/{a.dataset}.json {q} limit={ctx.page_size(PAGE_MAX)} "
                              f"(then page_meta.next)")
            return p
        rel = self._release(ctx)
        base = f"{FTP}/releases/chembl_{rel}/"
        listing = ctx.client.request("GET", base).text
        names = set(re.findall(r'href="([^"?/][^"]*)"', listing))
        p.release = f"chembl_{rel}"
        p.snapshot = f"chembl_{rel}"
        for asset in a.asset:
            suffix = ASSETS[asset]
            fname = suffix if not suffix.startswith(("_", ".")) else f"chembl_{rel}{suffix}"
            if fname not in names:
                raise UsageError(f"{fname} is not in {base} (asset {asset!r} not published for "
                                 f"this release)")
            head = ctx.client.head(base + fname)
            cl = head.headers.get("Content-Length", "") if head is not None else ""
            p.files.append(PlannedFile(base + fname, f"native/{fname}",
                                       expect="gzip" if fname.endswith(".gz") else "any",
                                       size=int(cl) if cl.isdigit() else None,
                                       description=f"ChEMBL {rel} {asset}"))
        for extra in ("checksums.txt", "README", "LICENSE", "REQUIRED.ATTRIBUTION"):
            if extra in names:
                p.files.append(PlannedFile(base + extra, f"native/{extra}",
                                           description="release metadata"))
        return p

    def run(self, ctx: Context, plan: Plan) -> None:
        if ctx.manifest.mode == "bulk":
            return self._run_bulk(ctx, plan)
        a = ctx.args
        ctx.manifest.snapshot_kind = "query"
        plural, key = RESOURCES[a.dataset]
        params: dict | None = {**a.filters, "limit": ctx.page_size(PAGE_MAX)}
        if a.only:
            params["only"] = a.only
        if a.order_by:
            params["order_by"] = a.order_by
        url: str | None = f"{API}/{a.dataset}.json"
        pages: list[Path] = []
        records = 0
        total = None
        while url:
            reason = ctx.cap(records, len(pages))
            if reason:
                ctx.manifest.truncate(reason)
                break
            data = ctx.client.get_json(url, params=params)
            items = data.get(plural) or []
            meta = data.get("page_meta") or {}
            total = meta.get("total_count", total)
            if a.limit and records + len(items) > a.limit:
                items = items[: a.limit - records]
                data = {**data, plural: items}
                ctx.manifest.truncate(f"--limit {a.limit} reached (of {total})")
                meta = {**meta, "next": None}
            pages.append(ctx.save_json(f"native/{plural}-{len(pages) + 1:04d}.json", data,
                                       url=f"{API}/{a.dataset}.json", records=len(items)))
            records += len(items)
            nxt = meta.get("next")
            url = urljoin("https://www.ebi.ac.uk", nxt) if nxt else None
            params = None
            if url and a.limit and records >= a.limit:
                ctx.manifest.truncate(f"--limit {a.limit} reached (of {total})")
                break
        ctx.manifest.pagination = {"pages_fetched": len(pages), "records": records,
                                   "total_reported": total,
                                   "complete": not ctx.manifest.truncated}
        try:
            st = ctx.client.get_json(f"{API}/status.json")
            ctx.manifest.source_release = st.get("chembl_db_version")
        except Exception:
            pass
        if ctx.want_csv and pages:
            flat = JsonFlattener(plural, lambda r, i: {key: r.get(key)}, skip_fields=(key,))
            tables = flat.write(lambda: iter_json_records(pages, plural),
                                ctx.path("normalized"))
            ctx.register_tables(tables, pages)
            ctx.manifest.transformations.append(
                f"{plural} JSON -> {plural}.csv + child tables keyed by {key}; every provider "
                f"field kept in its own column (standard_* and originally reported measurement "
                f"columns stay separate)")
            if flat.json_columns:
                ctx.manifest.transformations.append(
                    "kept as JSON text: " + ", ".join(sorted(flat.json_columns)))

    def _run_bulk(self, ctx: Context, plan: Plan) -> None:
        ctx.manifest.snapshot_kind = "full"
        sums: dict[str, str] = {}
        for pf in sorted(plan.files, key=lambda f: f.dest != "native/checksums.txt"):
            path = ctx.fetch(pf, role="native" if pf.description != "release metadata"
                             else "dictionary")
            if path.name == "checksums.txt":
                for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                    parts = line.split()
                    if len(parts) == 2 and re.fullmatch(r"[0-9a-f]{64}", parts[0]):
                        sums[parts[1]] = parts[0]
        for entry in ctx.manifest.files:
            name = Path(entry.path).name
            if name in sums:
                if sums[name] != entry.sha256:
                    raise DownloadError(f"{name}: SHA-256 does not match the release "
                                        f"checksums.txt -- delete it and rerun")
                entry.description = (entry.description or "") + " (SHA-256 verified)"
        ctx.manifest.transformations.append("assets verified against release checksums.txt "
                                            "where listed")
