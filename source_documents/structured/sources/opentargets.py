"""Open Targets Platform: bounded GraphQL entity retrieval (API) and Parquet
datasets from a pinned release (bulk).

Verified 2026-11-08: GraphQL POST https://api.platform.opentargets.org/api/v4/
graphql (live: meta.dataVersion {year month iteration} = 26.09; target(ensemblId)
.associatedDiseases(page:{index,size}) {count rows{score disease{id name}
datatypeScores{id score}}}; disease(efoId).associatedTargets(...); drug(chemblId)
{maximumClinicalStage mechanismsOfAction{rows{...}} indications{count rows{id
maxClinicalStage disease{id name}}}}). Releases:
https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/<release>/output/
<dataset>/part-*.parquet (snake_case dataset names since 25.03).

The GraphQL schema changes between releases (26.09 renamed several drug
fields); --query-file lets a user supply their own query, and the query used
is saved next to the results.
"""

from __future__ import annotations

import argparse
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
    register,
)
from .._manifest import utc_stamp
from .._tabular import CsvTable, JsonFlattener

GQL = "https://api.platform.opentargets.org/api/v4/graphql"
FTP = "https://ftp.ebi.ac.uk/pub/databases/opentargets/platform"
PAGE_MAX = 100

QUERIES = {
    "target": ("ensemblId", "associatedDiseases", """
query($id: String!, $index: Int!, $size: Int!) {
  target(ensemblId: $id) {
    id approvedSymbol approvedName biotype
    associatedDiseases(page: {index: $index, size: $size}) {
      count rows { score disease { id name } datatypeScores { id score } }
    }
  }
}"""),
    "disease": ("efoId", "associatedTargets", """
query($id: String!, $index: Int!, $size: Int!) {
  disease(efoId: $id) {
    id name description
    associatedTargets(page: {index: $index, size: $size}) {
      count rows { score target { id approvedSymbol } datatypeScores { id score } }
    }
  }
}"""),
    "drug": ("chemblId", None, """
query($id: String!) {
  drug(chemblId: $id) {
    id name drugType maximumClinicalStage description
    mechanismsOfAction { rows { mechanismOfAction actionType targets { id approvedSymbol } } }
    indications { count rows { id maxClinicalStage disease { id name } } }
  }
}"""),
}
META_QUERY = "{ meta { dataVersion { year month iteration } apiVersion { x y z } } }"
_RELEASE = re.compile(r"\d{2}\.\d{2}(\.\d+)?")


@register
class OpenTargetsSource(Source):
    name = "opentargets"
    title = "Open Targets Platform"
    domain = "pharma"
    summary = ("Target-disease evidence: GraphQL lookups of selected targets / diseases / drugs "
               "(bounded association pages) and Parquet datasets from a pinned release.")
    modes = {
        "api": ModeSpec("Open Targets GraphQL API",
                        {"target": "targets by Ensembl id (+ associated diseases)",
                         "disease": "diseases by EFO/MONDO id (+ associated targets)",
                         "drug": "drugs by ChEMBL id (+ mechanisms, indications)"},
                        formats=("native", "csv")),
        "bulk": ModeSpec("Parquet dataset from a pinned release (EBI FTP over HTTPS)",
                         {"<dataset>": "any folder under <release>/output/, e.g. target, "
                                       "disease, drug_molecule, association_overall_direct"},
                         formats=("native", "csv"), open_datasets=True),
    }
    docs_urls = ["https://platform-docs.opentargets.org/data-access/graphql-api",
                 "https://platform-docs.opentargets.org/data-access/datasets"]
    terms_urls = ["https://platform-docs.opentargets.org/licence"]
    attribution = "Open Targets Platform (CC0 1.0); cite the release used"
    rate_per_second = 2.0
    notes = ["Bulk CSV conversion needs pyarrow: uv sync --extra structured. Nested list/struct "
             "columns are written as JSON text and listed in the manifest.",
             "Use --max-files for a bounded bulk sample; --limit caps CSV rows, not the Parquet "
             "download."]

    def add_arguments(self, mode: str, parser: argparse.ArgumentParser) -> None:
        if mode == "api":
            parser.add_argument("--id", action="append", required=True,
                                help="entity ids (repeat or comma-separate)")
            parser.add_argument("--query-file", type=Path,
                                help="custom GraphQL query using $id (and $index/$size)")
        else:
            parser.add_argument("--release", required=True,
                                help="release like 26.09, or 'latest' (resolved and recorded)")
            parser.add_argument("--max-files", type=int,
                                help="download only the first N parquet parts (marked truncated)")

    def validate(self, mode, args):
        if mode == "api":
            ids = []
            for v in args.id:
                ids += [x.strip() for x in v.split(",") if x.strip()]
            if not ids or any(not re.fullmatch(r"[A-Za-z0-9_:.-]+", i) for i in ids):
                raise UsageError(f"invalid --id {args.id!r}")
            args.id = ids
            if args.query_file and not args.query_file.exists():
                raise UsageError(f"--query-file not found: {args.query_file}")
            return
        if args.dataset in (None, "<dataset>") or not re.fullmatch(r"[a-z0-9_]+", args.dataset):
            raise UsageError("pass --dataset <name> (a folder under <release>/output/)")
        if args.release != "latest" and not _RELEASE.fullmatch(args.release):
            raise UsageError("--release must look like 26.09 or be 'latest'")
        if args.max_files is not None and args.max_files < 1:
            raise UsageError("--max-files must be >= 1")

    def snapshot_label(self, mode, args):
        return utc_stamp() if mode == "api" else "pending"

    # ---------------------------------------------------------------- plan

    def _listing(self, ctx: Context, url: str) -> list[str]:
        html = ctx.client.request("GET", url).text
        return re.findall(r'href="([^"?/][^"]*)"', html)

    def plan(self, ctx: Context) -> Plan:
        a = ctx.args
        p = Plan(self.name, a.dataset, ctx.manifest.mode, ctx.root.name)
        if ctx.manifest.mode == "api":
            for i in a.id:
                p.requests.append(f"POST {GQL} {a.dataset}({QUERIES[a.dataset][0]}: {i})")
            if QUERIES[a.dataset][1]:
                p.notes.append(f"associations paged {PAGE_MAX} per request; --limit caps rows "
                               f"per entity")
            return p
        rel = a.release
        if rel == "latest":
            rels = [n.rstrip("/") for n in self._listing(ctx, f"{FTP}/")
                    if _RELEASE.fullmatch(n.rstrip("/"))]
            if not rels:
                raise UsageError("could not list Open Targets releases; pass --release")
            rel = max(rels, key=lambda r: [int(x) for x in r.split(".")])
        base = f"{FTP}/{rel}/output/{a.dataset}/"
        names = self._listing(ctx, base)
        parts = sorted(n for n in names if n.endswith(".parquet"))
        if not parts:
            raise UsageError(f"no parquet files at {base} -- check --dataset (names are "
                             f"snake_case since 25.03, e.g. association_overall_direct)")
        p.release = rel
        p.snapshot = rel
        if a.max_files and len(parts) > a.max_files:
            p.notes.append(f"--max-files {a.max_files}: {a.max_files} of {len(parts)} parts "
                           f"(sample, not the complete dataset)")
            parts = parts[: a.max_files]
        for name in parts:
            size = None
            if len(parts) <= 50:
                head = ctx.client.head(base + name)
                cl = head.headers.get("Content-Length", "") if head is not None else ""
                size = int(cl) if cl.isdigit() else None
            p.files.append(PlannedFile(base + name, f"native/{name}", expect="parquet",
                                       size=size, description=f"{a.dataset} parquet part"))
        return p

    # ----------------------------------------------------------------- run

    def run(self, ctx: Context, plan: Plan) -> None:
        if ctx.manifest.mode == "bulk":
            return self._run_bulk(ctx, plan)
        a = ctx.args
        ctx.manifest.snapshot_kind = "query"
        meta = ctx.client.post_json(GQL, {"query": META_QUERY})
        ctx.save_json("native/meta.json", meta, url=GQL, description="API data version")
        dv = ((meta.get("data") or {}).get("meta") or {}).get("dataVersion") or {}
        if dv:
            ctx.manifest.source_release = ".".join(str(dv[k]) for k in ("year", "month")
                                                   if dv.get(k)) + (
                f".{dv['iteration']}" if dv.get("iteration") else "")
        _arg, assoc, query = QUERIES[a.dataset]
        if a.query_file:
            query = a.query_file.read_text(encoding="utf-8")
            assoc = assoc if "$index" in query else None
        qpath = ctx.path("native/query.graphql")
        qpath.parent.mkdir(parents=True, exist_ok=True)
        qpath.write_text(query, encoding="utf-8")
        ctx.manifest.add_file(ctx.root, qpath, role="native", format="graphql",
                              description="GraphQL query used")
        natives: list[Path] = []
        requests = 0
        for ent in a.id:
            index = 0
            rows_seen = 0
            while True:
                variables = {"id": ent}
                size = PAGE_MAX
                if assoc:
                    if a.limit:
                        size = max(1, min(PAGE_MAX, a.limit - rows_seen))
                    variables |= {"index": index, "size": size}
                data = ctx.client.post_json(GQL, {"query": query, "variables": variables})
                requests += 1
                if data.get("errors"):
                    raise UsageError(f"GraphQL error for {ent}: {data['errors'][0].get('message')}"
                                     f" (the schema may have changed; try --query-file)")
                obj = (data.get("data") or {}).get(a.dataset)
                if obj is None:
                    ctx.manifest.warn(f"{a.dataset} {ent} not found")
                    break
                safe = re.sub(r"[^A-Za-z0-9_.-]", "_", ent)
                natives.append(ctx.save_json(f"native/{safe}-p{index:04d}.json", data, url=GQL))
                if not assoc:
                    break
                block = obj.get(assoc) or {}
                got = len(block.get("rows") or [])
                rows_seen += got
                count = block.get("count") or 0
                if got == 0 or rows_seen >= count:
                    break
                if a.limit and rows_seen >= a.limit:
                    ctx.manifest.truncate(f"--limit {a.limit} association rows for {ent} "
                                          f"(of {count})")
                    break
                if a.max_pages and index + 1 >= a.max_pages:
                    ctx.manifest.truncate(f"--max-pages {a.max_pages} for {ent} (of {count})")
                    break
                index += 1
        ctx.manifest.pagination = {"requests": requests, "complete": not ctx.manifest.truncated}
        if ctx.want_csv and natives:
            def records():
                for path in natives:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    obj = dict((data.get("data") or {}).get(a.dataset) or {})
                    page = int(path.stem.rsplit("-p", 1)[1])
                    if page > 0:             # later pages only add association rows
                        obj = {"id": obj.get("id"), **{k: v for k, v in obj.items()
                                                       if k == assoc}}
                    obj["_page"] = page
                    yield obj
            flat = JsonFlattener(a.dataset, lambda r, i: {"id": r.get("id"),
                                                          "page": r.get("_page")},
                                 skip_fields=("id", "_page"))
            tables = flat.write(records, ctx.path("normalized"))
            ctx.register_tables(tables, natives)
            ctx.manifest.transformations.append(
                f"GraphQL JSON -> {a.dataset}.csv + child tables (associations, datatype "
                f"scores, mechanisms...) keyed by id + page + ordinals")

    def _run_bulk(self, ctx: Context, plan: Plan) -> None:
        ctx.manifest.snapshot_kind = "full"
        if ctx.args.max_files:
            listed = [n for n in plan.notes if n.startswith("--max-files")]
            if listed:
                ctx.manifest.truncate(listed[0])
        parts = [ctx.fetch(pf, format="parquet") for pf in plan.files]
        if ctx.want_csv:
            self._parquet_to_csv(ctx, parts)

    def _parquet_to_csv(self, ctx: Context, parts: list[Path]) -> None:
        try:
            import pyarrow.parquet as pq
        except ImportError:
            raise UsageError("--format csv for Parquet needs pyarrow: run "
                             "`uv sync --extra structured` (native Parquet was kept)") from None
        out = ctx.path(f"normalized/{ctx.args.dataset}.csv")
        limit = ctx.args.limit
        table: CsvTable | None = None
        json_cols: set[str] = set()
        try:
            for part in parts:
                pf = pq.ParquetFile(part)
                for batch in pf.iter_batches(batch_size=10_000):
                    if table is None:
                        table = CsvTable(out, batch.schema.names)
                    for row in batch.to_pylist():
                        if limit is not None and table.rows >= limit:
                            raise StopIteration
                        for k, v in row.items():
                            if isinstance(v, (list, dict)):
                                json_cols.add(k)
                        table.write(row)
        except StopIteration:
            ctx.manifest.truncate(f"{out.name} capped at --limit {limit} rows (Parquet complete)")
        except BaseException:
            if table is not None:
                table.abort()
            raise
        if table is None:
            return
        table.close()
        ctx.register_tables({ctx.args.dataset: table}, parts,
                            description="Parquet -> CSV (streamed in 10k-row batches)")
        ctx.manifest.transformations.append(
            "Parquet -> CSV streamed batch by batch; nested list/struct columns written as JSON "
            "text: " + (", ".join(sorted(json_cols)) or "none"))
