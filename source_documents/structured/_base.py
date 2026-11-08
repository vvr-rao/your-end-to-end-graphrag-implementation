"""Source registry, shared CLI options, and the run lifecycle.

A source module subclasses :class:`Source`, declares its modes / datasets /
formats / credentials, adds its own argparse flags, and implements
``plan()`` (what would be fetched -- used by ``--dry-run``) and ``run()``.
The lifecycle in :func:`execute` handles everything common: validation,
the snapshot folder, resume / overwrite rules, the manifest, and turning
expected failures into clear messages and exit codes.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ._http import DownloadError, DownloadResult, HttpClient, write_json
from ._manifest import (
    MANIFEST_NAME,
    Manifest,
    load_manifest,
    redact_url,
    utc_stamp,
    verified_entry,
)

DEFAULT_CONTACT = "your-end-to-end-graphrag-implementation structured-download"
REPO_ROOT = Path(__file__).resolve().parents[2]


class UsageError(Exception):
    """Unsupported option combination or missing input (exit code 2)."""


class CredentialError(UsageError):
    """A required credential environment variable is not set."""


def load_env() -> None:
    """Load ``.env`` from the repo root if python-dotenv is available.
    Existing environment variables win (``override=False``)."""
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover
        return
    env = REPO_ROOT / ".env"
    if env.exists():
        load_dotenv(env, override=False)


def require_env(name: str, purpose: str, *, docs: str | None = None) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        where = f" Get one at {docs}." if docs else ""
        raise CredentialError(
            f"{name} is not set; it is required for {purpose}.{where} Add it to the "
            f"repo's .env file (never pass credentials on the command line).")
    return value


@dataclass
class EnvVar:
    name: str
    required_for: str          # e.g. "api mode", "all modes", "optional (higher rate limit)"
    description: str
    required: bool = True


@dataclass
class ModeSpec:
    description: str
    datasets: dict[str, str]                  # dataset -> description
    formats: tuple[str, ...] = ("native",)    # first = default
    bounded_by_limit: bool = True             # False for archive downloads
    open_datasets: bool = False               # dataset names validated by the source


@dataclass
class PlannedFile:
    url: str
    dest: str                       # relative to the snapshot folder
    expect: str = "any"
    description: str = ""
    size: int | None = None
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class Plan:
    source: str
    dataset: str
    mode: str
    snapshot: str                   # folder label (release / period / UTC stamp)
    requests: list[str] = field(default_factory=list)      # human lines, redacted
    files: list[PlannedFile] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    release: str | None = None

    def render(self) -> str:
        lines = [f"[plan] {self.source} {self.mode} dataset={self.dataset} "
                 f"snapshot={self.snapshot}"]
        if self.release:
            lines.append(f"  release: {self.release}")
        for r in self.requests:
            lines.append(f"  request: {r}")
        total = 0
        known = True
        for f in self.files:
            size = f"{f.size / 1e6:,.1f} MB" if f.size else "size unknown"
            if f.size:
                total += f.size
            else:
                known = False
            lines.append(f"  file: {f.dest}  <- {redact_url(f.url, f.params)}  ({size})")
        if self.files:
            lines.append(f"  total download: {total / 1e6:,.1f} MB"
                         + ("" if known else " + files of unknown size"))
        for n in self.notes:
            lines.append(f"  note: {n}")
        return "\n".join(lines)


class Context:
    """Everything a source needs while running."""

    def __init__(self, *, args: argparse.Namespace, source: Source, root: Path,
                 manifest: Manifest, client: HttpClient, prior: dict | None):
        self.args = args
        self.source = source
        self.root = root
        self.manifest = manifest
        self.client = client
        self.prior = prior
        self.log = client.log

    @property
    def want_csv(self) -> bool:
        return getattr(self.args, "format", "native") == "csv"

    def path(self, rel: str) -> Path:
        return self.root / rel

    def fetch(self, pf: PlannedFile, *, role: str = "native", format: str | None = None,
              delimiter: str | None = None) -> Path:
        """Download a planned file (skipping it if a prior manifest verifies
        it) and register it in the manifest."""
        dest = self.path(pf.dest)
        prior = verified_entry(self.prior, self.root, dest)
        fmt = format or _ext_format(dest)
        if prior and not self.args.overwrite:
            self.log(f"  skip (verified by prior manifest): {pf.dest}")
            self.manifest.add_file(self.root, dest, role=role, format=fmt,
                                   delimiter=delimiter, source_url=pf.url,
                                   description=pf.description, sha256=prior["sha256"])
            return dest
        self.log(f"  download: {pf.dest}")
        res: DownloadResult = self.client.download(pf.url, dest, params=pf.params or None,
                                                   expect=pf.expect, resume=self.args.resume)
        if res.resumed:
            self.manifest.transformations.append(f"resumed transfer of {pf.dest}")
        self.manifest.add_file(self.root, dest, role=role, format=fmt, delimiter=delimiter,
                               source_url=pf.url, description=pf.description,
                               sha256=res.sha256)
        return dest

    def save_json(self, rel: str, payload: Any, *, url: str | None = None,
                  records: int | None = None, description: str | None = None) -> Path:
        """Write an original API response verbatim (as JSON) and register it."""
        path = self.path(rel)
        write_json(path, payload)
        self.manifest.add_file(self.root, path, role="native", format="json", rows=records,
                               source_url=url, description=description)
        return path

    def register_tables(self, tables: dict, derived_from: list[Path], *,
                        description: str = "") -> None:
        for name, t in tables.items():
            self.manifest.add_file(self.root, t.path, role="normalized", format="csv",
                                   delimiter="comma", rows=t.rows, derived_from=derived_from,
                                   description=description or name)

    def cap(self, records: int, pages: int) -> str | None:
        return paginate_cap(self, records, pages)

    def page_size(self, maximum: int) -> int:
        limit = getattr(self.args, "limit", None)
        return max(1, min(maximum, limit)) if limit else maximum


def iter_json_records(paths: list[Path], key: str | Callable[[Any], list]) -> Iterator[dict]:
    """Re-read saved native page files one at a time and yield their records."""
    for p in paths:
        with p.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        items = key(payload) if callable(key) else (payload.get(key) or [])
        for item in items:
            if isinstance(item, dict):
                yield item


def ingest_archive(ctx: Context, archive: Path, *, delimiter: str | None,
                   table_suffixes: tuple[str, ...] = (".txt", ".csv"),
                   quoting: int | None = None, description: str = "",
                   convert: bool = False, null_markers: tuple[str, ...] = ()) -> list[Path]:
    """Copy a user-downloaded (or fetched) archive into the snapshot, extract
    it safely, register every member, and optionally convert delimited
    members to CSV. Native files are always kept."""
    import csv as _csv

    from ._archive import safe_extract
    from ._tabular import convert_delimited, count_data_rows, delimiter_name

    if not archive.exists():
        raise UsageError(f"archive not found: {archive}")
    native_dir = ctx.path("native")
    if archive.resolve().parent != native_dir.resolve():
        dest = native_dir / archive.name
        if not (verified_entry(ctx.prior, ctx.root, dest) and not ctx.args.overwrite):
            native_dir.mkdir(parents=True, exist_ok=True)
            part = dest.with_name(dest.name + ".part")
            shutil.copyfile(archive, part)
            os.replace(part, dest)
        archive = dest
    ctx.manifest.add_file(ctx.root, archive, role="native", format=_ext_format(archive),
                          description=description or "provider archive")
    q = _csv.QUOTE_NONE if quoting is None else quoting
    extracted = safe_extract(archive, ctx.path("extracted") / archive.name.split(".")[0])
    ctx.manifest.transformations.append(f"extracted {archive.name} ({len(extracted)} files)")
    out: list[Path] = []
    for f in sorted(extracted):
        is_table = delimiter is not None and f.name.lower().endswith(table_suffixes)
        rows = None
        if is_table:
            try:
                rows = count_data_rows(f, delimiter, quoting=q)
            except _csv.Error as exc:
                ctx.manifest.warn(f"could not count rows in {f.name}: {exc}")
        ctx.manifest.add_file(ctx.root, f, role="extracted", format=_ext_format(f),
                              delimiter=delimiter_name(delimiter) if is_table else None,
                              rows=rows, derived_from=[archive])
        out.append(f)
        if convert and is_table:
            dst = ctx.path("normalized") / (f.stem + ".csv")
            rej = ctx.path("normalized") / (f.stem + ".rejects.csv")
            conv = convert_delimited(f, dst, delimiter=delimiter, quoting=q,
                                     null_markers=null_markers, rejects=rej)
            ctx.manifest.add_file(ctx.root, dst, role="normalized", format="csv",
                                  delimiter="comma", rows=conv.rows, derived_from=[f])
            ctx.manifest.transformations.append(
                f"{f.name}: {delimiter_name(delimiter)} -> CSV (values copied as strings; "
                f"source encoding {conv.encoding})")
            if conv.rejected:
                ctx.manifest.add_file(
                    ctx.root, rej, role="rejected", format="csv", delimiter="comma",
                    rows=conv.rejected, derived_from=[f],
                    description="provider rows with the wrong field count, kept verbatim "
                                "and NOT included in the normalized table")
                ctx.manifest.warn(
                    f"{f.name}: {conv.rejected} of {conv.rows + conv.rejected} rows have the "
                    f"wrong field count (provider data error); quarantined in {rej.name}, "
                    f"left out of {dst.name}; the native file is unchanged")
    return out


def _ext_format(p: Path) -> str:
    name = p.name.lower()
    for suffix, fmt in ((".json", "json"), (".jsonl", "jsonl"), (".csv", "csv"),
                        (".tsv", "tsv"), (".txt", "txt"), (".zip", "zip"),
                        (".parquet", "parquet"), (".tar.gz", "tar.gz"), (".gz", "gz"),
                        (".xml", "xml")):
        if name.endswith(suffix):
            return fmt
    return p.suffix.lstrip(".") or "bin"


class Source:
    """Base class for a structured-data provider."""

    name: str = ""
    title: str = ""
    domain: str = ""                         # finance | pharma
    summary: str = ""
    modes: dict[str, ModeSpec] = {}
    env: list[EnvVar] = []
    docs_urls: list[str] = []
    terms_urls: list[str] = []
    attribution: str = ""
    rate_per_second: float | None = 2.0
    data_dictionaries: list[str] = []
    notes: list[str] = []                    # shown by `info`

    # ---- hooks for subclasses ------------------------------------------

    def add_arguments(self, mode: str, parser: argparse.ArgumentParser) -> None:
        """Add source-specific flags for ``mode``."""

    def validate(self, mode: str, args: argparse.Namespace) -> None:
        """Raise UsageError for unsupported combinations."""

    def credentials(self, mode: str, args: argparse.Namespace) -> dict[str, Any]:
        """Return ``{"params": {...}, "headers": {...}, "user_agent": str|None}``.
        Raise CredentialError when a required variable is missing."""
        return {}

    def snapshot_label(self, mode: str, args: argparse.Namespace) -> str:
        return utc_stamp()

    def dataset_of(self, mode: str, args: argparse.Namespace) -> str:
        ds = getattr(args, "dataset", None)
        return ds or next(iter(self.modes[mode].datasets))

    def plan(self, ctx: Context) -> Plan:
        raise NotImplementedError

    def run(self, ctx: Context, plan: Plan) -> None:
        raise NotImplementedError


SOURCES: dict[str, Source] = {}


def register(cls: type[Source]) -> type[Source]:
    inst = cls()
    SOURCES[inst.name] = inst
    return cls


# --------------------------------------------------------------- CLI glue

def add_common_arguments(parser: argparse.ArgumentParser, source: Source, mode: str) -> None:
    spec = source.modes[mode]
    g = parser.add_argument_group("common options")
    g.add_argument("-o", "--output", type=Path, default=None,
                   help=f"Output root (default: source_documents/structured_{source.name}/). "
                        f"Files land in <output>/<dataset>/<snapshot>/.")
    g.add_argument("--format", choices=spec.formats, default=spec.formats[0],
                   help="native = provider files only; csv = native files PLUS normalized "
                        f"CSV (supported here: {', '.join(spec.formats)}).")
    if spec.bounded_by_limit:
        g.add_argument("--limit", type=int, default=None,
                       help="Stop after N records (the result is marked as a truncated sample).")
        g.add_argument("--max-pages", type=int, default=None,
                       help="Stop after N API pages (marked truncated).")
    g.add_argument("--dry-run", action="store_true",
                   help="Print the download plan (URLs, expected sizes) and exit; no data "
                        "is downloaded (small metadata lookups may still be made).")
    g.add_argument("--resume", action="store_true",
                   help="Reuse an existing snapshot folder: skip files its manifest verifies "
                        "and continue .part transfers where the server supports ranges.")
    g.add_argument("--overwrite", action="store_true",
                   help="Replace an existing snapshot folder of the same name.")
    g.add_argument("--timeout", type=float, default=120.0,
                   help="Read timeout per request in seconds (default 120).")
    g.add_argument("--max-retries", type=int, default=4,
                   help="Retries for 429/5xx/connection errors (default 4).")


def default_output(source: Source) -> Path:
    return REPO_ROOT / "source_documents" / f"structured_{source.name}"


def _prepare_root(root: Path, args: argparse.Namespace) -> dict | None:
    """Apply the snapshot-folder rules; return the prior manifest (if resuming)."""
    if not root.exists() or not any(root.iterdir()):
        return None
    if args.overwrite and args.resume:
        raise UsageError("--overwrite and --resume are mutually exclusive")
    if args.resume:
        return load_manifest(root)
    if args.overwrite:
        if not (root / MANIFEST_NAME).exists():
            raise UsageError(
                f"{root} exists but has no {MANIFEST_NAME}; refusing to delete a folder "
                f"this tool did not create. Remove it by hand or choose another --output.")
        shutil.rmtree(root)
        return None
    raise UsageError(
        f"snapshot folder already exists: {root}\n"
        f"  pass --resume to complete/verify it, or --overwrite to replace it.")


def execute(source: Source, mode: str, args: argparse.Namespace,
            *, client_factory: Callable[..., HttpClient] = HttpClient) -> int:
    """Run one source/mode invocation. Returns a process exit code."""
    log = lambda m: print(m, file=sys.stderr)  # noqa: E731
    try:
        if mode not in source.modes:
            raise UsageError(f"{source.name} does not support {mode} mode "
                             f"(supported: {', '.join(source.modes)})")
        spec = source.modes[mode]
        dataset = source.dataset_of(mode, args)
        if not spec.open_datasets and dataset not in spec.datasets:
            raise UsageError(f"{source.name} {mode}: unknown dataset {dataset!r} "
                             f"(choose from: {', '.join(spec.datasets)})")
        args.dataset = dataset
        for flag in ("limit", "max_pages"):
            v = getattr(args, flag, None)
            if v is not None and v < 1:
                raise UsageError(f"--{flag.replace('_', '-')} must be >= 1")
        source.validate(mode, args)
        try:
            creds = source.credentials(mode, args) or {}
        except CredentialError as exc:
            if not args.dry_run:
                raise
            log(f"warning (dry run continues): {exc}")
            creds = {}
        client = client_factory(
            contact=DEFAULT_CONTACT,
            rate_per_second=source.rate_per_second,
            timeout=(15.0, float(args.timeout)),
            max_retries=args.max_retries,
            user_agent=creds.get("user_agent"),
            auth_params=creds.get("params"),
            auth_headers=creds.get("headers"),
            log=log,
        )
        label = source.snapshot_label(mode, args)
        out_root = args.output or default_output(source)
        root = out_root / dataset / label
        params = {k: _plain(v) for k, v in vars(args).items()
                  if k not in {"func", "output", "dry_run", "overwrite", "resume", "timeout",
                               "max_retries", "command", "mode"}
                  and v not in (None, False) and v != [] and v != {}}
        manifest = Manifest(source=source.name, dataset=dataset, mode=mode, params=params,
                            attribution=source.attribution or None,
                            terms_urls=list(source.terms_urls),
                            data_dictionaries=list(source.data_dictionaries))
        # The plan may resolve the snapshot label (e.g. a provider release date),
        # so it is computed before the snapshot folder is checked or created.
        ctx = Context(args=args, source=source, root=root, manifest=manifest, client=client,
                      prior=None)
        plan = source.plan(ctx)
        if plan.snapshot != root.name:
            if not re.fullmatch(r"[A-Za-z0-9._-]+", plan.snapshot):
                raise UsageError(f"unsafe snapshot label {plan.snapshot!r}")
            root = out_root / dataset / plan.snapshot
            ctx.root = root
        if args.dry_run:
            print(plan.render())
            print(f"[plan] output folder: {root}")
            return 0
        ctx.prior = _prepare_root(root, args)
        root.mkdir(parents=True, exist_ok=True)
        if plan.release:
            manifest.source_release = plan.release
        for n in plan.notes:
            log(f"  note: {n}")
        log(f"[{source.name}] {mode} {dataset} -> {root}")
        try:
            source.run(ctx, plan)
        except (DownloadError, UsageError, ValueError, OSError) as exc:
            manifest.errors.append(f"{type(exc).__name__}: {exc}")
            manifest.requests_made = client.requests_made
            manifest.finish()
            manifest.write(root)
            log(f"[{source.name}] FAILED: {exc}")
            log(f"[{source.name}] manifest (status={manifest.status}): {root / MANIFEST_NAME}")
            return 1
        manifest.requests_made = client.requests_made
        manifest.finish()
        manifest.write(root)
        _print_summary(source, manifest, root)
        return 0 if manifest.status in ("complete", "sample") else 1
    except UsageError as exc:
        log(f"error: {exc}")
        return 2
    except DownloadError as exc:
        log(f"error: {exc}")
        return 1


def _plain(v: Any) -> Any:
    """JSON-safe copy of a CLI value (paths become strings)."""
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, dict):
        return {k: _plain(x) for k, x in v.items()}
    return v


def _print_summary(source: Source, manifest: Manifest, root: Path) -> None:
    print(f"[{source.name}] status={manifest.status} files={len(manifest.files)} "
          f"requests={manifest.requests_made} -> {root}")
    if manifest.truncated:
        print(f"[{source.name}] TRUNCATED SAMPLE (not a complete dataset): "
              + "; ".join(manifest.truncation_reasons))
    for w in manifest.warnings:
        print(f"[{source.name}] warning: {w}")


def describe(source: Source) -> str:
    """Text for `structured_download.py info <source>`."""
    lines = [f"{source.name} -- {source.title} ({source.domain})", "", source.summary, ""]
    for mode, spec in source.modes.items():
        lines.append(f"mode {mode}: {spec.description}")
        lines.append(f"  formats: {', '.join(spec.formats)}"
                     + ("" if spec.bounded_by_limit else
                        "   (--limit not available: archives download whole)"))
        for ds, desc in spec.datasets.items():
            lines.append(f"  dataset {ds}: {desc}")
        lines.append("")
    if source.env:
        lines.append("credentials (environment variables / .env):")
        for e in source.env:
            req = "required" if e.required else "optional"
            lines.append(f"  {e.name} ({req}, {e.required_for}): {e.description}")
    else:
        lines.append("credentials: none")
    for n in source.notes:
        lines.append(f"note: {n}")
    if source.docs_urls:
        lines.append("docs: " + "  ".join(source.docs_urls))
    if source.terms_urls:
        lines.append("terms: " + "  ".join(source.terms_urls))
    if source.attribution:
        lines.append(f"attribution: {source.attribution}")
    lines.append(f"\nflags: structured_download.py {source.name} <mode> --help")
    return "\n".join(lines)


def paginate_cap(ctx: Context, records: int, pages: int) -> str | None:
    """Return a truncation reason if --limit/--max-pages has been reached."""
    limit = getattr(ctx.args, "limit", None)
    max_pages = getattr(ctx.args, "max_pages", None)
    if limit is not None and records >= limit:
        return f"--limit {limit} reached"
    if max_pages is not None and pages >= max_pages:
        return f"--max-pages {max_pages} reached"
    return None
