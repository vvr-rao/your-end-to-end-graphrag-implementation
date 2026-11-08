"""Download public structured (tabular / JSON / Parquet) datasets for YEGI.

Finance: SEC EDGAR, FRED/ALFRED, GLEIF, FFIEC / Federal Reserve NIC.
Pharma:  ClinicalTrials.gov / AACT, Drugs@FDA, FDA Orange Book, ChEMBL,
         Open Targets.

Each run writes provider-native files (and, with ``--format csv``,
normalized CSV alongside them) plus a ``manifest.json`` into
``<output>/<dataset>/<snapshot>/``. Small bounded samples are the default
way to use it; whole-dataset downloads need an explicit bulk selection.

Usage:
    uv run python source_documents/structured_download.py list
    uv run python source_documents/structured_download.py info sec
    uv run python source_documents/structured_download.py sec api --cik 320193 --format csv
    uv run python source_documents/structured_download.py sec bulk \\
        --dataset fsds --period 2024q1 --dry-run

See source_documents/structured/README.md for every source's flags,
credentials and limitations.
"""

from __future__ import annotations

import argparse
import sys

from structured._base import (
    SOURCES,
    add_common_arguments,
    describe,
    execute,
    load_env,
)
from structured.sources import load_all


def build_parser() -> argparse.ArgumentParser:
    load_all()
    p = argparse.ArgumentParser(
        prog="structured_download",
        description="Download selected public finance / pharma structured datasets "
                    "with a manifest per run.",
    )
    sub = p.add_subparsers(dest="command", required=True, metavar="{list,info,<source>}")
    sub.add_parser("list", help="List sources, modes and formats.")
    info = sub.add_parser("info", help="Show a source's datasets, credentials and notes.")
    info.add_argument("source", choices=sorted(SOURCES))
    for name, src in SOURCES.items():
        sp = sub.add_parser(name, help=f"{src.title} ({src.domain})",
                            description=src.summary)
        modes = sp.add_subparsers(dest="mode", required=True)
        for mode, spec in src.modes.items():
            mp = modes.add_parser(mode, help=spec.description, description=spec.description)
            single = len(spec.datasets) == 1 and not spec.open_datasets
            mp.add_argument("--dataset",
                            choices=None if spec.open_datasets else list(spec.datasets),
                            default=next(iter(spec.datasets)) if single else None,
                            help="; ".join(f"{k}: {v}" for k, v in spec.datasets.items()))
            src.add_arguments(mode, mp)
            add_common_arguments(mp, src, mode)
    return p


def _list() -> str:
    rows = []
    for name in sorted(SOURCES, key=lambda n: (SOURCES[n].domain, n)):
        src = SOURCES[name]
        modes = ", ".join(f"{m}[{'/'.join(s.formats)}]" for m, s in src.modes.items())
        creds = ", ".join(e.name + ("" if e.required else "?") for e in src.env) or "-"
        rows.append(f"  {name:<15} {src.domain:<8} {modes:<34} creds: {creds}")
    return ("sources (mode[formats]; creds ending in ? are optional):\n" + "\n".join(rows)
            + "\n\nNext: structured_download.py info <source>")


def main(argv: list[str] | None = None) -> int:
    load_env()
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "list":
        print(_list())
        return 0
    if args.command == "info":
        print(describe(SOURCES[args.source]))
        return 0
    src = SOURCES[args.command]
    if args.dataset is None:
        parser.error(f"{src.name} {args.mode}: --dataset is required "
                     f"({', '.join(src.modes[args.mode].datasets)})")
    return execute(src, args.mode, args)


if __name__ == "__main__":
    sys.exit(main())
