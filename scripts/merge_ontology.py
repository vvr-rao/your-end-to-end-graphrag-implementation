#!/usr/bin/env python3
"""Merge the 7 pinned core ontologies with a chosen domain ontology (or the user's
own, or none), record the merge, and print the resulting version-folder path.

Cross-platform replacement for the bash array + fetch + `merge` + `ls|head` block.

    uv run python scripts/merge_ontology.py --domain pharma
    uv run python scripts/merge_ontology.py --ontology /path/to/mine.owl
    uv run python scripts/merge_ontology.py --core-only

The 7 core ontologies are ALWAYS included. Exactly one source mode is required:
--domain (pharma|finance|manufacturing, fetched via fetch_ontology.py), --ontology
(one or more of the user's own .owl/.rdf/.ttl/.xml/.zip files), or --core-only.
Prints the merge folder as its LAST line so a caller can capture it.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _skillutil import ROOT, latest_version_dir  # noqa: E402
import build_state  # noqa: E402

CORE = [
    "source_ontologies/core_ontologies/viao_intelligence_artifact_ontology_v2.owl",
    "source_ontologies/core_ontologies/foaf.rdf",
    "source_ontologies/core_ontologies/org.ttl",
    "source_ontologies/core_ontologies/geography_ontology.owl",
    "source_ontologies/core_ontologies/time.ttl",
    "source_ontologies/core_ontologies/skos.rdf",
    "source_ontologies/core_ontologies/domain_concepts.owl",
]


def _resolve_domain(args: argparse.Namespace) -> list[str]:
    if args.domain:
        sys.path.insert(0, str(ROOT / "source_ontologies"))
        from fetch_ontology import fetch  # noqa: E402
        return [str(fetch(args.domain))]
    if args.ontology:
        out = []
        for p in args.ontology:
            path = Path(p)
            if not path.exists():
                print(f"ERROR: ontology file not found: {p}", file=sys.stderr)
                raise SystemExit(1)
            out.append(str(path))
        return out
    return []  # --core-only


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Merge core + domain ontologies.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--domain", choices=["pharma", "finance", "manufacturing"])
    g.add_argument("--ontology", action="append", help="Path to your own ontology (repeatable).")
    g.add_argument("--core-only", action="store_true")
    ap.add_argument("--output-dir", default="output_ontologies")
    args = ap.parse_args(argv)

    extra = _resolve_domain(args)
    ontologies = [str(ROOT / c) for c in CORE] + extra

    cli_args: list[str] = []
    for f in ontologies:
        cli_args += ["--ontology", f]
    cmd = [sys.executable, "-m", "backend.app.cli", "merge", *cli_args,
           "--output-dir", args.output_dir]

    print(f"[merge] {len(ontologies)} ontologies (7 core + {len(extra)} domain)")
    proc = subprocess.run(cmd, cwd=str(ROOT))
    if proc.returncode != 0:
        print("[merge] merge failed — see the error above.", file=sys.stderr)
        return proc.returncode

    merge_dir = latest_version_dir("merge")
    if merge_dir is None:
        print("[merge] merge reported success but no v*-merge folder found.", file=sys.stderr)
        return 1
    build_state.record_step("merge", {"path": str(merge_dir)})
    print(f"[merge] recorded. Open {merge_dir}/merged.owl in Protégé to verify.")
    print(merge_dir)  # LAST line: the caller captures this
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
