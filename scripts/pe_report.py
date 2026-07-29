#!/usr/bin/env python3
"""Confirm a prune-expand run finished, print its cost, and record it. Cross-platform
replacement for the bash sentinel check (`ls {stats,manifest,cost}.json`) + inline
python cost print + tracker record.

    uv run python scripts/pe_report.py               # newest v*-prune-expand/
    uv run python scripts/pe_report.py --dir <path>

A prune-expand folder gets stats.json / manifest.json / cost.json ONLY on success,
so their presence is the completion sentinel. Prints the folder as its last line.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _skillutil import latest_version_dir  # noqa: E402
import build_state  # noqa: E402

SENTINELS = ("stats.json", "manifest.json", "cost.json")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=None, help="prune-expand version folder (default: newest)")
    args = ap.parse_args(argv)

    pe = Path(args.dir) if args.dir else latest_version_dir("prune-expand")
    if pe is None or not pe.is_dir():
        print("[pe] no prune-expand version folder found.", file=sys.stderr)
        return 1

    missing = [s for s in SENTINELS if not (pe / s).exists()]
    if missing:
        print(f"[pe] NOT complete — missing {', '.join(missing)} in {pe}. "
              f"The run may still be going or may have died; check job_status.py.")
        print(pe)
        return 1

    print(f"[pe] COMPLETE: {pe}")
    try:
        cost = json.loads((pe / "cost.json").read_text(encoding="utf-8"))
        print("[pe] cost report:", json.dumps(cost, indent=2))
    except Exception:
        pass
    build_state.record_step("prune-expand", {"path": str(pe)})
    print(pe)  # LAST line
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
