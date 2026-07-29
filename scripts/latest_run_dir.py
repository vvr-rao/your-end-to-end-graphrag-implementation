#!/usr/bin/env python3
"""Print the newest output_ontologies/v*-<suffix>/ folder (or nothing if none).
Cross-platform replacement for `ls -dt output_ontologies/v*-<suffix>/ | head -1`.

    uv run python scripts/latest_run_dir.py merge
    uv run python scripts/latest_run_dir.py prune-expand
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _skillutil import latest_version_dir  # noqa: E402


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: latest_run_dir.py <suffix>  (e.g. merge | prune-expand)", file=sys.stderr)
        return 2
    d = latest_version_dir(argv[0])
    if d is None:
        return 1
    print(d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
