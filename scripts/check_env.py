#!/usr/bin/env python3
"""Presence-check .env keys WITHOUT ever printing their values. Cross-platform
replacement for the bash `grep -q` loops in the skills.

    uv run python scripts/check_env.py OPENAI_API_KEY ANTHROPIC_API_KEY

Prints one line per key (present / MISSING), and exits non-zero if any are
missing — so a skill can gate a paid step on it. A key set to a blank value or a
'...placeholder...' counts as MISSING.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _skillutil import env_present  # noqa: E402


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: check_env.py KEY [KEY ...]", file=sys.stderr)
        return 2
    missing: list[str] = []
    for key in argv:
        ok = env_present(key)
        print(f"  {key}: {'present' if ok else 'MISSING'}")
        if not ok:
            missing.append(key)
    if missing:
        print(f"MISSING: {', '.join(missing)} — add them to .env yourself "
              f"(never paste keys into the chat).")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
