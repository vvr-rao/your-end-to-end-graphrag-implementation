#!/usr/bin/env python3
"""Presence-check .env keys WITHOUT ever printing their values. Cross-platform
replacement for the bash `grep -q` loops in the skills.

    uv run python scripts/check_env.py OPENAI_API_KEY ANTHROPIC_API_KEY
    uv run python scripts/check_env.py --optional LANGSMITH_API_KEY HF_TOKEN --config langsmith.user

Prints one line per key (present / MISSING), and exits non-zero if any REQUIRED
key is missing — so a skill can gate a paid step on it. A key set to a blank
value or a '...placeholder...' counts as MISSING.

`--optional` marks the keys after it as optional: they are reported (present /
not set (optional)) but never make the exit code non-zero. `--config KEY`
reports whether a dotted key in config/config.yaml (or $APP_CONFIG_PATH) has a
value -- config.yaml holds no secrets, but only set / not set is printed.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _skillutil import ROOT, env_present  # noqa: E402


def config_present(dotted: str, path: Path | None = None) -> bool:
    """True when the dotted key in the app config has a non-blank value."""
    import yaml

    p = path or Path(os.environ.get("APP_CONFIG_PATH") or ROOT / "config" / "config.yaml")
    if not p.exists():
        return False
    try:
        node = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return False
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return False
        node = node[part]
    return node is not None and str(node).strip() != ""


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: check_env.py KEY [KEY ...] [--optional KEY ...] [--config a.b ...]",
              file=sys.stderr)
        return 2
    missing: list[str] = []
    optional_missing: list[str] = []
    mode = "required"
    for arg in argv:
        if arg in ("--optional", "--config"):
            mode = arg[2:]
            continue
        if mode == "config":
            ok = config_present(arg)
            print(f"  {arg} (config.yaml): {'set' if ok else 'not set'}")
            if not ok:
                optional_missing.append(arg)
            continue
        ok = env_present(arg)
        if mode == "optional":
            print(f"  {arg}: {'present' if ok else 'not set (optional)'}")
            if not ok:
                optional_missing.append(arg)
        else:
            print(f"  {arg}: {'present' if ok else 'MISSING'}")
            if not ok:
                missing.append(arg)
    if optional_missing:
        print(f"Optional, not set: {', '.join(optional_missing)} -- only needed for the "
              f"features that use them.")
    if missing:
        print(f"MISSING: {', '.join(missing)} — add them to .env yourself "
              f"(never paste keys into the chat).")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
