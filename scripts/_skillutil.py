"""Shared utilities for the cross-platform skill helpers.

The build skills used to embed bash (arrays, grep, ls|head). That broke on native
Windows and depended on a bash shell everywhere. These helpers move that logic into
Python so the skills only ever invoke `uv run python scripts/<helper>.py` — one
command that behaves identically on Linux, macOS, and Windows.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path


def repo_root() -> Path:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        if out:
            return Path(out)
    except Exception:
        pass
    return Path.cwd()


ROOT = repo_root()

# A value that is blank or still a "...placeholder..." counts as NOT configured.
_PLACEHOLDER = re.compile(r"\.\.\.")


def env_present(key: str, env_path: Path | None = None) -> bool:
    """True iff `key` is set to a real value in .env — not blank, not a placeholder.

    Reads the file directly and returns only a boolean; the VALUE is never returned
    or printed, so this is safe to use under the credential firewall.
    """
    p = env_path or (ROOT / ".env")
    if not p.exists():
        return False
    try:
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return False
    for line in lines:
        s = line.strip()
        if s.startswith(f"{key}="):
            val = s[len(key) + 1:].strip().strip('"').strip("'")
            return bool(val) and not _PLACEHOLDER.search(val)
    return False


def latest_version_dir(suffix: str, root: Path | None = None) -> Path | None:
    """Newest output_ontologies/v<TS>-<suffix>/ folder.

    Version folders are named v<UTC-ISO-timestamp>-<subcommand>, so a lexical sort
    on the name is a chronological sort — deterministic and cross-platform (unlike
    `ls -dt`, which depends on filesystem mtime)."""
    base = (root or ROOT) / "output_ontologies"
    if not base.is_dir():
        return None
    dirs = sorted(
        (d for d in base.glob(f"v*-{suffix}") if d.is_dir()),
        key=lambda d: d.name,
    )
    return dirs[-1] if dirs else None
