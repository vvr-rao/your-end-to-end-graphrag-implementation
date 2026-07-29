#!/usr/bin/env python3
"""Cross-session build-progress tracker for the guided app build.

Records which pipeline steps the user has completed (and where each artifact
landed) in `.build/state.jsonl`, so a NEW Claude Code session -- even after a
restart that wiped the conversation -- can tell the user what they did last and
where to continue.

  python3 scripts/build_state.py record <step> [key=value ...] [--note "text"]
  python3 scripts/build_state.py show  [--quiet-if-empty]

`record` appends one line; the log is append-only so history survives. `show`
prints the recent history, the last operation (+ its artifact path), and the
suggested next step derived from the canonical pipeline order.

The SessionStart hook runs `show --quiet-if-empty` so every fresh session is
reminded of progress automatically. `--quiet-if-empty` prints nothing on a clean
clone so it never adds noise before any work has been done.
"""
from __future__ import annotations

import datetime
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Env override lets tests isolate the tracker instead of touching the real one.
STATE = Path(os.environ.get("YEGI_STATE_FILE") or (ROOT / ".build" / "state.jsonl"))

# Canonical pipeline order (step key -> human label). Used to suggest "next".
ORDER: list[tuple[str, str]] = [
    ("llm-mode", "choose the LLM provider mode"),
    ("database", "set up the database"),
    ("merge", "merge the ontologies"),
    ("prune-expand", "build the ontology from documents (prune-expand)"),
    ("db-init", "load the ontology into the database"),
    ("register-documents", "chunk + embed the corpus"),
    ("extract-entities", "extract entities + relationships"),
    ("enrich-time", "temporal enrichment"),
    ("generate-artifacts", "generate intelligence artifacts"),
    ("deploy", "deploy to Render"),
]
ORDER_KEYS = [k for k, _ in ORDER]
LABELS = dict(ORDER)


def _records() -> list[dict]:
    if not STATE.exists():
        return []
    out: list[dict] = []
    for line in STATE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # tolerate a corrupt line rather than crash the hook
    return out


def record_step(step: str, details: dict | None = None, note: str | None = None) -> dict:
    """Append one step record to the tracker. Importable by the skill helpers so
    they record themselves directly instead of shelling out."""
    rec: dict = {
        "ts": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "step": step,
        "details": details or {},
    }
    if note:
        rec["note"] = note
    STATE.parent.mkdir(parents=True, exist_ok=True)
    with STATE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


def record(argv: list[str]) -> None:
    if not argv:
        sys.exit("usage: build_state.py record <step> [key=value ...] [--note text]")
    step = argv[0]
    details: dict[str, str] = {}
    note: str | None = None
    i = 1
    while i < len(argv):
        a = argv[i]
        if a == "--note":
            note = argv[i + 1] if i + 1 < len(argv) else None
            i += 2
            continue
        if "=" in a:
            k, v = a.split("=", 1)
            details[k] = v
        i += 1
    record_step(step, details, note)
    print(f"[build-state] recorded: {step}" + (f"  {details}" if details else ""))


def show(argv: list[str]) -> None:
    quiet_if_empty = "--quiet-if-empty" in argv
    recs = _records()
    if not recs:
        if not quiet_if_empty:
            print('No build steps recorded yet. Say "where do I start" to begin.')
        return

    print("Build progress (this working copy):")
    for r in recs[-12:]:
        d = r.get("details") or {}
        extra = ("  " + "  ".join(f"{k}={v}" for k, v in d.items())) if d else ""
        note = f"  — {r['note']}" if r.get("note") else ""
        print(f"  [{r['ts']}] {r['step']}{extra}{note}")

    last = recs[-1]
    print(f"\nLast operation: {last['step']} at {last['ts']}")
    ld = last.get("details") or {}
    if ld.get("path"):
        print(f"  artifact: {ld['path']}")

    done = {r["step"] for r in recs}
    nxt = next((k for k in ORDER_KEYS if k not in done), None)
    if nxt:
        print(f"Suggested next: {nxt} — {LABELS[nxt]}")
    else:
        print("All tracked steps recorded. You can query/serve the app, or redeploy.")


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("usage: build_state.py {record|show} ...")
    cmd = sys.argv[1]
    if cmd == "record":
        record(sys.argv[2:])
    elif cmd == "show":
        show(sys.argv[2:])
    else:
        sys.exit(f"unknown command: {cmd!r} (expected record|show)")


if __name__ == "__main__":
    main()
