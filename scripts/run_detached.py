#!/usr/bin/env python3
"""Cross-platform durable-run harness: launch a command DETACHED so it survives
a killed Claude Code session. Works on Linux, macOS, and Windows.

This replaces the earlier setsid-based bash version, which only ran on Linux
(`setsid` is a Linux-only binary; macOS ships neither the CLI nor a bash that
has it, and native Windows has no bash at all). The detach here uses the OS's
own process-session primitives via Python's subprocess:

  - POSIX (Linux, macOS): start_new_session=True — the setsid *syscall*, which
    macOS has even though the `setsid` command does not. The child leads a new
    session with no controlling terminal, so the SIGHUP that fires when the
    launching session dies never reaches it.
  - Windows: DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP creation flags detach
    the child from the parent console/process group.

A tiny supervisor (this same script re-invoked with --supervise) runs the real
command in that detached session, then records its exit code — so "finished"
and "hard-killed" stay distinguishable exactly as before.

Usage:
    python3 scripts/run_detached.py <run_id> <command...>
    # (internal) python3 scripts/run_detached.py --supervise <run_id> <command...>

Per-run state lands in .runs/<run_id>/ (gitignored):
    cmd  started_at  ended_at  pid  status  log
`status` is RUNNING until the command exits, then the exit code; if the job is
hard-killed the supervisor dies before writing it, so it stays RUNNING — which
job_status.py reads (pid dead + status RUNNING) as DIED.
"""
from __future__ import annotations

import datetime
import os
import subprocess
import sys
from pathlib import Path


def _root() -> Path:
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


ROOT = _root()


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _runs_root() -> Path:
    # Env override lets tests isolate runs instead of touching the real .runs/.
    return Path(os.environ.get("YEGI_RUNS_DIR") or (ROOT / ".runs"))


def _run_dir(run_id: str) -> Path:
    return _runs_root() / run_id


def _launch(run_id: str, cmd: list[str]) -> int:
    if not run_id or "/" in run_id or "\\" in run_id or run_id.startswith("."):
        print("run_id must be a simple name (no '/', '\\', or leading '.')", file=sys.stderr)
        return 2
    if not cmd:
        print("usage: run_detached.py <run_id> <command...>", file=sys.stderr)
        return 2

    d = _run_dir(run_id)
    if d.exists():
        print(f"run id '{run_id}' already exists at {d} -- pick a fresh id", file=sys.stderr)
        return 3
    d.mkdir(parents=True)

    # subprocess.list2cmdline gives a readable, re-runnable rendering on both
    # POSIX and Windows without importing shlex only-on-POSIX.
    (d / "cmd").write_text(subprocess.list2cmdline(cmd) + "\n", encoding="utf-8")
    (d / "started_at").write_text(_now() + "\n", encoding="utf-8")
    (d / "status").write_text("RUNNING\n", encoding="utf-8")

    # Detach the supervisor from this (Claude) session.
    popen_kwargs: dict = {
        "cwd": str(ROOT),
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    if os.name == "nt":
        popen_kwargs["creationflags"] = (
            subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
        )
    else:
        popen_kwargs["start_new_session"] = True

    sup = subprocess.Popen(
        [sys.executable, os.path.abspath(__file__), "--supervise", run_id, *cmd],
        **popen_kwargs,
    )
    (d / "pid").write_text(f"{sup.pid}\n", encoding="utf-8")

    print(f"started run '{run_id}' (pid {sup.pid}) -- detached, survives this session")
    print(f"  log:    .runs/{run_id}/log")
    print(f"  status: .runs/{run_id}/status")
    print(f"  watch:  python3 scripts/job_status.py {run_id}")
    return 0


def _supervise(run_id: str, cmd: list[str]) -> int:
    """Runs in the detached session: execute the command, capture output to log,
    then record the exit code + end time."""
    d = _run_dir(run_id)
    code = 1
    try:
        with open(d / "log", "wb") as log:
            # PYTHONUNBUFFERED: stdout here is a FILE, not a tty, so CPython
            # block-buffers print() in ~4-8 KB chunks. In a detached run that
            # means a stage can emit progress for many minutes and the log stays
            # empty -- observed live: summarization ran for 11 minutes with zero
            # [evaluated-summary] lines while logging-based warnings (stderr,
            # unbuffered) streamed normally. "Silent" then looks identical to
            # "hung", which is what makes people kill healthy runs.
            #
            # Set on the environment rather than adding `-u` to the argv so it
            # applies to any python entrypoint, however the caller spelled it.
            env = {**os.environ, "PYTHONUNBUFFERED": "1"}
            proc = subprocess.Popen(
                cmd, cwd=str(ROOT), env=env,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            )
            code = proc.wait()
    except Exception as exc:  # command not found, etc.
        try:
            (d / "log").open("a", encoding="utf-8").write(f"\n[run_detached] launch error: {exc}\n")
        except Exception:
            pass
        code = 127
    (d / "status").write_text(f"{code}\n", encoding="utf-8")
    (d / "ended_at").write_text(_now() + "\n", encoding="utf-8")
    return 0


def main(argv: list[str]) -> int:
    if argv and argv[0] == "--supervise":
        if len(argv) < 3:
            print("usage: run_detached.py --supervise <run_id> <command...>", file=sys.stderr)
            return 2
        return _supervise(argv[1], argv[2:])
    if len(argv) < 2:
        print("usage: python3 scripts/run_detached.py <run_id> <command...>", file=sys.stderr)
        return 2
    return _launch(argv[0], argv[1:])


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
