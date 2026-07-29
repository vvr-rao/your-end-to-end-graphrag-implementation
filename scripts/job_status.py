#!/usr/bin/env python3
"""Report on detached runs started by run_detached.py. Cross-platform.

This is how Claude re-attaches after a session death: it does not hold the
process, it reads this. With no run_id, lists every run and its state.

State is derived from (supervisor pid alive?) x (status file):
    status "RUNNING" + pid alive  -> RUNNING
    status "RUNNING" + pid dead   -> DIED   (killed/crashed before finishing)
    status "0"                    -> SUCCESS
    status "<non-zero>"           -> FAILED (exit <code>)

Usage:
    python3 scripts/job_status.py [run_id] [n_log_lines]
"""
from __future__ import annotations

import errno
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


# Env override lets tests isolate runs instead of touching the real .runs/.
RUNS = Path(os.environ.get("YEGI_RUNS_DIR") or (_root() / ".runs"))


def _pid_alive(pid: int) -> bool:
    """Portable 'is this pid alive'. POSIX uses signal 0; Windows uses the Win32
    API via ctypes (os.kill with signal 0 is unreliable there)."""
    if os.name == "nt":
        try:
            import ctypes
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            STILL_ACTIVE = 259
            k = ctypes.windll.kernel32  # type: ignore[attr-defined]
            h = k.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if not h:
                return False
            code = ctypes.c_ulong()
            ok = k.GetExitCodeProcess(h, ctypes.byref(code))
            k.CloseHandle(h)
            return bool(ok) and code.value == STILL_ACTIVE
        except Exception:
            return False
    try:
        os.kill(pid, 0)
    except OSError as e:
        return e.errno == errno.EPERM  # alive but not ours
    return True


def _read(path: Path, default: str = "") -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception:
        return default


def _state(d: Path) -> str:
    pid_s = _read(d / "pid")
    status = _read(d / "status", "?")
    alive = False
    if pid_s.isdigit():
        alive = _pid_alive(int(pid_s))
    if status == "RUNNING" and alive:
        return "RUNNING"
    if status == "RUNNING" and not alive:
        return "DIED (no exit code — killed/crashed)"
    if status == "0":
        return "SUCCESS"
    if status == "?":
        return "UNKNOWN (no status file)"
    return f"FAILED (exit {status})"


def main(argv: list[str]) -> int:
    run_id = argv[0] if argv else ""
    lines = int(argv[1]) if len(argv) > 1 and argv[1].isdigit() else 20

    if not run_id:
        if not RUNS.is_dir() or not any(RUNS.iterdir()):
            print("no runs yet (.runs/ is empty)")
            return 0
        print(f"{'RUN':<24} {'PID':<8} STATE")
        for d in sorted(RUNS.iterdir()):
            if d.is_dir():
                print(f"{d.name:<24} {_read(d / 'pid', '?'):<8} {_state(d)}")
        return 0

    d = RUNS / run_id
    if not d.is_dir():
        print(f"no such run: {run_id}", file=sys.stderr)
        return 1
    print(f"run:     {run_id}")
    print(f"state:   {_state(d)}")
    print(f"pid:     {_read(d / 'pid', '?')}")
    print(f"cmd:     {_read(d / 'cmd', '?')}")
    print(f"started: {_read(d / 'started_at', '?')}")
    ended = _read(d / "ended_at")
    if ended:
        print(f"ended:   {ended}")
    print(f"--- last {lines} log lines ---")
    log = d / "log"
    if log.exists():
        content = log.read_text(encoding="utf-8", errors="replace").splitlines()
        print("\n".join(content[-lines:]) if content else "(log empty)")
    else:
        print("(no log yet)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
