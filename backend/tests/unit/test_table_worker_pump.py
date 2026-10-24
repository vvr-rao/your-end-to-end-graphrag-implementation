"""_pump_worker never waits forever on a table worker.

2026-10-23: register-documents sat 25 minutes after its third worker had
exited -- no EOF, no exit status, no child process left. These tests drive
real subprocesses through the pump with the poll interval shrunk."""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

import pytest

from backend.app.services import table_extract as te


def _run(code: str, *, max_silent_s: float = 60.0) -> tuple[int, float]:
    async def go() -> tuple[int, float]:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-c", code,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        )
        t0 = time.monotonic()
        rc = await te._pump_worker(proc, Path("x.pdf"), max_silent_s=max_silent_s)
        return rc, time.monotonic() - t0

    return asyncio.run(go())


@pytest.fixture(autouse=True)
def fast_poll(monkeypatch):
    monkeypatch.setattr(te, "_WORKER_POLL_S", 0.2)
    monkeypatch.setattr(te, "_WORKER_EXIT_WAIT_S", 2.0)


def test_normal_worker_output_is_forwarded(capsys):
    rc, _ = _run("print('[worker] result: n_tables=3'); import sys; sys.exit(0)")
    assert rc == 0
    assert "[worker] result: n_tables=3" in capsys.readouterr().out


def test_failing_worker_reports_its_exit_code():
    rc, _ = _run("import sys; sys.exit(2)")
    assert rc == 2


def test_live_but_silent_worker_is_killed(capsys):
    rc, took = _run("import time; time.sleep(30)", max_silent_s=0.6)
    assert rc != 0 and took < 10
    assert "killing the worker" in capsys.readouterr().out


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="/proc liveness check is Linux-only")
def test_exited_worker_whose_output_never_closes_does_not_hang(capsys):
    # The worker exits at once, but a grandchild inherited its stdout and
    # keeps it open for 20 s -- EOF never arrives while the pump waits.
    code = ("import subprocess, sys; "
            "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(20)']); "
            "print('[worker] result: n_tables=0', flush=True); sys.exit(0)")
    rc, took = _run(code)
    assert took < 10, "pump waited for EOF from an exited worker"
    assert rc == 0
    assert "exited without closing its output" in capsys.readouterr().out


def test_max_silent_default_and_override(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(te, "get_settings", lambda: SimpleNamespace(app_config={}))
    assert te._worker_max_silent_s() == te._DEFAULT_WORKER_MAX_SILENT_S
    monkeypatch.setattr(te, "get_settings", lambda: SimpleNamespace(
        app_config={"tables": {"worker_max_silent_seconds": 42}}))
    assert te._worker_max_silent_s() == 42.0
