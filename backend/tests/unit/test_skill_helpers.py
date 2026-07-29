"""Cross-platform tests for the skill helper scripts (scripts/*.py).

These run on Linux, macOS, and Windows in CI to verify the portable building
blocks the build skills rely on — no bash, no `python3`-only commands. The star
is the durable-run harness: launching a detached job and reading its status must
work on every OS (Windows uses DETACHED_PROCESS + a Win32 liveness check).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]  # backend/tests/unit -> repo root
SCRIPTS = REPO / "scripts"
sys.path.insert(0, str(SCRIPTS))

import _skillutil  # noqa: E402
import build_state  # noqa: E402


def test_env_present_is_placeholder_aware(tmp_path):
    env = tmp_path / ".env"
    env.write_text("REAL=sk-abc123\nPLACE=your-key-...\nBLANK=\n", encoding="utf-8")
    assert _skillutil.env_present("REAL", env) is True
    assert _skillutil.env_present("PLACE", env) is False   # '...' placeholder
    assert _skillutil.env_present("BLANK", env) is False   # empty value
    assert _skillutil.env_present("ABSENT", env) is False  # not in file


def test_latest_version_dir_sorts_chronologically(tmp_path):
    out = tmp_path / "output_ontologies"
    out.mkdir()
    for name in (
        "v20260101T000000Z-merge",
        "v20260202T000000Z-merge",
        "v20260101T000000Z-prune-expand",
    ):
        (out / name).mkdir()
    assert _skillutil.latest_version_dir("merge", root=tmp_path).name == "v20260202T000000Z-merge"
    assert _skillutil.latest_version_dir("prune-expand", root=tmp_path).name == "v20260101T000000Z-prune-expand"
    assert _skillutil.latest_version_dir("missing", root=tmp_path) is None


def test_build_state_record_and_readback(tmp_path, monkeypatch):
    state = tmp_path / "state.jsonl"
    monkeypatch.setattr(build_state, "STATE", state)
    build_state.record_step("merge", {"path": "/x/y"})
    build_state.record_step("llm-mode", {"mode": "openai"})
    recs = [json.loads(line) for line in state.read_text(encoding="utf-8").splitlines()]
    assert [r["step"] for r in recs] == ["merge", "llm-mode"]
    assert recs[0]["details"]["path"] == "/x/y"


def _run(args: list[str], env: dict | None = None) -> subprocess.CompletedProcess:
    e = dict(os.environ)
    e.update(env or {})
    return subprocess.run(
        [sys.executable, *args], cwd=str(REPO), env=e,
        capture_output=True, text=True,
    )


def test_check_env_missing_key_exits_nonzero():
    r = _run([str(SCRIPTS / "check_env.py"), "DEFINITELY_NOT_A_REAL_KEY_9Z"])
    assert r.returncode == 1
    assert "MISSING" in r.stdout


@pytest.mark.parametrize("code,expect", [(0, "SUCCESS"), (3, "FAILED")])
def test_durable_harness_launch_and_status(tmp_path, code, expect):
    """Launch a detached job and poll its status to completion — the cross-platform
    heart of the harness (Windows exercises DETACHED_PROCESS + the Win32 pid check)."""
    env = {"YEGI_RUNS_DIR": str(tmp_path / "runs")}
    run_id = f"t_{expect.lower()}"
    prog = f"import sys; sys.exit({code})"

    launch = _run([str(SCRIPTS / "run_detached.py"), run_id, sys.executable, "-c", prog], env=env)
    assert launch.returncode == 0, launch.stderr

    reached = None
    for _ in range(40):  # up to ~20s
        status = _run([str(SCRIPTS / "job_status.py"), run_id], env=env)
        if expect in status.stdout:
            reached = expect
            break
        time.sleep(0.5)
    assert reached == expect, f"harness never reached {expect}"
