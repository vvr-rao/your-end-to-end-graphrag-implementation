"""A cached PDF must not start a table worker.

2026-10-23, FinQA (148 one-page PDFs, all cache hits): register-documents spent
632 of its 670 s starting one worker per PDF, each only to copy its cache entry
into the run cache -- ~10 min of a paid GPU deployment idling. The parent now
does that copy itself and spawns workers only for uncached PDFs."""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from backend.app.services import table_cache
from backend.app.services import table_extract as te


@pytest.fixture
def env(monkeypatch, tmp_path):
    user = tmp_path / "user"
    user.mkdir()
    monkeypatch.setattr(table_cache, "user_cache_dir", lambda: user)
    spawned: list[str] = []

    class _Proc:
        returncode = 0

    async def fake_exec(*cmd, **kw):
        spawned.append(Path(cmd[cmd.index("--pdf") + 1]).name)
        return _Proc()

    async def fake_pump(proc, p, *, max_silent_s):
        return 0

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(te, "_pump_worker", fake_pump)
    return tmp_path, user, spawned


def _pdf(folder: Path, name: str) -> Path:
    p = folder / name
    p.write_bytes(b"%PDF-1.4 " + name.encode())
    return p


def _cache(user: Path, pdf: Path, n_tables: int) -> str:
    doc_sha, key = te._hash_pdf_streaming(pdf)
    table_cache.save(user, key, doc_sha=doc_sha, doc_path=str(pdf),
                     tables=[{"@id": f"t{i}"} for i in range(n_tables)],
                     manifest={"n_tables": n_tables, "cost_usd": 0.01, "source": "fresh"})
    return key


def _drive(pdfs, run_cache):
    return asyncio.run(te._drive_subprocess_workers(
        pdfs, run_cache_dir=run_cache, use_vision=True, concurrency=4))


def test_cached_pdf_spawns_no_worker_and_reaches_the_run_cache(env, capsys) -> None:
    tmp, user, spawned = env
    cached, fresh = _pdf(tmp, "cached.pdf"), _pdf(tmp, "fresh.pdf")
    key = _cache(user, cached, n_tables=2)
    run = tmp / "run"
    out = _drive([cached, fresh], run)
    assert spawned == ["fresh.pdf"]
    assert (run / f"{key}.jsonld").exists(), "run cache must still hold every PDF"
    assert out[str(cached)]["source"] == "cache"
    assert out[str(cached)]["n_tables"] == 2
    assert "1 cache-hits" in capsys.readouterr().out


def test_all_cached_spawns_nothing(env) -> None:
    tmp, user, spawned = env
    pdfs = [_pdf(tmp, f"p{i}.pdf") for i in range(5)]
    for p in pdfs:
        _cache(user, p, n_tables=1)
    _drive(pdfs, tmp / "run")
    assert spawned == []


def test_no_run_cache_dir_still_skips_the_worker(env) -> None:
    tmp, user, spawned = env
    p = _pdf(tmp, "a.pdf")
    _cache(user, p, n_tables=1)
    out = _drive([p], None)
    assert spawned == [] and out[str(p)]["source"] == "cache"
