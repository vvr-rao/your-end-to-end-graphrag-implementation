"""Memory detection must work on Linux, macOS and Windows.

The run-sizing advice (how many table-extraction subprocesses fit) is derived
from free memory, and the repo supports all three platforms. Only Linux can be
executed in CI here, so the platform-specific PARSERS are tested against
captured real output instead.

The macOS case is the one that matters most: counting only "Pages free"
reports a few hundred MB on a machine with gigabytes available, because macOS
keeps most RAM in `inactive` (evictable). That would advise
`table_extraction: 1` on a Mac that could run 20.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "tpm_check", Path(__file__).resolve().parents[3] / "scripts" / "tpm_check.py"
)
tpm = importlib.util.module_from_spec(_SPEC)
sys.modules["tpm_check"] = tpm
_SPEC.loader.exec_module(tpm)  # type: ignore[union-attr]


# Real `vm_stat` output from an Apple Silicon Mac (16 KB pages), trimmed.
_VM_STAT = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                               45000.
Pages active:                            180000.
Pages inactive:                           60000.
Pages speculative:                         5000.
Pages throttled:                              0.
Pages wired down:                         90000.
Pages purgeable:                           1200.
"""


def test_macos_counts_evictable_pages_not_just_free() -> None:
    """free(45k) + inactive(60k) + speculative(5k) = 110k pages x 16 KB."""
    mb = tpm.parse_vm_stat(_VM_STAT, 16384)
    assert mb == (110_000 * 16384) // (1024 * 1024) == 1718


def test_macos_free_only_would_badly_understate() -> None:
    """Guards the specific mistake: 'Pages free' alone is 2.4x too low here,
    and on a busy Mac the gap is far wider."""
    free_only = (45_000 * 16384) // (1024 * 1024)
    assert tpm.parse_vm_stat(_VM_STAT, 16384) > free_only * 2


def test_macos_handles_4k_pages() -> None:
    """Intel Macs use 4 KB pages; the page size is parsed, not assumed."""
    assert tpm.parse_vm_stat(_VM_STAT, 4096) == (110_000 * 4096) // (1024 * 1024)


def test_macos_tolerates_missing_categories() -> None:
    """Older/newer vm_stat variants may omit a line; it must not raise."""
    assert tpm.parse_vm_stat("Pages free:  1000.\n", 4096) == (1000 * 4096) // (1024 * 1024)


def test_parser_survives_junk() -> None:
    assert tpm.parse_vm_stat("not vm_stat output at all", 4096) == 0


def test_mem_mb_returns_a_plausible_pair_on_this_platform() -> None:
    avail, total = tpm._mem_mb()
    if avail < 0:
        pytest.skip("memory probe unsupported on this platform")
    assert 0 < avail <= total, f"available {avail} MB, total {total} MB"
    assert total > 256, "implausibly small total memory"


def test_advice_degrades_gracefully_when_memory_is_unknown(monkeypatch, capsys) -> None:
    """On an unrecognised platform the tool must say so, not crash or silently
    print nothing -- the skill relies on this output existing."""
    monkeypatch.setattr(tpm, "_mem_mb", lambda: (-1, -1))
    tpm.advise_memory(batch_size=8, table_conc=1)
    assert "skipping memory-based advice" in capsys.readouterr().out


def test_swap_absence_is_reported_as_a_hard_kill_risk(monkeypatch, capsys) -> None:
    monkeypatch.setattr(tpm, "_mem_mb", lambda: (2000, 4000))
    monkeypatch.setattr(tpm, "_has_swap", lambda: False)
    tpm.advise_memory(batch_size=8, table_conc=1)
    out = capsys.readouterr().out
    assert "swap: NONE" in out
    assert "HARD KILL" in out


def test_recommendation_scales_with_available_memory(monkeypatch, capsys) -> None:
    """More free RAM must recommend more table workers, and a tight box must
    not be told to run several."""
    monkeypatch.setattr(tpm, "_has_swap", lambda: True)

    monkeypatch.setattr(tpm, "_mem_mb", lambda: (800, 4000))
    tpm.advise_memory(batch_size=8, table_conc=1)
    tight = capsys.readouterr().out

    monkeypatch.setattr(tpm, "_mem_mb", lambda: (8000, 16000))
    tpm.advise_memory(batch_size=8, table_conc=1)
    roomy = capsys.readouterr().out

    def rec(text: str) -> int:
        line = next(l for l in text.splitlines() if "table_extraction" in l)
        return int(line.rsplit("suggest", 1)[1].strip())

    assert rec(tight) == 1, "a 800 MB box must not be told to run several workers"
    assert rec(roomy) >= 6, "an 8 GB box should be told it can parallelise"
