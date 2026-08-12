"""Every LLM-bound stage resolves its concurrency from config, not a hardcoded 4.

Before this, only register-documents read config. extract-entities,
generate-artifacts, regenerate-stale-artifacts and evaluate-queries each had
`default=4` in argparse, so config was never consulted and those stages sat at
4 no matter what the user configured -- adding hours to a large run while using
a fraction of a percent of the account's throughput.
"""

from __future__ import annotations

import argparse

import pytest

from backend.app.cli.main import _resolve_concurrency

STAGES = ["summarization", "entity_extraction", "artifact_generation", "evaluation"]


class _Settings:
    def __init__(self, cfg: dict) -> None:
        self.app_config = cfg


@pytest.fixture
def cfg(monkeypatch):
    """Patch the settings the resolver reads, without touching the real file."""

    def _apply(config: dict):
        import backend.app.core.config as C

        monkeypatch.setattr(C, "get_settings", lambda: _Settings(config))
        return config

    return _apply


@pytest.mark.parametrize("stage", STAGES)
def test_cli_flag_always_wins(stage, cfg) -> None:
    cfg({"concurrency": {s: 16 for s in STAGES}})
    assert _resolve_concurrency(argparse.Namespace(concurrency=64), stage) == 64


@pytest.mark.parametrize("stage", STAGES)
def test_reads_per_stage_config(stage, cfg) -> None:
    cfg({"concurrency": {s: (7 if s == stage else 1) for s in STAGES}})
    assert _resolve_concurrency(argparse.Namespace(concurrency=None), stage) == 7


def test_stages_are_independent(cfg) -> None:
    """The whole point: a cheap mini-model stage must not be pinned to the
    value that is safe for gpt-4.1 at 32k max_tokens."""
    cfg({"concurrency": {"summarization": 32, "entity_extraction": 32,
                         "artifact_generation": 32, "evaluation": 4},
         "expansion": {"max_concurrent_llm_calls": 4}})
    none = argparse.Namespace(concurrency=None)
    assert _resolve_concurrency(none, "summarization") == 32
    assert _resolve_concurrency(none, "evaluation") == 4


def test_legacy_summarization_key_still_honoured(cfg) -> None:
    """Configs written before the `concurrency:` block must keep working."""
    cfg({"summarization": {"concurrency": 12},
         "expansion": {"max_concurrent_llm_calls": 4}})
    assert _resolve_concurrency(argparse.Namespace(concurrency=None), "summarization") == 12


@pytest.mark.parametrize("stage", STAGES)
def test_falls_back_to_expansion_cap(stage, cfg) -> None:
    cfg({"expansion": {"max_concurrent_llm_calls": 6}})
    assert _resolve_concurrency(argparse.Namespace(concurrency=None), stage) == 6


@pytest.mark.parametrize("stage", STAGES)
def test_final_fallback_is_the_old_default(stage, cfg) -> None:
    cfg({})
    assert _resolve_concurrency(argparse.Namespace(concurrency=None), stage) == 4


def test_missing_attr_is_treated_as_unset(cfg) -> None:
    cfg({"concurrency": {"entity_extraction": 9}})
    assert _resolve_concurrency(argparse.Namespace(), "entity_extraction") == 9


def test_shipped_config_defines_every_stage() -> None:
    """A stage missing from config.example.yaml would silently fall back to 4."""
    import yaml

    example = yaml.safe_load(open("config/config.example.yaml"))
    block = example.get("concurrency") or {}
    missing = [s for s in STAGES if s not in block]
    assert not missing, f"config.example.yaml is missing concurrency keys: {missing}"
