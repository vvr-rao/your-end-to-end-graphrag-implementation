"""The concurrency gate sizes each STAGE from the models that stage really calls.

It used to bucket models by a TPM threshold (>= 5M = "mini tier"). On an
account where gpt-4.1 has 30M TPM that put every model in the mini bucket, so
class_proposal / dedup got no suggestion at all -- and the mini suggestion was
applied to extract-entities even though its orphan check runs on gpt-4.1. In
Anthropic mode it probed OpenAI models, because it only looked at OpenAI tasks.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
_SPEC = importlib.util.spec_from_file_location("tpm_check", ROOT / "scripts" / "tpm_check.py")
tpm = importlib.util.module_from_spec(_SPEC)
sys.modules["tpm_check"] = tpm
_SPEC.loader.exec_module(tpm)  # type: ignore[union-attr]

PRESETS = [
    "config/models.example.yaml",
    "config/models.openai.example.yaml",
    "config/models.anthropic.example.yaml",
]

# The modules whose LLM calls run under each stage's semaphore.
STAGE_MODULES = {
    "entity_extraction": ["backend/app/services/db_entity_extract.py"],
    "artifact_generation": [
        "backend/app/services/db_artifact_gen.py",
        "backend/app/services/db_artifact_rollup.py",
        "backend/app/services/db_insight_gen.py",
    ],
    "summarization": ["backend/app/services/evaluated_summarizer.py"],
    "evaluation": ["backend/app/services/eval_judge.py"],
}


def _tasks(preset: str) -> dict:
    return yaml.safe_load((ROOT / preset).read_text())["tasks"]


# --------------------------------------------------------------------------
# The mapping must not drift from models.yaml or from the code.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("preset", PRESETS)
def test_every_mapped_task_is_routed_in_every_preset(preset):
    routed = _tasks(preset)
    missing = [t for ts in tpm.STAGE_TASKS.values() for t in ts if t not in routed]
    assert not missing, f"{preset} does not route: {missing}"


@pytest.mark.parametrize("stage", sorted(STAGE_MODULES))
def test_every_task_a_stage_calls_is_mapped_to_it(stage):
    """A task missing from STAGE_TASKS is a model the gate never checks, so a
    stage could be sized from a loose limit while one of its calls sits on a
    tight one -- exactly how extract-entities' gpt-4.1 orphan check was missed."""
    all_tasks = set(_tasks("config/models.example.yaml"))
    used: set[str] = set()
    for mod in STAGE_MODULES[stage]:
        src = (ROOT / mod).read_text()
        used |= {t for t in all_tasks if re.search(r"[\"']%s[\"']" % re.escape(t), src)}
        # rollup builds f"{_fam}_evaluate" / f"{_fam}_revise"
        for fam in re.findall(r"_fam = \"(\w+)\"", src) + re.findall(r"else \"(\w+)\"", src):
            used |= {f"{fam}_evaluate", f"{fam}_revise"} & all_tasks
    unmapped = sorted(used - set(tpm.STAGE_TASKS[stage]))
    assert not unmapped, f"{stage} calls {unmapped} but STAGE_TASKS omits them"


# --------------------------------------------------------------------------
# suggest(): each provider's limit shape
# --------------------------------------------------------------------------

def test_unknown_limits_are_not_treated_as_unlimited():
    assert tpm.suggest({}, 8192) is None
    assert tpm.suggest({"rpm": 10_000}, 8192) is None


def test_openai_combined_tpm_and_rpm():
    # 10M TPM at 25% over 20k-token calls -> 125; RPM 500 at 25% -> 125.
    assert tpm.suggest({"tpm": 10_000_000, "rpm": 500}, 8_000) == 125
    # A tight RPM binds even with TPM to spare.
    assert tpm.suggest({"tpm": 150_000_000, "rpm": 40}, 8_000) == 10


def test_anthropic_meters_input_and_output_separately():
    lim = {"itpm": 2_000_000, "otpm": 400_000, "rpm": 4_000}
    # input: 2M*.25/12k = 41; output: 400k*.25/4096 = 24 -> output binds
    assert tpm.suggest(lim, 4_096) == 24


def test_the_suggestion_is_capped():
    assert tpm.suggest({"tpm": 10**12, "rpm": 10**9}, 1_000) == tpm.MAX_SUGGESTION


def test_large_output_calls_are_capped_even_with_tpm_to_spare():
    """class_proposal / match_dedup (32k output on gpt-4.1): TPM alone says 128
    on a 30M account, but concurrent large calls throttle."""
    assert tpm.suggest({"tpm": 30_000_000, "rpm": 10_000}, 32_768) == tpm.LARGE_CALL_CAP


# --------------------------------------------------------------------------
# stage_suggestions(): a stage is bound by its TIGHTEST task
# --------------------------------------------------------------------------

def test_a_mixed_stage_is_bound_by_its_tight_model(monkeypatch):
    specs = {
        "entity_extract": {"provider": "openai", "model": "mini", "max_tokens": 8192},
        "relationship_orphan_check": {"provider": "openai", "model": "big",
                                      "max_tokens": 8192},
    }
    monkeypatch.setattr(tpm, "_task_specs", lambda: specs)
    monkeypatch.setattr(tpm, "STAGE_TASKS", {"entity_extraction": tuple(specs)})
    limits = {
        ("openai", "mini"): {"tpm": 150_000_000, "rpm": 30_000},
        ("openai", "big"): {"tpm": 1_000_000, "rpm": 10_000},
    }
    sg = tpm.stage_suggestions(limits)["entity_extraction"]
    assert sg["task"] == "relationship_orphan_check"
    assert sg["value"] == tpm.suggest(limits[("openai", "big")], 8192)


def test_a_failed_probe_is_reported_not_guessed(monkeypatch):
    specs = {"entity_extract": {"provider": "anthropic", "model": "haiku",
                                "max_tokens": 4096}}
    monkeypatch.setattr(tpm, "_task_specs", lambda: specs)
    monkeypatch.setattr(tpm, "STAGE_TASKS", {"entity_extraction": ("entity_extract",)})
    sg = tpm.stage_suggestions(
        {("anthropic", "haiku"): {"error": "ANTHROPIC_API_KEY not set"}}
    )["entity_extraction"]
    assert sg["value"] is None
    assert sg["unknown"] == ["entity_extract (anthropic/haiku)"]


def test_models_in_use_includes_every_provider(monkeypatch):
    specs = {
        "entity_extract": {"provider": "anthropic", "model": "claude-haiku-4-5"},
        "chunk_classification": {"provider": "groq", "model": "llama"},
        "artifact_chunk_extract": {"provider": "openai", "model": "gpt-4o-mini"},
    }
    monkeypatch.setattr(tpm, "_task_specs", lambda: specs)
    assert set(tpm._models_in_use()) == {
        ("anthropic", "claude-haiku-4-5"), ("groq", "llama"), ("openai", "gpt-4o-mini"),
    }


# --------------------------------------------------------------------------
# apply_concurrency(): writes values, keeps the file's comments
# --------------------------------------------------------------------------

_CFG = """\
chunking:
  streaming_batch_size: 8

# Why these numbers: measured 2026-09-26.
concurrency:
  # mini tier
  summarization: 64   # measured ok
  entity_extraction: 12

database:
  pool_size: 10
"""


def test_apply_updates_in_place_and_keeps_comments(tmp_path):
    f = tmp_path / "config.yaml"
    f.write_text(_CFG)
    changes = tpm.apply_concurrency(f, {"entity_extraction": 64, "summarization": 64})
    out = f.read_text()
    assert "  entity_extraction: 64\n" in out
    assert "  summarization: 64   # measured ok\n" in out
    assert "# Why these numbers: measured 2026-09-26." in out
    assert "  # mini tier\n" in out
    assert changes == ["concurrency.entity_extraction: 12 -> 64"]
    assert yaml.safe_load(out)["database"] == {"pool_size": 10}


def test_apply_adds_missing_keys_inside_the_block(tmp_path):
    f = tmp_path / "config.yaml"
    f.write_text(_CFG)
    tpm.apply_concurrency(f, {"dedup": 16})
    parsed = yaml.safe_load(f.read_text())
    assert parsed["concurrency"]["dedup"] == 16
    assert parsed["database"] == {"pool_size": 10}


def test_set_rejects_an_unknown_stage():
    with pytest.raises(SystemExit):
        tpm._parse_sets(["--set", "entity_extractoin=64"])
    assert tpm._parse_sets(["--set", "dedup=8", "--set", "table_extraction=3"]) == {
        "dedup": 8, "table_extraction": 3,
    }
