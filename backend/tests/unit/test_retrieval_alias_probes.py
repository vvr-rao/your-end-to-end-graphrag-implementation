"""Alias probe injection + distance-aware fusion trimming.

Pure-unit: no DB, no LLM, no embeddings, no cost. Both features are
additive, so the load-bearing tests here are the regression guards --
with no aliases and no distance knobs, probe lists and rankings must come
out byte-identical to what the pipeline produced before either existed.
"""
from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest

from backend.app.services import retrieval


@pytest.fixture
def qa_config(monkeypatch):
    """Override the `qa:` config block for one test."""

    def _apply(**values):
        monkeypatch.setattr(
            retrieval,
            "get_settings",
            lambda: SimpleNamespace(app_config={"qa": values}),
        )

    return _apply


# --------------------------------------------------------------------------
# Alias injection into probes
# --------------------------------------------------------------------------

# The shape from the reported failure: the corpus chunk holding
# tirzepatide's real maximum dose says "MOUNJARO" and never "tirzepatide".
_ALIASES = {"tirzepatide": ["Mounjaro"], "dulaglutide": ["Trulicity"]}
_PROBES = [
    "Compare the dosing schedules of tirzepatide and dulaglutide",
    "Dosing schedule of tirzepatide",
    "Dosing schedule of dulaglutide",
]


def test_blended_mode_rewrites_in_place(qa_config):
    """Default mode: aliases ride along in the existing probe text, so no
    extra rerank pass is added."""
    qa_config(alias_probe_mode="blended")
    out = retrieval._apply_aliases_to_probes(_PROBES, _ALIASES)

    assert len(out) == len(_PROBES), "blended mode must not add probes"
    assert out[1] == "Dosing schedule of tirzepatide (Mounjaro)"
    assert out[2] == "Dosing schedule of dulaglutide (Trulicity)"
    # Both entities named in one probe -> both aliases injected.
    assert "Mounjaro" in out[0] and "Trulicity" in out[0]


def test_separate_mode_adds_dedicated_probes(qa_config):
    """The higher-recall mode: a probe that says only the brand name."""
    qa_config(alias_probe_mode="separate", max_alias_probes=4)
    out = retrieval._apply_aliases_to_probes(_PROBES, _ALIASES)

    assert _PROBES[1] in out, "original probes are preserved verbatim"
    assert "Dosing schedule of Mounjaro" in out
    assert "Dosing schedule of Trulicity" in out


def test_separate_mode_respects_probe_budget(qa_config):
    """Each added probe is one sequential id-filtered rerank, so the cap
    has to actually bind."""
    qa_config(alias_probe_mode="separate", max_alias_probes=1)
    out = retrieval._apply_aliases_to_probes(_PROBES, _ALIASES)
    assert len(out) == len(_PROBES) + 1


def test_off_mode_leaves_probes_untouched(qa_config):
    """`off` still lets aliases widen graph seeding; it just keeps them
    out of the probe list."""
    qa_config(alias_probe_mode="off")
    assert retrieval._apply_aliases_to_probes(_PROBES, _ALIASES) == _PROBES


def test_alias_already_present_is_not_repeated(qa_config):
    qa_config(alias_probe_mode="blended")
    probes = ["Dosing schedule of tirzepatide (Mounjaro)"]
    assert retrieval._apply_aliases_to_probes(probes, _ALIASES) == probes


def test_alias_matching_is_case_insensitive_and_preserves_case(qa_config):
    """Matching ignores case, but the probe's own wording is kept: the
    rewrite must not lowercase "TIRZEPATIDE" to the parse output."""
    qa_config(alias_probe_mode="blended")
    (out,) = retrieval._apply_aliases_to_probes(
        ["Maximum dose of TIRZEPATIDE"], {"tirzepatide": ["Mounjaro"]}
    )
    assert out == "Maximum dose of TIRZEPATIDE (Mounjaro)"


def test_probe_without_the_term_is_untouched(qa_config):
    qa_config(alias_probe_mode="blended")
    probes = ["Adverse events in elderly patients"]
    assert retrieval._apply_aliases_to_probes(probes, _ALIASES) == probes


@pytest.mark.parametrize("aliases", [{}, None])
def test_no_aliases_is_a_noop(qa_config, aliases):
    """REGRESSION GUARD: an empty term_aliases table (nobody has run
    `mine-aliases`) must reproduce today's probe list exactly."""
    qa_config(alias_probe_mode="blended")
    assert retrieval._apply_aliases_to_probes(_PROBES, aliases or {}) == _PROBES


def test_unknown_mode_is_a_noop(qa_config):
    qa_config(alias_probe_mode="nonsense")
    assert retrieval._apply_aliases_to_probes(_PROBES, _ALIASES) == _PROBES


# --------------------------------------------------------------------------
# Distance-aware trimming
# --------------------------------------------------------------------------


def _ranked(*dists: float) -> list[tuple[uuid.UUID, float]]:
    return [(uuid.uuid4(), d) for d in dists]


def test_no_knobs_keeps_every_result_in_order(qa_config):
    """REGRESSION GUARD: unset knobs == the previous rank-only behavior."""
    qa_config()
    ranked = _ranked(0.34, 0.36, 0.80, 0.95)
    assert retrieval._trim_by_distance(ranked) == [cid for cid, _ in ranked]


def test_probe_that_found_nothing_casts_no_votes(qa_config):
    """The core fix. A probe whose BEST hit is beyond the floor found
    nothing relevant; previously it still voted rank-1..N at full
    strength, indistinguishable from a probe that found the answer."""
    qa_config(probe_dist_floor=0.55)
    assert retrieval._trim_by_distance(_ranked(0.61, 0.68, 0.72)) == []


def test_probe_with_a_real_hit_survives_the_floor(qa_config):
    qa_config(probe_dist_floor=0.55)
    ranked = _ranked(0.34, 0.61, 0.72)
    assert len(retrieval._trim_by_distance(ranked)) == 3


def test_band_trims_the_weak_tail(qa_config):
    """Keep the tight cluster around the probe's own best hit; drop the
    long tail that only looks good because nothing better exists."""
    qa_config(probe_dist_band=0.15)
    ranked = _ranked(0.34, 0.36, 0.45, 0.80, 0.95)
    kept = retrieval._trim_by_distance(ranked)
    assert kept == [cid for cid, d in ranked if d <= 0.34 + 0.15]
    assert len(kept) == 3


def test_floor_and_band_together(qa_config):
    qa_config(probe_dist_band=0.15, probe_dist_floor=0.55)
    # Best hit clears the floor; band then trims the tail.
    assert len(retrieval._trim_by_distance(_ranked(0.34, 0.40, 0.90))) == 2
    # Best hit fails the floor -> nothing survives, band irrelevant.
    assert retrieval._trim_by_distance(_ranked(0.60, 0.62)) == []


def test_empty_ranking(qa_config):
    qa_config(probe_dist_band=0.15, probe_dist_floor=0.55)
    assert retrieval._trim_by_distance([]) == []


def test_config_read_failure_falls_back_to_documented_defaults(monkeypatch):
    """An unreadable config must degrade to each knob's documented
    default, never to an exception.

    For the distance knobs that default is "off", so trimming becomes a
    no-op. For alias mode it is "blended" -- which still changes nothing
    on a database where `mine-aliases` has never run, since the alias
    dict is empty there.
    """

    def _boom():
        raise RuntimeError("no config")

    monkeypatch.setattr(retrieval, "get_settings", _boom)

    ranked = _ranked(0.9, 0.95)
    assert retrieval._trim_by_distance(ranked) == [cid for cid, _ in ranked]

    # No mined aliases (the pre-`mine-aliases` state) -> untouched.
    assert retrieval._apply_aliases_to_probes(_PROBES, {}) == _PROBES
    # With mined aliases -> the default mode still applies.
    blended = retrieval._apply_aliases_to_probes(_PROBES, _ALIASES)
    assert len(blended) == len(_PROBES)
    assert blended[1] == "Dosing schedule of tirzepatide (Mounjaro)"
