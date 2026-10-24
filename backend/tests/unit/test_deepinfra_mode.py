"""DeepInfra/Qwen mode: provider wiring, embedder selection, the embedding-
model guard's decision table, threshold profiles, and GPU sizing maths.

All offline -- no network, no database.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from backend.app.core.config import apply_embedding_profile
from backend.app.db.embedding_guard import LEGACY_EMBEDDING_MODEL, resolve_stamp
from backend.app.services import embeddings as emb_mod
from backend.app.services.embeddings import Embedder, _l2_normalize
from backend.app.services.llm_router import (
    DEEPINFRA_BASE_URL,
    ChatResult,
    DeepInfraProvider,
    LLMRouter,
    _strip_thinking,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
import deepinfra_deploy as deploy
import deepinfra_sizing as sizing

PRESET = "config/models.deepinfra.example.yaml"


# --------------------------------------------------------------------------- #
# Provider
# --------------------------------------------------------------------------- #

class _FakeCompletions:
    def __init__(self, content: str) -> None:
        self.kwargs: dict | None = None
        self._content = content

    async def create(self, **kwargs):
        self.kwargs = kwargs
        msg = SimpleNamespace(content=self._content)
        usage = SimpleNamespace(
            prompt_tokens=1000, completion_tokens=100,
            prompt_tokens_details=SimpleNamespace(cached_tokens=400),
        )
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")], usage=usage)


def _provider(content: str) -> tuple[DeepInfraProvider, _FakeCompletions]:
    p = DeepInfraProvider(api_key="test")
    fake = _FakeCompletions(content)
    p._client = SimpleNamespace(chat=SimpleNamespace(completions=fake))  # type: ignore[assignment]
    return p, fake


def test_deepinfra_client_targets_the_openai_compatible_base_url() -> None:
    assert str(DeepInfraProvider(api_key="test")._client.base_url).rstrip("/") == DEEPINFRA_BASE_URL


@pytest.mark.asyncio
async def test_thinking_is_off_by_default_and_on_when_asked() -> None:
    p, fake = _provider("ok")
    await p.chat(model="Qwen/Qwen3.8-27B", system="s", user="u", temperature=0.0,
                 max_tokens=10, timeout=10, response_format=None)
    assert fake.kwargs["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    await p.chat(model="Qwen/Qwen3.8-27B", system="s", user="u", temperature=0.0,
                 max_tokens=10, timeout=10, response_format=None, thinking=True)
    assert fake.kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True


@pytest.mark.asyncio
async def test_reasoning_and_fences_are_stripped_from_json_replies() -> None:
    p, _ = _provider('<think>let me see</think>\n```json\n{"a": 1}\n```')
    r = await p.chat(model="Qwen/Qwen3.8-27B", system="s", user="u", temperature=0.0,
                     max_tokens=10, timeout=10, response_format="json_object")
    assert r.text == '{"a": 1}'
    assert r.provider == "deepinfra"
    # 600 uncached @ $0.15/M + 400 cached @ 0.25x + 100 out @ $1.875/M
    assert r.cost_usd == pytest.approx(600 * 0.15e-6 + 400 * 0.15e-6 * 0.25 + 100 * 1.875e-6)


def test_template_opened_think_block_is_stripped() -> None:
    # Qwen templates open <think> in the PROMPT; the reply has only the close.
    assert _strip_thinking('reasoning here\n</think>\n\n{"ok": false}') == '{"ok": false}'


@pytest.mark.asyncio
async def test_thinking_requests_drop_json_mode() -> None:
    p, fake = _provider('r\n</think>\n{"a": 1}')
    r = await p.chat(model="m", system="s", user="u", temperature=0.0, max_tokens=10,
                     timeout=10, response_format="json_object", thinking=True)
    assert "response_format" not in fake.kwargs and r.text == '{"a": 1}'
    await p.chat(model="m", system="s", user="u", temperature=0.0, max_tokens=10,
                 timeout=10, response_format="json_object")
    assert fake.kwargs["response_format"] == {"type": "json_object"}


def test_unterminated_think_block_yields_no_answer() -> None:
    assert _strip_thinking("<think>still reasoning when cut off") == ""
    assert _strip_thinking("plain answer") == "plain answer"


@pytest.mark.asyncio
async def test_dedicated_model_costs_zero_but_still_reports_tokens() -> None:
    p, _ = _provider("ok")
    r = await p.chat(model="me/flash-next", system="s", user="u", temperature=0.0,
                     max_tokens=10, timeout=10, response_format=None)
    assert r.cost_usd == 0.0
    assert r.prompt_tokens == 1000


def _bare_router(tasks: dict) -> LLMRouter:
    router = LLMRouter.__new__(LLMRouter)
    router.reset_counters()
    router._settings = None  # type: ignore[assignment]
    router._tasks = tasks
    router._retry = {"max_attempts": 1, "initial_wait_seconds": 0, "max_wait_seconds": 0}
    return router


def test_thinking_on_a_non_deepinfra_task_is_rejected_at_load() -> None:
    router = _bare_router({"t": {"provider": "openai", "model": "gpt-4.1", "thinking": True}})
    with pytest.raises(ValueError, match="thinking"):
        router._validate_thinking()


@pytest.mark.asyncio
async def test_zero_cost_tasks_still_appear_in_the_cost_report() -> None:
    class _Stub:
        async def chat(self, **kw):
            return ChatResult(text="x", model=kw["model"], provider="deepinfra",
                              prompt_tokens=50, completion_tokens=7, cost_usd=0.0)

    router = _bare_router({"class_proposal": {"provider": "deepinfra", "model": "me/flash"}})
    router._providers = {"deepinfra": _Stub()}
    await router.chat("class_proposal", system="s", user="u")
    row = router.cost_report()["by_task"]["class_proposal"]
    assert (row["calls"], row["in_tokens"], row["out_tokens"]) == (1, 50, 7)


@pytest.mark.asyncio
async def test_thinking_kwarg_is_not_sent_to_tasks_that_do_not_set_it() -> None:
    seen: dict = {}

    class _Stub:
        async def chat(self, **kw):
            seen.update(kw)
            return ChatResult(text="x", model=kw["model"], provider="openai")

    router = _bare_router({"t": {"provider": "openai", "model": "gpt-4.1"}})
    router._providers = {"openai": _Stub()}
    await router.chat("t", system="s", user="u")
    assert "thinking" not in seen


# --------------------------------------------------------------------------- #
# Preset
# --------------------------------------------------------------------------- #

def test_preset_routes_creation_to_dedicated_qa_to_deepseek() -> None:
    tasks = yaml.safe_load(open(PRESET))["tasks"]
    assert tasks["class_proposal"]["model"] == "__DEEPINFRA_FLASH_MODEL__"
    assert tasks["entity_extract"]["provider"] == "deepinfra"
    # Q&A runs on serverless DeepSeek-V4.1-Flash (won the 2026-10-17 head to
    # head against Qwen3.8-27B); no Q&A task may point at the dedicated GPU.
    for t in ("answer_simple_qa", "answer_deep_research", "question_parse", "answer_conversation_turn"):
        assert tasks[t]["model"] == "deepseek-ai/DeepSeek-V4.1-Flash", t
    assert not any(v.get("model") == "Qwen/Qwen3.8-27B" for v in tasks.values())
    # Serverless 27B only as the alias judge's fallback (user decision
    # 2026-10-22): delete-document refreshes aliases with no GPU up.
    assert [t for t, v in tasks.items() if v.get("fallback_model") == "Qwen/Qwen3.8-27B"] == ["alias_judge"]
    assert tasks["judge_consistency"]["model"] == "deepseek-ai/DeepSeek-V4.1-Flash"
    assert tasks["embeddings"] == {
        "provider": "deepinfra", "model": "Qwen/Qwen3-Embedding-4B", "dim": 1024, "batch_size": 100,
        # The "facts" instruction; config.yaml's embedding_profiles thresholds
        # are calibrated for exactly this prefix.
        "query_prefix": ("Instruct: Given a question about a document collection, retrieve the "
                         "passages that contain the facts needed to answer it\nQuery: "),
    }


def test_select_llm_mode_substitutes_the_deployment_id(tmp_path, monkeypatch) -> None:
    import select_llm_mode as sel

    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "models.deepinfra.example.yaml").write_text(
        "tasks:\n  a:\n    model: __DEEPINFRA_FLASH_MODEL__\n")
    (tmp_path / ".env").write_text("DEEPINFRA_FLASH_MODEL=me/flash-next\n")
    monkeypatch.setattr(sel, "ROOT", tmp_path)
    monkeypatch.setattr(sel, "env_value_nonsecret", lambda k: "me/flash-next")
    monkeypatch.setattr(sel, "env_present", lambda k: True)
    monkeypatch.setattr(sel.build_state, "record_step", lambda *a, **k: None)
    assert sel.main(["deepinfra"]) == 0
    assert "me/flash-next" in (tmp_path / "config" / "models.yaml").read_text()


def test_nonsecret_reader_refuses_credentials() -> None:
    from _skillutil import env_value_nonsecret

    with pytest.raises(ValueError):
        env_value_nonsecret("DEEPINFRA_API_KEY")


# --------------------------------------------------------------------------- #
# Embedder
# --------------------------------------------------------------------------- #

def test_embedder_defaults_are_unchanged_without_config(monkeypatch) -> None:
    monkeypatch.setattr(emb_mod, "get_settings", lambda: (_ for _ in ()).throw(RuntimeError()))
    cfg = emb_mod.configured_embedding()
    assert cfg == {"provider": "openai", "model": "text-embedding-3-small", "dim": 1024, "batch_size": 100}


def test_qwen_embedder_targets_deepinfra_and_prefixes_queries() -> None:
    e = Embedder(provider="deepinfra", model="Qwen/Qwen3-Embedding-8B", dim=1024, batch_size=100,
                 settings=SimpleNamespace(deepinfra_api_key="k", openai_api_key=None))  # type: ignore[arg-type]
    assert str(e._client.base_url).rstrip("/") == DEEPINFRA_BASE_URL
    assert e.query_prefix.startswith("Instruct:")
    o = Embedder(provider="openai", model="text-embedding-3-small", dim=1024, batch_size=100,
                 settings=SimpleNamespace(openai_api_key="k"))  # type: ignore[arg-type]
    assert o.query_prefix == ""


def test_qwen_embedder_without_key_fails_loudly() -> None:
    with pytest.raises(RuntimeError, match="DEEPINFRA_API_KEY"):
        Embedder(provider="deepinfra", model="Qwen/Qwen3-Embedding-8B", dim=1024, batch_size=100,
                 settings=SimpleNamespace(deepinfra_api_key=None))  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_non_openai_vectors_are_unit_normalised() -> None:
    e = Embedder(provider="deepinfra", model="Qwen/Qwen3-Embedding-8B", dim=2, batch_size=100,
                 settings=SimpleNamespace(deepinfra_api_key="k"))  # type: ignore[arg-type]

    class _Emb:
        async def create(self, **kw):
            return SimpleNamespace(data=[SimpleNamespace(embedding=[3.0, 4.0])],
                                   usage=SimpleNamespace(prompt_tokens=1_000_000))

    e._client = SimpleNamespace(embeddings=_Emb())  # type: ignore[assignment]
    (vec,) = await e.embed(["x"])
    assert vec == pytest.approx([0.6, 0.8])
    assert e.total_cost_usd == pytest.approx(0.010)
    assert _l2_normalize([0.0, 0.0]) == [0.0, 0.0]


# --------------------------------------------------------------------------- #
# Guard + profiles
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("stamped,has_vectors,configured,expected", [
    (None, False, "Qwen/Qwen3-Embedding-8B", ("Qwen/Qwen3-Embedding-8B", True)),   # empty DB adopts
    (None, True, LEGACY_EMBEDDING_MODEL, (LEGACY_EMBEDDING_MODEL, True)),         # legacy, same model
    (None, True, "Qwen/Qwen3-Embedding-8B", (LEGACY_EMBEDDING_MODEL, False)),     # legacy, MISMATCH
    ("Qwen/Qwen3-Embedding-8B", True, "Qwen/Qwen3-Embedding-8B", ("Qwen/Qwen3-Embedding-8B", False)),
])
def test_guard_decision_table(stamped, has_vectors, configured, expected) -> None:
    assert resolve_stamp(stamped, has_vectors, configured) == expected


def test_embedding_profile_overrides_only_for_the_active_model() -> None:
    app = {"qa": {"relationship_similarity": 0.55, "probe_dist_floor": 1.05},
           "embedding_profiles": {"Qwen/Qwen3-Embedding-8B": {"qa.relationship_similarity": 0.62}}}
    qwen = {"tasks": {"embeddings": {"model": "Qwen/Qwen3-Embedding-8B"}}}
    out = apply_embedding_profile(app, qwen)
    assert out["qa"] == {"relationship_similarity": 0.62, "probe_dist_floor": 1.05}
    assert app["qa"]["relationship_similarity"] == 0.55  # input untouched
    assert apply_embedding_profile(app, {"tasks": {}}) is app


# --------------------------------------------------------------------------- #
# Sizing + deploy
# --------------------------------------------------------------------------- #

def test_workload_scales_linearly_with_corpus_tokens() -> None:
    a = sizing.project_workload(100_000, sizing.DEFAULT_RATIOS)
    b = sizing.project_workload(200_000, sizing.DEFAULT_RATIOS)
    assert b["class_proposal"]["out_tokens"] == pytest.approx(2 * a["class_proposal"]["out_tokens"])


def test_gpu_options_only_include_configurations_that_fit() -> None:
    opts = sizing.gpu_options(model="flash-next")
    labels = {o.label() for o in opts}
    assert "2x H200-141GB fp8" in labels           # 185.5 GB fits in 2x141
    assert "1x H200-141GB fp8" not in labels       # does not
    assert "2x H200-141GB bf16" not in labels      # 360 GB does not
    assert all(o.total_gpus <= sizing.MAX_GPUS_PER_ACCOUNT for o in opts)
    assert all(o.kv_free_gb >= 8 for o in opts)


def test_default_model_is_the_deployed_dense_27b() -> None:
    """The preset deploys Qwen3.8-27B-FP8 (30.9 GB): every preferred pair fits,
    and decode reads ALL weights each step (dense), unlike Flash-Next."""
    opts = {o.label(): o for o in sizing.gpu_options()}
    for label in ("2x B200-180GB fp8", "2x H200-141GB fp8", "2x H100-80GB fp8"):
        assert label in opts and opts[label].dense
        assert opts[label].repo == "Qwen/Qwen3.8-27B-FP8"
    arch = sizing.MODELS["27b"]["fallback_arch"]
    assert sizing.kv_bytes_per_token(arch) == 17 * 2 * 4 * 256 * 2   # 16 full-attn + 1 MTP layer


def test_recurrent_state_dominates_short_sequences() -> None:
    arch = sizing.FALLBACK_ARCH
    assert sizing.state_bytes_per_seq(arch) > sizing.kv_bytes_per_token(arch) * 2000


def test_recommendation_follows_the_preferred_gpu_order() -> None:
    """2x B200, then 2x H200, then 2x H100 -- whatever is cheaper."""
    b = sizing.GpuOption("B200-180GB", 2, 1, "fp8", "r", 30.9, 283, 7.38)
    h2 = sizing.GpuOption("H200-141GB", 2, 1, "fp8", "r", 30.9, 213, 5.38)
    h1 = sizing.GpuOption("H100-80GB", 2, 1, "fp8", "r", 30.9, 103, 4.40)
    cheap = sizing.GpuOption("H100-80GB", 1, 1, "fp8", "r", 30.9, 36, 2.20)
    rows = [(cheap, {"hours": 2, "usd": 4.4}), (h1, {"hours": 1.3, "usd": 5.9}),
            (h2, {"hours": 1.2, "usd": 6.9}), (b, {"hours": 1, "usd": 7.8})]
    assert sizing.recommend(rows, None)[0] is b
    assert [o for o, _ in sizing.preferred_chain(rows, None)] == [b, h2, h1]
    # A deadline only B200 misses moves the recommendation down the chain.
    assert sizing.recommend(rows, 1.25, preferred=(("H200-141GB", 2), ("H100-80GB", 2)))[0] is h2


def test_four_gpu_options_come_first_as_two_instances() -> None:
    """2026-10-19 order: 4x B200, 4x H200 (each as 2 instances of 2 GPUs),
    then 2x B200, 2x H200, 2x H100."""
    assert sizing.PREFERRED_GPUS == (("B200-180GB", 2, 2), ("H200-141GB", 2, 2), ("B200-180GB", 2, 1),
                                     ("H200-141GB", 2, 1), ("H100-80GB", 2, 1))
    opts = {o.label(): o for o in sizing.gpu_options()}
    chain_labels = ["2x B200-180GB fp8 x2 inst", "2x H200-141GB fp8 x2 inst",
                    "2x B200-180GB fp8", "2x H200-141GB fp8", "2x H100-80GB fp8"]
    assert all(lbl in opts for lbl in chain_labels)
    rows = [(opts[lbl], {"hours": 1.0, "usd": 1.0}) for lbl in reversed(chain_labels)]
    assert [o.label() for o, _ in sizing.preferred_chain(rows, None)] == chain_labels
    assert sizing.recommend(rows, None)[0].instances == 2
    assert opts["2x B200-180GB fp8 x2 inst"].total_gpus <= sizing.MAX_GPUS_PER_ACCOUNT


def test_recommendation_without_a_preferred_fit_is_cheapest_and_skips_caveats() -> None:
    a = sizing.GpuOption("A100-80GB", 4, 1, "fp8", "r", 185.5, 80, 3.56, note="caveat")
    h = sizing.GpuOption("H200-141GB", 4, 1, "fp8", "r", 185.5, 58, 10.76)
    b = sizing.GpuOption("B200-180GB", 4, 1, "fp8", "r", 185.5, 128, 14.76)
    rows = [(a, {"hours": 1, "usd": 3.5}), (h, {"hours": 2, "usd": 21.5}), (b, {"hours": 1, "usd": 14.8})]
    assert sizing.recommend(rows, None)[0] is b
    assert sizing.recommend(rows, None, allow_noted=True)[0] is a
    assert sizing.recommend(rows, 0.5) is None


def test_a_priori_concurrency_per_gpu_tracks_the_measured_builds() -> None:
    """Prune-expand on 2x H100 ran ~9 requests in flight at class_proposal 12 /
    dedup 8 and took ~1 h; the GPU had far more room. Each preferred GPU gets
    its own values: class_proposal (32k-token replies) is KV-bound, so it
    shrinks with the card's memory; nothing exceeds MAX_BATCH."""
    stages = sizing.project_workload(500_000, sizing.DEFAULT_RATIOS)
    arch = sizing.MODELS["27b"]["fallback_arch"]
    opts = {o.label(): o for o in sizing.gpu_options()}
    conc = {g: sizing.a_priori_concurrency(opts[g], arch, stages)
            for g in ("2x B200-180GB fp8", "2x H200-141GB fp8", "2x H100-80GB fp8")}
    cp = [conc[g]["class_proposal"] for g in conc]
    assert cp == sorted(cp, reverse=True) and cp[-1] > 12
    assert all(v <= sizing.MAX_BATCH for c in conc.values() for v in c.values())
    assert conc["2x B200-180GB fp8"]["entity_extraction"] >= 64


def test_timeout_cap_uses_each_tasks_own_limit() -> None:
    """A short-timeout task caps concurrency only by ITS slow call."""
    stages = sizing.project_workload(500_000, sizing.DEFAULT_RATIOS)
    arch = sizing.MODELS["27b"]["fallback_arch"]
    opt = {o.label(): o for o in sizing.gpu_options()}["2x H100-80GB fp8"]
    free = sizing.a_priori_concurrency(opt, arch, stages)
    tight = sizing.a_priori_concurrency(opt, arch, stages, {"artifact_generation": [(10, 4000.0)]})
    assert tight["artifact_generation"] < free["artifact_generation"]
    assert tight["entity_extraction"] == free["entity_extraction"]


def test_knee_is_smallest_concurrency_near_peak_throughput() -> None:
    curve = [{"concurrency": c, "tok_s": t, "p95_s": p} for c, t, p in
             [(1, 100, 3), (8, 700, 4), (16, 1300, 5), (32, 1900, 7), (64, 2000, 14), (128, 2010, 30)]]
    assert sizing.find_knee(curve) == 32


def test_probe_concurrency_respects_kv_slots_and_timeouts() -> None:
    curve = [{"concurrency": c, "tok_s": t, "p95_s": p} for c, t, p in
             [(8, 700, 4), (32, 1900, 7), (64, 2000, 14)]]
    stages = {"class_proposal": {"mean_out": 6300}, "entity_extraction": {"mean_out": 500}}
    out = sizing.probe_concurrency(curve, stages, {"class_proposal": 16, "entity_extraction": 128},
                                   {"class_proposal": 2400, "entity_extraction": 20}, probe_out=256)
    assert out["class_proposal"] == 16        # KV-capped below the knee (32)
    assert out["entity_extraction"] == 8      # 7s*1.95 > 10s at 32; 4s*1.95 fits at 8


def test_deploy_payload_matches_the_sized_option() -> None:
    sz = {"option": {"gpu": "H200-141GB", "num_gpus": 2, "instances": 1,
                     "repo": "Qwen/Qwen3.8-Flash-Next-FP8"},
          "concurrency": {"class_proposal": 40, "summarization": 96}}
    p = deploy.build_payload(sz, "flash-next")
    assert p == {"model_name": "flash-next", "gpu": "H200-141GB", "num_gpus": 2, "max_batch_size": 96,
                 "hf": {"repo": "Qwen/Qwen3.8-Flash-Next-FP8"},
                 "settings": {"min_instances": 1, "max_instances": 1}}


def test_create_walks_the_fallback_chain_until_one_runs(tmp_path, monkeypatch) -> None:
    """B200: 409 unavailable. H200: accepted, then `no-gpu-available` -> deleted.
    H100: runs -> recorded as `active`; its OWN concurrency is the one used."""
    def entry(gpu, usd, cp):
        return {"option": {"gpu": gpu, "num_gpus": 2, "instances": 1, "repo": "Qwen/Qwen3.8-27B-FP8",
                           "usd_hr": usd, "label": f"2x {gpu} fp8"},
                "estimate": {"hours": 1, "usd": usd}, "concurrency": {"class_proposal": cp}, "max_batch_size": cp}
    sz = {**entry("B200-180GB", 7.38, 96), "fallbacks": [entry("H200-141GB", 5.38, 48), entry("H100-80GB", 4.4, 32)]}
    sp = tmp_path / "sizing.json"
    sp.write_text(json.dumps(sz))
    calls = []

    def fake_call(method, path, body=None):
        calls.append((method, path, (body or {}).get("gpu")))
        if method == "POST" and body["gpu"] == "B200-180GB":
            raise deploy.DeployHTTPError(method, path, 409, '{"error":"GPU 2xB200-180GB is currently unavailable"}')
        if method == "POST":
            return {"deploy_id": "d-" + body["gpu"][:4]}
        return {}

    states = {"d-H200": ("failed", "no-gpu-available"), "d-H100": ("running", "")}
    monkeypatch.setattr(deploy, "_call", fake_call)
    monkeypatch.setattr(deploy, "wait_until_running", lambda did, minutes: states[did])
    rc = deploy.main(["create", "--from-sizing", str(sp), "--yes", "--ledger", str(tmp_path / "l.jsonl")])
    assert rc == 0
    assert ("DELETE", "/deploy/d-H200", None) in calls
    active = json.loads(sp.read_text())["active"]
    assert active["deploy_id"] == "d-H100" and active["concurrency"] == {"class_proposal": 32}
    assert active["model_name"].startswith("qwen27b-h100-")
    posts = [c for c in calls if c[0] == "POST"]
    assert [p[2] for p in posts] == ["B200-180GB", "H200-141GB", "H100-80GB"]


def test_wait_until_running_stops_on_terminal_states() -> None:
    seq = iter([("creating", ""), ("initializing", ""), ("running", "")])
    assert deploy.wait_until_running("d", 10, poll_s=0, state=lambda d: next(seq), sleep=lambda s: None) == ("running", "")
    assert deploy.wait_until_running("d", 10, state=lambda d: ("failed", "no-gpu-available"),
                                     sleep=lambda s: None) == ("failed", "no-gpu-available")
    assert deploy.wait_until_running("d", 0, state=lambda d: ("creating", ""), sleep=lambda s: None) == ("timeout", "")


def test_gpu_ledger_pairs_create_and_delete(tmp_path) -> None:
    led = tmp_path / "l.jsonl"
    led.write_text(
        '{"ts": "2026-10-12T10:00:00", "event": "create", "deploy_id": "d1", "usd_hr": 5.38, "label": "x"}\n'
        '{"ts": "2026-10-12T12:30:00", "event": "delete", "deploy_id": "d1"}\n'
        '{"ts": "2026-10-12T13:00:00", "event": "create", "deploy_id": "d2", "usd_hr": 1.0}\n')
    rows = {r["deploy_id"]: r for r in deploy.gpu_hours(led)}
    assert rows["d1"]["hours"] == 2.5 and rows["d1"]["usd"] == 13.45
    assert "STILL RUNNING" in rows["d2"]["note"]


# --------------------------------------------------------------------------- #
# Benchmark scorers
# --------------------------------------------------------------------------- #

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "bench"))
import eval_arm  # noqa: E402


@pytest.mark.parametrize("pred,gold,qtype,ok", [
    ("Yes, both outlets report it.", "Yes", "comparison_query", True),
    ("No.", "Yes", "comparison_query", False),
    ("Sam Bankman-Fried was accused of fraud.", "Sam Bankman-Fried", "inference_query", True),
    ("The corpus does not contain this.", "Insufficient information.", "null_query", True),
    (None, "Yes", "comparison_query", False),
])
def test_multihop_answer_scoring(pred, gold, qtype, ok) -> None:
    assert eval_arm.answer_correct(pred, gold, qtype) is ok


def test_multihop_retrieval_ranks_by_first_gold_document() -> None:
    titles = {"some headline": "mhrag_doc_001"}
    ev = [{"document_title": "Some Headline"}, {"document_title": "mhrag_doc_009"},
          {"document_title": "Some Headline"}, {"document_title": "mhrag_doc_002"}]
    got = eval_arm.retrieval_scores(ev, ["mhrag_doc_002", "mhrag_doc_404"], titles)
    assert got == {"recall": 0.5, "mrr": pytest.approx(1 / 3), "gold_at5": 1}


def test_gold_questions_exclude_vacuous_null_queries() -> None:
    qs = [{"query": "a", "evidence_list": [{"title": "T1"}]},
          {"query": "b", "evidence_list": [{"title": "T1"}, {"title": "Elsewhere"}]},
          {"query": "c", "evidence_list": []}]
    got = eval_arm.gold_questions(qs, {"t1": "doc1"})
    assert [q["query"] for q in got] == ["a"] and got[0]["gold_docs"] == ["doc1"]


def test_cancellation_is_never_retried() -> None:
    import asyncio

    from backend.app.services import retry_policy

    assert retry_policy.is_retryable(asyncio.CancelledError()) is False
    assert retry_policy.is_retryable(KeyboardInterrupt()) is False


@pytest.mark.asyncio
async def test_glm_models_get_the_thinking_disabled_switch() -> None:
    p, fake = _provider("ok")
    await p.chat(model="zai-org/GLM-5.3", system="s", user="u", temperature=0.0,
                 max_tokens=10, timeout=10, response_format=None)
    # ONLY the GLM switch: adding enable_thinking=false makes GLM leak its
    # reasoning into the answer.
    assert fake.kwargs["extra_body"] == {"thinking": {"type": "disabled"}}
    await p.chat(model="Qwen/Qwen3.8-27B", system="s", user="u", temperature=0.0,
                 max_tokens=10, timeout=10, response_format=None)
    assert "thinking" not in fake.kwargs["extra_body"]


def test_qwen_class_proposal_prompt_is_separate() -> None:
    from backend.app.services.prompts import PROMPTS
    from backend.app.services.prompts_qwen import QWEN_OUTPUT_RULES, qwen_class_proposal

    sl = {"http://x#Drug": {"label": "Drug"}}
    shared_sys, shared_user = PROMPTS["class_proposal"](sl, "Semaglutide is a drug.")
    qwen_sys, qwen_user = qwen_class_proposal(sl, "Semaglutide is a drug.")
    assert QWEN_OUTPUT_RULES not in shared_sys            # other models unchanged
    assert QWEN_OUTPUT_RULES in qwen_sys and qwen_sys != shared_sys
    assert "each IRI at most once" in qwen_sys and "each IRI at most once" not in shared_sys
    # same data (ontology slice + text); only the closing key-order request differs
    assert qwen_user.split("Return JSON")[0] == shared_user.split("Return JSON")[0]
    assert qwen_user.index('"MATCH NOT FOUND"') < qwen_user.index('"MATCHES FOUND"')
    assert "EXHAUSTIVE" not in QWEN_OUTPUT_RULES           # v1 over-generated
    assert "NEVER walk through DATA_CLASSES" in QWEN_OUTPUT_RULES


def test_preset_selects_qwen_prompt_only_for_class_proposal() -> None:
    tasks = yaml.safe_load(open(PRESET))["tasks"]
    assert tasks["class_proposal"]["prompt_set"] == "qwen"
    assert tasks["relationship_extract"]["prompt_set"] == "deepseek"
    assert tasks["relationship_extract"]["model"] == "deepseek-ai/DeepSeek-V4.1-Flash"
    assert sorted(t for t, v in tasks.items() if v.get("prompt_set")) == [
        "alias_judge", "class_proposal", "class_synonym_judge", "judge_comprehensiveness", "judge_consistency",
        "judge_gap_detection", "judge_no_hallucination", "relationship_extract", "relationship_repair"]


def test_deepinfra_preset_is_deepinfra_only() -> None:
    """User decision 2026-10-17: no OpenAI or Anthropic model anywhere in this mode."""
    doc = yaml.safe_load(open(PRESET))
    assert {v["provider"] for v in doc["tasks"].values()} == {"deepinfra"}
    import select_llm_mode as sel

    assert sel.MODES["deepinfra"][1] == ["DEEPINFRA_API_KEY", "DEEPINFRA_FLASH_MODEL"]


def test_deepseek_prompt_set_selects_deepseek_repair_and_judges() -> None:
    from backend.app.services import eval_judge
    from backend.app.services.db_entity_extract import _repair_prompt
    from backend.app.services.prompts import PROMPTS
    from backend.app.services.prompts_deepseek import DEEPSEEK_JUDGES, deepseek_relationship_repair

    ds = SimpleNamespace(task_spec=lambda t: {"prompt_set": "deepseek"})
    shared = SimpleNamespace(task_spec=lambda t: {})
    assert _repair_prompt(ds) is deepseek_relationship_repair
    assert _repair_prompt(shared) is PROMPTS["relationship_repair"]
    for t in DEEPSEEK_JUDGES:
        assert eval_judge._judge_prompt(ds, t) is DEEPSEEK_JUDGES[t]
        assert eval_judge._judge_prompt(shared, t) is PROMPTS[t]
    # same scoring contract as the shared judges
    system, user = DEEPSEEK_JUDGES["judge_gap_detection"]("q", [{"kind": "chunk", "iri": "x", "text": "t"}], "a", True)
    assert '"score"' in system and "EXPECTED_GAP: True" in user


def test_bench_pins_one_gpt41_judge_for_every_arm(tmp_path, monkeypatch) -> None:
    import run_arm

    monkeypatch.setenv("DEEPINFRA_FLASH_MODEL", "me/x")
    for arm in ("openai", "deepinfra-b1", "sl:Qwen/Qwen3.8-27B"):
        tasks = yaml.safe_load(run_arm.models_file(arm, tmp_path).read_text())["tasks"]
        for t in run_arm.JUDGE_TASKS:
            assert (tasks[t]["provider"], tasks[t]["model"]) == ("openai", "gpt-4.1"), (arm, t)
            assert "prompt_set" not in tasks[t]


def test_non_qwen_serverless_arms_drop_the_qwen_prompt(tmp_path) -> None:
    import run_arm

    other = yaml.safe_load(run_arm.models_file("sl:deepseek-ai/DeepSeek-V4-Pro", tmp_path).read_text())["tasks"]
    assert not any(v.get("prompt_set") for v in other.values())
    assert other["relationship_extract"]["model"] == "deepseek-ai/DeepSeek-V4-Pro"  # one model everywhere
    qwen = yaml.safe_load(run_arm.models_file("sl:Qwen/Qwen3.8-27B", tmp_path).read_text())["tasks"]
    assert qwen["class_proposal"]["prompt_set"] == "qwen"
    assert "prompt_set" not in qwen["relationship_extract"]  # deepseek prompt not given to Qwen
    assert qwen["embeddings"]["model"] == "Qwen/Qwen3-Embedding-4B"


def test_evidence_check_ignores_markdown_but_not_words() -> None:
    from backend.app.services.db_entity_extract import _evidence_norm

    chunk = "*   **Mechanism:** Mimic hormones (GLP-1, GIP) made by the small intestine."
    hay = _evidence_norm(chunk)
    assert _evidence_norm("Mechanism: Mimic hormones (GLP-1, GIP) made by the small intestine") in hay
    assert _evidence_norm("**Mechanism:** Mimic hormones") in hay          # copied markdown still fine
    assert _evidence_norm("Mechanism: Copies hormones made by the gut") not in hay  # paraphrase still rejected
    assert _evidence_norm("GLP-1, GIP") in hay                              # hyphens kept


def test_qwen_relationship_prompt_is_separate_and_selected_by_preset() -> None:
    from backend.app.services.db_entity_extract import _relationship_prompt
    from backend.app.services.prompts import PROMPTS
    from backend.app.services.prompts_qwen import (
        QWEN_RELATIONSHIP_RULES, qwen_class_proposal, qwen_relationship_extract,
    )

    args = ("**Ozempic:** Approved for type 2 diabetes.",
            [{"canonical_name": "Ozempic", "class_label": "Drug"}],
            [{"iri": "x#treats", "label": "treats", "domain_label": "Drug", "range_label": "Condition"}])
    shared, _ = PROMPTS["relationship_extract"](*args)
    qwen, _ = qwen_relationship_extract(*args)
    assert QWEN_RELATIONSHIP_RULES not in shared and qwen == shared + QWEN_RELATIONSHIP_RULES
    assert "CLASS OR INSTANCE" in qwen_class_proposal({"a": {"label": "A"}}, "t")[0]
    assert "CLASS OR INSTANCE" not in PROMPTS["class_proposal"]({"a": {"label": "A"}}, "t")[0]

    class _R:
        def __init__(self, spec): self.spec = spec
        def task_spec(self, t): return self.spec
    assert _relationship_prompt(_R({"prompt_set": "qwen"})) is qwen_relationship_extract
    assert _relationship_prompt(_R({})) is PROMPTS["relationship_extract"]
    assert _relationship_prompt(object()) is PROMPTS["relationship_extract"]


def test_deepseek_relationship_prompt_is_its_own_module() -> None:
    from backend.app.services import prompts_deepseek, prompts_qwen
    from backend.app.services.db_entity_extract import _relationship_prompt

    args = ("**Ozempic:** Approved for type 2 diabetes.",
            [{"canonical_name": "Ozempic", "class_label": "Drug"}],
            [{"iri": "x#treats", "label": "treats", "domain_label": "Drug", "range_label": "Condition"}])
    ds, _ = prompts_deepseek.deepseek_relationship_extract(*args)
    qw, _ = prompts_qwen.qwen_relationship_extract(*args)
    assert prompts_deepseek.DEEPSEEK_RELATIONSHIP_RULES in ds
    assert prompts_deepseek.DEEPSEEK_RELATIONSHIP_RULES is not prompts_qwen.QWEN_RELATIONSHIP_RULES
    assert ds == qw  # starts as a clone; diverges only when edited

    class _R:
        def task_spec(self, t): return {"prompt_set": "deepseek"}
    assert _relationship_prompt(_R()) is prompts_deepseek.deepseek_relationship_extract



# --------------------------------------------------------------------------- #
# Truncated-reply salvage (class_proposal)
# --------------------------------------------------------------------------- #

def test_salvage_keeps_complete_entries_and_drops_the_partial_one() -> None:
    from backend.app.helpers.json_salvage import salvage_truncated_json

    cut = ('{"MATCH NOT FOUND":[{"LABEL":"SleepApnea","PARENT_LABEL":"Disease"},'
           '{"LABEL":"Hypoglyc')
    assert salvage_truncated_json(cut) == {"MATCH NOT FOUND": [
        {"LABEL": "SleepApnea", "PARENT_LABEL": "Disease"}]}
    cut2 = ('{"MATCH NOT FOUND":[{"LABEL":"A"}],"MATCH NOT FOUND RELATIONS":[{"LABEL":"treats",'
            '"DOMAIN":"Drug"},{"LABEL":"cau')
    assert salvage_truncated_json(cut2) == {
        "MATCH NOT FOUND": [{"LABEL": "A"}],
        "MATCH NOT FOUND RELATIONS": [{"LABEL": "treats", "DOMAIN": "Drug"}]}
    assert salvage_truncated_json('{"A":[{"x":"a } ] { [ b"},{"x":"c') == {"A": [{"x": "a } ] { [ b"}]}
    assert salvage_truncated_json('{"A":[1,2]}') == {"A": [1, 2]}
    assert salvage_truncated_json('{"A":[{"x":') == {"A": []}
    assert salvage_truncated_json("no json here") is None


def test_salvaged_class_proposal_dedupes_and_drops_unevidenced_matches() -> None:
    from types import SimpleNamespace

    from backend.app.services.pipeline_llm import _salvage_class_proposal

    chunk = SimpleNamespace(index=3, source_name="doc.txt",
                            text="**Mechanism:** Ozempic causes nausea and fat loss.")
    reply = ('{"MATCH NOT FOUND":[{"LABEL":"Nausea"},{"LABEL":"Nausea"}],'
             '"MATCHES FOUND":[{"IRI":"x#Drug","TEXT_SNIPPET":"Ozempic"},'
             '{"IRI":"x#OCRE400015","TEXT_SNIPPET":"Collection"},{"IRI":"x#Trial","TEXT_SNIPPET":"fat los')
    out = _salvage_class_proposal(reply, chunk)
    assert out["MATCH NOT FOUND"] == [{"LABEL": "Nausea"}]
    assert out["MATCHES FOUND"] == [{"IRI": "x#Drug", "TEXT_SNIPPET": "Ozempic"}]


def test_salvage_is_opt_in() -> None:
    from backend.app.services.pipeline_llm import _salvage_enabled

    class _R:
        def __init__(self, spec): self.spec = spec
        def task_spec(self, t): return self.spec
    assert _salvage_enabled(_R({})) is False
    assert _salvage_enabled(object()) is False
    assert _salvage_enabled(_R({"salvage_truncated": True})) is True
    tasks = yaml.safe_load(open(PRESET))["tasks"]
    assert tasks["class_proposal"]["salvage_truncated"] is True
    for other in ("config/models.openai.example.yaml", "config/models.example.yaml",
                  "config/models.anthropic.example.yaml"):
        assert "salvage_truncated" not in yaml.safe_load(open(other))["tasks"]["class_proposal"]


# --------------------------------------------------------------------------- #
# Stage 3b: duplicate-class merge (similarity proposes, a judge confirms)
# --------------------------------------------------------------------------- #

def test_merge_groups_collapse_chains_to_one_canonical() -> None:
    from backend.app.services.class_synonym_merge import merge_groups

    # 0=1 (keep 0), 1=2 (keep 0's group member 1 -> votes favour 0)
    assert merge_groups(4, [(0, 1, 0), (1, 2, 1)]) in ({1: 0, 2: 0}, {0: 1, 2: 1})
    assert merge_groups(3, []) == {}


def test_apply_merges_rewrites_every_reference() -> None:
    from backend.app.services.class_synonym_merge import apply_merges

    payload = {
        "MATCH NOT FOUND": [
            {"LABEL": "Hypertension", "PARENT_LABEL": "Disease"},
            {"LABEL": "HighBloodPressure", "PARENT_LABEL": "Disease"},
            {"LABEL": "SevereHypertension", "PARENT_LABEL": "HighBloodPressure"},
        ],
        "MATCH NOT FOUND RELATIONS": [{"LABEL": "treats", "DOMAIN": "Drug", "RANGE": "High Blood Pressure"}],
        "MATCH NOT FOUND INSTANCES": [{"LABEL": "Stage 2", "TYPE_LABEL": ["HighBloodPressure", "Disease"]}],
    }
    out = apply_merges(payload, {1: 0})
    assert [c["LABEL"] for c in out["MATCH NOT FOUND"]] == ["Hypertension", "SevereHypertension"]
    assert out["MATCH NOT FOUND"][1]["PARENT_LABEL"] == "Hypertension"
    assert out["MATCH NOT FOUND RELATIONS"][0]["RANGE"] == "Hypertension"
    assert out["MATCH NOT FOUND INSTANCES"][0]["TYPE_LABEL"] == ["Hypertension", "Disease"]


@pytest.mark.asyncio
async def test_semantic_merge_only_merges_what_the_judge_confirms() -> None:
    from backend.app.services.class_synonym_merge import semantic_class_merge

    payload = {"MATCH NOT FOUND": [
        {"LABEL": "BeforeBreakfast"}, {"LABEL": "AfterBreakfast"},
        {"LABEL": "Hypertension"}, {"LABEL": "HighBloodPressure"}]}

    class _Emb:  # both pairs look near-identical to the embedder
        async def embed(self, texts):
            return [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]

    class _Router:
        def task_spec(self, t):
            return {"enabled": True, "prompt_set": "deepseek", "threshold": 0.83}

        async def chat(self, task, *, system, user):
            assert task == "class_synonym_judge" and "BeforeBreakfast" in user
            ids = {ln.split(".")[0]: ln for ln in user.splitlines() if ln[:1].isdigit()}
            dec = [{"id": int(k), "same": "Hypertension" in ln, "keep": "A", "reason": "r"} for k, ln in ids.items()]
            return ChatResult(text=json.dumps({"decisions": dec}), model="m", provider="deepinfra")

    out, merges = await semantic_class_merge(_Router(), payload, embedder=_Emb(), log=lambda *_: None)
    labels = [c["LABEL"] for c in out["MATCH NOT FOUND"]]
    assert "BeforeBreakfast" in labels and "AfterBreakfast" in labels      # opposites kept
    assert ("Hypertension" in labels) != ("HighBloodPressure" in labels)   # synonyms merged
    assert len(merges) == 1


@pytest.mark.asyncio
async def test_semantic_merge_is_off_unless_enabled_with_a_matching_prompt() -> None:
    from backend.app.services.class_synonym_merge import semantic_class_merge

    p = {"MATCH NOT FOUND": [{"LABEL": "A"}, {"LABEL": "B"}]}

    class _R:
        def __init__(self, spec): self.spec = spec
        def task_spec(self, t): return self.spec
    for spec in ({}, {"enabled": False, "prompt_set": "deepseek"}, {"enabled": True, "prompt_set": None}):
        out, merges = await semantic_class_merge(_R(spec), p, embedder=object(), log=lambda *_: None)
        assert out is p and merges == []



def test_identifier_shaped_strings_are_never_minted_as_classes() -> None:
    from backend.app.helpers.ontology_pruning import _is_garbage_endpoint, looks_like_iri

    for bad in ("http://purl.org/net/OCRe/OCRe.owl#OCRE400015",
                "http_purl_org_net_ocre_ocre_owl_ocre400010", "https://x.org/a#B"):
        assert looks_like_iri(bad) and _is_garbage_endpoint(bad)
    for good in ("Kharg Island", "HttpRequest", "Hypertension", "GLP-1 receptor agonist"):
        assert not looks_like_iri(good) and not _is_garbage_endpoint(good)


# --------------------------------------------------------------------------- #
# Table vision route
# --------------------------------------------------------------------------- #

class _FakePage:
    """pdfplumber page stand-in: crop -> to_image -> save(PNG)."""

    def crop(self, bbox):
        return self

    def to_image(self, resolution):
        return self

    def save(self, buf, format):
        buf.write(b"\x89PNG fake")


def _vision_settings(provider, model):
    return SimpleNamespace(models_config={"tasks": {"table_extract_vision": {
        "provider": provider, "model": model, "timeout": 5, "max_tokens": 64}}})


def test_vision_deepinfra_uses_deepinfra_client_and_prices_by_model(monkeypatch):
    import asyncio

    from backend.app.services import table_extract as te

    calls = []

    class _Comp:
        async def create(self, **kw):
            calls.append(kw)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content='```json\n{"no_table": true}\n```'))],
                usage=SimpleNamespace(prompt_tokens=1000, completion_tokens=100))

    fake = SimpleNamespace(chat=SimpleNamespace(completions=_Comp()))
    monkeypatch.setattr(te, "get_settings", lambda: _vision_settings("deepinfra", "Qwen/Qwen3-VL-30B-A3B-Instruct"))
    monkeypatch.setattr(te, "_get_deepinfra_vision_client", lambda: fake)
    monkeypatch.setattr(te, "_get_vision_client", lambda: pytest.fail("OpenAI client used for deepinfra"))
    payload, cost = asyncio.run(te._extract_table_via_vision(
        _FakePage(), (0, 0, 1, 1), doc_sha="x", table_index=0, page_number=1, caption_hint=None, router=None))
    assert payload is None  # fenced "no_table" reply parsed, then dropped
    assert calls[0]["model"] == "Qwen/Qwen3-VL-30B-A3B-Instruct"
    assert cost == pytest.approx(1000 * 0.15e-6 + 100 * 0.6e-6)


def test_vision_call_failure_raises_instead_of_silent_drop(monkeypatch):
    import asyncio

    from backend.app.services import table_extract as te

    class _Comp:
        async def create(self, **kw):
            raise RuntimeError("model not found")

    monkeypatch.setattr(te, "get_settings", lambda: _vision_settings("deepinfra", "Qwen/Qwen3-VL-30B-A3B-Instruct"))
    monkeypatch.setattr(te, "_get_deepinfra_vision_client",
                        lambda: SimpleNamespace(chat=SimpleNamespace(completions=_Comp())))
    with pytest.raises(te.VisionCallError):
        asyncio.run(te._extract_table_via_vision(
            _FakePage(), (0, 0, 1, 1), doc_sha="x", table_index=0, page_number=1, caption_hint=None, router=None))


def test_vision_unknown_provider_is_a_config_error(monkeypatch):
    import asyncio

    from backend.app.services import table_extract as te

    monkeypatch.setattr(te, "get_settings", lambda: _vision_settings("groq", "llama"))
    with pytest.raises(ValueError, match="not supported"):
        asyncio.run(te._extract_table_via_vision(
            _FakePage(), (0, 0, 1, 1), doc_sha="x", table_index=0, page_number=1, caption_hint=None, router=None))


def test_deepinfra_preset_vision_is_a_priced_serverless_model():
    from backend.app.services.llm_router import _PRICING_PER_1K

    spec = yaml.safe_load(Path(PRESET).read_text())["tasks"]["table_extract_vision"]
    assert spec["provider"] == "deepinfra"
    assert spec["model"] in _PRICING_PER_1K


def test_query_prefix_default_override_and_config(monkeypatch):
    from backend.app.services import embeddings as em

    s = SimpleNamespace(deepinfra_api_key="k", openai_api_key="k")
    q = em.Embedder(provider="deepinfra", model="Qwen/Qwen3-Embedding-4B", dim=1024, batch_size=10, settings=s)
    monkeypatch.setattr(em, "configured_embedding", lambda: {"provider": "deepinfra", "model": "Qwen/Qwen3-Embedding-4B",
                                                              "dim": 1024, "batch_size": 10})
    assert q.query_prefix.startswith("Instruct: ")              # default unchanged
    monkeypatch.setattr(em, "configured_embedding", lambda: {"provider": "deepinfra", "model": "Qwen/Qwen3-Embedding-4B",
                                                              "dim": 1024, "batch_size": 10, "query_prefix": ""})
    assert q.query_prefix == ""                                  # models.yaml can disable it
    o = em.Embedder(provider="openai", model="text-embedding-3-small", dim=1024, batch_size=10, settings=s)
    assert o.query_prefix == ""                                  # spec for another model is ignored
    q2 = em.Embedder(provider="deepinfra", model="Qwen/Qwen3-Embedding-4B", dim=1024, batch_size=10,
                     settings=s, query_prefix_override="Q: ")
    assert q2.query_prefix == "Q: "


# --------------------------------------------------------------------------- #
# render-init per LLM mode
# --------------------------------------------------------------------------- #

def test_render_env_for_deepinfra_mode_has_no_openai_key() -> None:
    from backend.app.cli.main import _render_backend_env

    env_file = {"DATABASE_URL": "postgresql://x", "BEARER_TOKEN": "t", "DEEPINFRA_API_KEY": "d",
                "OPENAI_API_KEY": "o", "GROQ_API_KEY": "g"}
    env = {e["key"]: e["value"] for e in _render_backend_env(env_file, "deepinfra", "https://f")}
    assert env["MODELS_CONFIG_PATH"] == "config/models.deepinfra.example.yaml"
    assert env["DEEPINFRA_API_KEY"] == "d" and env["FRONTEND_ORIGIN"] == "https://f"
    assert "OPENAI_API_KEY" not in env and "GROQ_API_KEY" not in env


def test_render_env_default_mode_is_unchanged() -> None:
    from backend.app.cli.main import _render_backend_env

    env_file = {"DATABASE_URL": "postgresql://x", "BEARER_TOKEN": "t", "OPENAI_API_KEY": "o", "GROQ_API_KEY": "g"}
    keys = [e["key"] for e in _render_backend_env(env_file, "groq-openai", "https://f")]
    assert set(keys) == {"DATABASE_URL", "BEARER_TOKEN", "FRONTEND_ORIGIN", "LOG_LEVEL", "ENV",
                         "OPENAI_API_KEY", "GROQ_API_KEY"}


def test_synonym_candidate_pairs_blockwise_match_the_full_matrix(monkeypatch) -> None:
    """Row blocks + running trim (the full n x n product OOM-killed the 27B
    web-search prune-expand at ~11k classes) return the same best pairs."""
    import numpy as np

    import backend.app.services.class_synonym_merge as csm

    rng = np.random.default_rng(0)
    m = rng.normal(size=(300, 16)).astype(np.float32)
    m[:40] = m[40:80] + 0.01 * rng.normal(size=(40, 16))
    m /= np.linalg.norm(m, axis=1, keepdims=True)
    sims = m @ m.T
    iu, ju = np.triu_indices(len(m), k=1)
    vals = sims[iu, ju]
    keep = np.nonzero(vals >= 0.9)[0]
    want = keep[np.argsort(-vals[keep], kind="stable")][:25]
    monkeypatch.setattr(csm, "_PAIR_BLOCK_ROWS", 7)
    got = csm.candidate_pairs(m, 0.9, 25)
    assert [(i, j) for i, j, _ in got] == [(int(iu[k]), int(ju[k])) for k in want]
    assert all(i < j for i, j, _ in got)
    assert csm.candidate_pairs(m[:1], 0.9, 25) == []


# --------------------------------------------------------------------------- #
# Qwen class budget + declared lists
# --------------------------------------------------------------------------- #

def test_class_rules_prompt_and_preset_guard() -> None:
    """v4 (2026-10-18): no class number in the prompt; the preset keeps a
    hidden runaway guard; the per-list instance bound matches the prompt."""
    from backend.app.services import prompts_qwen as pq

    spec = yaml.safe_load(open(PRESET))["tasks"]["class_proposal"]
    assert spec["max_new_classes"] == 60 and "max_new_classes_with_lists" not in spec
    assert spec["max_instances_per_list"] == pq.QWEN_INSTANCES_PER_LIST
    system, user = pq.qwen_class_proposal({}, "text")
    assert "AT MOST 25" not in system and "new classes in total" not in system
    for rule in ("EXISTING CLASSES FIRST", "WHERE RELEVANT", "DRUGS ARE NEVER CLASSES", '"LISTS"',
                 "clinical studies and trials"):
        assert rule in system, rule
    # The contradictory example (a class for 'Washington' / 'BMW') is gone.
    assert "'Country' for 'Washington'" not in system
    assert user.index('"LISTS"') < user.index('"MATCH NOT FOUND"')


def _chunk():
    from types import SimpleNamespace

    return SimpleNamespace(index=0, source_name="doc.txt")


def test_class_budget_keeps_the_first_entries() -> None:
    from backend.app.services.pipeline_llm import _apply_class_budget

    spec = {"max_new_classes": 25, "max_new_classes_with_lists": 40, "max_instances_per_list": 40}
    parsed = {"MATCH NOT FOUND": [{"LABEL": f"C{i}"} for i in range(300)],
              "MATCH NOT FOUND INSTANCES": [{"LABEL": f"I{i}"} for i in range(10)]}
    out = _apply_class_budget(parsed, spec, _chunk())
    assert [c["LABEL"] for c in out["MATCH NOT FOUND"]] == [f"C{i}" for i in range(25)]
    assert len(out["MATCH NOT FOUND INSTANCES"]) == 10


def test_a_declared_list_raises_the_budget_and_allows_its_instances() -> None:
    from backend.app.services.pipeline_llm import _apply_class_budget

    spec = {"max_new_classes": 25, "max_new_classes_with_lists": 40, "max_instances_per_list": 40,
            "max_new_instances": 80}
    parsed = {"LISTS": [{"WHAT": "oil refineries in Asia", "ITEM_KIND": "OilRefinery", "ITEM_COUNT": 300}],
              "MATCH NOT FOUND": [{"LABEL": f"C{i}"} for i in range(60)],
              "MATCH NOT FOUND INSTANCES": [{"LABEL": f"R{i}"} for i in range(300)]}
    out = _apply_class_budget(parsed, spec, _chunk())
    assert len(out["MATCH NOT FOUND"]) == 40
    assert len(out["MATCH NOT FOUND INSTANCES"]) == 80 + 40


def test_no_budget_configured_changes_nothing() -> None:
    from backend.app.services.pipeline_llm import _apply_class_budget

    parsed = {"MATCH NOT FOUND": [{"LABEL": f"C{i}"} for i in range(300)]}
    assert len(_apply_class_budget(parsed, {}, _chunk())["MATCH NOT FOUND"]) == 300
    assert _apply_class_budget(None, {"max_new_classes": 25}, _chunk()) is None


def test_both_class_prompts_state_the_adverse_event_hierarchy() -> None:
    from backend.app.services import adverse_event_taxonomy as ae
    from backend.app.services.prompts_deepseek import deepseek_class_proposal
    from backend.app.services.prompts_qwen import qwen_class_proposal

    for fn in (qwen_class_proposal, deepseek_class_proposal):
        system, _ = fn({}, "text")
        assert "only the hierarchy's CATEGORIES are classes" in system
        assert "never a class. A list of adverse reactions has ITEMS_ARE_KINDS false" in system
        assert "DRUGS ARE NEVER CLASSES" in system and "relation endpoint" in system
        assert all(t in system for t in ae.TYPES) and ae.ROOT in system
    spec = yaml.safe_load(open(PRESET))["tasks"]["class_proposal"]
    assert spec["adverse_event_taxonomy"] is True
    assert spec["ontology_view"] == {"full_max_classes": 600, "subtree_max_classes": 600}


def test_ontology_view_sends_whole_small_ontology_and_subtrees_of_a_large_one() -> None:
    from backend.app.services.pipeline_llm import _ontology_view

    def cls(name, parent=None):
        return {"name": name, "labels": [name],
                "superclasses": [{"iri": parent}] if parent else [], "comments": ["a " + name]}
    C = {"r1": cls("Root1"), "a": cls("A", "r1"), "a1": cls("A1", "a"), "a2": cls("A2", "a1"),
         "r2": cls("Root2"), "b": cls("B", "r2")}
    onto = {"classes_dict": C}
    full = _ontology_view(onto, ["r1"], 1, {"full_max_classes": 10})
    assert set(full) == set(C) and full["a2"] == {"label": "A2", "parents": ["a1"], "desc": "a A2"}
    sub = _ontology_view(onto, ["r1"], 1, {"full_max_classes": 3, "subtree_max_classes": 10})
    assert set(sub) == {"r1", "a", "a1", "a2"}            # whole subtree, nothing from Root2
    capped = _ontology_view(onto, ["r1"], 1, {"full_max_classes": 3, "subtree_max_classes": 2})
    assert set(capped) == {"r1", "a"}


def test_concurrency_is_capped_by_local_memory() -> None:
    """The GPUs set how many requests the deployment can serve; this machine
    must also hold every in-flight prompt + reply (2026-10-19). A request is
    ~0.6 MB even at a 30k-token context, so this only binds on small boxes
    (on the 2.7 GB dev VM ~700 class proposals fit; the GPUs take 192)."""
    stages = {"class_proposal": {"mean_ctx": 30000}, "entity_extraction": {"mean_ctx": 3000},
              "dedup": {"mean_ctx": 24000}}
    gpu = {"class_proposal": 192, "entity_extraction": 192, "dedup": 192}
    small = sizing.local_memory((900, 1024))                      # a 1 GB box: 300 MB usable
    big = sizing.local_memory((30000, 32768))
    c_small, caps = sizing.memory_capped_concurrency(gpu, stages, small)
    c_big, caps_big = sizing.memory_capped_concurrency(gpu, stages, big)
    assert caps["class_proposal"] < 192 and c_small["class_proposal"] == caps["class_proposal"]
    assert c_small["entity_extraction"] > c_small["class_proposal"]       # short contexts fit more
    assert caps_big == {} and c_big["class_proposal"] == 192              # a big box keeps the GPU value
    assert c_big["class_judges"] == sizing.CLASS_JUDGE_MAX                # judges: dedup, at most 64
    assert 1 <= c_small["table_extraction"] <= c_big["table_extraction"] == 8
    no_mem, none_caps = sizing.memory_capped_concurrency(gpu, stages, None)
    assert none_caps == {} and "table_extraction" not in no_mem and no_mem["class_judges"] == 64
    assert sizing.max_streaming_batch(small) < sizing.max_streaming_batch(big)
    assert sizing.max_streaming_batch(None) is None


def test_deepinfra_mode_uses_only_deepinfra() -> None:
    """DeepInfra mode assumes the user holds NO other key: every task (judges,
    compaction, table vision, embeddings) and every fallback is DeepInfra, and
    selecting the mode asks for no other provider's key."""
    root = Path(__file__).resolve().parents[3]
    doc = yaml.safe_load((root / "config" / "models.deepinfra.example.yaml").read_text(encoding="utf-8"))
    off = {name: spec.get("provider") for name, spec in doc["tasks"].items()
           if spec.get("provider") != "deepinfra" or spec.get("fallback_provider", "deepinfra") != "deepinfra"}
    assert not off, f"non-DeepInfra task(s) in DeepInfra mode: {off}"
    import select_llm_mode

    keys = select_llm_mode.MODES["deepinfra"][1]
    assert not [k for k in keys if k.split("_")[0] in {"OPENAI", "ANTHROPIC", "GROQ"}], keys
