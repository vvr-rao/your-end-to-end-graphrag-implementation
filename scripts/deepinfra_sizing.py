#!/usr/bin/env python3
"""Size a DEDICATED DeepInfra deployment for the deepinfra/Qwen mode, then
recommend `concurrency:` -- the GPU counterpart of tpm_check.py.

    uv run python scripts/deepinfra_sizing.py <docs-dir>[,<docs-dir>...] [--deadline-hours H]
        [--selection F] [--precision fp8|bf16|any] [--gpu H200-141GB] [--out sizing.json]
    uv run python scripts/deepinfra_sizing.py --probe [--apply]   # deployment is UP
    uv run python scripts/deepinfra_sizing.py --calibrate         # refit workload ratios

WHY A DIFFERENT TOOL: OpenAI/Anthropic/Groq meter you by TPM/RPM, so
tpm_check sizes concurrency from rate-limit headers. A dedicated deployment
has no rate limit -- you rent GPUs by the minute -- so the questions become:

  1. How much work is this corpus?   corpus tokens -> calls + in/out tokens
                                     per creation stage (calibrated ratios).
  2. Which GPUs?                     every configuration that FITS the model
                                     within the account's 4-GPU limit, with
                                     KV-cache headroom, $/hour, projected wall
                                     time and total cost. The recommendation
                                     follows PREFERRED_GPUS (2x B200, then
                                     2x H200, then 2x H100); the others are the
                                     fallback chain deepinfra_deploy.py walks
                                     when a GPU type has no capacity.
  3. How many requests at once?      per configuration, the concurrency that
                                     maximises throughput within its KV cache
                                     (model calibrated on the measured builds);
                                     a --probe load sweep on the live
                                     deployment refines it. That number, not a
                                     rate limit, sets `concurrency:`.

Step 2 is an A-PRIORI estimate (memory-bandwidth / FLOP model, roughly
+-50%); it picks the hardware. Step 3's a-priori values are written for every
configuration in the chain, so whichever GPU deepinfra_deploy.py obtains gets
its own; --probe MEASURES. Nothing is deployed here: deepinfra_deploy.py
consumes the `sizing.json` this writes, and only after the user approves the
spend.

The deployed model is Qwen3.8-27B (FP8, dense; --model flash-next sizes the
original Qwen3.8-Flash-Next MoE). Q&A runs on SERVERLESS DeepSeek-V4.1-Flash
(per token, no GPUs to size); its `evaluation` concurrency is bounded by
DeepInfra's per-model concurrent request limit (--serverless-limit).

Reads only whether keys are present; never prints key values.
"""
from __future__ import annotations

import asyncio
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parent.parent
CALIBRATION_FILE = ROOT / ".build" / "sizing_calibration.json"
SPEC_URL = "https://huggingface.co/{repo}/resolve/main/config.json"

# --------------------------------------------------------------------------- #
# Hardware + model facts
# --------------------------------------------------------------------------- #

# DeepInfra custom-LLM GPUs (deepinfra.com/pricing, 2026-10; billed per minute).
# mem_gb is what the deploy API names; bw = HBM TB/s; tflops dense bf16/fp8.
GPU_CATALOG: dict[str, dict] = {
    "A100-80GB": {"mem_gb": 80, "usd_hr": 0.89, "bw": 2.0, "bf16": 312, "fp8": None},
    "H100-80GB": {"mem_gb": 80, "usd_hr": 2.20, "bw": 3.35, "bf16": 989, "fp8": 1979},
    "H200-141GB": {"mem_gb": 141, "usd_hr": 2.69, "bw": 4.8, "bf16": 989, "fp8": 1979},
    "B200-180GB": {"mem_gb": 180, "usd_hr": 3.69, "bw": 8.0, "bf16": 2250, "fp8": 4500},
    # Names must match the deploy API's `gpu` enum exactly (it 422s otherwise);
    # the docs say "B300-288GB" but the API accepts only B300-270GB.
    "B300-270GB": {"mem_gb": 270, "usd_hr": 4.89, "bw": 8.0, "bf16": 2250, "fp8": 4500},
}
MAX_GPUS_PER_ACCOUNT = 4  # DeepInfra default; support can raise it.

# Preferred hosting, in order (user-set 2026-10-17). The first that fits is
# recommended; the rest are the fallback chain for `no-gpu-available` / 409.
PREFERRED_GPUS: tuple[tuple[str, int], ...] = (
    ("B200-180GB", 2), ("H200-141GB", 2), ("H100-80GB", 2),
)

# Models that can be deployed. Checkpoint sizes from the Hugging Face
# safetensors index; fallback architecture from config.json (both 2026-10-12/17),
# used when the live config cannot be fetched.
MODELS: dict[str, dict] = {
    # What the deepinfra preset deploys. Dense 27B, but hybrid like Flash-Next:
    # 16 full-attention (+1 MTP) layers carry per-token KV, 48 Gated-DeltaNet
    # layers carry a fixed recurrent state.
    "27b": {
        "label": "Qwen3.8-27B", "dense": True, "active_params": 27e9,
        "checkpoints": {"fp8": {"repo": "Qwen/Qwen3.8-27B-FP8", "weights_gb": 30.9}},
        "fallback_arch": {
            "full_attention_layers": 17, "linear_attention_layers": 48,
            "num_key_value_heads": 4, "head_dim": 256,
            "linear_num_value_heads": 48, "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
        },
    },
    "flash-next": {
        "label": "Qwen3.8-Flash-Next", "dense": False,
        "active_params": 6e9,  # MoE active per token (125B total, 512 experts, top-10)
        "checkpoints": {
            "bf16": {"repo": "Qwen/Qwen3.8-Flash-Next", "weights_gb": 360.0},
            "fp8": {"repo": "Qwen/Qwen3.8-Flash-Next-FP8", "weights_gb": 185.5},
        },
        "fallback_arch": {
            "full_attention_layers": 13,  # 12 + 1 MTP layer
            "linear_attention_layers": 36,
            "num_key_value_heads": 2, "head_dim": 256,
            "linear_num_value_heads": 48, "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
        },
    },
}
DEFAULT_MODEL = "27b"
CHECKPOINTS = MODELS[DEFAULT_MODEL]["checkpoints"]
FALLBACK_ARCH = MODELS[DEFAULT_MODEL]["fallback_arch"]
ACTIVE_PARAMS = MODELS[DEFAULT_MODEL]["active_params"]
N_EXPERTS, TOP_K = 512, 10          # Flash-Next routing
DENSE_FRACTION = 0.08  # MoE: attention/shared/embeddings read on EVERY decode step

GPU_MEMORY_UTILIZATION = 0.90   # vLLM default
RUNTIME_OVERHEAD_GB_PER_GPU = 5  # activations, CUDA graphs, NCCL buffers
TP_EFFICIENCY = 0.85             # tensor-parallel communication loss
BW_EFFICIENCY = 0.60
# Effective compute efficiency under the pipeline's real load (prefill-heavy:
# extraction sends ~8 input tokens per output token). Fitted 2026-10-17 on the
# dedicated-27B builds: 2x B200 generate-artifacts at 63 in flight = 3,760 out
# + 7,600 in tok/s; 2x H100 extract-entities at 39 in flight = 850 out +
# 6,800 in tok/s (0.07-0.15 across steps). Single-stream decode matched the
# bandwidth model (B200 125, H200 ~115, H100 ~70 tok/s measured).
MFU = 0.12
STEP_OVERHEAD_S = 0.004
# The deploy API accepted max_batch_size 128 in every run; higher is untested,
# so a-priori concurrency stops here (a --probe sweep may justify more).
MAX_BATCH = 128
DEFAULT_SPINUP_MIN = 20          # pull ~185-360 GB + load; billed
# GPU time billed while the pipeline is NOT calling the model: DB writes,
# embedding, chunking, the gaps between CLI commands. Per document, plus a
# fixed floor. Rough; the benchmark measures it (GPU ledger vs call_seconds).
DEFAULT_NON_LLM_MIN = 10
NON_LLM_MIN_PER_DOC = 0.3

# --------------------------------------------------------------------------- #
# Workload model: per creation task, calls per 1M corpus tokens and mean
# in/out tokens per call. prune-expand rows were fitted (--calibrate) on the
# 20 saved OpenAI prune-expand runs (2026-09-05..09-27, 5k-425k-token
# corpora); summarization rows match the structural count (1 window per
# 11.5k tokens; question-gen + 1 chunk + <=3 evaluate + <=2 revise per window).
# Phase-2 rows are STRUCTURAL ESTIMATES per summary chunk until a benchmark
# run's GRAPHRAG_COST_REPORT_DIR dumps refit them -- the report says which.
# --------------------------------------------------------------------------- #

DEFAULT_RATIOS: dict[str, dict] = {
    # stage: summarization
    "evaluated_summary_chunk": {"stage": "summarization", "per_mtok": 87, "in": 6346, "out": 1633, "src": "fit"},
    "summary_question_gen": {"stage": "summarization", "per_mtok": 87, "in": 3115, "out": 773, "src": "fit"},
    "summary_evaluate": {"stage": "summarization", "per_mtok": 198, "in": 2877, "out": 777, "src": "fit"},
    "summary_revise": {"stage": "summarization", "per_mtok": 113, "in": 6437, "out": 2105, "src": "fit"},
    # prune-expand
    "chunk_classification": {"stage": "chunk_classification", "per_mtok": 440, "in": 3564, "out": 412, "src": "fit"},
    "class_proposal": {"stage": "class_proposal", "per_mtok": 440, "in": 20192, "out": 6298, "src": "fit"},
    "match_dedup": {"stage": "dedup", "per_mtok": 78, "in": 15497, "out": 8412, "src": "fit"},
    "concept_grouping": {"stage": "class_proposal", "per_mtok": 15, "in": 4691, "out": 1674, "src": "fit"},
    "classification_audit": {"stage": "class_proposal", "per_mtok": 15, "in": 1838, "out": 1042, "src": "fit"},
    "table_concept_grouping": {"stage": "table_mining", "per_mtok": 100, "in": 822, "out": 336, "src": "fit"},
    # Phase 2: ~1 summary chunk (800 tok) per 2.4k corpus tokens -> ~420/Mtok.
    "entity_extract": {"stage": "entity_extraction", "per_mtok": 420, "in": 2500, "out": 600, "src": "structural"},
    "concept_extract": {"stage": "entity_extraction", "per_mtok": 420, "in": 2500, "out": 400, "src": "structural"},
    "entity_validate": {"stage": "entity_extraction", "per_mtok": 420, "in": 2000, "out": 300, "src": "structural"},
    "relationship_extract": {"stage": "entity_extraction", "per_mtok": 420, "in": 3000, "out": 700, "src": "structural"},
    "relationship_verify": {"stage": "entity_extraction", "per_mtok": 420, "in": 2500, "out": 400, "src": "structural"},
    "relationship_orphan_check": {"stage": "entity_extraction", "per_mtok": 40, "in": 4000, "out": 800, "src": "structural"},
    "artifact_chunk_extract_with_entities": {"stage": "artifact_generation", "per_mtok": 420, "in": 3000, "out": 1200, "src": "structural"},
    "artifact_document_summary": {"stage": "artifact_generation", "per_mtok": 30, "in": 8000, "out": 1500, "src": "structural"},
    "artifact_merge": {"stage": "artifact_generation", "per_mtok": 60, "in": 6000, "out": 2000, "src": "structural"},
    "summary_merge": {"stage": "artifact_generation", "per_mtok": 30, "in": 8000, "out": 2500, "src": "structural"},
    "insight_gen": {"stage": "artifact_generation", "per_mtok": 20, "in": 6000, "out": 1500, "src": "structural"},
}

# Stages whose calls can carry a 32k-token output budget (models.yaml).
LARGE_OUTPUT_STAGES = {"class_proposal", "dedup"}
LARGE_OUTPUT_TOKENS = 32_768


# --------------------------------------------------------------------------- #
# 1. Corpus -> workload
# --------------------------------------------------------------------------- #

def corpus_tokens(docs_dir: str, selection: str | None = None) -> tuple[int, int, list[int]]:
    """(documents, total tokens, per-doc tokens) using the summarizer's encoder."""
    from backend.app.services.document_io import load_documents
    from backend.app.services.evaluated_summarizer import get_encoder

    enc = get_encoder()
    docs = list(load_documents(Path(docs_dir)))
    if selection:
        from tpm_check import _selected_paths

        chosen = _selected_paths(selection)
        if chosen is not None:
            docs = [d for d in docs if str(d.path.resolve()) in chosen]
    toks = [len(enc.encode(d.text, disallowed_special=())) for d in docs]
    return len(docs), sum(toks), toks


def load_ratios() -> dict[str, dict]:
    ratios = {k: dict(v) for k, v in DEFAULT_RATIOS.items()}
    if CALIBRATION_FILE.exists():
        try:
            fitted = json.loads(CALIBRATION_FILE.read_text())["ratios"]
            for task, row in fitted.items():
                ratios.setdefault(task, {}).update(row)
        except (OSError, ValueError, KeyError):
            pass
    return ratios


def project_workload(total_tokens: int, ratios: dict[str, dict]) -> dict[str, dict]:
    """Per STAGE: calls, input/output tokens, mean context per call."""
    mtok = total_tokens / 1e6
    stages: dict[str, dict] = {}
    for task, r in ratios.items():
        calls = r["per_mtok"] * mtok
        st = stages.setdefault(
            r["stage"],
            {"calls": 0.0, "in_tokens": 0.0, "out_tokens": 0.0, "tasks": [], "structural": False},
        )
        st["calls"] += calls
        st["in_tokens"] += calls * r["in"]
        st["out_tokens"] += calls * r["out"]
        st["tasks"].append(task)
        st["structural"] |= r.get("src") == "structural"
    for st in stages.values():
        n = max(st["calls"], 1e-9)
        st["mean_ctx"] = (st["in_tokens"] + st["out_tokens"]) / n
        st["mean_out"] = st["out_tokens"] / n
    return stages


# --------------------------------------------------------------------------- #
# 2. GPU configurations
# --------------------------------------------------------------------------- #

def fetch_arch(repo: str, fallback: dict | None = None) -> dict:
    """KV-relevant architecture from the HF config; `fallback` (default
    FALLBACK_ARCH) on failure."""
    try:
        import urllib.request

        with urllib.request.urlopen(SPEC_URL.format(repo=repo), timeout=15) as r:
            cfg = json.load(r)
        t = cfg.get("text_config", cfg)
        types = t["layer_types"]
        mtp = int(t.get("mtp_num_hidden_layers", 0))
        return {
            "full_attention_layers": types.count("full_attention") + mtp,
            "linear_attention_layers": types.count("linear_attention"),
            "num_key_value_heads": t["num_key_value_heads"], "head_dim": t["head_dim"],
            "linear_num_value_heads": t["linear_num_value_heads"],
            "linear_key_head_dim": t["linear_key_head_dim"],
            "linear_value_head_dim": t["linear_value_head_dim"],
        }
    except Exception:
        return dict(fallback or FALLBACK_ARCH)


def kv_bytes_per_token(arch: dict) -> float:
    """Full-attention KV (bf16) per token; linear layers keep no per-token KV."""
    return arch["full_attention_layers"] * 2 * arch["num_key_value_heads"] * arch["head_dim"] * 2


def state_bytes_per_seq(arch: dict) -> float:
    """Gated-DeltaNet recurrent state per sequence (fp32), independent of length.

    ~113 MB for Flash-Next -- it dominates short calls, so slot counts must not
    be computed from per-token KV alone.
    """
    return (arch["linear_attention_layers"] * arch["linear_num_value_heads"]
            * arch["linear_key_head_dim"] * arch["linear_value_head_dim"] * 4)


def seq_gb(arch: dict, ctx_tokens: float) -> float:
    return (kv_bytes_per_token(arch) * ctx_tokens + state_bytes_per_seq(arch)) / 1e9


@dataclass
class GpuOption:
    gpu: str
    num_gpus: int          # per instance
    instances: int
    precision: str
    repo: str
    weights_gb: float
    kv_free_gb: float      # per instance
    usd_hr: float          # all instances
    note: str = ""
    dense: bool = False    # every decode step reads ALL weights (no expert routing)
    active_params: float = MODELS["flash-next"]["active_params"]

    @property
    def total_gpus(self) -> int:
        return self.num_gpus * self.instances

    def label(self) -> str:
        inst = f" x{self.instances} inst" if self.instances > 1 else ""
        return f"{self.num_gpus}x {self.gpu} {self.precision}{inst}"


def gpu_options(precision: str = "any", max_gpus: int = MAX_GPUS_PER_ACCOUNT,
                model: str = DEFAULT_MODEL) -> list[GpuOption]:
    """Every (gpu, count, replicas, precision) whose weights fit with KV room."""
    m = MODELS[model]
    out: list[GpuOption] = []
    for prec, ck in m["checkpoints"].items():
        if precision not in ("any", prec):
            continue
        for gpu, g in GPU_CATALOG.items():
            note = ""
            if prec == "fp8" and g["fp8"] is None:
                note = "no native FP8 (weight-only Marlin path, slower prefill)"
            for n in (1, 2, 4, 8):
                if n > max_gpus:
                    break
                usable = n * g["mem_gb"] * GPU_MEMORY_UTILIZATION - n * RUNTIME_OVERHEAD_GB_PER_GPU
                kv = usable - ck["weights_gb"]
                # Require room for at least 8 mean-length sequences, else the
                # config "fits" but serves almost nothing concurrently.
                if kv < 8:
                    continue
                for inst in range(1, max_gpus // n + 1):
                    out.append(GpuOption(
                        gpu=gpu, num_gpus=n, instances=inst, precision=prec,
                        repo=ck["repo"], weights_gb=ck["weights_gb"],
                        kv_free_gb=round(kv, 1), usd_hr=round(n * inst * g["usd_hr"], 2),
                        note=note, dense=m["dense"], active_params=m["active_params"],
                    ))
    return out


def kv_slots(opt: GpuOption, arch: dict, ctx_tokens: float) -> int:
    """Concurrent sequences of `ctx_tokens` that fit, across all instances."""
    return int(opt.kv_free_gb // seq_gb(arch, ctx_tokens)) * opt.instances


def _flops(opt: GpuOption) -> float:
    g = GPU_CATALOG[opt.gpu]
    return opt.num_gpus * (g["fp8"] if opt.precision == "fp8" and g["fp8"] else g["bf16"]) * 1e12


def step_seconds(opt: GpuOption, batch: int, prefill_ratio: float = 0.0) -> float:
    """One decode step of ONE instance with `batch` sequences in flight.

    Weight reads (memory-bound): a dense model reads every weight each step;
    a MoE reads the dense part plus every expert ANY sequence routes to
    (1-(1-k/E)^b of them). Compute: each in-flight sequence's token, plus the
    prefill work its request carries -- `prefill_ratio` input tokens per output
    token, interleaved with decode (continuous batching). Under the pipeline's
    prefill-heavy load the compute term is what saturates, not bandwidth.
    """
    g = GPU_CATALOG[opt.gpu]
    touched = 1.0 if opt.dense else (
        DENSE_FRACTION + (1 - DENSE_FRACTION) * (1 - (1 - TOP_K / N_EXPERTS) ** batch))
    bytes_step = opt.weights_gb * 1e9 * touched
    bw = opt.num_gpus * g["bw"] * 1e12 * BW_EFFICIENCY * (TP_EFFICIENCY if opt.num_gpus > 1 else 1)
    compute = batch * (1 + prefill_ratio) * 2 * opt.active_params / (_flops(opt) * MFU)
    return bytes_step / bw + compute + STEP_OVERHEAD_S


def decode_tps(opt: GpuOption, batch: int, prefill_ratio: float = 0.0) -> float:
    """Aggregate output tokens/s of ONE instance at `batch` concurrent requests."""
    return batch / step_seconds(opt, batch, prefill_ratio)


def prefill_tps(opt: GpuOption) -> float:
    return _flops(opt) * MFU / (2 * opt.active_params)


def estimate_run(opt: GpuOption, arch: dict, stages: dict[str, dict],
                 spinup_min: float = DEFAULT_SPINUP_MIN, non_llm_min: float = DEFAULT_NON_LLM_MIN) -> dict:
    """Wall time + cost for all creation stages, run one stage after another.

    A stage takes the longer of (a) its tokens at the batch the KV cache and
    its call count allow, and (b) its long tail: the slowest single call
    (~2x the mean output) decoding at single-stream speed. (b) is why a
    15-call dedup stage cannot be shortened by adding GPUs.
    """
    seconds = 0.0
    per_stage: dict[str, dict] = {}
    single = decode_tps(opt, 1)
    for name, st in stages.items():
        if st["calls"] < 0.5:
            continue
        slots = kv_slots(opt, arch, st["mean_ctx"]) // opt.instances
        batch = max(1, min(slots, MAX_BATCH, math.ceil(st["calls"] / opt.instances)))
        ratio = st["in_tokens"] / max(st["out_tokens"], 1.0)
        bulk = st["out_tokens"] / (decode_tps(opt, batch, ratio) * opt.instances)
        tail = 2 * st["mean_out"] / single
        s = max(bulk, tail)
        per_stage[name] = {"seconds": round(s), "batch": batch, "tail_bound": tail > bulk}
        seconds += s
    hours = seconds / 3600 + (spinup_min + non_llm_min) / 60
    return {"hours": round(hours, 2), "usd": round(hours * opt.usd_hr, 2),
            "llm_hours": round(seconds / 3600, 2), "stages": per_stage}


def preferred_chain(rows: list[tuple[GpuOption, dict]], deadline_h: float | None,
                    preferred: tuple[tuple[str, int], ...] = PREFERRED_GPUS) -> list[tuple[GpuOption, dict]]:
    """The PREFERRED_GPUS configurations that fit (single instance, FP8 when
    the model has it) and meet the deadline, in preference order."""
    out = []
    for gpu, n in preferred:
        cands = [r for r in rows if r[0].gpu == gpu and r[0].num_gpus == n and r[0].instances == 1
                 and (deadline_h is None or r[1]["hours"] <= deadline_h)]
        cands.sort(key=lambda r: (r[0].precision != "fp8", r[1]["usd"]))
        if cands:
            out.append(cands[0])
    return out


def recommend(rows: list[tuple[GpuOption, dict]], deadline_h: float | None,
              allow_noted: bool = False,
              preferred: tuple[tuple[str, int], ...] = PREFERRED_GPUS) -> tuple[GpuOption, dict] | None:
    """The first PREFERRED_GPUS configuration that fits and meets the deadline;
    if none does, the cheapest option that does. Options carrying a caveat
    (A100 FP8 = an unproven weight-only kernel path) are listed but not
    recommended unless `allow_noted`."""
    chain = preferred_chain(rows, deadline_h, preferred)
    if chain:
        return chain[0]
    ok = [r for r in rows
          if (deadline_h is None or r[1]["hours"] <= deadline_h)
          and (allow_noted or not r[0].note)]
    if not ok:
        return None
    return min(ok, key=lambda r: (r[1]["usd"], r[1]["hours"]))


# --------------------------------------------------------------------------- #
# 3. Concurrency
# --------------------------------------------------------------------------- #

CONCURRENCY_LEVELS = (8, 16, 24, 32, 48, 64, 96, 128)


def a_priori_concurrency(opt: GpuOption, arch: dict, stages: dict[str, dict],
                         timeouts: dict[str, list[tuple[int, float]]] | None = None) -> dict[str, int]:
    """Per-stage concurrency that maximises throughput before a probe.

    The smallest level reaching 90% of the modeled peak (find_knee on the
    step model at the stage's own input/output ratio), capped by the KV slots
    the stage's context leaves and by MAX_BATCH. `timeouts` (from
    nonstreamed_timeouts: per stage, (timeout, mean output) of each
    non-streamed task, whose limit is the WHOLE call) also caps it so each
    task's slow call (~2x its mean output) finishes inside half its timeout.

    Measured on the 27B builds: at 64 in flight 2x B200 was still scaling
    linearly, while prefill-heavy extraction on 2x H100 ran at ~1/3 of
    single-stream speed per request -- so B200 gets KV-limited values and
    H100 compute-limited ones.
    """
    out: dict[str, int] = {}
    for name, st in stages.items():
        ctx = st["mean_ctx"]
        if name in LARGE_OUTPUT_STAGES:
            # A reply may run to the full budget; reserve for in + p95-ish out.
            ctx = st["in_tokens"] / max(st["calls"], 1e-9) + LARGE_OUTPUT_TOKENS / 2
        cap = min(MAX_BATCH, kv_slots(opt, arch, ctx) // opt.instances)
        ratio = st["in_tokens"] / max(st["out_tokens"], 1.0)
        levels = [c for c in CONCURRENCY_LEVELS if c <= cap] or [max(1, cap)]
        curve = [{"concurrency": c, "tok_s": decode_tps(opt, c, ratio)} for c in levels]
        c = find_knee(curve)
        for limit, slow_out in (timeouts or {}).get(name, ()):
            while c > 1 and slow_out * step_seconds(opt, c, ratio) > 0.5 * limit:
                c = max(1, c // 2)
        out[name] = max(1, c * opt.instances)
    return out


def find_knee(curve: list[dict], frac: float = 0.90) -> int:
    """Smallest concurrency reaching `frac` of the best measured throughput.

    `curve` rows: {"concurrency", "tok_s", "p95_s"}. Beyond the knee extra
    in-flight requests only queue -- latency grows, throughput does not.
    """
    good = [r for r in curve if r.get("tok_s")]
    if not good:
        return 1
    best = max(r["tok_s"] for r in good)
    return min(r["concurrency"] for r in good if r["tok_s"] >= frac * best)


def probe_concurrency(curve: list[dict], stages: dict[str, dict], kv_caps: dict[str, int],
                      timeouts: dict[str, int], probe_out: int) -> dict[str, int]:
    """Measured knee, capped per stage by KV slots and by the task timeout.

    A stage's calls emit `mean_out` tokens vs the probe's `probe_out`; p95
    latency scales with output length, and the projected p95 must stay under
    half the stage's timeout or retries start.
    """
    knee = find_knee(curve)
    out: dict[str, int] = {}
    for name, st in stages.items():
        c = min(knee, kv_caps.get(name, knee))
        limit = timeouts.get(name)
        if limit:
            scale = max(1.0, st["mean_out"] / max(probe_out, 1))
            fits = [r["concurrency"] for r in curve
                    if r.get("p95_s") and r["p95_s"] * scale <= 0.5 * limit]
            if fits:
                c = min(c, max(fits))
        out[name] = max(1, c)
    return out


async def run_probe(model: str, levels: list[int], *, prompt_tokens: int = 4000,
                    out_tokens: int = 256, per_level: int | None = None) -> list[dict]:
    """Load sweep against the live deployment. Costs GPU minutes, not tokens."""
    from openai import AsyncOpenAI

    from backend.app.core.config import get_settings
    from backend.app.services.llm_router import DEEPINFRA_BASE_URL

    s = get_settings()
    if not s.deepinfra_api_key:
        raise SystemExit("DEEPINFRA_API_KEY not set in .env")
    client = AsyncOpenAI(api_key=s.deepinfra_api_key, base_url=DEEPINFRA_BASE_URL, max_retries=0)
    filler = ("The quarterly report discusses revenue, regulatory filings, clinical "
              "trial phases and supply chain risks across several regions. ") * (prompt_tokens // 22)

    async def one(i: int) -> tuple[float, int]:
        t0 = time.monotonic()
        r = await client.chat.completions.create(
            model=model, max_tokens=out_tokens, temperature=0.0, timeout=600,
            messages=[{"role": "user", "content": f"[{i}] {filler}\nList 40 entities mentioned."}],
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        return time.monotonic() - t0, (r.usage.completion_tokens if r.usage else 0)

    curve: list[dict] = []
    for c in levels:
        n = per_level or max(2 * c, 8)
        sem = asyncio.Semaphore(c)

        async def gated(i: int, sem=sem, c=c) -> tuple[float, int] | None:
            async with sem:
                try:
                    return await one(i)
                except Exception as exc:  # report, keep sweeping
                    print(f"    c={c}: {type(exc).__name__}: {str(exc)[:80]}")
                    return None

        t0 = time.monotonic()
        res = [r for r in await asyncio.gather(*[gated(i) for i in range(n)]) if r]
        wall = time.monotonic() - t0
        if not res:
            curve.append({"concurrency": c, "tok_s": 0, "p95_s": None, "errors": n})
            break
        lat = sorted(r[0] for r in res)
        row = {
            "concurrency": c,
            "tok_s": round(sum(r[1] for r in res) / wall, 1),
            "p95_s": round(lat[min(len(lat) - 1, int(len(lat) * 0.95))], 2),
            "errors": n - len(res),
        }
        curve.append(row)
        print(f"    c={c:>4}: {row['tok_s']:>9} tok/s   p95 {row['p95_s']:>7}s   errors {row['errors']}")
        # Past the knee and latency exploding -> stop paying for the sweep.
        if len(curve) >= 3 and row["tok_s"] < 1.02 * curve[-2]["tok_s"] and row["p95_s"] > 2 * curve[-2]["p95_s"]:
            break
    return curve


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #

def calibrate() -> dict:
    """Refit ratios from saved runs: output_ontologies/*/cost.json (prune-
    expand) and benchmark cost dumps that carry a `corpus.json` sidecar
    ({"tokens": N}) next to them (GRAPHRAG_COST_REPORT_DIR)."""
    import collections

    from backend.app.services.document_io import load_document
    from backend.app.services.evaluated_summarizer import get_encoder

    enc = get_encoder()
    by_name = {}
    for p in (ROOT / "source_documents").rglob("*"):
        if p.is_file():
            by_name.setdefault(p.name, p)
    agg: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)

    def add(report: dict, corpus_tok: int) -> None:
        for task, v in report.get("by_task", {}).items():
            a = agg[task]
            a["corpus"] += corpus_tok
            a["calls"] += v.get("calls", 0)
            if "in_tokens" in v:
                a["in"] += v["in_tokens"]
                a["out"] += v.get("out_tokens", 0)
                a["measured"] += v.get("calls", 0)

    runs = 0
    for cost in sorted((ROOT / "output_ontologies").glob("v*-prune-expand/cost.json")):
        man = cost.parent / "manifest.json"
        if not man.exists():
            continue
        docs = json.loads(man.read_text()).get("input_documents", [])
        tok = 0
        for d in docs:
            p = by_name.get(d.get("name", ""))
            if p is None:
                tok = 0
                break
            tok += len(enc.encode(load_document(p, warn_unreadable=False).text, disallowed_special=()))
        if tok:
            add(json.loads(cost.read_text()), tok)
            runs += 1
    for side in (ROOT / "benchmarks").rglob("corpus.json") if (ROOT / "benchmarks").exists() else []:
        tok = int(json.loads(side.read_text()).get("tokens", 0))
        for rep in side.parent.glob("*.json"):
            if rep.name != "corpus.json" and tok:
                add(json.loads(rep.read_text()), tok)
                runs += 1

    fitted: dict[str, dict] = {}
    for task, a in agg.items():
        if not a["corpus"] or not a["calls"]:
            continue
        row = {"per_mtok": round(a["calls"] / a["corpus"] * 1e6, 1)}
        if a["measured"]:
            row["in"] = round(a["in"] / a["measured"])
            row["out"] = round(a["out"] / a["measured"])
            row["src"] = "fit"
        fitted[task] = row
    CALIBRATION_FILE.parent.mkdir(parents=True, exist_ok=True)
    CALIBRATION_FILE.write_text(json.dumps({"runs": runs, "ratios": fitted}, indent=2))
    return {"runs": runs, "tasks": len(fitted)}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _arg(argv: list[str], name: str, default=None):
    if name in argv:
        i = argv.index(name)
        if i + 1 < len(argv):
            return argv[i + 1]
    return default


def _fmt_h(h: float) -> str:
    return f"{int(h)}h{round((h % 1) * 60):02d}m"


def report(docs_dir: str, argv: list[str]) -> int:
    deadline = _arg(argv, "--deadline-hours")
    deadline_h = float(deadline) if deadline else None
    precision = _arg(argv, "--precision", "any")
    spinup = float(_arg(argv, "--spinup-min", DEFAULT_SPINUP_MIN))
    non_llm_arg = _arg(argv, "--non-llm-min")
    out_path = Path(_arg(argv, "--out", str(ROOT / ".build" / "sizing.json")))

    # Several directories (comma-separated) size ONE deployment window for all
    # of them -- spin-up is paid once, so batching corpora is the main saving.
    n, total, per_doc = 0, 0, []
    for d in docs_dir.split(","):
        print(f"1. CORPUS  {d}  (extracting + tokenising...)")
        dn, dt, dp = corpus_tokens(d, _arg(argv, "--selection"))
        print(f"   {dn} document(s), {dt:,} tokens")
        n, total, per_doc = n + dn, total + dt, per_doc + dp
    if not n:
        print("   no documents found.")
        return 1
    print(f"   {n} document(s), {total:,} tokens (largest {max(per_doc):,})")
    non_llm = float(non_llm_arg) if non_llm_arg else DEFAULT_NON_LLM_MIN + NON_LLM_MIN_PER_DOC * n

    model = _arg(argv, "--model", DEFAULT_MODEL)
    if model not in MODELS:
        print(f"unknown --model {model}; one of {', '.join(MODELS)}")
        return 2
    ratios = load_ratios()
    stages = project_workload(total, ratios)
    print(f"\n   Projected creation workload ({MODELS[model]['label']}, dedicated):")
    print(f"   {'stage':<22} {'calls':>8} {'in tok':>12} {'out tok':>11} {'ctx/call':>9}")
    for name, st in sorted(stages.items(), key=lambda kv: -kv[1]["out_tokens"]):
        flag = "  (structural estimate)" if st["structural"] else ""
        print(f"   {name:<22} {st['calls']:>8.0f} {st['in_tokens']:>12,.0f} "
              f"{st['out_tokens']:>11,.0f} {st['mean_ctx']:>9,.0f}{flag}")
    tot_in = sum(s["in_tokens"] for s in stages.values())
    tot_out = sum(s["out_tokens"] for s in stages.values())
    print(f"   {'TOTAL':<22} {'':>8} {tot_in:>12,.0f} {tot_out:>11,.0f}")

    arch = fetch_arch(next(iter(MODELS[model]["checkpoints"].values()))["repo"],
                      MODELS[model]["fallback_arch"])
    print(f"\n2. GPU OPTIONS  {MODELS[model]['label']}  (KV {kv_bytes_per_token(arch)/1024:.0f} KB/token + "
          f"{state_bytes_per_seq(arch)/1e6:.0f} MB recurrent state per sequence; "
          f"{MAX_GPUS_PER_ACCOUNT}-GPU account limit)")
    print(f"   Billed beyond LLM time: spin-up {spinup:.0f} min + ~{non_llm:.0f} min of pipeline "
          "work between calls (--spinup-min / --non-llm-min).")
    # --gpu restricts to one GPU type (e.g. when DeepInfra has no capacity
    # for the cheapest one: deployments fail with `no-gpu-available`).
    only_gpu = _arg(argv, "--gpu")
    rows = [(o, estimate_run(o, arch, stages, spinup, non_llm)) for o in gpu_options(precision, model=model)
            if not only_gpu or o.gpu == only_gpu]
    chain = preferred_chain(rows, deadline_h)
    rank = {id(o): i + 1 for i, (o, _) in enumerate(chain)}
    rows.sort(key=lambda r: (rank.get(id(r[0]), 99), r[1]["usd"], r[1]["hours"]))
    print(f"   {'pref':<5}{'configuration':<30} {'$/hr':>6} {'KV GB':>7} {'slots@ctx':>9} {'LLM time':>9} {'billed':>8} {'est. $':>8}")
    for o, est in rows:
        slots = kv_slots(o, arch, stages.get("class_proposal", {}).get("mean_ctx", 26000))
        late = "  > deadline" if deadline_h and est["hours"] > deadline_h else ""
        pref = f"#{rank[id(o)]}" if id(o) in rank else ""
        print(f"   {pref:<5}{o.label():<30} {o.usd_hr:>6.2f} {o.kv_free_gb:>7.0f} {slots:>9} "
              f"{_fmt_h(est['llm_hours']):>9} {_fmt_h(est['hours']):>8} {est['usd']:>8.2f}{late}"
              + (f"  [{o.note}]" if o.note else ""))
    best = recommend(rows, deadline_h, allow_noted="--allow-a100-fp8" in argv)
    if best is None:
        print(f"\n   No configuration meets --deadline-hours {deadline_h}. Relax it, or ask "
              "DeepInfra support to raise the 4-GPU limit.")
        return 1
    opt, est = best
    print(f"\n   RECOMMEND {opt.label()}  ~{_fmt_h(est['hours'])} billed, ~${est['usd']:.2f} "
          f"(a-priori estimate, +-50%; --probe measures the real curve)")
    fallbacks = [r for r in chain if r[0] is not opt]
    if fallbacks:
        print("   Fallback chain if it has no capacity (deepinfra_deploy.py walks it): "
              + " -> ".join(f"{o.label()} (${o.usd_hr:.2f}/h)" for o, _ in fallbacks))
    tails = [k for k, v in est["stages"].items() if v["tail_bound"]]
    if tails:
        print(f"   Long-tail bound (more GPUs would not shorten): {', '.join(tails)}")
    if est["hours"] > 0 and (spinup + non_llm) / 60 > 0.5 * est["hours"]:
        print("   Most billed time is spin-up + pipeline gaps, not inference: the cheapest "
              "$/hour configuration wins, and batching corpora into ONE window saves most.")
    if any(s["structural"] for s in stages.values()):
        print("   Phase-2 stages use structural estimates until a benchmark run is calibrated "
              "(--calibrate).")

    timeouts = nonstreamed_timeouts(ratios)

    def entry(o: GpuOption, e: dict) -> dict:
        c = a_priori_concurrency(o, arch, stages, timeouts)
        return {"option": {**asdict(o), "label": f"{o.label()} ({MODELS[model]['label']})"},
                "estimate": e, "concurrency": c, "max_batch_size": max(8, min(MAX_BATCH, max(c.values())))}

    primary = entry(opt, est)
    print(f"\n3. CONCURRENCY for {opt.label()} (a priori: throughput knee within the KV cache; "
          "--probe refines):")
    _print_concurrency(primary["concurrency"])
    chained = [entry(o, e) for o, e in fallbacks]
    for fb in chained:
        print(f"   {fb['option']['label']}: " + ", ".join(f"{k} {v}" for k, v in sorted(fb["concurrency"].items())))
    payload = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "docs_dir": docs_dir, "documents": n, "corpus_tokens": total, "model": model,
        **primary, "fallbacks": chained, "arch": arch,
        "stages": {k: {kk: vv for kk, vv in v.items() if kk != "tasks"} for k, v in stages.items()},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\n   wrote {out_path.relative_to(ROOT) if out_path.is_relative_to(ROOT) else out_path}"
          " -> scripts/deepinfra_deploy.py create --from-sizing <it>  (needs your approval: paid)")
    return 0


def nonstreamed_timeouts(ratios: dict[str, dict] | None = None) -> dict[str, list[tuple[int, float]]]:
    """Per stage, (whole-call timeout, slow-call output tokens) for each of its
    NON-streamed tasks that run on the dedicated deployment in the deepinfra
    preset. Each task is checked against its OWN timeout and output length: a
    stage-wide minimum let a 90 s short-reply task cap a stage of long merges.
    Streamed tasks time out on the gap between tokens, and serverless tasks are
    not on these GPUs, so neither sets a cap."""
    import yaml
    from tpm_check import STAGE_TASKS

    ratios = ratios or load_ratios()
    preset = ROOT / "config" / "models.deepinfra.example.yaml"
    specs = (yaml.safe_load(preset.read_text()) or {}).get("tasks") or {}
    out: dict[str, list[tuple[int, float]]] = {}
    for stage, tasks in STAGE_TASKS.items():
        for t in tasks:
            s = specs.get(t) or {}
            if (s.get("timeout") and not s.get("stream") and t in ratios
                    and "__DEEPINFRA_FLASH_MODEL__" in str(s.get("model", ""))):
                # The slow call: ~2x the mean output, but never past the task's own
                # max_tokens (a 1,024-token summary cannot run to 3,000).
                slow = min(2 * float(ratios[t]["out"]), float(s.get("max_tokens") or 1e9))
                out.setdefault(stage, []).append((int(s["timeout"]), slow))
    return out


def _print_concurrency(conc: dict[str, int]) -> None:
    from backend.app.core.config import get_settings

    cur = get_settings().app_config.get("concurrency") or {}
    for stage, v in sorted(conc.items()):
        print(f"   {stage:<22} now {cur.get(stage, 'unset')!s:>6}  ->  {v}")


async def probe_main(argv: list[str]) -> int:
    from backend.app.core.config import get_settings

    sizing_path = Path(_arg(argv, "--sizing", str(ROOT / ".build" / "sizing.json")))
    if not sizing_path.exists():
        print(f"no sizing file at {sizing_path}; run the corpus step first")
        return 1
    sizing = json.loads(sizing_path.read_text())
    specs = get_settings().models_config.get("tasks") or {}
    model = _arg(argv, "--model") or (specs.get("class_proposal") or {}).get("model")
    if not model or model.startswith("__"):
        print("no dedicated model id: set DEEPINFRA_FLASH_MODEL and re-run select_llm_mode.py deepinfra")
        return 1
    levels = [int(x) for x in _arg(argv, "--levels", "1,8,16,32,64,128").split(",")]
    probe_out = int(_arg(argv, "--probe-out", 256))
    print(f"Probing {model} at concurrency {levels} (GPU minutes only)...")
    curve = await run_probe(model, levels, out_tokens=probe_out)

    from tpm_check import STAGE_TASKS

    timeouts = {st: max(int((specs.get(t) or {}).get("timeout", 120)) for t in tasks)
                for st, tasks in STAGE_TASKS.items()}
    # KV caps of the configuration actually running (deepinfra_deploy.py
    # records it as `active` -- it may be a fallback, not the recommendation).
    caps = (sizing.get("active") or sizing)["concurrency"]
    conc = probe_concurrency(curve, sizing["stages"], caps, timeouts, probe_out)
    conc["evaluation"] = min(32, int(_arg(argv, "--serverless-limit", 200)) // 4)
    sizing.update({"probe": curve, "knee": find_knee(curve), "concurrency": conc})
    print(f"\n   knee: {sizing['knee']} concurrent requests")
    _print_concurrency(conc)

    # streaming_batch_size caps summarization: windows per batch must cover it.
    n, tok = sizing["documents"], sizing["corpus_tokens"]
    windows = max(1, tok // 11500)
    batch = max(4, min(n, math.ceil(conc.get("summarization", 8) * n / windows)))
    print(f"   chunking.streaming_batch_size -> {batch} (so ~{windows} windows keep "
          f"{conc.get('summarization', 8)} slots busy)")
    sizing_path.write_text(json.dumps(sizing, indent=2))

    if "--apply" in argv:
        from tpm_check import apply_concurrency, db_pool_report

        path = ROOT / "config" / "config.yaml"
        changes = apply_concurrency(path, conc)
        print(f"\nWROTE {path.relative_to(ROOT)}:")
        for c in changes or ["(no changes -- already at these values)"]:
            print(f"  {c}")
        print(f"  (set chunking.streaming_batch_size: {batch} by hand if it differs)")
        db_pool_report(conc.get("evaluation"))
    else:
        print("\n   Nothing written. Re-run with --apply to write these into config/config.yaml.")
    return 0


def main() -> int:
    argv = sys.argv[1:]
    if "--calibrate" in argv:
        res = calibrate()
        print(f"calibrated from {res['runs']} run(s), {res['tasks']} task(s) -> "
              f"{CALIBRATION_FILE.relative_to(ROOT)}")
        return 0
    if "--probe" in argv:
        return asyncio.run(probe_main(argv))
    positional = [a for i, a in enumerate(argv)
                  if not a.startswith("-") and (i == 0 or not argv[i - 1].startswith("--"))]
    if not positional:
        print(__doc__)
        return 2
    return report(positional[0], argv)


if __name__ == "__main__":
    raise SystemExit(main())
