#!/usr/bin/env python3
"""Run Stage 1-2 of prune-expand (class proposal) for ONE model on IDENTICAL
inputs, to compare models head to head. Stops before Stage 3 (dedup).

    uv run python scripts/bench/class_proposal_compare.py --arm deepseek --corpus websearch3x \
        --docs source_documents/websearch-geo-time-3x --merge output_ontologies/<merge> \
        --cache-root benchmarks/cache/deepinfra-ws --summary-model VenkatRamRao/<id that made the summaries> \
        --class-model deepseek-ai/DeepSeek-V4.1-Flash --prompt-set deepseek \
        --stage1 benchmarks/<date>/classcmp/<corpus>_stage1.json --record-stage1 \
        --out benchmarks/<date>/classcmp/<arm>/<corpus> [--concurrency 32] --go

Identical inputs, so only the model (and its own prompt version) differs:
  - CHUNKS: summaries come from an existing cache. Every summarization task
    is pointed at a placeholder id whose cache key is computed with
    --summary-model, the id that produced those summaries. A miss calls the
    placeholder and fails loudly instead of re-summarizing.
  - STAGE 1 (chunk -> ontology roots): recorded once (--record-stage1, run on
    the --stage1-model) and replayed for every other arm, keyed by chunk
    text. Replay never calls a model; a missing chunk is an error.
  - STAGE 2 runs the real `_propose_for_chunk` with the deepinfra preset's
    class_proposal settings (ontology_view, guard, salvage, streaming) and
    the arm's model + prompt_set. The audit log (llm_audit.jsonl) holds every
    parsed reply; costs/ holds tokens, seconds and finish reasons per task.

PAID: Stage 2 calls (and Stage 1 when recording).
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

PLACEHOLDER = "VenkatRamRao/summary-cache-only"
SUMMARY_TASKS = ("document_summarize", "evaluated_summary_chunk", "summary_question_gen",
                 "summary_evaluate", "summary_revise", "embed_compress")


class StopAfterStage2(Exception):
    pass


def _arg(argv, name, default=None):
    return argv[argv.index(name) + 1] if name in argv and argv.index(name) + 1 < len(argv) else default


def render_models(out: Path, class_model: str, prompt_set: str, stage1_model: str) -> Path:
    import yaml

    doc = yaml.safe_load((ROOT / "config" / "models.deepinfra.example.yaml").read_text())
    t = doc["tasks"]
    for name in SUMMARY_TASKS:
        if name in t:
            t[name]["model"] = PLACEHOLDER
    t["chunk_classification"].update({"provider": "deepinfra", "model": stage1_model})
    t["chunk_classification"].pop("prompt_set", None)
    t["class_proposal"].update({"provider": "deepinfra", "model": class_model, "prompt_set": prompt_set})
    for name, spec in t.items():            # any other dedicated-model task: unused here
        if "__DEEPINFRA_FLASH_MODEL__" in str(spec.get("model", "")):
            spec["model"] = PLACEHOLDER
    path = out / "models.yaml"
    path.write_text(yaml.safe_dump(doc, sort_keys=False))
    return path


def main(argv: list[str]) -> int:
    need = ("--arm", "--corpus", "--docs", "--merge", "--cache-root", "--summary-model",
            "--class-model", "--prompt-set", "--stage1", "--out")
    if any(_arg(argv, k) is None for k in need):
        print(__doc__)
        return 2
    out = Path(_arg(argv, "--out"))
    stage1_path = Path(_arg(argv, "--stage1"))
    record = "--record-stage1" in argv
    conc = int(_arg(argv, "--concurrency", "32"))
    if "--go" not in argv:
        print(f"(dry run) {_arg(argv, '--arm')} on {_arg(argv, '--corpus')} -> {out}; add --go (PAID)")
        return 0
    if out.exists():
        shutil.rmtree(out)
    (out / "pe").mkdir(parents=True)
    (out / "costs").mkdir()

    models = render_models(out, _arg(argv, "--class-model"), _arg(argv, "--prompt-set"),
                           _arg(argv, "--stage1-model", "deepseek-ai/DeepSeek-V4.1-Flash"))
    app_cfg = out / "config.yaml"
    shutil.copy(ROOT / "config" / "config.yaml", app_cfg)
    from tpm_check import apply_concurrency

    apply_concurrency(app_cfg, {"chunk_classification": conc, "class_proposal": conc})
    os.environ.update({
        "MODELS_CONFIG_PATH": str(models), "APP_CONFIG_PATH": str(app_cfg),
        "GRAPHRAG_CACHE_ROOT": str(Path(_arg(argv, "--cache-root")).resolve()),
        "GRAPHRAG_COST_REPORT_DIR": str(out / "costs"), "PYTHONUNBUFFERED": "1",
    })

    import backend.app.services.evaluated_summarizer as es
    import backend.app.services.pipeline_llm as pl

    old = _arg(argv, "--summary-model")
    ek, dk = es._cache_key, pl._doc_summary_cache_key
    es._cache_key = lambda text, model, *a: ek(text, old if model == PLACEHOLDER else model, *a)
    pl._doc_summary_cache_key = lambda text, model: dk(text, old if model == PLACEHOLDER else model)

    stage1 = json.loads(stage1_path.read_text()) if stage1_path.exists() else {}
    if not record and not stage1:
        raise SystemExit(f"no recorded Stage 1 at {stage1_path}; run the first arm with --record-stage1")
    real_classify = pl._classify_chunk

    async def classify(router, branches, chunk, *a, **kw):
        key = hashlib.sha256(f"{chunk.source_name}\n{chunk.text}".encode()).hexdigest()
        if key in stage1:
            return list(stage1[key])
        if not record:
            raise RuntimeError(f"Stage 1 replay miss: chunk #{chunk.index} of {chunk.source_name}")
        iris = await real_classify(router, branches, chunk, *a, **kw)
        stage1[key] = iris
        return iris

    async def stop(*a, **kw):
        raise StopAfterStage2()

    pl._classify_chunk = classify
    pl._dedup = stop
    from backend.app.cli.main import main as cli

    try:
        cli(["prune-expand", "--input", _arg(argv, "--merge"), "--documents", _arg(argv, "--docs"),
             "--tables", "--max-cost-usd", _arg(argv, "--max-cost-usd", "10"), "--output-dir", str(out / "pe")])
        print("WARNING: prune-expand returned without reaching Stage 3")
    except StopAfterStage2:
        print("STAGE 2 DONE (stopped before dedup)")
    finally:
        if record:
            stage1_path.parent.mkdir(parents=True, exist_ok=True)
            stage1_path.write_text(json.dumps(stage1))
            print(f"recorded Stage 1 for {len(stage1)} chunk(s) -> {stage1_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
