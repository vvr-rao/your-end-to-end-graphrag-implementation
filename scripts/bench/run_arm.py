#!/usr/bin/env python3
"""Build one benchmark arm's knowledge graph for one corpus, isolated.

    uv run python scripts/bench/run_arm.py --arm openai|qwen|qwen27b|qwen27b-ded|qwen27b-think|qwen27b-fast --corpus pharma \
        --docs source_documents/pharma-30-identity-test --merge output_ontologies/<v>-merge \
        [--limit N] [--out benchmarks/<date>] [--max-cost-usd 15] [--go]

Without --go it prints the plan and exits: every step here is PAID.

Isolation per (arm, corpus):
  DATABASE_URL              its own database in the bench container (bench_db.py)
  MODELS_CONFIG_PATH        the arm's preset (qwen: rendered with the dedicated
                            deployment id from .env DEEPINFRA_FLASH_MODEL)
  GRAPHRAG_CACHE_ROOT       its own summary/table caches -- the table cache is
                            keyed on document bytes only and would otherwise
                            hand one arm's extractions to the other
  GRAPHRAG_COST_REPORT_DIR  per-task tokens/cost/seconds from every command
Outputs land in <out>/<arm>/<corpus>/: steps.json (exit/seconds per step),
costs/*.json, costs/corpus.json (token count, for sizing calibration), and
prune_expand_dir.txt.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "bench"))

CLI = ["uv", "run", "python", "-m", "backend.app.cli"]
PRESETS = {
    "openai": ROOT / "config" / "models.openai.example.yaml",
    "qwen": ROOT / "config" / "models.deepinfra.example.yaml",
    # Same preset, but every creation task on SERVERLESS 27B -- no dedicated
    # deployment. Isolates "Flash-Next" from "Qwen + this prompt set".
    "qwen27b": ROOT / "config" / "models.deepinfra.example.yaml",
    # Creation on a DEDICATED 27B deployment (id from DEEPINFRA_FLASH_MODEL).
    "qwen27b-ded": ROOT / "config" / "models.deepinfra.example.yaml",
    # As qwen27b-ded on faster hardware with higher concurrency (APP_CONFIG_PATH).
    "qwen27b-fast": ROOT / "config" / "models.deepinfra.example.yaml",
    # As qwen27b-fast, but class_proposal uses the Qwen-only prompt set.
    "qwen27b-qprompt": ROOT / "config" / "models.deepinfra.example.yaml",
    "qwen27b-qprompt2": ROOT / "config" / "models.deepinfra.example.yaml",
    # v2 + class-vs-instance rule + Qwen relationship prompt.
    "qwen27b-qprompt3": ROOT / "config" / "models.deepinfra.example.yaml",
    # Mixed: 27B everywhere except relationship_extract -> DeepSeek-V4.1-Flash.
    "qwen27b-mixed-ds": ROOT / "config" / "models.deepinfra.example.yaml",
    # The deepinfra preset exactly as shipped (mixed mode, v3 + salvage).
    "preset-v3": ROOT / "config" / "models.deepinfra.example.yaml",
    # Full benchmark: two independent builds of the shipped deepinfra preset.
    "deepinfra-b1": ROOT / "config" / "models.deepinfra.example.yaml",
    "deepinfra-b2": ROOT / "config" / "models.deepinfra.example.yaml",
    # b2's ontology + summaries, re-extracted with the short_name merge guard
    # and evaluated with the "facts" prefix + rerank Qwen profile.
    "deepinfra-b3": ROOT / "config" / "models.deepinfra.example.yaml",
    # The shipped deepinfra preset (Q&A on DeepSeek-V4.1-Flash) on the web-search
    # corpus; this build is the one restored into Supabase for Render.
    "deepinfra-ws": ROOT / "config" / "models.deepinfra.example.yaml",
    # 2026-10-19: the same preset after the Stage 2-4 scaling work, built on
    # 4x H200 (two 2-GPU instances); the build pushed to Supabase for Render.
    "deepinfra-ws2": ROOT / "config" / "models.deepinfra.example.yaml",
    # deepinfra-b3's graph, every Q&A task on DeepSeek-V4.1-Flash (QA_TASKS).
    "deepinfra-b3-dsqa": ROOT / "config" / "models.deepinfra.example.yaml",
    # Embedding isolation: the OpenAI build's DB re-embedded with Qwen3-Embedding-4B,
    # OpenAI models for every chat task (scripts/bench/reembed_copy.py).
    "openai-qwenemb": ROOT / "config" / "models.openai.example.yaml",
    # As qwen27b-ded, with reasoning ON for the two ontology-deciding steps.
    "qwen27b-think": ROOT / "config" / "models.deepinfra.example.yaml",
}
# Per-arm task overrides merged into the rendered models.yaml. max_tokens is
# raised so the reasoning trace cannot eat the answer budget; dedup keeps its
# batch size pinned to the thinking-off run's 32768 so only `thinking` changes.
TASK_PATCHES = {
    "openai-qwenemb": {
        "embeddings": {"provider": "deepinfra", "model": "Qwen/Qwen3-Embedding-4B", "dim": 1024, "batch_size": 100},
    },
    "qwen27b-mixed-ds": {
        "relationship_extract": {"provider": "deepinfra", "model": "deepseek-ai/DeepSeek-V4.1-Flash",
                                 "prompt_set": "deepseek"},
    },
    "qwen27b-think": {
        "class_proposal": {"thinking": True, "max_tokens": 49152},
        "match_dedup": {"thinking": True, "max_tokens": 49152, "batch_budget_tokens": 32768},
    },
}
# Every task the Q&A path calls (question parsing through answer synthesis).
# In the deepinfra preset these all run on serverless Qwen3.8-27B with the
# shared prompts -- there is no Qwen-only Q&A prompt set.
QA_TASKS = ("qa_synthesis", "question_parse", "question_relations", "concept_expansion",
            "query_decompose", "retrieval_rounds_plan", "entity_probes", "chunk_relevance_filter",
            "answer_simple_qa", "answer_deep_research", "follow_up_resolution",
            "answer_conversation_turn")
# deepinfra-b3's graph (a template copy of its DB), Q&A on DeepSeek-V4.1-Flash.
TASK_PATCHES["deepinfra-b3-dsqa"] = {
    t: {"provider": "deepinfra", "model": "deepseek-ai/DeepSeek-V4.1-Flash"} for t in QA_TASKS
}
# ONE judge for every arm, so scores compare across modes. The deepinfra
# preset judges with DeepSeek (it is DeepInfra-only); a benchmark must not let
# each arm grade itself with its own model.
JUDGE_TASKS = ("judge_comprehensiveness", "judge_no_hallucination", "judge_gap_detection", "judge_consistency")
BENCH_JUDGE = {"provider": "openai", "model": "gpt-4.1"}


def pin_judges(doc: dict) -> dict:
    for t in JUDGE_TASKS:
        if t in doc.get("tasks", {}):
            spec = doc["tasks"][t]
            spec.pop("prompt_set", None)
            spec.update(BENCH_JUDGE)
    return doc


# Model that replaces the dedicated-deployment placeholder, per arm.
CREATION_MODEL_OVERRIDE = {"qwen27b": "Qwen/Qwen3.8-27B"}


def _arg(argv, name, default=None):
    return argv[argv.index(name) + 1] if name in argv and argv.index(name) + 1 < len(argv) else default


def serverless_model(arm: str) -> str | None:
    """`sl:<deepinfra model id>` = that ONE serverless model for every chat
    task (creation AND retrieval); embeddings/judges as in the preset."""
    return arm[3:] if arm.startswith("sl:") else None


def arm_slug(arm: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", arm.lower()).strip("-")


def models_file(arm: str, out: Path) -> Path:
    sl = serverless_model(arm)
    if sl:
        # ONE model for every DeepInfra chat task (embeddings/judges untouched).
        # A model-specific prompt set is kept only for its own family.
        import yaml

        doc = yaml.safe_load(PRESETS["qwen"].read_text())
        family = {"qwen": "Qwen/", "deepseek": "deepseek-ai/DeepSeek-V4.1-Flash"}
        for name, spec in doc["tasks"].items():
            if spec.get("provider") != "deepinfra" or name == "embeddings":
                continue
            spec["model"] = sl
            ps = spec.get("prompt_set")
            if ps and not sl.startswith(family.get(ps, "\0")):
                spec.pop("prompt_set")
        pin_judges(doc)
        (out / "models.yaml").write_text(yaml.safe_dump(doc, sort_keys=False))
        return out / "models.yaml"
    src = PRESETS[arm].read_text()
    if arm in CREATION_MODEL_OVERRIDE:
        src = src.replace("__DEEPINFRA_FLASH_MODEL__", CREATION_MODEL_OVERRIDE[arm])
    if "__DEEPINFRA_FLASH_MODEL__" in src:
        from _skillutil import env_value_nonsecret

        # Process env wins over .env, so a run can target a just-created
        # deployment before the user has edited .env.
        flash = os.environ.get("DEEPINFRA_FLASH_MODEL") or env_value_nonsecret("DEEPINFRA_FLASH_MODEL")
        if not flash:
            raise SystemExit("set DEEPINFRA_FLASH_MODEL in .env (the dedicated deployment id) first")
        src = src.replace("__DEEPINFRA_FLASH_MODEL__", flash)
    import yaml

    doc = yaml.safe_load(src)
    for task, patch in TASK_PATCHES.get(arm, {}).items():
        doc["tasks"][task].update(patch)
    src = yaml.safe_dump(pin_judges(doc), sort_keys=False)
    dst = out / "models.yaml"
    dst.write_text(src)
    return dst


def arm_env(arm: str, corpus: str, out: Path) -> dict:
    """Environment that points every CLI command at this arm's isolated state."""
    import bench_db

    bench_db.up()
    costs = out / "costs"
    costs.mkdir(parents=True, exist_ok=True)
    return {
        **os.environ,
        # stdout goes to a log FILE, so CPython block-buffers it: per-chunk
        # progress lines would only appear at exit (same fix as run_detached.py).
        "PYTHONUNBUFFERED": "1",
        "DATABASE_URL": bench_db.create(f"{arm_slug(arm)}_{corpus}".replace("-", "_")[:60]),
        "MODELS_CONFIG_PATH": str(models_file(arm, out)),
        "GRAPHRAG_CACHE_ROOT": str(ROOT / "benchmarks" / "cache" / arm_slug(arm)),
        "GRAPHRAG_COST_REPORT_DIR": str(costs),
    }


DOC_SUFFIXES = {".pdf", ".pptx", ".docx", ".txt", ".md"}


def subset_dir(docs: Path, n: int, dst: Path) -> Path:
    """First `n` documents (sorted by name) of `docs`, symlinked into `dst`."""
    dst.mkdir(parents=True, exist_ok=True)
    for old in dst.iterdir():
        old.unlink()
    files = sorted(p for p in docs.iterdir() if p.is_file() and p.suffix.lower() in DOC_SUFFIXES)
    for p in files[:n]:
        (dst / p.name).symlink_to(p.resolve())
    return dst


def step(name: str, args: list[str], env: dict, log: Path, timeout: int) -> dict:
    t0 = time.monotonic()
    print(f">>> {name}: {' '.join(args)}", flush=True)
    with (log / f"{name}.log").open("w") as fh:
        try:
            code = subprocess.run(CLI + args, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT,
                                  timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            code = 124
    row = {"name": name, "exit": code, "seconds": round(time.monotonic() - t0, 1)}
    print(f"    exit {code} in {row['seconds']}s", flush=True)
    return row


def main(argv: list[str]) -> int:
    arm, corpus = _arg(argv, "--arm"), _arg(argv, "--corpus")
    docs, merge = _arg(argv, "--docs"), _arg(argv, "--merge")
    if (arm not in PRESETS and not serverless_model(arm)) or not (corpus and docs and merge):
        print(__doc__)
        return 2
    limit = _arg(argv, "--limit")
    cap = _arg(argv, "--max-cost-usd", "15")
    out = Path(_arg(argv, "--out", str(ROOT / "benchmarks" / time.strftime("%Y-%m-%d")))) / arm_slug(arm) / corpus
    plan = ["db-migrate", "prune-expand", "db-init(replace)", "register-documents",
            "extract-entities", "enrich-time", "enrich-geo", "embed-relationships",
            "generate-artifacts", "db-size"]
    print(f"arm={arm} corpus={corpus} docs={docs} limit={limit or 'ALL'} cap=${cap}/step\n"
          f"steps: {' -> '.join(plan)}\nout: {out}")
    if "--go" not in argv:
        print("\n(dry run -- add --go to execute; PAID)")
        return 0

    from deepinfra_sizing import corpus_tokens

    out.mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)
    env = arm_env(arm, corpus, out)
    if limit:
        # prune-expand has no --limit, so a smoke run gets a folder of the
        # first N documents (symlinks) for EVERY step -- otherwise the
        # "smoke" would pay prune-expand on the whole corpus.
        docs = str(subset_dir(Path(docs), int(limit), out / "docs_subset"))
    n, tok, _ = corpus_tokens(docs)
    (out / "costs" / "corpus.json").write_text(json.dumps(
        {"documents": n, "tokens": tok, "docs": docs, "limit": limit}))
    resume = _arg(argv, "--resume-from")  # a post-prune-expand step name
    prior = json.loads((out / "steps.json").read_text()) if resume and (out / "steps.json").exists() else []
    rows = [r for r in prior if r["exit"] == 0] if resume else []
    if not resume:
        rows.append(step("db-migrate", ["db-migrate"], env, out / "logs", 600))
        # Each run gets its OWN parent folder: two parallel runs picking "the
        # newest output_ontologies/ folder" grabbed each other's (2026-10-13).
        pe_parent = out / "prune_expand"
        pe_parent.mkdir(exist_ok=True)
        rows.append(step("prune-expand", ["prune-expand", "--input", merge, "--documents", docs,
                                          "--max-cost-usd", cap, "--tables", "--output-dir", str(pe_parent)],
                         env, out / "logs", 4 * 3600))
        made = sorted(d for d in pe_parent.glob("v*-prune-expand") if (d / "merged.json").exists())
        pe_dir = str(made[-1]) if made else ""
        (out / "prune_expand_dir.txt").write_text(pe_dir)
    else:
        pe_dir = _arg(argv, "--pe-dir") or (out / "prune_expand_dir.txt").read_text().strip()
        (out / "prune_expand_dir.txt").write_text(pe_dir)
    later = [
        ("db-init", ["db-init", "--input", pe_dir, "--mode", "replace", "--yes"], 3600),
        ("register-documents", ["register-documents", "--input", docs, "--tables"], 4 * 3600),
        ("extract-entities", ["extract-entities", "--max-cost-usd", cap], 4 * 3600),
        ("enrich-time", ["enrich-time"], 3600),
        # Step 4b of the ingest-corpus skill: place containment edges, then the
        # embeddings the relation-matched hop needs for them. Builds before
        # 2026-10-19 skipped both, so their retrieval fell back silently.
        ("enrich-geo", ["enrich-geo"], 3600),
        ("embed-relationships", ["embed-relationships"], 3600),
        ("generate-artifacts", ["generate-artifacts", "--max-cost-usd", cap], 4 * 3600),
        ("db-size", ["db-size"], 300),
    ]
    if resume:
        names = [n for n, _, _ in later]
        later = later[names.index(resume):]
        rows = [r for r in rows if r["name"] not in {n for n, _, _ in later}]
    if (not rows or rows[-1]["exit"] == 0) and pe_dir:
        for name, args, to in later:
            rows.append(step(name, args, env, out / "logs", to))
            if rows[-1]["exit"] != 0:
                print(f"stopping: {name} failed (see {out / 'logs' / (name + '.log')})")
                break
    (out / "steps.json").write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out / 'steps.json'}")
    return 0 if all(r["exit"] == 0 for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
