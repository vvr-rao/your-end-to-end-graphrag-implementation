---
name: choose-llm-mode
description: >
  Select the LLM provider mode for this GraphRAG app (Groq+OpenAI, OpenAI-only,
  Anthropic+OpenAI, or DeepInfra/Qwen) and activate the matching config preset. Use when setting
  up the app, switching providers, or when a build step reports a missing
  provider/model config. Handles API keys by pointing the user to .env — never
  accepts or displays credentials in chat.
---

# Choose the LLM provider mode

This app's per-task provider/model map lives in `config/models.yaml`. Four
shipped presets cover the supported modes. Your job: help the user pick one,
copy it into place, and confirm the required API keys are present in `.env` —
**without ever seeing a key value**.

## SECURITY — non-negotiable, applies to every step
- **Never** read `.env` with the Read tool (it is denied) and never `cat`/print it.
- **Never** ask the user to paste a key into the chat. Keys go in `.env` only,
  edited by the user in their own editor.
- Presence is checked by a helper that reads `.env` and returns only present/MISSING,
  never a value: `uv run python scripts/check_env.py KEY [KEY ...]`. A key counts as
  configured only when present, non-empty, AND not a `...placeholder...`.
- If you ever need to show the user what to edit, show the **variable name**,
  never a value.

## Step 1 — Ask which mode
Present these four options (AskUserQuestion). Embeddings run on OpenAI
(`text-embedding-3-small`) in every mode except DeepInfra, which embeds with
`Qwen/Qwen3-Embedding-4B`. **Every mode except DeepInfra requires `OPENAI_API_KEY`**;
DeepInfra mode is DeepInfra-only (judges and table vision included).

**Rule: DeepInfra mode uses ONLY DeepInfra models, and the user is assumed to
hold no other key.** Every task in `config/models.deepinfra.example.yaml` —
creation, description compaction, alias judge, Q&A, eval judges, table vision,
embeddings — is `provider: deepinfra`, and `select_llm_mode.py deepinfra` asks
for `DEEPINFRA_API_KEY` alone (plus the non-secret `DEEPINFRA_FLASH_MODEL`).
Never suggest an OpenAI/Anthropic/Groq model or key as a fix or fallback in
this mode, and never run a DeepInfra-mode step under another mode's
`config/models.yaml` (e.g. a standalone `summarize-descriptions` while the
active preset is OpenAI). `test_deepinfra_mode_uses_only_deepinfra` enforces it.

| Mode | `<mode>` arg | Keys required in `.env` | Notes |
|---|---|---|---|
| **Groq + OpenAI** (default) | `groq-openai` | `OPENAI_API_KEY`, `GROQ_API_KEY` | Groq runs chunk_classification (best disambiguation); OpenAI everything else. |
| **OpenAI only** | `openai` | `OPENAI_API_KEY` | Simplest; one provider. chunk_classification → `gpt-4.1-mini`. |
| **Anthropic + OpenAI** | `anthropic` | `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` | Claude for chat (haiku cheap tasks / sonnet synthesis); OpenAI for embeddings only. |
| **DeepInfra (Qwen + DeepSeek)** | `deepinfra` | `DEEPINFRA_API_KEY`, `DEEPINFRA_FLASH_MODEL` | Knowledge creation on a **dedicated** Qwen3.8-27B deployment (rented GPUs, billed per GPU-hour while it exists); relationships, repair, judges and **all Q&A** on serverless DeepSeek-V4.1-Flash; table vision `Qwen3-VL-235B-A22B` (only when pdfplumber cannot build the table; **host it on its own short-lived deployment — recommended**, serverless is the slow fallback); embeddings `Qwen3-Embedding-4B`@1024. No OpenAI/Anthropic key. Needs its OWN database (vectors can't mix with OpenAI ones; a guard refuses). See "DeepInfra mode" below. |

How to describe DeepInfra when presenting the options: **cheapest to run, rented
GPUs to build.** Q&A costs ~$0.009 per deep-research question (vs ~$0.10 on
OpenAI) and the deployed app needs no GPU; but every build or document add
rents a GPU deployment for the length of the run (a 66-document corpus is ~1 h,
~$5-10), and DeepInfra sometimes has no free GPUs, so a build can wait.
Measured on the pharma benchmark against OpenAI: answer quality on par
(audit 0.973 vs 0.959), better grounding, higher retrieval recall/precision,
lower MRR, ~1.3x slower per question.

## Step 2 — Activate the mode (one command)
```bash
uv run python scripts/select_llm_mode.py <mode>   # groq-openai | openai | anthropic | deepinfra
```
This scaffolds `.env` + `config/config.yaml` from the templates if missing, copies
the matching `models.*.example.yaml` into `config/models.yaml`, presence-checks the
mode's required keys (values never shown), and records `llm-mode` to the tracker.
It exits non-zero and names any missing keys.

## Step 3 — Handle missing keys
- If the helper reports all keys present → report ready, hand back to the orchestrator.
- If it names any MISSING → tell the user exactly which variable(s) to fill in
  `.env` (by name), e.g. *"Open `.env` in your editor and set `ANTHROPIC_API_KEY=`
  to your key. I can't see or accept the value — add it there and tell me when
  done."* Then re-check with `uv run python scripts/check_env.py OPENAI_API_KEY
  ANTHROPIC_API_KEY`. Do not proceed to any paid step until keys pass.

## DeepInfra mode — size, rent, build, delete (paid)
`DEEPINFRA_FLASH_MODEL` is not a secret: it is the dedicated deployment's model
id (`<deepinfra-username>/<deploy-name>`), which exists only after a deployment
is created. The account needs a DeepInfra display name (username) first —
`create` says so if it is missing. Order:
1. User adds `DEEPINFRA_API_KEY` to `.env` themselves.
2. **Size from the corpus** (free; reads documents only):
   `uv run python scripts/deepinfra_sizing.py "<DOCS>" [--deadline-hours H]`
   It prints the projected workload per stage and every GPU configuration that
   fits Qwen3.8-27B-FP8 within the 4-GPU account limit, ranked in the
   **preferred hosting order: #1 4x B200, #2 4x H200, #3 2x B200, #4 2x H200,
   #5 2x H100** (then the rest by cost), each with $/hour, estimated billed
   time and cost. The 4-GPU options run as **two 2-GPU instances behind one
   model id** (the model fits on two GPUs; two instances roughly halve the
   GPU-bound stages for about the same total cost). It recommends #1 and writes
   `.build/sizing.json` holding #1 AND its fallbacks (#2-#5), each with **its
   own concurrency**, sized from two memories:
   - **GPU memory** — the per-stage throughput knee inside that
     configuration's KV cache (×2 for two instances), calibrated on measured
     builds — e.g. class_proposal 192 on 4x B200, 48 on 2x H200;
   - **this machine's memory** — every in-flight request's prompt and reply
     must fit in RAM (Linux, macOS and Windows are read; a 600 MB reserve is
     kept), which also sizes `table_extraction` workers and
     `table_page_vision` (page-vision calls per worker) and caps
     `chunking.streaming_batch_size`. It adds `class_judges` (Stage-3 pair and
     validity judges) from the GPU's dedup concurrency. Any value lowered by
     local memory is printed as "capped by local memory".
   Show the user the preferred rows and their rates. Several corpora: pass them
   comma-separated to size ONE window (spin-up is billed per deployment).
3. **Only with explicit approval of the hourly rates** (all of them — a
   fallback may be the one that runs):
   `uv run python scripts/deepinfra_deploy.py create --from-sizing .build/sizing.json --apply-concurrency --yes`
   It tries 4x B200, 4x H200, 2x B200, 2x H200, then 2x H100: a 409 "currently unavailable"
   or a deployment that fails with `no-gpu-available` is deleted at once and the
   next is tried; it waits for `running` (weights load in ~4-8 min). It records
   the winner as `active` in the sizing file and writes **that GPU's**
   concurrency into `config/config.yaml`. Exit 3 = nothing had capacity and
   nothing is left running — tell the user, and offer to retry every ~10 min
   (capacity comes and goes within the hour). Logs to `.build/gpu_ledger.jsonl`.
4. User sets `DEEPINFRA_FLASH_MODEL=<username>/<printed model_name>` in `.env`;
   re-run `select_llm_mode.py deepinfra`. Record:
   `uv run python scripts/build_state.py record sizing applied=<yes|custom|kept>`.
   **Check the concurrency actually in use before the first paid step.**
   `--apply-concurrency` writes `config/config.yaml`; any OTHER config the
   build reads (`APP_CONFIG_PATH`, a copied bench config, a config from an
   earlier, smaller GPU) keeps its old values. Compare its `concurrency:`
   block with the sizing output for the GPU that won (`active` in the sizing
   file). Measured 2026-11-07: a copied config held class_proposal 24 /
   dedup 16 / summarization 64 against the 2x H200 sizing of 48 / 48 / 96;
   with the sized values, two corpora (318k tokens, 41 docs) built in ONE
   1.31 h window for $7.05 of 27B time (prune-expand 27 min and 20 min).
   The sizing printout also says where the time goes: when "most billed
   time is spin-up + pipeline gaps", batch the corpora into one window
   rather than adding GPUs.
5. Optional (~5 GPU-minutes): `uv run python scripts/deepinfra_sizing.py --probe`
   measures the real throughput knee on the running deployment; `--apply`
   writes it. Worth it for a long build; skip for a short one.
6. Run the creation steps back to back (prune-expand, register-documents,
   extract-entities, enrich-time, enrich-geo, generate-artifacts, and the final
   real `mine-aliases`) — the GPU bills while idle.
7. **Delete the deployment as soon as the last creation step finishes**:
   `uv run python scripts/deepinfra_deploy.py delete <deploy_id>` (the id is in
   the sizing file's `active`). Q&A keeps working with no GPU — it runs on
   serverless DeepSeek-V4.1-Flash. `deepinfra_deploy.py status` shows billed
   hours per deployment and flags any still running.

**Where each model runs (fixed by the preset, not by sizing):** Qwen3.8-27B on
the dedicated GPUs for every knowledge-creation task (summaries, classification,
class proposal, dedup and its judges, entities, artifacts); **DeepSeek-V4.1-Flash
always on serverless** (relationships, repair, all Q&A, eval judges), as are
the embeddings (Qwen3-Embedding-4B). Never move DeepSeek onto the dedicated
deployment. Table vision (Qwen3-VL) is the one exception: see below.

**Table vision (`--tables` on PDFs): HOST it — recommended over serverless.**

*Which model — Qwen3-VL-235B, not 30B.* Measured 2026-10-24 on 16 hard fin-gen
statement pages (LGES/SDI/24220c53), each twice, scored against the PDF's own
text layer (serverless, 8 pages in flight, production page-vision prompt,
150 dpi, 16k output cap):

| Model | Rows right | Precision | Same rows both runs | Figures under a wrong label | Median s/page | $/page |
|---|---|---|---|---|---|---|
| Qwen3-VL-30B-A3B | 0.66 | 0.73 | 0.60 | 11 | 98 | 0.0022 |
| **Qwen3-VL-235B-A22B** | **0.87** | **0.87** | 0.92 | 2 | 124 | 0.0026 |
| gemma-4-31B | 0.75 | 0.85 | 0.95 | - | 70 | 0.0014 |
| gemini-3.7-flash (proprietary, resold) | 0.86 | 0.86 | 0.99 | - | 25 | 0.016 |

The full 3-PDF re-extraction confirmed it (rows right 235B vs 30B: 235 vs 201,
596 vs 520, 483 vs 399). Never fall back to the 30B to save time.

*Why host.* Serverless speed drifts badly: the 124 s/page above was a good day
(3 fin-gen PDFs in ~25 min, ~$0.26); on 2026-10-26 it was ~7 tok/s (217 s for
1.5k tokens) and on 2026-11-07 ~200-230 s/page with dense statement pages
timing out at 2 x 750 s and left PARTIAL. Hosted Qwen3-VL measured **~205 tok/s
per stream vs ~33 serverless, identical output** (2026-10-24, the 30B on 1x H100).

*Where to host the 235B* (FP8 repo `Qwen/Qwen3-VL-235B-A22B-Instruct-FP8`, ~236 GB
of weights; DeepInfra bills **per GPU**, so N GPUs = N x the per-GPU rate):

| Configuration | GPU memory | $/hour | Result |
|---|---|---|---|
| 2x H200-141GB | 282 GB | 5.38 | **FAILS** — `wont-start: couldn't start model` after ~10 min (~$0.85 billed), 2026-11-07 |
| 1x B300-270GB | 270 GB | 4.89 | do not try: less headroom than the failed 2x H200 |
| **2x B200-180GB** | 360 GB | **7.38** | **recommended, WORKS** (2026-11-07): ~12 min to load; 3 fin-gen PDFs (LGES 58 p, SDI 99 p, 24220c53 55 p) -> 206 tables in **337 s** with 24 pages in flight (40-70 s per dense statement page), 0 failed / 0 partial; **$2.18 billed in total** (~18 min) |
| 4x H200-141GB | 564 GB | 10.76 | **WORKS, first fallback** (2026-11-07, when 2x B200 had no capacity for ~20 min): ~10 min to load; 5 pharma-12 PDFs (incl. a 125-page label) -> 129 tables in **566 s** at 24 pages in flight, 0 failed; **$3.57 billed in total** |
| 4x H100-80GB | 320 GB | 8.80 ($2.20 per H100; 2x H100 is $4.40) | untested; second fallback (tighter than 4x H200) |

Cost for 3 audited filings: **$2.18 measured** on 2x B200 (load dominates;
extraction itself ~6 min), vs ~$0.26 serverless on a good day — and serverless
took >30 min for 2 pages on 2026-11-07. Budget ~$2.50 + ~$0.10 per extra filing. The account limit is 4 GPUs across all live deployments; running
the VL deployment first and deleting it before the 27B one (step 1) keeps
them from overlapping.

Procedure:
1. Run it as a SEPARATE deployment, **before** the Qwen3.8-27B deployment
   exists (tables need no 27B; otherwise the 27B bills idle while pages run).
   **Hosting order: 2x B200 first, then 4x H200, then 4x H100** — shipped as
   `config/deepinfra_vl235.sizing.example.json` (B200 + both fallbacks):
   `uv run python scripts/deepinfra_deploy.py create --from-sizing config/deepinfra_vl235.sizing.example.json --name qwenvl235 --wait-min 25 --yes`
   No `--no-fallback`: a 409 "no capacity" (B200 had none for ~20 min on
   2026-11-07) or a failed start is deleted at once and the next entry is
   tried. Never add 2x H200 to this chain (the 235B does not start on it).
   If nothing has capacity (exit 3), retry every ~4-10 min; when the corpus
   also has non-PDF work, start the 27B build of a txt-only corpus meanwhile
   and keep the PDF corpus waiting for the tables (bench recipe:
   `benchmarks/2026-11-07/ep/vl_retry.sh` + the pharma gate in `ep/arm.py`).
   Give it a hard time cap and delete it on exit (the bench recipe
   `benchmarks/2026-11-07/dcv2/vl_phase.sh` traps EXIT and deletes).
2. Point BOTH `table_extract_vision` and `table_extract_page_vision` in
   `config/models.yaml` at `<username>/<printed model_name>` (provider stays
   `deepinfra`), set `concurrency.table_page_vision` to **24** on the hosted
   235B (measured on 2x B200 and 4x H200: 0 failures; use ~8 serverless) with
   `concurrency.table_extraction: 3` PDFs at a time (memory-bound — see tpm_check),
   and warm the table cache: run the `--tables` step (or extract tables on
   their own) with the deployment up.
3. Delete the VL deployment as soon as extraction finishes.
4. **Keep that hosted model id in both vision tasks for every later `--tables`
   step** (prune-expand AND register-documents) even after the deployment is
   gone. The table cache key hashes `provider:model` of both vision tasks
   (`table_cache.vision_models_tag`): switching back to the serverless name
   misses every cached PDF and re-extracts it. Cached PDFs need no calls; a
   PDF cached as PARTIAL re-sends only its failed pages, which then fail
   against the deleted deployment (set a short `timeout` / `max_retries: 0`
   so they fail fast instead of holding the 27B GPU idle).
Serverless stays acceptable for a handful of short PDFs, or when no GPU budget
is approved; then run 8 pages in flight. Ask for approval of the hourly rate
as for any deployment, quoting the PER-GPU rate x GPU count.

**One task can use either:** `alias_judge` (mine-aliases) runs on the dedicated
deployment while it is `running` and falls back to serverless `Qwen/Qwen3.8-27B`
by itself otherwise (`fallback_model` in the preset; verdicts are shared). For a
stand-alone alias refresh with many new pairs, a single GPU is enough:
`config/deepinfra_1gpu.sizing.example.json` (1x H100 -> 1x H200 -> 1x B200) --
see **ingest-corpus**, "Alias judge". No capacity (exit 3) -> run on serverless.

Concurrency note: `config/config.yaml` `concurrency:` is shared by every mode.
DeepInfra's values (up to 192) would hit OpenAI/Anthropic rate limits — after
switching back to another mode, re-size with `tpm_check.py`.

## Notes
- A wrong/expired key cannot be detected by presence check — only a real call
  fails. If a later step 401s on a provider, the fix is almost always the key in
  `.env`; re-point the user there (never ask for it in chat).
- To switch modes later, just re-run this skill — it overwrites `config/models.yaml`.
