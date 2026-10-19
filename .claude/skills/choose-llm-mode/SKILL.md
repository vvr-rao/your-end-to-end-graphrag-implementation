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

| Mode | `<mode>` arg | Keys required in `.env` | Notes |
|---|---|---|---|
| **Groq + OpenAI** (default) | `groq-openai` | `OPENAI_API_KEY`, `GROQ_API_KEY` | Groq runs chunk_classification (best disambiguation); OpenAI everything else. |
| **OpenAI only** | `openai` | `OPENAI_API_KEY` | Simplest; one provider. chunk_classification → `gpt-4.1-mini`. |
| **Anthropic + OpenAI** | `anthropic` | `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` | Claude for chat (haiku cheap tasks / sonnet synthesis); OpenAI for embeddings only. |
| **DeepInfra (Qwen + DeepSeek)** | `deepinfra` | `DEEPINFRA_API_KEY`, `DEEPINFRA_FLASH_MODEL` | Knowledge creation on a **dedicated** Qwen3.8-27B deployment (rented GPUs, billed per GPU-hour while it exists); relationships, repair, judges and **all Q&A** on serverless DeepSeek-V4.1-Flash; table vision `Qwen3-VL-30B-A3B`; embeddings `Qwen3-Embedding-4B`@1024. No OpenAI/Anthropic key. Needs its OWN database (vectors can't mix with OpenAI ones; a guard refuses). See "DeepInfra mode" below. |

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
   **preferred hosting order: #1 2x B200, #2 2x H200, #3 2x H100** (then the
   rest by cost), each with $/hour, estimated billed time and cost. It
   recommends #1 and writes `.build/sizing.json` holding #1 AND its fallbacks
   (#2, #3), each with **its own throughput-optimised concurrency** (the
   per-stage throughput knee inside that GPU's KV cache, calibrated on measured
   builds — e.g. class_proposal 96 / 48 / 32). Show the user the three
   preferred rows and their rates. Several corpora: pass them comma-separated
   to size ONE window (spin-up is billed per deployment).
3. **Only with explicit approval of the hourly rates** (all three — the
   fallback may be the one that runs):
   `uv run python scripts/deepinfra_deploy.py create --from-sizing .build/sizing.json --apply-concurrency --yes`
   It tries 2x B200, then 2x H200, then 2x H100: a 409 "currently unavailable"
   or a deployment that fails with `no-gpu-available` is deleted at once and the
   next is tried; it waits for `running` (weights load in ~4-8 min). It records
   the winner as `active` in the sizing file and writes **that GPU's**
   concurrency into `config/config.yaml`. Exit 3 = nothing had capacity and
   nothing is left running — tell the user, and offer to retry every ~10 min
   (capacity comes and goes within the hour). Logs to `.build/gpu_ledger.jsonl`.
4. User sets `DEEPINFRA_FLASH_MODEL=<username>/<printed model_name>` in `.env`;
   re-run `select_llm_mode.py deepinfra`. Record:
   `uv run python scripts/build_state.py record sizing applied=<yes|custom|kept>`.
5. Optional (~5 GPU-minutes): `uv run python scripts/deepinfra_sizing.py --probe`
   measures the real throughput knee on the running deployment; `--apply`
   writes it. Worth it for a long build; skip for a short one.
6. Run the creation steps back to back (prune-expand, register-documents,
   extract-entities, enrich-time, generate-artifacts) — the GPU bills while idle.
7. **Delete the deployment as soon as the last creation step finishes**:
   `uv run python scripts/deepinfra_deploy.py delete <deploy_id>` (the id is in
   the sizing file's `active`). Q&A keeps working with no GPU — it runs on
   serverless DeepSeek-V4.1-Flash. `deepinfra_deploy.py status` shows billed
   hours per deployment and flags any still running.

Concurrency note: `config/config.yaml` `concurrency:` is shared by every mode.
DeepInfra's values (up to 96) would hit OpenAI/Anthropic rate limits — after
switching back to another mode, re-size with `tpm_check.py`.

## Notes
- A wrong/expired key cannot be detected by presence check — only a real call
  fails. If a later step 401s on a provider, the fix is almost always the key in
  `.env`; re-point the user there (never ask for it in chat).
- To switch modes later, just re-run this skill — it overwrites `config/models.yaml`.
