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
| **DeepInfra / Qwen** | `deepinfra` | `DEEPINFRA_API_KEY`, `DEEPINFRA_FLASH_MODEL` | Knowledge creation on a **dedicated** Qwen3.8-27B deployment (billed per GPU-hour); relationships, repair and judges on serverless DeepSeek-V4.1-Flash; retrieval on serverless `Qwen/Qwen3.8-27B`; table vision `Qwen3-VL-30B-A3B`; embeddings `Qwen3-Embedding-4B`@1024. No OpenAI/Anthropic key. Needs its OWN database (vectors can't mix with OpenAI ones; a guard refuses). See "DeepInfra mode" below. |

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

## DeepInfra mode — size the GPUs BEFORE deploying (paid)
`DEEPINFRA_FLASH_MODEL` is not a secret: it is the dedicated deployment's model
id (`<deepinfra-username>/<deploy-name>`), which exists only after a deployment
is created. Order:
1. User adds `DEEPINFRA_API_KEY` to `.env` themselves.
2. Size from the corpus (free; reads documents only):
   `uv run python scripts/deepinfra_sizing.py "<DOCS>" [--deadline-hours H]`
   → projected workload per stage, every GPU configuration that fits Flash-Next
   within the 4-GPU account limit, $/hour, estimated billed time and cost, and a
   recommendation. Show the table; let the user pick.
3. **Only with explicit approval of the hourly rate**:
   `uv run python scripts/deepinfra_deploy.py create --from-sizing .build/sizing.json`
   (bills per minute until deleted; logs to `.build/gpu_ledger.jsonl`).
4. User sets `DEEPINFRA_FLASH_MODEL=` in `.env`; re-run `select_llm_mode.py deepinfra`.
5. Once `deepinfra_deploy.py status` shows it running:
   `uv run python scripts/deepinfra_sizing.py --probe` (≈5 min of GPU time) measures
   the throughput knee; with the user's go-ahead re-run with `--apply` to write
   `concurrency:`. Record: `uv run python scripts/build_state.py record sizing applied=<yes|custom|kept>`.
6. **Delete the deployment as soon as creation stages finish**:
   `uv run python scripts/deepinfra_deploy.py delete <deploy_id>`. Retrieval keeps
   working — it runs on serverless 27B.

## Notes
- A wrong/expired key cannot be detected by presence check — only a real call
  fails. If a later step 401s on a provider, the fix is almost always the key in
  `.env`; re-point the user there (never ask for it in chat).
- To switch modes later, just re-run this skill — it overwrites `config/models.yaml`.
