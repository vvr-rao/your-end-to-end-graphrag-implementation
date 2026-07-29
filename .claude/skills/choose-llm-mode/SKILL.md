---
name: choose-llm-mode
description: >
  Select the LLM provider mode for this GraphRAG app (Groq+OpenAI, OpenAI-only,
  or Anthropic+OpenAI) and activate the matching config preset. Use when setting
  up the app, switching providers, or when a build step reports a missing
  provider/model config. Handles API keys by pointing the user to .env — never
  accepts or displays credentials in chat.
---

# Choose the LLM provider mode

This app's per-task provider/model map lives in `config/models.yaml`. Three
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
Present these three options (AskUserQuestion). Embeddings ALWAYS run on OpenAI
(`text-embedding-3-small`); Anthropic/Voyage are not used for embeddings. So
**every mode requires `OPENAI_API_KEY`.**

| Mode | `<mode>` arg | Keys required in `.env` | Notes |
|---|---|---|---|
| **Groq + OpenAI** (default) | `groq-openai` | `OPENAI_API_KEY`, `GROQ_API_KEY` | Groq runs chunk_classification (best disambiguation); OpenAI everything else. |
| **OpenAI only** | `openai` | `OPENAI_API_KEY` | Simplest; one provider. chunk_classification → `gpt-4.1-mini`. |
| **Anthropic + OpenAI** | `anthropic` | `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` | Claude for chat (haiku cheap tasks / sonnet synthesis); OpenAI for embeddings only. |

## Step 2 — Activate the mode (one command)
```bash
uv run python scripts/select_llm_mode.py <mode>   # groq-openai | openai | anthropic
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

## Notes
- A wrong/expired key cannot be detected by presence check — only a real call
  fails. If a later step 401s on a provider, the fix is almost always the key in
  `.env`; re-point the user there (never ask for it in chat).
- To switch modes later, just re-run this skill — it overwrites `config/models.yaml`.
