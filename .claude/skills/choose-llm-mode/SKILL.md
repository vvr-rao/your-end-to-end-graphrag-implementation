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
- Check for a key using ONLY a quiet check that prints no value. A key counts as
  configured when it is present, non-empty, AND not a template placeholder (the
  `.env.example` placeholders are empty or end in `...`, e.g. `sk-...`, `gsk_...`):
  ```bash
  key_configured() {  # $1 = var name; exit 0 if a REAL value is set. Prints nothing.
    grep -qE "^$1=.+" .env && ! grep -qE "^$1=.*\.\.\.[[:space:]]*$" .env
  }
  ```
  A blank `ANTHROPIC_API_KEY=` fails `.+` (MISSING); a placeholder `sk-...` matches
  the second grep and is rejected (MISSING); a real `sk-proj-…` passes (present).
- If you ever need to show the user what to edit, show the **variable name**,
  never a value.

## Step 1 — Ask which mode
Present these three options (AskUserQuestion). Embeddings ALWAYS run on OpenAI
(`text-embedding-3-small`); Anthropic/Voyage are not used for embeddings. So
**every mode requires `OPENAI_API_KEY`.**

| Mode | Preset file | Keys required in `.env` | Notes |
|---|---|---|---|
| **Groq + OpenAI** (default) | `config/models.example.yaml` | `OPENAI_API_KEY`, `GROQ_API_KEY` | Groq runs chunk_classification (best disambiguation); OpenAI everything else. |
| **OpenAI only** | `config/models.openai.example.yaml` | `OPENAI_API_KEY` | Simplest; one provider. chunk_classification → `gpt-4.1-mini`. |
| **Anthropic + OpenAI** | `config/models.anthropic.example.yaml` | `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` | Claude for chat (haiku cheap tasks / sonnet synthesis); OpenAI for embeddings only. |

## Step 2 — Activate the preset
```bash
# .env scaffolding (safe: template has placeholders, no real secrets)
[ -f .env ] || cp .env.example .env
# copy the chosen preset into the active config
cp config/models.<chosen>.example.yaml config/models.yaml
```
Confirm it landed:
```bash
grep -m1 -A2 'chunk_classification:' config/models.yaml | grep provider:
python3 scripts/build_state.py record llm-mode mode=<chosen>   # cross-session tracker
```
(`models.yaml` has no secrets — reading it is fine and expected.)

## Step 3 — Verify required keys are present (presence only)
Using the `key_configured` helper above, check each key the chosen mode needs.
Example for Anthropic+OpenAI mode:
```bash
key_configured() { grep -qE "^$1=.+" .env && ! grep -qE "^$1=.*\.\.\.[[:space:]]*$" .env; }
for k in OPENAI_API_KEY ANTHROPIC_API_KEY; do
  if key_configured "$k"; then echo "$k: present"; else echo "$k: MISSING"; fi
done
```
- If all present → report ready, hand back to the orchestrator.
- If any MISSING → tell the user exactly which variable(s) to fill in `.env`
  (by name), e.g. *"Open `.env` in your editor and set `ANTHROPIC_API_KEY=...`.
  I can't see or accept the value — add it there and tell me when done."* Then
  re-run the presence check. Do not proceed to any paid step until keys pass.

## Notes
- A wrong/expired key cannot be detected by presence check — only a real call
  fails. If a later step 401s on a provider, the fix is almost always the key in
  `.env`; re-point the user there (never ask for it in chat).
- To switch modes later, just re-run this skill — it overwrites `config/models.yaml`.
