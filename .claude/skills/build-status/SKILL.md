---
name: build-status
description: >
  Report where the user is in building the app and what to do next, using the
  cross-session progress tracker. Use when the user asks "what's the status",
  "what did we do last", "where were we", "what's next", or wants to resume a
  build after restarting their session.
---

# Build status — resume where the user left off

The build progress is tracked on disk in `.build/state.jsonl` (written as each
step completes), so it survives a closed/restarted session. A SessionStart hook
already surfaces it automatically at the start of each session, but use this skill
whenever the user explicitly asks about status or wants to continue.

## Report status
```bash
python3 scripts/build_state.py show
```
Relay it plainly: the recent steps, the **last operation** (and its artifact
path), and the **suggested next step**. Then offer to continue from there.

Example: if the last operation is `merge` with `path=output_ontologies/v..-merge/`,
tell the user the merge is done and where it landed, and offer to run prune-expand
against a document corpus next (hand off to **run-prune-expand**, whose `--input`
is that merge folder).

## Continue
Based on the suggested next step, hand off to the right skill:
- `llm-mode` → **choose-llm-mode**
- `database` → **setup-database**
- `merge` / `prune-expand` → **run-prune-expand**
- anything past prune-expand (`db-init`, `register-documents`, …) → **build-app**
  drives the sequence.

Never re-run a completed step without asking — the tracker shows what's already
done, and steps like prune-expand are paid. Confirm before repeating anything.

## If nothing is recorded
`show` will say so. Point the user to **build-app** ("where do I start") to begin.

## Note
The tracker records what the assistant has confirmed completing. If the user did
something outside these skills (e.g. ran a CLI command themselves), it won't be in
the log — check the actual artifacts (`output_ontologies/`, `db-status`) if the
user says they did something the tracker doesn't show, and record it with
`python3 scripts/build_state.py record <step> path=<...>` once confirmed.
