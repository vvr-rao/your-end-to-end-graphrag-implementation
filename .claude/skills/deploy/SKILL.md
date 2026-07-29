---
name: deploy
description: >
  Deploy this GraphRAG app to Render — guide the user through forking the repo,
  setting RENDER_API_KEY + generating a BEARER_TOKEN in .env, running render-init
  to create the backend + frontend services, monitoring the build, and the final
  manual UI token paste. Use when the user wants to deploy, publish, or push the
  app to Render. Never accepts credentials in chat.
---

# Deploy to Render

The last step. It creates two Render services from `render.yaml` — a Docker
**backend** (FastAPI + MCP) and a static **frontend** (the React UI) — wired to a
cloud Postgres. Render pulls source from a GitHub repo, so the user deploys from
**their own fork**, and the CLI drives Render's **REST API** (no "Render MCP" and
no dashboard blueprint click-through required).

## SECURITY — credentials never touch the chat
- Never read `.env`; never ask the user to paste a key or token into the chat.
- The user generates `BEARER_TOKEN` and sets `RENDER_API_KEY` in `.env`
  **themselves**. You only presence-check that a var exists (exit code, never the
  value). Do NOT run a token generator yourself — its output would print the
  secret into this conversation.

## Step 0 — Preconditions (check all before doing anything)

**(a) They deploy from their OWN fork.** `render-init` reads the `origin` remote
and tells Render to build from it, so `origin` must be a repo on the user's own
GitHub account:
```bash
git remote get-url origin
```
If `origin` is the upstream project rather than the user's account, stop and tell
them to **fork it on GitHub**, then either clone their fork or
`git remote set-url origin <their-fork-url>`. A public fork lets Render deploy
straightaway; a private fork requires the user to connect GitHub in the Render
dashboard first.

**(b) DATABASE_URL points at a CLOUD database, not the local docker one.** The
deployed backend cannot reach a laptop's localhost. Check the build tracker and
look at the `database` line:
```bash
uv run python scripts/build_state.py show
```
If the database step was recorded `kind=local`, **STOP**: a local docker DB won't
work in production. The user must set `DATABASE_URL` to a cloud Postgres (e.g. a
Supabase Session Pooler URL) in `.env`; and if the ontology/corpus was built
against the local DB, the pipeline must be re-run against the cloud DB before
deploying (see **setup-database**).

**(c) Required secrets present** (presence-check only — never print a value):
```bash
uv run python scripts/check_env.py DATABASE_URL OPENAI_API_KEY BEARER_TOKEN RENDER_API_KEY
```

## Step 1 — BEARER_TOKEN (the app's own API auth)
This token protects the deployed REST/MCP API and is later pasted into the UI. If
it's missing, ask the **user** to generate one and add it to `.env` themselves —
do not run this yourself:
> In your terminal, run this and paste the result into `.env` as
> `BEARER_TOKEN=...`:
> `python -c "import secrets; print(secrets.token_urlsafe(32))"`

## Step 2 — RENDER_API_KEY (auth to Render)
If missing, point the user to
https://dashboard.render.com/u/settings#api-keys → create a key → add
`RENDER_API_KEY=...` to `.env`. If they belong to more than one Render team/owner,
`render-init` will ask for `RENDER_OWNER_ID` — have them add that too.

## Step 3 — Create the services with render-init
```bash
uv run python -m backend.app.cli render-init
```
Reads `render.yaml` + `.env`, creates the `backend` + `frontend` services against
your fork's current branch, pushes the env vars, and triggers the first deploys.
Idempotent — existing services are picked up, not duplicated.
- Multiple Render owners reported → add `RENDER_OWNER_ID` to `.env` and re-run.
- `--branch <name>` deploys a branch other than the current one; `--no-deploy`
  creates services without triggering a build.

## Step 4 — Monitor the build
Builds take several minutes (Docker backend + static frontend). Poll:
```bash
uv run python -m backend.app.cli render-status
uv run python -m backend.app.cli render-logs --service backend --since 10m
```
Or block on one service until it reaches live/failed:
```bash
uv run python -m backend.app.cli render-deploy --service backend --wait
```
Report state transitions; if a build fails, surface the logs tail and diagnose
(missing env var, DB unreachable, Docker build error).

## Step 5 — Final manual step: connect the UI
`render-init` prints the frontend URL. This one is a browser action the user must
do themselves — you cannot. Tell them to:
1. open the **frontend URL**,
2. go to **Settings**, and
3. paste their `BEARER_TOKEN` (the value from `.env`) so the UI can call the
   backend API.

## Step 6 — Record + confirm
Once both services are live, read the URLs from `render-status` and record the
step for the cross-session tracker:
```bash
uv run python scripts/build_state.py record deploy backend=<backend-url> frontend=<frontend-url>
```
Report both URLs and note the backend serves REST at `/`, MCP at `/mcp`, and a
health check at `/health`, and that the UI needs the bearer token pasted (Step 5).

## Managing a running deployment
- `render-status` — current state of both services.
- `render-logs --service <backend|frontend> --since 10m` — tail logs.
- `render-deploy --service <name> [--wait] [--clear-cache]` — redeploy.
- `render-suspend --all` / `render-resume --all` — pause / resume compute
  (free-tier hours stop accruing while suspended; fully reversible).
