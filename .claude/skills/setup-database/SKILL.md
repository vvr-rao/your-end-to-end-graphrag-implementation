---
name: setup-database
description: >
  Set up the Postgres+pgvector database for this GraphRAG app. Asks whether the
  user already has a database (e.g. a Supabase connection string) and, if not,
  brings up the local docker-compose Postgres automatically, then runs the
  schema migration. Use during initial build or when db-status reports no
  connection/schema. Never accepts a connection string in chat.
---

# Set up the database

The app stores everything in Postgres with the pgvector extension. This step
gets a working database and creates the `graphrag` schema. It does NOT load the
ontology yet — that happens after prune-expand (via `db-init --input <folder>`).

## SECURITY
- A connection string contains a password — it is a credential. **Never** accept
  it in chat and never print `.env`. The user puts `DATABASE_URL` in `.env`
  themselves; you validate by making the app connect, not by reading the value.
- You may run `cp .env.example .env` once (template, no secrets). After that the
  user owns `.env`.

## Step 1 — Ask which database, and warn about the deploy implication
Ask the user (AskUserQuestion):

> Do you already have a Postgres database (e.g. a Supabase connection string),
> or should I set up a local one for you?

**Say this warning at the fork — it saves a wasted multi-hour rebuild:**
> A **local** database is great for trying the whole pipeline on your machine,
> but it only exists here — a later Render deploy talks to a *cloud* database, so
> you'd have to re-run the pipeline against Supabase before deploying. If you
> already know you want to deploy, give me a **Supabase** connection string now
> and we build once.

Ensure the env file exists either way:
```bash
[ -f .env ] || cp .env.example .env
```

## Step 2a — External database (Supabase or any Postgres)
1. Ask the user to open `.env` in their editor and set `DATABASE_URL`. For
   Supabase, it MUST be the **Session Pooler** URL (the direct `db.*.supabase.co`
   host is IPv6-only and will hang). Format (project-ref goes in the username):
   ```
   DATABASE_URL=postgresql://postgres.<project-ref>:<password>@aws-0-<region>.pooler.supabase.com:5432/postgres?sslmode=require
   ```
   Tell them: *"Paste it into `.env` — I can't see or accept it here. Tell me when
   it's saved."* (The app auto-appends `sslmode=require` for supabase hosts, but
   including it is fine.)
2. For Supabase, remind them to enable pgvector **once per database**:
   Dashboard → Database → Extensions → search "vector" → toggle ON.
3. Go to Step 3.

## Step 2b — Local docker-compose database (automatic)
Bring Postgres (pgvector/pgvector:pg16) up and wait until it's healthy:
```bash
uv run python scripts/db_up.py
```
(Requires Docker Desktop running. The helper runs `docker compose up -d postgres`
and polls the container health cross-platform.)
Then tell the user to set this **exact, non-secret local** line in `.env` (they
paste it — keep `.env` owned by them; you may display this value because it is a
throwaway local credential):
```
DATABASE_URL=postgresql://ontologist:ontologist@localhost:5432/ontologist
```
Local pgvector is preinstalled in that image — no extension toggle needed. Go to
Step 3.

## Step 3 — Migrate + verify (both paths)
`db-init` with no `--input` migrates the schema and prints status — the app reads
`DATABASE_URL` from `.env` itself, so no credential passes through you:
```bash
uv run python -m backend.app.cli db-init
```
On success, record it for the cross-session tracker (`local` or `external`):
```bash
uv run python scripts/build_state.py record database kind=<local|external>
```
- Success looks like: alembic upgrade to head + a db-status report (schema
  present, tables listed, size comfortably within the database's storage limit).
- If it fails to connect: the fix is almost always `DATABASE_URL` in `.env`
  (typo, wrong host, direct-vs-pooler for Supabase, or pgvector not enabled).
  Point the user back to `.env`; never ask for the string in chat.

## Done
Report the DB is ready and which target was used (local vs external). Note for
the orchestrator: the ontology import (`db-init --input <prune-expand folder>`)
runs later, once prune-expand has produced a version folder.
