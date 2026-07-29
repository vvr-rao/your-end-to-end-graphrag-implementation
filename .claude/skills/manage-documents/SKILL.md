---
name: manage-documents
description: >
  Add, list, update, or delete documents in an already-built knowledge graph
  (after the initial ingest). Handles soft vs hard delete, the STALE-artifact
  fallout of updates/deletes, and re-running extraction for new or changed docs.
  Use when the corpus changes — new documents to add, or ones to remove/replace.
---

# Manage documents (add / list / update / delete)

Once the graph is built (see **ingest-corpus**), the corpus can change. This skill
drives the document lifecycle. Every command is `uv run python -m backend.app.cli
<cmd>` (cross-platform). Requires the DB set up + the ontology loaded (`db-init`).

**Destructive actions need explicit confirmation** — deletes and updates have
knock-on effects on chunks and intelligence artifacts. Confirm before running them.

## Find a document's IRI first
Update/delete key off the document IRI. List to find it:
```bash
uv run python -m backend.app.cli list-documents                  # all
uv run python -m backend.app.cli list-documents --status ACTIVE  # or STALE / DELETED
```

## Add documents to an existing graph
`register-documents` is **idempotent — it skips any doc already ingested (matched
by sha256)**, so pointing it at a folder containing new files adds only the new
ones. Smoke-test a big batch with `--limit` first, then launch detached:
```bash
# choose a fresh RUN_ID, e.g. add_docs_<date+time>
uv run python scripts/run_detached.py <RUN_ID> \
  uv run python -m backend.app.cli register-documents --input "<folder-with-new-docs>" [--tables] [--full-text-chunks]
uv run python scripts/job_status.py <RUN_ID> 40
```
**Then fold them into the graph.** New docs create chunks but NOT yet entities or
artifacts. Re-run the extraction steps — each is idempotent and processes only the
new, unprocessed chunks: `extract-entities` (skips chunks that already have edges),
`enrich-time`, `generate-artifacts`. See **ingest-corpus** Steps 3–5. Record it:
```bash
uv run python scripts/build_state.py record add-documents docs=<n>
```

## Update (replace) a document
Replaces a document's content. **No-op if the new file's sha256 matches the old.**
```bash
uv run python -m backend.app.cli update-document --iri "<doc-iri>" --path "<new-file>"
```
Effect (by design): a new version supersedes the old, the old document → `DELETED`,
and artifacts whose **ALL** sources were in that document → **STALE** (mixed-source
artifacts stay `ACTIVE`). Then re-run register/extract/generate so the new content
is chunked, mined, and re-covered. See "Stale artifacts" below.

## Delete a document
```bash
# Soft delete (DEFAULT) — status -> DELETED; rows kept so existing citations resolve
uv run python -m backend.app.cli delete-document --iri "<doc-iri>"

# Hard delete — removes the document row and CASCADES to its chunks (IRREVERSIBLE)
uv run python -m backend.app.cli delete-document --iri "<doc-iri>" --hard
```
- **Soft (default)** is the safe choice: the document is flagged `DELETED` but its
  rows remain. Artifacts whose ALL sources were in it → **STALE**; mixed-source
  artifacts stay `ACTIVE`. Retrieval excludes DELETED/STALE rows immediately.
- **`--hard`** is irreversible — the row and its chunks are removed via cascade.
  **Get explicit user confirmation before using it.**

Record the action, e.g. `uv run python scripts/build_state.py record delete-document iri=<iri> mode=<soft|hard>`.

## Stale artifacts (the follow-on that bites)
After an update or soft-delete, artifacts that depended **entirely** on the
affected document are set `status='STALE'` (mixed-source ones stay `ACTIVE`).
Retrieval simply **skips STALE rows**, so answers won't cite them — but the rows
persist. **There is currently no one-command `regenerate-stale-artifacts`.**
Regeneration is manual: re-run `generate-artifacts` (optionally `--rollup`) so
fresh artifacts replace the stale coverage. **Tell the user this explicitly** — so
they know stale artifacts linger (excluded from retrieval) until regenerated.

## Notes
- `list-documents --status STALE` and `--status DELETED` show what a change left
  behind.
- Deletes/updates change retrieval results immediately (DELETED/STALE excluded);
  artifact coverage catches up only when you regenerate.
- All long adds run detached (survive a killed session); short ops run directly.
