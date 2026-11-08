"""Structured-data download helpers (finance + pharma public datasets).

Entry point: ``source_documents/structured_download.py``. Each provider lives
in ``structured/sources/<name>.py`` and registers itself with the registry in
``structured/_base.py``. Shared behaviour (HTTP retries, atomic streaming
downloads, safe archive extraction, manifests, delimited-table conversion)
lives in the underscore modules so source modules stay focused on the
provider's own endpoints and data meaning.

Nothing here calls an LLM and nothing writes to the database; the output is
local files + a ``manifest.json`` per run, for a later ingestion upgrade.
"""
