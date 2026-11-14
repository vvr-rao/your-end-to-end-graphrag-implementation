"""Structured-data support: profile flat tables (CSV/TSV/JSON/Parquet) with an
LLM, cache the per-table metadata, add classes to the ontology, load the tables
into Postgres, link them to the graph and answer questions through SQL.

Every entry point is opt-in (``--structured-data`` / ``--use-structured-data``);
without those flags no module here is imported by the pipelines.
"""
