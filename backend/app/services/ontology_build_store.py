"""Staging store for class-proposal cleanup at scale (schema `ontology_build`,
migration 0011). Everything streams: proposals are upserted per Stage-2 reply
as ONE row per distinct folded label (occurrences counted), embeddings are
filled in batches, and nearest-neighbour candidates come from HNSW lookups in
id batches -- n*k index probes, never an n x n matrix. Memory stays at one
batch whatever the corpus size; the vectors live on disk in Postgres.

Uses the application's database (`DATABASE_URL`) through `session_scope`.
"""
from __future__ import annotations

import re
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass
from typing import Any

from sqlalchemy import text

from backend.app.db.session import session_scope

SCHEMA = "ontology_build"
CHUNK_REFS_CAP = 20


def fold_key(label: Any) -> str:
    """Case / punctuation / spacing / CamelCase / plural-insensitive key: the
    exact-merge step. 'Refineries', 'refinery', 'Refinery ' fold together."""
    s = re.sub(r"[^a-z0-9]", "", str(label or "").lower())
    if len(s) > 4 and s.endswith("ies"):
        return s[:-3] + "y"
    if len(s) > 4 and s.endswith("s") and not s.endswith("ss"):
        return s[:-1]
    return s


def embed_text(label: str, description: str | None) -> str:
    """What a label is embedded as: its words plus a short description, so
    'Mercury' the planet and 'Mercury' the metal can be told apart."""
    words = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(label)).replace("_", " ")
    desc = " ".join(str(description or "").split()[:20])
    return f"{words}: {desc}" if desc else words


@dataclass
class Pair:
    a_id: int
    b_id: int
    cosine: float


def _scalar(v: Any) -> str | None:
    """Text column value: a list (merged across chunks) -> its first item."""
    if isinstance(v, (list, tuple)):
        v = next((x for x in v if x), None)
    return None if v is None else str(v)


class BuildStore:
    def __init__(self, run_id: str) -> None:
        self.run_id = run_id

    async def clear(self) -> None:
        async with session_scope() as s:
            await s.execute(text(f"DELETE FROM {SCHEMA}.pair_decisions WHERE run_id = :r"), {"r": self.run_id})
            await s.execute(text(f"DELETE FROM {SCHEMA}.label_vectors WHERE run_id = :r"), {"r": self.run_id})

    async def upsert(self, kind: str, rows: Iterable[dict[str, Any]], chunk_ref: int | None = None) -> int:
        """Fold and upsert one reply's labels. Returns rows touched."""
        params = []
        seen: set[str] = set()
        for r in rows:
            key = fold_key(r.get("label"))
            if not key or key in seen:
                continue
            seen.add(key)
            params.append({
                "r": self.run_id, "k": kind, "key": key, "label": str(r.get("label")),
                "parent": _scalar(r.get("parent")), "desc": _scalar(r.get("description")),
                "iri": _scalar(r.get("iri")),
                "chunks": [chunk_ref] if chunk_ref is not None else [],
            })
        if not params:
            return 0
        async with session_scope() as s:
            await s.execute(text(f"""
                INSERT INTO {SCHEMA}.label_vectors
                    (run_id, kind, label_key, label, parent, description, iri, chunk_refs)
                VALUES (:r, :k, :key, :label, :parent, :desc, :iri, CAST(:chunks AS integer[]))
                ON CONFLICT (run_id, kind, label_key) DO UPDATE SET
                    occurrences = {SCHEMA}.label_vectors.occurrences + 1,
                    parent = COALESCE({SCHEMA}.label_vectors.parent, EXCLUDED.parent),
                    description = COALESCE({SCHEMA}.label_vectors.description, EXCLUDED.description),
                    chunk_refs = CASE WHEN cardinality({SCHEMA}.label_vectors.chunk_refs) >= {CHUNK_REFS_CAP}
                                      THEN {SCHEMA}.label_vectors.chunk_refs
                                      ELSE {SCHEMA}.label_vectors.chunk_refs || EXCLUDED.chunk_refs END
            """), params)
        return len(params)

    async def embed_missing(self, embedder: Any, batch: int = 500) -> int:
        """Embed every row of this run without a vector, `batch` at a time."""
        from backend.app.services.semantic_grouping import _embed_compact

        done = 0
        while True:
            async with session_scope() as s:
                rows = (await s.execute(text(f"""
                    SELECT id, label, description FROM {SCHEMA}.label_vectors
                    WHERE run_id = :r AND embedding IS NULL ORDER BY id LIMIT :n
                """), {"r": self.run_id, "n": batch})).all()
            if not rows:
                return done
            vecs = await _embed_compact(embedder, [embed_text(r.label, r.description) for r in rows])
            async with session_scope() as s:
                await s.execute(text(f"UPDATE {SCHEMA}.label_vectors SET embedding = CAST(:v AS vector) WHERE id = :id"),
                                [{"id": r.id, "v": "[" + ",".join(f"{x:.6f}" for x in v) + "]"}
                                 for r, v in zip(rows, vecs, strict=True)])
            done += len(rows)

    async def set_status(self, ids: Iterable[int], status: str, reason: str | None = None,
                         merged_into: int | None = None) -> None:
        ids = list(ids)
        if not ids:
            return
        async with session_scope() as s:
            await s.execute(text(f"""
                UPDATE {SCHEMA}.label_vectors SET status = :st, status_reason = :why, merged_into = :into
                WHERE run_id = :r AND id = ANY(:ids)
            """), {"st": status, "why": reason, "into": merged_into, "r": self.run_id, "ids": ids})

    async def ids(self, kind: str, status: str = "kept") -> list[int]:
        async with session_scope() as s:
            return [r[0] for r in (await s.execute(text(f"""
                SELECT id FROM {SCHEMA}.label_vectors WHERE run_id = :r AND kind = :k AND status = :st ORDER BY id
            """), {"r": self.run_id, "k": kind, "st": status})).all()]

    async def _knn_batch(self, ids: list[int], to_kind: str, k: int,
                         ef: int = 100) -> dict[int, list[tuple[int, float]]]:
        """Top-k `to_kind` neighbours for every id in ONE round trip
        (`knn_batch`, migration 0011): HNSW on the 256-d prefix over-fetching
        3k, re-scored on the full vector."""
        out: dict[int, list[tuple[int, float]]] = {a: [] for a in ids}
        if not ids:
            return out
        async with session_scope() as s:
            rows = (await s.execute(text(f"SELECT a_id, b_id, cos FROM {SCHEMA}.knn_batch("
                                         f"CAST(:ids AS bigint[]), :r, :tk, :k, :over, :ef)"),
                                    {"ids": ids, "r": self.run_id, "tk": to_kind, "k": k,
                                     "over": 3 * k, "ef": ef})).all()
        for r in rows:
            out[int(r.a_id)].append((int(r.b_id), float(r.cos)))
        return out

    async def _knn(self, aid: int, to_kind: str, k: int) -> list[tuple[int, float]]:
        return (await self._knn_batch([aid], to_kind, k))[aid]

    async def neighbours(self, from_kind: str, to_kind: str, k: int = 10, min_cosine: float = 0.75,
                         batch: int = 250, concurrency: int | None = None) -> AsyncIterator[list[Pair]]:
        """Top-k nearest `to_kind` labels for every kept `from_kind` label,
        yielded per batch of source labels. Same-kind pairs are emitted once
        (a_id < b_id). Each batch is one server-side call; `concurrency`
        batches run at once (default: the CPU count -- the search is
        CPU-bound in Postgres, so more only queues). Memory holds one round
        of batches."""
        import asyncio
        import os

        src = await self.ids(from_kind)
        same = from_kind == to_kind
        conc = concurrency or max(1, min(8, os.cpu_count() or 2))
        parts = [src[i:i + batch] for i in range(0, len(src), batch)]
        for i in range(0, len(parts), conc):
            got = await asyncio.gather(*[self._knn_batch(p, to_kind, k) for p in parts[i:i + conc]])
            yield [Pair(a, b, c) for hits in got for a, bs in hits.items() for b, c in bs
                   if c >= min_cosine and (not same or a < b)]

    async def record(self, pairs: Iterable[tuple[int, int, float, str, str | None]]) -> None:
        rows = [{"r": self.run_id, "a": a, "b": b, "c": c, "d": d, "why": why} for a, b, c, d, why in pairs]
        if not rows:
            return
        async with session_scope() as s:
            await s.execute(text(f"""
                INSERT INTO {SCHEMA}.pair_decisions (run_id, a_id, b_id, cosine, decision, reason)
                VALUES (:r, :a, :b, :c, :d, :why)
                ON CONFLICT (run_id, a_id, b_id) DO NOTHING
            """), rows)

    async def counts(self) -> dict[str, int]:
        async with session_scope() as s:
            rows = (await s.execute(text(f"""
                SELECT kind || ':' || status, count(*), sum(occurrences) FROM {SCHEMA}.label_vectors
                WHERE run_id = :r GROUP BY 1
            """), {"r": self.run_id})).all()
        return {r[0]: int(r[1]) for r in rows} | {f"{r[0]}:occurrences": int(r[2]) for r in rows}

    async def rows(self, ids: list[int]) -> dict[int, dict[str, Any]]:
        if not ids:
            return {}
        async with session_scope() as s:
            res = (await s.execute(text(f"""
                SELECT id, kind, label, parent, description, iri, occurrences FROM {SCHEMA}.label_vectors
                WHERE id = ANY(:ids)
            """), {"ids": ids})).mappings().all()
        return {r["id"]: dict(r) for r in res}

    async def exact_existing_matches(self) -> list[tuple[int, int]]:
        """(class id, existing_class id) pairs with the same folded label."""
        async with session_scope() as s:
            rows = (await s.execute(text(f"""
                SELECT c.id, e.id FROM {SCHEMA}.label_vectors c
                JOIN {SCHEMA}.label_vectors e
                  ON e.run_id = c.run_id AND e.kind = 'existing_class' AND e.label_key = c.label_key
                WHERE c.run_id = :r AND c.kind = 'class' AND c.status = 'kept'
            """), {"r": self.run_id})).all()
        return [(int(a), int(b)) for a, b in rows]

    async def iter_rows(self, kind: str, status: str = "kept", batch: int = 100) -> AsyncIterator[list[dict[str, Any]]]:
        """This run's rows of `kind`/`status` in id order, `batch` at a time
        (keyset pagination: constant memory, safe while statuses change)."""
        last = 0
        while True:
            async with session_scope() as s:
                rows = (await s.execute(text(f"""
                    SELECT id, kind, label, parent, description, iri, occurrences, status, status_reason
                    FROM {SCHEMA}.label_vectors
                    WHERE run_id = :r AND kind = :k AND status = :st AND id > :last
                    ORDER BY id LIMIT :n
                """), {"r": self.run_id, "k": kind, "st": status, "last": last, "n": batch})).mappings().all()
            if not rows:
                return
            last = rows[-1]["id"]
            yield [dict(r) for r in rows]

    async def sample(self, kind: str, status: str, n: int, seed: float = 0.42,
                     reason: str | None = None) -> list[dict[str, Any]]:
        """A reproducible random sample of rows (setseed + random())."""
        async with session_scope() as s:
            await s.execute(text("SELECT setseed(:s)"), {"s": seed})
            rows = (await s.execute(text(f"""
                SELECT id, kind, label, parent, description, iri, occurrences, status, status_reason, merged_into
                FROM {SCHEMA}.label_vectors
                WHERE run_id = :r AND kind = :k AND status = :st
                  AND (CAST(:why AS text) IS NULL OR status_reason = :why)
                ORDER BY random() LIMIT :n
            """), {"r": self.run_id, "k": kind, "st": status, "n": n, "why": reason})).mappings().all()
        return [dict(r) for r in rows]

    async def count(self, kind: str, status: str) -> int:
        async with session_scope() as s:
            return int((await s.execute(text(f"""
                SELECT count(*) FROM {SCHEMA}.label_vectors WHERE run_id = :r AND kind = :k AND status = :st
            """), {"r": self.run_id, "k": kind, "st": status})).scalar() or 0)

    async def knn_for(self, aid: int, to_kind: str, k: int = 10) -> list[tuple[int, float]]:
        """Public form of the per-label lookup (used by the audit's sample)."""
        return await self._knn(aid, to_kind, k)
