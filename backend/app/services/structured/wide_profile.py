"""Profiling a WIDE table in column batches (resumable).

One ``table_profile`` call must describe every column, list every candidate
quantity and write the query catalog; on a wide table that reply exceeds the
output cap. Above ``column_batch`` columns the work is split:

1. **Column batches** (``column_batch`` columns per call, id = source column
   name): each call gets a compact schema of the WHOLE table, the measured
   structural evidence (single-column keys, column families sharing a stem,
   candidate indicator/value pairs, unit-like columns) and the detailed
   profile of its own columns, and must return one record per assigned
   column. Related fields on opposite sides of a batch boundary are linked
   through the schema and the evidence, which every batch sees.
2. **Synthesis**: table description, grain, cross-column relationships and
   the candidate quantities, from a short digest of the column results (no
   column description is repeated). Continued while the model reports more
   candidates.
3. **Query capabilities**, batched by candidate id: metrics / filters /
   queries per candidate (each tagged with its ``capability_ids``).

Every call is checkpointed (source hash + prompt version + batch config +
model) so a resumed run only does what is unfinished. Ids that could not be
completed carry an incomplete status (:mod:`continuation`); the table is
``complete`` only when every column and candidate is accounted for.
Domain-neutral: nothing here names a dataset, column or value.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from backend.app.services.structured.continuation import Budget, parse_reply, run_by_ids

WIDE_VERSION = "wide-1"
COLUMN_BATCH = 40
CANDIDATE_BATCH = 12
MAX_SYNTHESIS_CALLS = 6
_YEAR_SUFFIX = re.compile(r"^(?P<stem>.*?)[_\-\s]?(?P<suffix>(?:19|20)\d{2}(?:[_\-](?:19|20)?\d{2})?|\d{2}[_\-]\d{2})$")
_UNITISH = re.compile(r"[$%/€£¥]|\(|per\b", re.I)


def column_families(names: list[str]) -> dict[str, list[str]]:
    """Columns sharing a stem with a year / period suffix (``x_2021``,
    ``x_2022``...), plus the bare stem when it exists: one family."""
    fam: dict[str, list[str]] = {}
    for n in names:
        m = _YEAR_SUFFIX.match(n)
        if m and m.group("stem"):
            fam.setdefault(m.group("stem").rstrip("_- "), []).append(n)
    for stem in list(fam):
        if stem in names:
            fam[stem].insert(0, stem)
    return {s: cols for s, cols in fam.items() if len(cols) >= 2}


def structural_evidence(profile) -> dict[str, Any]:
    """Measured facts every batch sees (no LLM)."""
    rows = profile.row_count or 0
    cols = profile.columns
    keys = [c.name for c in cols if rows and c.distinct == rows and not c.distinct_capped and not c.nulls]
    text = [c for c in cols if c.type in {"text", "boolean"}]
    nums = [c.name for c in cols if c.type in {"integer", "numeric"}]
    pairs = [c.name for c in text if rows and 1 < c.distinct <= 500 and c.distinct < rows / 2] if nums else []
    unitish = [c.name for c in text if c.top_values and c.distinct <= 50
               and sum(1 for v, _n in c.top_values if _UNITISH.search(str(v))) >= max(1, len(c.top_values) // 2)]
    dates = [c.name for c in cols if c.type in {"date", "timestamp"}]
    return {"row_count": rows, "single_column_keys": keys,
            "column_families": column_families([c.name for c in cols]),
            "possible_indicator_columns": pairs, "numeric_columns": nums,
            "unit_like_columns": unitish, "date_columns": dates}


def compact_schema(profile) -> list[str]:
    return [f"{c.name} [{c.pg_type}] distinct={c.distinct}{'+' if c.distinct_capped else ''} nulls={c.nulls}"
            for c in profile.columns]


def _batch_markdown(profile, names: list[str], max_sample: int = 8) -> str:
    from backend.app.services.structured.profiling import TOP_VALUES

    want = set(names)
    out = []
    for c in profile.columns:
        if c.name not in want:
            continue
        bits = [f"- {c.name} [{c.pg_type}]"]
        if c.source_name != c.name:
            bits.append(f"header='{c.source_name}'")
        bits.append(f"non_null={c.non_null} nulls={c.nulls} distinct={c.distinct}{'+' if c.distinct_capped else ''}")
        if c.min is not None:
            bits.append(f"min={c.min} max={c.max}")
        if c.top_values:
            bits.append("top: " + ", ".join(f"{v!r} ({n})" for v, n in c.top_values[:TOP_VALUES]))
        out.append(" ".join(bits))
    keep = want | set(structural_evidence(profile)["single_column_keys"][:2])
    rows = [{k: v for k, v in r.items() if k in keep} for r in profile.sample_rows[:max_sample]]
    return "\n".join(out) + "\n\nSAMPLE ROWS (these columns + a key)\n" + "\n".join(
        json.dumps(r, ensure_ascii=False, default=str) for r in rows)


def _fingerprint(src, model: str, column_batch: int) -> str:
    from backend.app.services.structured.profiler import PROMPT_VERSION

    raw = f"{src.file_sha}|{PROMPT_VERSION}|{WIDE_VERSION}|{column_batch}|{CANDIDATE_BATCH}|{model}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


class Checkpoint:
    """Completed work of one table's wide profiling, on disk."""

    def __init__(self, path: Path | None) -> None:
        self.path = path
        self.data: dict[str, Any] = {"columns": {}, "synthesis": None, "candidates": [], "capabilities": {}}
        if path and path.exists():
            try:
                self.data.update(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                pass

    def save(self) -> None:
        if self.path is None:
            return
        from backend.app.services.structured.cache import atomic_write_json

        self.path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(self.path, self.data)


def checkpoint_path(cache, src, model: str, column_batch: int) -> Path | None:
    root = getattr(cache, "user_dir", None) or getattr(cache, "root", None)
    if root is None:
        return None
    return Path(root) / "partial" / f"{src.table_name}-{_fingerprint(src, model, column_batch)}.json"


def _valid_column(_id: str, r: dict) -> str | None:
    from backend.app.services.structured.metadata import COLUMN_ROLES

    if r.get("role") not in COLUMN_ROLES and not r.get("roles"):
        return "no role"
    if not str(r.get("description") or "").strip():
        return "no description"
    return None


def _valid_capability(_id: str, r: dict) -> str | None:
    if not any(isinstance(r.get(k), list) for k in ("metrics", "filters", "queries")) and not r.get("status"):
        return "neither queries nor a status"
    return None


def candidate_id(c: dict, i: int) -> str:
    """Stable id of a profiled candidate: its own id, else from its label."""
    if str(c.get("candidate_id") or "").startswith("cand_"):
        return str(c["candidate_id"])
    raw = str(c.get("id") or c.get("label") or f"candidate {i + 1}")
    slug = re.sub(r"[^a-z0-9]+", "_", raw.lower()).strip("_")
    slug = slug[5:] if slug.startswith("cand_") else slug
    return "cand_" + (slug[:48] or str(i + 1))


async def profile_wide(src, profile, router, *, cands: list[dict], cache=None,
                       column_batch: int = COLUMN_BATCH, budget: Budget | None = None,
                       log: Callable[[str], None] = print) -> tuple[dict, dict]:
    """(reply in the single-call shape, report {stages, coverage, complete})."""
    from backend.app.services.prompts import PROMPTS
    from backend.app.services.structured.profiler import _model_tag

    model = _model_tag(router, "table_profile")
    ck = Checkpoint(checkpoint_path(cache, src, model, column_batch))
    log(f"[structured] {src.table_name}: wide table ({len(profile.columns)} columns) -> column batches of "
        f"{column_batch}, synthesis, capabilities by candidate id; checkpoint {ck.path}"
        + (" (exists: resuming)" if ck.path and ck.path.exists() else ""))
    evidence = structural_evidence(profile)
    schema = compact_schema(profile)
    header = {"table": src.table_name, "file": src.rel_path, "rows": profile.row_count,
              "columns": len(profile.columns)}
    stages = []

    # ---- 1. column batches
    async def _cols(batch: list[str], _n: int):
        system, user = PROMPTS["table_profile_columns"](header, schema, evidence,
                                                         _batch_markdown(profile, batch), batch, cands)
        res = await router.chat("table_profile", system=system, user=user)
        obj, _salvaged = parse_reply(res.text)
        recs = [dict(r, id=r.get("name")) for r in obj.get("columns") or [] if isinstance(r, dict)]
        return recs, res

    def _save_cols(done: dict) -> None:
        ck.data["columns"] = done
        ck.save()

    names = [c.name for c in profile.columns]
    cols, col_fail, st = await run_by_ids("profile_columns", names, _cols, validate=_valid_column,
                                          batch_size=column_batch, budget=budget,
                                          done=ck.data.get("columns") or {}, on_progress=_save_cols, log=log)
    stages.append(st.to_dict())
    _save_cols(cols)

    # ---- 2. synthesis (description, grain, relationships, candidates)
    digest = [{"name": n, "type": next(c.pg_type for c in profile.columns if c.name == n),
               **({k: cols[n].get(k) for k in ("role", "roles", "unit", "family") if cols.get(n, {}).get(k)}),
               "summary": str(cols.get(n, {}).get("description") or "")[:80]}
              for n in names]
    syn = ck.data.get("synthesis")
    from backend.app.services.structured.continuation import StageStats

    sst = StageStats(stage="profile_synthesis")
    if syn and not syn.get("remaining"):
        log(f"[structured] profile_synthesis: from the checkpoint ({len(syn.get('candidates') or [])} candidates)")
    if not syn or syn.get("remaining"):
        syn = syn or {"candidates": []}
        for _i in range(MAX_SYNTHESIS_CALLS):
            listed = [candidate_id(c, i) for i, c in enumerate(syn.get("candidates") or [])]
            system, user = PROMPTS["table_profile_synthesis"](header, digest, evidence,
                                                               profile.sample_rows[:5], cands,
                                                               already_listed=listed)
            res = await router.chat("table_profile", system=system, user=user)
            sst.add_call(res, len(names))
            log(f"[structured] profile_synthesis: call {sst.calls} -> finish={res.finish_reason}, "
                f"out={res.completion_tokens} tok, cap={getattr(res, 'effective_max_tokens', None)}")
            obj, salvaged = parse_reply(res.text)
            for k in ("description", "grain"):
                if obj.get(k) and not syn.get(k):
                    syn[k] = obj[k]
            syn.setdefault("relationships", []).extend(r for r in obj.get("relationships") or [] if r)
            have = set(listed)
            for c in obj.get("candidates") or []:
                if isinstance(c, dict) and candidate_id(c, len(syn["candidates"])) not in have:
                    syn["candidates"].append(c)
            if isinstance(obj.get("countables"), dict):
                syn["countables"] = {**(syn.get("countables") or {}), **obj["countables"]}
            syn["remaining"] = bool(obj.get("more_candidates")) or (salvaged and bool(obj))
            ck.data["synthesis"] = syn
            ck.save()
            if not syn["remaining"]:
                break
    sst.ids_total = sst.ids_done = len(syn.get("candidates") or [])
    sst.stop_reason = "complete" if not syn.get("remaining") else "budget"
    stages.append(sst.to_dict())
    candidates = [{**c, "id": candidate_id(c, i)} for i, c in enumerate(syn.get("candidates") or [])]

    # ---- 3. query capabilities, batched by candidate id
    by_id = {c["id"]: c for c in candidates}

    async def _caps(batch: list[str], _n: int):
        system, user = PROMPTS["table_profile_capabilities"](header, digest, evidence,
                                                              [by_id[i] for i in batch],
                                                              profile.sample_rows[:5])
        res = await router.chat("table_profile", system=system, user=user)
        obj, _s = parse_reply(res.text)
        return [r for r in obj.get("capabilities") or [] if isinstance(r, dict)], res

    def _save_caps(done: dict) -> None:
        ck.data["capabilities"] = done
        ck.save()

    caps, cap_fail, cst = await run_by_ids("profile_capabilities", list(by_id), _caps,
                                           validate=_valid_capability, batch_size=CANDIDATE_BATCH,
                                           budget=budget, done=ck.data.get("capabilities") or {},
                                           on_progress=_save_caps, log=log)
    stages.append(cst.to_dict())
    _save_caps(caps)

    reply: dict[str, Any] = {
        "description": syn.get("description"), "grain": syn.get("grain"),
        "relationships": syn.get("relationships") or [],
        "columns": [{**cols[n], "name": n} for n in names if n in cols],
        "candidates": candidates, "countables": syn.get("countables") or {},
        "remaining_candidates": [by_id[i].get("label") or i for i in cap_fail],
        "metrics": [], "filters": [], "queries": []}
    for cid, rec in caps.items():
        for sec in ("metrics", "filters", "queries"):
            for it in rec.get(sec) or []:
                if isinstance(it, dict):
                    reply[sec].append({**it, "capability_ids": list(dict.fromkeys(
                        [*(it.get("capability_ids") or []), cid]))})
    coverage = {"columns": {n: ("done" if n in cols else col_fail.get(n, "incomplete_due_to_budget"))
                            for n in names},
                "candidates": {i: (caps[i].get("status") or "done") if i in caps
                               else cap_fail.get(i, "incomplete_due_to_budget") for i in by_id},
                "synthesis": "done" if not syn.get("remaining") else "incomplete_due_to_budget"}
    complete = (all(v == "done" for v in coverage["columns"].values())
                and coverage["synthesis"] == "done"
                and all(i in caps for i in by_id))
    return reply, {"stages": stages, "coverage": coverage, "complete": complete,
                   "checkpoint": str(ck.path) if ck.path else None}
