"""The MODULAR STEP: profile every table in a folder into a METADATA FILE.

``profile_folder`` is called by ``prune-expand --structured-data`` (after
Stage 4, with the final ontology so new classes can be added) and by
``extract-entities --structured-data`` (which normally finds everything in the
cache and makes no LLM call at all).

Per table (task ``table_profile``): description, grain, column descriptions
and roles, METRICS, FILTERS, QUERIES as JSON query specs (parameters listed
separately), and the ontology concept behind each column/metric/filter. The
model never writes SQL: :mod:`query_builder` compiles each spec. A spec that
does not compile gets one repair call; what still fails is dropped and
recorded.
Across tables (task ``table_lookups``): shared keys -> LOOKUP join queries,
stored in BOTH tables' metadata.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from backend.app.services.structured import cache as tcache
from backend.app.services.structured.discovery import TableSource, discover_tables, sanitize_identifier
from backend.app.services.structured.metadata import (
    COLUMN_ROLES,
    SECTIONS,
    column_types,
    finalize_item,
    rename_table,
)
from backend.app.services.structured.profiling import ProfileError, profile_markdown, profile_table
from backend.app.services.structured.query_builder import compact

PROMPT_VERSION = "table_profile@v2"
ANCHOR_LABELS = ("Metric", "Measure", "Dimension", "TimePeriod")
_MAX_CANDIDATES = 40
_STOP = {"the", "of", "and", "a", "an", "in", "for", "to", "by", "id", "no", "nr", "num",
         "code", "name", "type", "date", "value", "values", "rows", "row", "data", "table"}

AuditFn = Callable[[str, dict], None]


@dataclass
class ProfileRun:
    tables: list[dict] = field(default_factory=list)       # metadata (lookups merged in)
    sources: dict[str, TableSource] = field(default_factory=dict)
    profiled: list[str] = field(default_factory=list)
    cached: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)
    skipped_files: list[tuple[str, str]] = field(default_factory=list)
    lookups: list[dict] = field(default_factory=list)
    lookups_cached: bool = False
    llm_calls: int = 0

    def by_name(self) -> dict[str, dict]:
        return {m["table_name"]: m for m in self.tables}


# ------------------------------------------------------------------ helpers


def _model_tag(router, task: str) -> str:
    try:
        spec = router.task_spec(task)
        return f"{spec.get('provider')}:{spec.get('model')}"
    except Exception:
        return "unknown"


def _parse_json(text: str, salvage: bool) -> dict | None:
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    if salvage:
        from backend.app.helpers.json_salvage import salvage_truncated_json

        return salvage_truncated_json(text)
    return None


def _split_tokens(text: str) -> list[str]:
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(text or ""))
    return [t for t in re.split(r"[^A-Za-z0-9]+", text.lower()) if len(t) > 2 and t not in _STOP]


def _label_of(entry: dict) -> str:
    labels = entry.get("labels") or []
    return str(labels[0]) if labels else str(entry.get("name") or "")


class ClassIndex:
    """Lexical index over ontology class labels (no embeddings, no cost)."""

    def __init__(self, classes: dict[str, dict] | None) -> None:
        self.by_iri: dict[str, str] = {}
        self.by_label: dict[str, str] = {}
        self.tokens: dict[str, set[str]] = {}
        for iri, entry in (classes or {}).items():
            label = _label_of(entry)
            if not label:
                continue
            self.by_iri[iri] = label
            self.by_label.setdefault(label.lower(), iri)
            self.by_label.setdefault(re.sub(r"[^a-z0-9]", "", label.lower()), iri)
            self.tokens[iri] = set(_split_tokens(label))

    def resolve_label(self, label: str | None) -> str | None:
        if not label:
            return None
        key = label.strip().lower()
        return self.by_label.get(key) or self.by_label.get(re.sub(r"[^a-z0-9]", "", key))

    def candidates(self, words: list[str], k: int = _MAX_CANDIDATES) -> list[dict[str, str]]:
        want = set(words)
        scored = []
        for iri, toks in self.tokens.items():
            hit = len(toks & want)
            if hit:
                scored.append((hit / (1 + 0.1 * len(toks)), self.by_iri[iri], iri))
        scored.sort(key=lambda x: (-x[0], x[1]))
        out = [{"iri": iri, "label": label} for _s, label, iri in scored[:k]]
        for anchor in ANCHOR_LABELS:
            iri = self.resolve_label(anchor)
            if iri and all(c["iri"] != iri for c in out):
                out.append({"iri": iri, "label": self.by_iri[iri]})
        return out


def _profile_words(profile) -> list[str]:
    words = _split_tokens(profile.table_name) + _split_tokens(profile.rel_path)
    for c in profile.columns:
        words += _split_tokens(c.name) + _split_tokens(c.source_name)
        if c.type == "text" and c.distinct <= 200:
            for v, _n in c.top_values[:5]:
                words += _split_tokens(v)
    return words


def _norm_id(raw: Any, used: set[str], prefix: str = "") -> str:
    base = sanitize_identifier(str(raw or "item"))[:48]
    if prefix and not base.startswith(prefix):
        base = f"{prefix}{base}"
    cand, i = base, 2
    while cand in used:
        cand = f"{base}_{i}"
        i += 1
    used.add(cand)
    return cand


def _norm_params(raw) -> list[dict]:
    out = []
    for p in raw or []:
        if not isinstance(p, dict) or not p.get("name"):
            continue
        out.append({"name": str(p["name"]).lstrip(":"), "type": str(p.get("type") or "text"),
                    "column": p.get("column") or None, "description": str(p.get("description") or ""),
                    "example": None if p.get("example") in (None, "") else str(p.get("example"))})
    return out


def _concept(raw) -> dict | None:
    if not isinstance(raw, dict) or not (raw.get("label") or raw.get("iri")):
        return None
    return {"iri": raw.get("iri") or None, "label": raw.get("label") or None,
            "parent_label": raw.get("parent_label") or None}


def _norm_item(raw: dict, used: set[str], *, with_concept: bool) -> dict | None:
    if not isinstance(raw, dict) or not isinstance(raw.get("query"), dict):
        return None
    item = {
        "id": _norm_id(raw.get("id") or raw.get("label"), used),
        "label": str(raw.get("label") or raw.get("id") or "").strip() or None,
        "description": str(raw.get("description") or "").strip(),
        "columns": [str(c) for c in raw.get("columns") or [] if c],
        "class_iri": None, "class_label": None,
        "query": raw["query"],
        "params": _norm_params(raw.get("params")),
        "sql": "",
    }
    if with_concept:
        item["_concept"] = _concept(raw.get("concept"))
    return item


def build_metadata(src: TableSource, profile, reply: dict, *, key: str, model: str) -> dict:
    """LLM reply + profile -> metadata dict (SQL not yet validated)."""
    by_name = {c.get("name"): c for c in reply.get("columns") or [] if isinstance(c, dict)}
    columns = []
    for c in profile.columns:
        r = by_name.get(c.name) or {}
        role = r.get("role") if r.get("role") in COLUMN_ROLES else None
        columns.append({
            "name": c.name, "source_name": c.source_name, "type": c.type,
            "role": role, "description": str(r.get("description") or "").strip(),
            "unit": r.get("unit") or None, "entity_kind": r.get("entity_kind") or None,
            "examples": [v for v, _n in c.top_values[:5]] if c.type in {"text", "boolean"} else [],
            "class_iri": None, "class_label": None, "_concept": _concept(r.get("concept")),
        })
    used: set[str] = set()
    meta = {
        "format": tcache.FORMAT, "status": "complete", "cache_key": key,
        "profile_version": tcache.PROFILE_VERSION, "prompt_version": PROMPT_VERSION,
        "table_name": src.table_name,
        "source": {"rel_path": src.rel_path, "fmt": src.fmt, "file_sha": src.file_sha,
                   "dataset": src.dataset, "snapshot": src.snapshot},
        "row_count": profile.row_count,
        "description": str(reply.get("description") or "").strip(),
        "grain": str(reply.get("grain") or "").strip() or None,
        "columns": columns,
        "metrics": [], "filters": [], "queries": [], "lookups": [],
        "ontology": {"MATCHES FOUND": [], "MATCH NOT FOUND": []},
        "dropped": [], "model": model, "edited": False,
        "profile": profile.to_dict(),
    }
    for sec in ("metrics", "filters", "queries"):
        for raw in reply.get(sec) or []:
            it = _norm_item(raw, used, with_concept=sec != "queries")
            if it:
                meta[sec].append(it)
    return meta


def _columns_md(meta: dict) -> str:
    return "\n".join(f"- {c['name']} [{c['type']}]" for c in meta["columns"])


async def _validate_and_repair(meta: dict, router, audit: AuditFn | None, salvage: bool) -> int:
    """Compile every item's spec; one repair call for failures; drop what still fails."""
    types = {meta["table_name"]: column_types(meta)}
    failures = []
    for sec in ("metrics", "filters", "queries"):
        for it in meta[sec]:
            err = finalize_item(meta, sec, it, types)
            if err:
                failures.append((sec, it, err))
    calls = 0
    if failures:
        from backend.app.services.prompts import PROMPTS

        system, user = PROMPTS["table_sql_repair"](
            meta["table_name"], _columns_md(meta),
            [{"id": it["id"], "error": err, "params": it["params"], "query": it["query"]}
             for _s, it, err in failures])
        try:
            res = await router.chat("table_profile", system=system, user=user)
            calls += 1
            fixed = {str(x.get("id")): x for x in (_parse_json(res.text, salvage) or {}).get("items") or []
                     if isinstance(x, dict)}
        except Exception as exc:  # repair is best-effort
            fixed = {}
            if audit:
                audit("table_sql_repair_error", {"table": meta["table_name"], "error": str(exc)})
        for sec, it, err in failures:
            fx = fixed.get(it["id"])
            new_err = err
            if fx and isinstance(fx.get("query"), dict):
                trial = {**it, "query": fx["query"],
                         "params": _norm_params(fx.get("params")) if fx.get("params") is not None
                         else it["params"]}
                new_err = finalize_item(meta, sec, trial, types)
                if not new_err:
                    it.update(trial)
                    it["repaired"] = True
                    continue
            meta[sec].remove(it)
            meta["dropped"].append({"id": it["id"], "section": sec, "query": compact(it["query"]),
                                    "error": new_err or err})
    return calls


def _apply_concepts(meta: dict, index: ClassIndex) -> None:
    """Turn the model's concepts into class refs + a Stage-2-shaped proposal."""
    found, not_found = [], []

    def one(owner: dict, desc: str) -> None:
        c = owner.pop("_concept", None)
        if not c:
            return
        iri = c.get("iri") if c.get("iri") in index.by_iri else None
        label = c.get("label")
        if not iri and label:
            iri = index.resolve_label(label)
        if iri:
            owner["class_iri"], owner["class_label"] = iri, index.by_iri[iri]
            found.append({"IRI": iri, "TEXT_SNIPPET": f"{meta['table_name']}: {desc}"[:300]})
        elif label and not _looks_bad_label(label) and not new_class_blocked(owner, label):
            owner["class_iri"], owner["class_label"] = None, label
            not_found.append({"LABEL": label, "DESCRIPTION": desc[:500],
                              "PARENT_LABEL": c.get("parent_label") or "Metric"})

    for col in meta["columns"]:
        one(col, col.get("description") or col["name"])
    for sec in ("metrics", "filters"):
        for it in meta[sec]:
            one(it, it.get("description") or it.get("label") or it["id"])
    for it in meta["queries"]:
        it.pop("_concept", None)
    meta["ontology"] = {"MATCHES FOUND": found, "MATCH NOT FOUND": not_found}


_AGGREGATION_LABEL = re.compile(
    r"^(Total|Average|Avg|Mean|Median|Sum|Count|NumberOf|Num|Max|Maximum|Min|Minimum)(?=[A-Z])"
    r"|(Count|Total|Sum|Average)$")


def new_class_blocked(owner: dict, label: str) -> bool:
    """No NEW ontology class for an identifier/code column (a registration
    number or an LEI is a key, not a concept -- it may still MATCH an existing
    class such as LegalEntity), nor one named after an aggregation
    (TotalPrice, FactCount): the concept is the quantity, not the SUM of it."""
    if owner.get("role") == "identifier":
        return True
    return bool(_AGGREGATION_LABEL.search(re.sub(r"[^A-Za-z0-9]", "", label)))


def _looks_bad_label(label: str) -> bool:
    from backend.app.helpers.ontology_pruning import looks_like_iri

    s = label.strip()
    return (not s or looks_like_iri(s) or len(s) > 60 or len(s.split()) > 6
            or not re.search(r"[A-Za-z]", s))


# ------------------------------------------------------------------ lookups


def _lookup_summary(meta: dict) -> dict:
    return {"table": meta["table_name"], "description": meta.get("description"),
            "grain": meta.get("grain"),
            "columns": [{"name": c["name"], "type": c["type"], "role": c.get("role"),
                         "entity_kind": c.get("entity_kind"), "description": c.get("description"),
                         "examples": (c.get("examples") or [])[:3]}
                        for c in meta["columns"]
                        if c.get("role") in {"identifier", "dimension", "time", None}
                        or c.get("entity_kind")]}


def _flip_lookup(lk: dict) -> dict:
    out = {**lk, "other_table": lk["table"],
           "join": [{"column": j["other_column"], "other_column": j["column"]} for j in lk["join"]]}
    if isinstance(lk.get("query"), dict):
        # Same statement from the partner's side: pin the original base table
        # and join so the compiled SQL is identical in both files.
        out["query"] = {**lk["query"], "from": lk["table"],
                        "join": lk["query"].get("join") or {"table": lk["other_table"], "on": lk["join"]}}
    return out


def _attach_lookups(metas: dict[str, dict], lookups: list[dict]) -> None:
    for meta in metas.values():
        if meta.get("edited"):
            continue  # hand-edited lookups win
        meta["lookups"] = []
    for lk in lookups:
        a, b = lk["table"], lk["other_table"]
        for owner, item in ((a, lk), (b, _flip_lookup(lk))):
            meta = metas.get(owner)
            if meta is None or meta.get("edited"):
                continue
            it = {k: v for k, v in item.items() if k != "table"}
            meta["lookups"].append(it)


async def _profile_lookups(metas: dict[str, dict], router, audit: AuditFn | None) -> tuple[list[dict], int, list]:
    from backend.app.services.prompts import PROMPTS

    system, user = PROMPTS["table_lookups"]([_lookup_summary(m) for m in metas.values()])
    res = await router.chat("table_lookups", system=system, user=user)
    reply = _parse_json(res.text, salvage=True) or {}
    cols = {n: {c["name"] for c in m["columns"]} for n, m in metas.items()}
    types = {n: column_types(m) for n, m in metas.items()}
    used: set[str] = set()
    for m in metas.values():
        used |= {it["id"] for s in ("metrics", "filters", "queries") for it in m[s]}
    kept, dropped = [], []
    for raw in reply.get("lookups") or []:
        if not isinstance(raw, dict):
            continue
        a, b = raw.get("table"), raw.get("other_table")
        if a not in metas or b not in metas or a == b:
            dropped.append({"lookup": raw, "error": "unknown table pair"})
            continue
        joins = [{"column": j.get("column"), "other_column": j.get("other_column")}
                 for j in raw.get("join") or [] if isinstance(j, dict)]
        if not joins or any(j["column"] not in cols[a] or j["other_column"] not in cols[b] for j in joins):
            dropped.append({"lookup": raw, "error": "join columns do not exist"})
            continue
        it = {"id": _norm_id(raw.get("id") or f"{a}_{b}", used, prefix="lk_"),
              "label": raw.get("label") or f"{a} ↔ {b}", "description": str(raw.get("description") or ""),
              "columns": [j["column"] for j in joins], "class_iri": None, "class_label": None,
              "query": raw.get("query") if isinstance(raw.get("query"), dict) else None,
              "params": _norm_params(raw.get("params")), "sql": "",
              "table": a, "other_table": b, "join": joins}
        if it["query"] is not None:
            it["query"] = {**it["query"], "from": a,
                           "join": it["query"].get("join") or {"table": b, "on": joins}}
        err = finalize_item(metas[a], "lookups", it, types)
        if err:
            dropped.append({"lookup": raw, "error": err})
            continue
        kept.append(it)
    if audit:
        audit("table_lookups", {"kept": len(kept), "dropped": dropped})
    return kept, 1, dropped


# ------------------------------------------------------------------ main


async def profile_folder(
    folder: Path,
    router,
    *,
    cache: tcache.MetadataCache | None = None,
    ontology: dict | None = None,
    class_index: ClassIndex | None = None,
    concurrency: int = 4,
    limit: int | None = None,
    refresh: bool = False,
    allow_llm: bool = True,
    audit: AuditFn | None = None,
    log: Callable[[str], None] = print,
) -> ProfileRun:
    """Profile every table under ``folder`` (cache first).

    ``ontology`` (a loaded-ontology dict) supplies the candidate classes for
    concept matching; when absent, ``class_index`` may be passed instead.
    ``allow_llm=False`` turns a cache miss into a failure (used by tests and
    by callers that must not spend).
    """
    cache = cache or tcache.MetadataCache()
    disc = discover_tables(Path(folder))
    sources = disc.tables[:limit] if limit else disc.tables
    run = ProfileRun(sources={s.table_name: s for s in sources}, skipped_files=disc.skipped)
    index = class_index or ClassIndex((ontology or {}).get("classes_dict"))
    salvage = False
    try:
        salvage = bool(router.task_spec("table_profile").get("salvage_truncated")) if router else False
    except Exception:
        salvage = False
    use_qwen = False
    try:
        use_qwen = router is not None and router.task_spec("table_profile").get("prompt_set") == "qwen"
    except Exception:
        pass

    metas: dict[str, dict] = {}
    todo: list[TableSource] = []
    for src in sources:
        key = tcache.table_key(src.file_sha)
        hit = cache.get(key, src.table_name)
        if hit and refresh and not hit.get("edited"):
            hit = None
        if hit:
            if hit["table_name"] != src.table_name:
                log(f"[structured] {hit['table_name']} -> {src.table_name}: cached metadata renamed")
                hit = rename_table(hit, hit["table_name"], src.table_name)
                cache.put(hit)
            metas[src.table_name] = hit
            run.cached.append(src.table_name)
        else:
            todo.append(src)

    sem = asyncio.Semaphore(max(1, concurrency))

    async def _one(src: TableSource) -> None:
        async with sem:
            key = tcache.table_key(src.file_sha)
            try:
                profile = await asyncio.to_thread(profile_table, src)
            except (ProfileError, OSError, UnicodeError, ValueError) as exc:
                run.failed.append((src.table_name, f"profiling failed: {exc}"))
                return
            if not allow_llm:
                run.failed.append((src.table_name, "not in the metadata cache (LLM disabled)"))
                return
            from backend.app.services.prompts import PROMPTS

            cands = index.candidates(_profile_words(profile))
            max_sample = 10 if len(profile.columns) > 30 else 20
            builder = PROMPTS["table_profile"]
            if use_qwen:
                from backend.app.services.prompts_qwen import qwen_table_profile as builder
            system, user = builder(profile_markdown(profile, max_sample=max_sample), cands)
            try:
                res = await router.chat("table_profile", system=system, user=user)
            except Exception as exc:
                run.failed.append((src.table_name, f"LLM call failed: {exc}"))
                return
            run.llm_calls += 1
            reply = _parse_json(res.text, salvage)
            if not reply or not isinstance(reply.get("columns"), list):
                run.failed.append((src.table_name, "LLM reply was not the expected JSON"))
                if audit:
                    audit("table_profile_unparsed", {"table": src.table_name, "reply_head": res.text[:500]})
                return
            meta = build_metadata(src, profile, reply, key=key, model=_model_tag(router, "table_profile"))
            run.llm_calls += await _validate_and_repair(meta, router, audit, salvage)
            _apply_concepts(meta, index)
            cache.put(meta)
            metas[src.table_name] = meta
            run.profiled.append(src.table_name)
            if audit:
                audit("table_profile", {
                    "table": src.table_name, "rows": meta["row_count"],
                    "items": {s: len(meta[s]) for s in ("metrics", "filters", "queries")},
                    "dropped": meta["dropped"], "ontology": meta["ontology"]})
            log(f"[structured] profiled {src.table_name}: "
                f"{len(meta['metrics'])} metrics, {len(meta['filters'])} filters, "
                f"{len(meta['queries'])} queries, {len(meta['dropped'])} dropped")

    if todo:
        log(f"[structured] {len(run.cached)} table(s) from cache, profiling {len(todo)} with the LLM")
        await asyncio.gather(*(_one(s) for s in todo))
    else:
        log(f"[structured] all {len(run.cached)} table(s) from cache (no LLM calls)")

    # ---- cross-table lookups (cached on the set of table keys)
    if len(metas) > 1:
        lk_key = tcache.lookups_key([m["cache_key"] for m in metas.values()])
        hit = None if refresh else cache.get_lookups(lk_key)
        if hit is not None:
            lookups = hit["lookups"]
            run.lookups_cached = True
        elif allow_llm:
            try:
                lookups, calls, dropped = await _profile_lookups(metas, router, audit)
                run.llm_calls += calls
                cache.put_lookups({"format": tcache.FORMAT, "cache_key": lk_key,
                                   "tables": sorted(metas), "lookups": lookups, "dropped": dropped,
                                   "model": _model_tag(router, "table_lookups")})
                log(f"[structured] lookups: {len(lookups)} kept, {len(dropped)} dropped")
            except Exception as exc:
                lookups = []
                log(f"[structured] WARN lookups failed: {exc}")
        else:
            lookups = []
        run.lookups = lookups
        _attach_lookups(metas, lookups)
        if cache.run_dir is not None:
            for meta in metas.values():  # readable run-folder copy with lookups merged in
                tcache.atomic_write_json(cache.run_dir / f"{meta['table_name']}.metadata.json", meta)

    run.tables = [metas[s.table_name] for s in sources if s.table_name in metas]
    return run


def stage2_proposals(run: ProfileRun) -> dict[str, list]:
    """All tables' ontology proposals as one Stage-2-shaped dict (deduped)."""
    found, seen_f = [], set()
    nf, seen_nf = [], set()
    for meta in run.tables:
        for m in (meta.get("ontology") or {}).get("MATCHES FOUND") or []:
            if m["IRI"] not in seen_f:
                seen_f.add(m["IRI"])
                found.append(m)
        for m in (meta.get("ontology") or {}).get("MATCH NOT FOUND") or []:
            k = re.sub(r"[^a-z0-9]", "", m["LABEL"].lower())
            if k and k not in seen_nf:
                seen_nf.add(k)
                nf.append(m)
    return {"MATCHES FOUND": found, "MATCH NOT FOUND": nf}


def add_classes_to_ontology(run: ProfileRun, ontology: dict, *, base_iri: str,
                            default_parent_iri: str | None, cache: tcache.MetadataCache | None = None,
                            ) -> list[str]:
    """Mint the proposed classes (existing helper: fuzzy dedup + parent
    resolution), then write the resulting IRIs back into each metadata file."""
    from backend.app.helpers.ontology_pruning import add_new_classes_from_match_not_found

    proposals = stage2_proposals(run)
    if not proposals["MATCH NOT FOUND"]:
        created: list[str] = []
    else:
        _ont, created = add_new_classes_from_match_not_found(
            loaded_ontology=ontology, match_results=proposals, new_class_base_iri=base_iri,
            default_parent_iri=default_parent_iri, inplace=True)
    index = ClassIndex(ontology.get("classes_dict"))
    for meta in run.tables:
        changed = False
        owners = list(meta["columns"]) + [it for s in ("metrics", "filters") for it in meta[s]]
        for owner in owners:
            if owner.get("class_label") and not owner.get("class_iri"):
                iri = index.resolve_label(owner["class_label"])
                if iri:
                    owner["class_iri"] = iri
                    changed = True
        if changed and cache is not None:
            cache.put(meta)
    return list(created)


__all__ = ["SECTIONS", "ClassIndex", "ProfileRun", "add_classes_to_ontology", "build_metadata",
           "profile_folder", "stage2_proposals"]
