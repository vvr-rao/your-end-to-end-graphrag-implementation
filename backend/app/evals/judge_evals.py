"""LLM-as-judge evaluations of the graph and the ontology (yes/no per item).

Two evaluations, both paid but cheap (one serverless call per ~20 items):

* `eval-typing` (after extract-entities): is each entity's class right, and
  each concept's concept class? The judge sees the name, the class, its
  definition and parents, and a passage where the entity is mentioned.
* `eval-ontology <prune-expand folder>` (after prune-expand): does each class
  definition fit its class (label, parents)? Classes with no definition are
  not sent; they are counted as `no_definition`.

Reports give Yes / judged and Yes / all items, split by group (entity vs
concept; concept vs other classes, new vs source classes). Every verdict and
its one-line reason goes to a local JSONL log (`.build/evals/`, gitignored)
for review. A failed call leaves its items `unanswered`; nothing is written to
the database or the ontology.
"""
from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import text

from backend.app.evals.common import REPORT_ROOT, database_name, entity_kinds

TYPING_TASK = "eval_typing_judge"
ONTOLOGY_TASK = "eval_definition_judge"
YES, NO = "yes", "no"
SNIPPET = 360  # characters of passage around the mention
SKOS_DEFINITION = "http://www.w3.org/2004/02/skos/core#definition"  # FIBO's definitions


# ---------------------------------------------------------------- shared runner


def parse_verdicts(got: Any, batch: list[dict[str, Any]]) -> dict[int, tuple[str, str]]:
    """{batch index: (verdict, reason)} for the valid answers."""
    out: dict[int, tuple[str, str]] = {}
    for d in got if isinstance(got, list) else []:
        if not isinstance(d, dict):
            continue
        try:
            i = int(d["id"])
        except (KeyError, ValueError, TypeError):
            continue
        verdict = str(d.get("verdict") or "").strip().lower()
        if 0 <= i < len(batch) and verdict in (YES, NO):
            out[i] = (verdict, str(d.get("reason") or "").strip()[:300])
    return out


async def judge_items(router: Any, task: str, items: list[dict[str, Any]],
                      build: Callable[[list[dict[str, Any]]], tuple[str, str]],
                      view: Callable[[dict[str, Any]], dict[str, Any]],
                      *, log=print) -> dict[str, Any]:
    """Ask the judge about every item, in batches. Sets `verdict` and `reason`
    on each item (`unanswered` when the call failed or skipped it)."""
    from backend.app.services.db_artifact_gen import _extract_json

    spec = router.task_spec(task)
    size = int(spec.get("items_per_call", 20))
    sem = asyncio.Semaphore(int(spec.get("concurrency", 8)))
    start = router.total_cost_usd
    failed = 0

    async def one(batch: list[dict[str, Any]]) -> None:
        nonlocal failed
        system, user = build([{"id": k, **view(it)} for k, it in enumerate(batch)])
        async with sem:
            try:
                res = await router.chat(task, system=system, user=user)
                parsed = _extract_json(res.text)
            except Exception as exc:
                failed += 1
                log(f"[{task}] call failed ({type(exc).__name__}); {len(batch)} left unanswered")
                parsed = None
        got = parse_verdicts(parsed.get("decisions") if isinstance(parsed, dict) else None,
                             batch)
        for k, it in enumerate(batch):
            it["verdict"], it["reason"] = got.get(k, ("unanswered", ""))

    await asyncio.gather(*(one(items[i:i + size]) for i in range(0, len(items), size)))
    # One retry, in smaller batches, for items the judge skipped or a failed
    # call left (a long batch sometimes comes back with a few ids missing).
    again = [it for it in items if it.get("verdict") == "unanswered"]
    if again:
        small = max(1, size // 4)
        await asyncio.gather(*(one(again[i:i + small]) for i in range(0, len(again), small)))
    return {"cost_usd": round(router.total_cost_usd - start, 4), "failed_calls": failed,
            "retried": len(again), "model": spec.get("model", ""), "items_per_call": size}


def tally(items: list[dict[str, Any]], groups: dict[str, Callable[[dict], bool]]
          ) -> dict[str, dict[str, Any]]:
    """Per group: yes / no / unanswered / no_definition counts and the two
    percentages (Yes / judged, Yes / all)."""
    out: dict[str, dict[str, Any]] = {}
    for name, keep in {"all": lambda _it: True, **groups}.items():
        sel = [it for it in items if keep(it)]
        n = {v: sum(1 for it in sel if it.get("verdict") == v)
             for v in (YES, NO, "unanswered", "no_definition")}
        judged = n[YES] + n[NO]
        out[name] = {"items": len(sel), **n, "judged": judged,
                     "yes_pct_of_judged": round(100 * n[YES] / judged, 1) if judged else 0.0,
                     "yes_pct_of_all": round(100 * n[YES] / len(sel), 1) if sel else 0.0}
    return out


def write_log(path: Path, items: list[dict[str, Any]], keys: tuple[str, ...]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for it in items:
            fh.write(json.dumps({k: it.get(k) for k in keys}, default=str) + "\n")
    return path


def _stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def snippet(passage: str, name: str, width: int = SNIPPET) -> str:
    """The part of a passage around the first mention of `name`."""
    passage = re.sub(r"\s+", " ", passage or "").strip()
    i = passage.lower().find((name or "").lower())
    if i < 0:
        return passage[:width]
    lo = max(0, i - width // 2)
    return ("..." if lo else "") + passage[lo:lo + width] + ("..." if lo + width < len(passage) else "")


# ---------------------------------------------------------------- 1. entity typing

_TYPED_SQL = text("""
    SELECT e.id, e.name, oc.label AS class_label,
           coalesce(nullif(oc.description, ''),
                    oc.extra_metadata->'annotations'->'http://www.w3.org/2004/02/skos/core#definition'->>0,
                    '') AS class_definition,
           (SELECT string_agg(DISTINCT p.label, ', ')
              FROM graphrag.graph_relationships r
              JOIN graphrag.ontology_classes p ON p.id = r.target_node_id
             WHERE r.source_node_type = 'ontology_class' AND r.source_node_id = oc.id
               AND r.predicate_label = 'rdfs:subClassOf') AS parents,
           (SELECT c.text
              FROM graphrag.graph_relationships m
              JOIN graphrag.chunks c ON c.id = m.source_node_id
             WHERE m.source_node_type = 'chunk' AND m.target_node_type = 'entity'
               AND m.target_node_id = e.id AND m.predicate_label = 'viao:assertsAbout'
               AND c.status = 'ACTIVE'
             ORDER BY c.kind = 'summary' DESC LIMIT 1) AS passage
      FROM graphrag.entities e
      JOIN graphrag.ontology_classes oc ON oc.id = e.class_id
     WHERE e.status = 'ACTIVE'
     ORDER BY md5(e.id::text)
     LIMIT :limit
""")


async def load_typed_entities(session: Any, limit: int | None = None) -> list[dict[str, Any]]:
    """Active entities with their class, in a fixed pseudo-random order (so a
    --limit run is a sample, not the alphabet's first letters)."""
    kinds = await entity_kinds(session)
    rows = (await session.execute(_TYPED_SQL, {"limit": limit})).mappings().all()
    out = []
    for r in rows:
        kind = kinds.get(r["id"], ("named", None))[0]
        out.append({"entity_id": str(r["id"]), "name": r["name"],
                    "kind": "concept" if kind == "concept" else "entity",
                    "place": kind == "place", "class": r["class_label"] or "",
                    "class_definition": (r["class_definition"] or "")[:400],
                    "parents": r["parents"] or "",
                    "passage": snippet(r["passage"] or "", r["name"])})
    return out


def _typing_prompt(prompt_set: str | None):
    if prompt_set == "deepseek":
        from backend.app.services.prompts_deepseek import deepseek_eval_typing_judge

        return deepseek_eval_typing_judge
    from backend.app.services.prompts import eval_typing_judge

    return eval_typing_judge


async def typing_eval(session: Any, router: Any, *, limit: int | None = None,
                      log=print) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    items = await load_typed_entities(session, limit)
    spec = router.task_spec(TYPING_TASK)
    run = await judge_items(
        router, TYPING_TASK, items, _typing_prompt(spec.get("prompt_set")),
        lambda it: {k: it[k] for k in ("name", "kind", "class", "class_definition",
                                       "parents", "passage")}, log=log)
    groups = {"entities": lambda it: it["kind"] == "entity",
              "concepts": lambda it: it["kind"] == "concept",
              "places": lambda it: it["place"]}
    report = {"evaluation": "typing", "database": database_name(),
              "sample": limit is not None, **run, "groups": tally(items, groups)}
    return report, items


TYPING_LOG_KEYS = ("entity_id", "name", "kind", "class", "parents", "verdict", "reason",
                   "class_definition", "passage")


def save_typing(report: dict[str, Any], items: list[dict[str, Any]]) -> tuple[Path, Path]:
    base = REPORT_ROOT / database_name() / f"typing-{_stamp()}"
    log_path = write_log(base.with_suffix(".jsonl"), items, TYPING_LOG_KEYS)
    report["log"] = str(log_path)
    base.with_suffix(".json").write_text(json.dumps(report, indent=2, default=str))
    return base.with_suffix(".json"), log_path


# ---------------------------------------------------------------- 2. class definitions


def _first(values: Any) -> str:
    for v in values or []:
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def load_class_definitions(folder: Path, concept_roots: list[str],
                           limit: int | None = None) -> list[dict[str, Any]]:
    """Every class of a prune-expand (or merge/build) output folder, with its
    label, definition and parents. `concept` = under a concept root (by label);
    `new` = minted by prune-expand (not from a source ontology file)."""
    classes = json.loads((Path(folder) / "merged.json").read_text())["classes_dict"]
    by_name: dict[str, dict] = {}
    for iri, c in classes.items():
        by_name.setdefault(c.get("name") or iri, c)

    def label(c: dict) -> str:
        return _first(c.get("labels")) or c.get("name") or ""

    roots = {r.strip().lower() for r in concept_roots if r and r.strip()}
    memo: dict[str, bool] = {}

    def under_root(c: dict, seen: frozenset = frozenset()) -> bool:
        key = c.get("iri") or c.get("name")
        if key in memo:
            return memo[key]
        if label(c).lower() in roots or (c.get("name") or "").lower() in roots:
            memo[key] = True
            return True
        hit = False
        for p in c.get("superclasses") or []:
            parent = classes.get(p.get("iri")) or by_name.get(p.get("name"))
            if parent is not None and (parent.get("iri") not in seen):
                if under_root(parent, seen | {key}):
                    hit = True
                    break
        memo[key] = hit
        return hit

    out = []
    for iri, c in sorted(classes.items()):
        role = ((c.get("semantic_role") or {}).get("role") or "")
        new = bool(c.get("auto_created_via")) or (not c.get("sources") and role.startswith("proposed"))
        definition = (_first(c.get("descriptions")) or _first(c.get("comments"))
                      or _first((c.get("annotations") or {}).get(SKOS_DEFINITION)))
        out.append({"iri": iri, "class": label(c), "definition": definition[:600],
                    "parents": ", ".join(p.get("name") or "" for p in c.get("superclasses") or []
                                         if p.get("name") and p.get("name") != "Thing"),
                    "concept": under_root(c) and label(c).lower() not in roots,
                    "new": new,
                    "verdict": None if definition else "no_definition", "reason": ""})
    if limit is not None:
        out = sorted(out, key=lambda it: hash_key(it["iri"]))[:limit]
    return out


def hash_key(s: str) -> str:
    import hashlib

    return hashlib.md5(s.encode("utf-8")).hexdigest()


def _definition_prompt(prompt_set: str | None):
    if prompt_set == "deepseek":
        from backend.app.services.prompts_deepseek import deepseek_eval_definition_judge

        return deepseek_eval_definition_judge
    from backend.app.services.prompts import eval_definition_judge

    return eval_definition_judge


async def ontology_eval(folder: Path, router: Any, concept_roots: list[str], *,
                        limit: int | None = None, log=print
                        ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    items = load_class_definitions(folder, concept_roots, limit)
    to_judge = [it for it in items if it["verdict"] is None]
    spec = router.task_spec(ONTOLOGY_TASK)
    run = await judge_items(
        router, ONTOLOGY_TASK, to_judge, _definition_prompt(spec.get("prompt_set")),
        lambda it: {k: it[k] for k in ("class", "definition", "parents")}, log=log)
    groups = {"concept_classes": lambda it: it["concept"],
              "other_classes": lambda it: not it["concept"],
              "new_classes": lambda it: it["new"],
              "source_classes": lambda it: not it["new"]}
    report = {"evaluation": "ontology", "folder": str(folder), "sample": limit is not None,
              **run, "groups": tally(items, groups)}
    return report, items


ONTOLOGY_LOG_KEYS = ("iri", "class", "parents", "concept", "new", "verdict", "reason",
                     "definition")


def save_ontology(report: dict[str, Any], items: list[dict[str, Any]]) -> tuple[Path, Path]:
    base = REPORT_ROOT / "ontology" / Path(report["folder"]).name / f"definitions-{_stamp()}"
    log_path = write_log(base.with_suffix(".jsonl"), items, ONTOLOGY_LOG_KEYS)
    report["log"] = str(log_path)
    base.parent.mkdir(parents=True, exist_ok=True)
    base.with_suffix(".json").write_text(json.dumps(report, indent=2, default=str))
    return base.with_suffix(".json"), log_path


# ---------------------------------------------------------------- report text


def format_judge_eval(report: dict[str, Any]) -> str:
    what = "entity and concept typing" if report["evaluation"] == "typing" else "class definitions"
    lines = [f"{what}: model {report.get('model', '')}, cost ${report.get('cost_usd', 0):.4f}, "
             f"{report.get('failed_calls', 0)} failed call(s)"
             + (" (sample)" if report.get("sample") else "")]
    for name, g in report["groups"].items():
        extra = f", no definition {g['no_definition']}" if g["no_definition"] else ""
        lines.append(
            f"  {name:16s} {g['items']:5d} items: yes {g['yes']}, no {g['no']}, "
            f"unanswered {g['unanswered']}{extra} | yes/judged {g['yes_pct_of_judged']}%, "
            f"yes/all {g['yes_pct_of_all']}%")
    if report.get("log"):
        lines.append(f"  log: {report['log']}")
    return "\n".join(lines)


def judge_scores(report: dict[str, Any]) -> dict[str, float]:
    """Flat LangSmith feedback scores for a judge report."""
    out: dict[str, float] = {}
    for name, g in report["groups"].items():
        for key in ("items", "yes", "no", "unanswered", "no_definition",
                    "yes_pct_of_judged", "yes_pct_of_all"):
            out[f"{report['evaluation']}.{name}.{key}"] = g[key]
    out[f"{report['evaluation']}.cost_usd"] = report.get("cost_usd", 0.0)
    return out
