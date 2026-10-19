"""Measure how clean a class build is, instead of assuming it.

Three estimates for one staging-store run, each from a reproducible sample:

  leftover duplicates  sample surviving class labels, look up their nearest
                       neighbours among surviving classes, group the pairs
                       into cosine bands, judge up to `per_band` pairs per
                       band, and extrapolate:  est. duplicate pairs =
                       sum over bands of (judged-same rate x estimated pairs
                       in that band), with a Wilson 95% interval on each rate.
  over-merge rate      merged labels re-judged against the class they were
                       merged into (a 'same' verdict is a correct merge).
  junk precision       dropped labels (rules and judge) re-judged by the
                       validity judge: the share that really is junk or a
                       named thing.

The same judge that made the decisions also audits them, which biases the
over-merge number for judged merges towards 'correct'. It is unbiased for
auto-merges and folds, and `export_csv` writes every sample for a human.
"""
from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

BANDS = ((0.95, 1.01), (0.90, 0.95), (0.85, 0.90), (0.80, 0.85))


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (c - m) / d), min(1.0, (c + m) / d))


def band_name(cos: float) -> str | None:
    for lo, hi in BANDS:
        if lo <= cos < hi:
            return f"{lo:.2f}-{min(hi, 1.0):.2f}"
    return None


@dataclass
class AuditReport:
    labels_kept: int = 0
    labels_sampled: int = 0
    bands: dict[str, dict[str, Any]] = field(default_factory=dict)
    est_leftover_duplicate_pairs: float = 0.0
    est_leftover_duplicate_rate: float = 0.0       # duplicate pairs per kept label
    over_merge: dict[str, Any] = field(default_factory=dict)
    junk_precision: dict[str, Any] = field(default_factory=dict)
    samples: dict[str, list[dict[str, Any]]] = field(default_factory=dict)


async def audit(store: Any, pair_judge: Any, validity_router: Any | None = None, *,
                label_sample: int = 400, per_band: int = 40, merged_sample: int = 60,
                junk_sample: int = 60, k: int = 10, seed: int = 42) -> AuditReport:
    """Run the three estimates. `pair_judge(batch)` -> SAME decisions (the
    pairwise_dedup judge); `validity_router` enables the junk check."""
    rep = AuditReport()
    rng = random.Random(seed)
    rep.labels_kept = await store.count("class", "kept")

    # ---- leftover duplicates --------------------------------------------
    sampled = await store.sample("class", "kept", label_sample, seed=seed / 100)
    rep.labels_sampled = len(sampled)
    by_band: dict[str, list[tuple[int, int, float]]] = {}
    seen: set[tuple[int, int]] = set()
    for r in sampled:
        for to_kind in ("class", "existing_class"):
            for b_id, cos in await store.knn_for(r["id"], to_kind, k):
                key = (min(r["id"], b_id), max(r["id"], b_id))
                name = band_name(cos)
                if name and key not in seen:
                    seen.add(key)
                    by_band.setdefault(name, []).append((r["id"], b_id, cos))
    rows = await store.rows(list({i for v in by_band.values() for a, b, _ in v for i in (a, b)}))
    total = 0.0
    for name, pairs in sorted(by_band.items(), reverse=True):
        pick = rng.sample(pairs, min(per_band, len(pairs)))
        batch = [{"id": n, "a_label": rows[a]["label"], "a_desc": rows[a].get("description"),
                  "a_parent": rows[a].get("parent"), "b_label": rows[b]["label"],
                  "b_desc": rows[b].get("description"), "b_parent": rows[b].get("parent"),
                  "b_existing": rows[b]["kind"] == "existing_class"} for n, (a, b, _c) in enumerate(pick)]
        same_ids = {str(d.get("id")) for d in (await pair_judge(batch) if batch else [])}
        same = sum(1 for n in range(len(pick)) if str(n) in same_ids)
        rate = same / len(pick) if pick else 0.0
        # Pairs seen from a sampled label scale to the whole kept set; a pair
        # can be reached from either end, hence the / 2 for class-class.
        est_pairs = len(pairs) * rep.labels_kept / max(1, rep.labels_sampled) / 2
        lo, hi = wilson(same, len(pick))
        rep.bands[name] = {"pairs_in_sample": len(pairs), "judged": len(pick), "same": same,
                           "same_rate": round(rate, 3), "rate_95ci": [round(lo, 3), round(hi, 3)],
                           "est_pairs": round(est_pairs), "est_duplicates": round(rate * est_pairs)}
        total += rate * est_pairs
        rep.samples[f"leftover {name}"] = [
            {"a": p["a_label"], "b": p["b_label"], "judged_same": str(n) in same_ids} for n, p in enumerate(batch)]
    rep.est_leftover_duplicate_pairs = round(total)
    rep.est_leftover_duplicate_rate = round(total / max(1, rep.labels_kept), 4)

    # ---- over-merge ------------------------------------------------------
    merged = await store.sample("class", "merged", merged_sample, seed=seed / 100)
    canon = await store.rows([m["merged_into"] for m in merged if m.get("merged_into")])
    batch = [{"id": n, "a_label": m["label"], "a_desc": m.get("description"), "a_parent": m.get("parent"),
              "b_label": canon[m["merged_into"]]["label"], "b_desc": canon[m["merged_into"]].get("description"),
              "b_parent": canon[m["merged_into"]].get("parent"),
              "b_existing": canon[m["merged_into"]]["kind"] == "existing_class"}
             for n, m in enumerate(merged) if m.get("merged_into") in canon]
    same_ids = {str(d.get("id")) for d in (await pair_judge(batch) if batch else [])}
    wrong = sum(1 for n in range(len(batch)) if str(n) not in same_ids)
    rep.over_merge = {"sampled": len(batch), "judged_wrong": wrong,
                      "over_merge_rate": round(wrong / len(batch), 3) if batch else None,
                      "rate_95ci": [round(x, 3) for x in wilson(wrong, len(batch))]}
    rep.samples["merged"] = [{"a": p["a_label"], "b": p["b_label"], "judged_same": str(n) in same_ids}
                             for n, p in enumerate(batch)]

    # ---- junk precision ----------------------------------------------------
    if validity_router is not None:
        from backend.app.helpers.ontology_pruning import extract_json_from_output
        from backend.app.services.class_junk_filter import VALIDITY_TASK
        from backend.app.services.prompts_qwen import qwen_class_validity_judge

        junk = await store.sample("class", "junk", junk_sample, seed=seed / 100)
        items = [{"id": n, "label": r["label"], "parent": r.get("parent"), "desc": r.get("description")}
                 for n, r in enumerate(junk)]
        verdicts: dict[str, str] = {}
        if items:
            system, user = qwen_class_validity_judge(items)
            out = await validity_router.chat(VALIDITY_TASK, system=system, user=user)
            data = extract_json_from_output(out.text) or {}
            verdicts = {str(v.get("id")): str(v.get("verdict", "")).lower()
                        for v in (data.get("verdicts") or []) if isinstance(v, dict)}
        right = sum(1 for n in range(len(items)) if verdicts.get(str(n)) in ("junk", "named_thing"))
        rep.junk_precision = {"sampled": len(items), "confirmed": right,
                              "precision": round(right / len(items), 3) if items else None,
                              "rate_95ci": [round(x, 3) for x in wilson(right, len(items))]}
        rep.samples["junk"] = [{"label": r["label"], "reason": r.get("status_reason"),
                                "judge": verdicts.get(str(n))} for n, r in enumerate(junk)]
    return rep


def export_csv(rep: AuditReport, path: Path) -> Path:
    """Every sampled item, one row each, with a blank `human` column."""
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sample", "a_or_label", "b_or_reason", "judge", "human"])
        for name, rows in rep.samples.items():
            for r in rows:
                w.writerow([name, r.get("a", r.get("label")), r.get("b", r.get("reason")),
                            r.get("judged_same", r.get("judge")), ""])
    return path
