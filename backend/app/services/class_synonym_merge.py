"""Merge duplicate NEW classes that survive Stage-3 dedup (opt-in).

Stage 3 dedups with an LLM over semantically ordered batches, but synonyms
still survive (2026-10-14 preset run: Hypertension / HighBloodPressure,
Fatigue / Tiredness, BMI / BodyMassIndex) and proposals the dedup model
drops are restored unmerged. This pass:

  1. embeds each proposed class ("label words: description"),
  2. takes pairs at or above `threshold` cosine as CANDIDATES only --
     similarity also pairs opposites and siblings (BeforeBreakfast /
     AfterBreakfast 0.930, CleftLip / CleftPalate 0.890),
  3. asks a judge model which candidates are the SAME concept,
  4. merges confirmed groups into one class and rewrites every reference
     (PARENT_LABEL, relation DOMAIN/RANGE, instance TYPE_LABEL).

Enabled per preset by the `class_synonym_judge` task (`enabled: true`).
The judge prompt is model-specific (`prompt_set`); with no prompt for the
configured set the pass is skipped, never run with a mismatched prompt.
"""
from __future__ import annotations

import json
import re
from typing import Any

TASK = "class_synonym_judge"
CLASSES = "MATCH NOT FOUND"


def _words(label: str) -> str:
    return re.sub(r"([a-z])([A-Z])", r"\1 \2", str(label)).replace("_", " ").lower().strip()


def _key(label: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(label).lower())


_PAIR_BLOCK_ROWS = 1024


def candidate_pairs(vectors, threshold: float, max_pairs: int) -> list[tuple[int, int, float]]:
    """Index pairs (i < j) with cosine >= threshold, most similar first (unit
    vectors); ties keep row-major (i, j) order.

    Matrix products, not a Python double loop (that took many minutes at
    ~2,800 classes, 2026-10-17 pharma benchmark) -- but in ROW BLOCKS, not
    one n x n product: the full matrix plus its triu index arrays is ~1.5 GB
    at 11k classes, which OOM-killed the 27B web-search prune-expand on a
    2.7 GB machine (2026-10-17). A block is _PAIR_BLOCK_ROWS x n floats, and
    the running candidate list is trimmed to the best `max_pairs` as it goes.
    """
    import numpy as np

    m = np.asarray(vectors, dtype=np.float32)
    n = len(m)
    if n < 2 or max_pairs <= 0:
        return []
    cols = np.arange(n)
    acc_i = np.empty(0, dtype=np.int64)
    acc_j = np.empty(0, dtype=np.int64)
    acc_v = np.empty(0, dtype=np.float32)

    def best(i, j, v):
        order = np.lexsort((j, i, -v))[:max_pairs]      # -v first, then i, then j
        return i[order], j[order], v[order]

    for s in range(0, n, _PAIR_BLOCK_ROWS):
        block = m[s:s + _PAIR_BLOCK_ROWS] @ m.T
        rows = np.arange(s, s + len(block))[:, None]
        r, c = np.nonzero((block >= threshold) & (cols[None, :] > rows))
        if len(r):
            acc_i = np.concatenate([acc_i, r + s])
            acc_j = np.concatenate([acc_j, c])
            acc_v = np.concatenate([acc_v, block[r, c]])
            if len(acc_v) > 4 * max_pairs:
                acc_i, acc_j, acc_v = best(acc_i, acc_j, acc_v)
        del block
    acc_i, acc_j, acc_v = best(acc_i, acc_j, acc_v)
    return [(int(i), int(j), float(v)) for i, j, v in zip(acc_i, acc_j, acc_v, strict=True)]


def merge_groups(n: int, confirmed: list[tuple[int, int, int]]) -> dict[int, int]:
    """Union-find over confirmed (i, j, keep_index) pairs -> {member: canonical}.

    The canonical of a group is the index most often chosen as `keep`
    (ties: lowest index), so a chain A=B, B=C collapses to one class.
    """
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    votes: dict[int, int] = {}
    for i, j, keep in confirmed:
        parent[find(i)] = find(j)
        votes[keep] = votes.get(keep, 0) + 1
    groups: dict[int, list[int]] = {}
    for x in range(n):
        groups.setdefault(find(x), []).append(x)
    mapping: dict[int, int] = {}
    for members in groups.values():
        if len(members) < 2:
            continue
        canon = max(members, key=lambda m: (votes.get(m, 0), -m))
        for m in members:
            if m != canon:
                mapping[m] = canon
    return mapping


def apply_merges(payload: dict[str, Any], mapping: dict[int, int]) -> dict[str, Any]:
    """Drop merged classes and redirect every reference to the canonical one."""
    items = payload.get(CLASSES) or []
    rename = {_key(items[m].get("LABEL")): items[c].get("LABEL") for m, c in mapping.items()}

    def fix(v: Any) -> Any:
        if isinstance(v, list):
            return [fix(x) for x in v]
        return rename.get(_key(v), v) if isinstance(v, str) else v

    out = dict(payload)
    kept = []
    for idx, it in enumerate(items):
        if idx in mapping:
            continue
        if not isinstance(it, dict):
            kept.append(it)
            continue
        it = dict(it)
        if "PARENT_LABEL" in it:
            it["PARENT_LABEL"] = fix(it["PARENT_LABEL"])
        kept.append(it)
    out[CLASSES] = kept
    for section, fields in (("MATCH NOT FOUND RELATIONS", ("DOMAIN", "RANGE")),
                            ("MATCH NOT FOUND INSTANCES", ("TYPE_LABEL",))):
        if isinstance(payload.get(section), list):
            new = []
            for it in payload[section]:
                if isinstance(it, dict):
                    it = dict(it)
                    for f in fields:
                        if f in it:
                            it[f] = fix(it[f])
                new.append(it)
            out[section] = new
    return out


def _judge_prompt(prompt_set: str | None):
    if prompt_set == "deepseek":
        from backend.app.services.prompts_deepseek import deepseek_class_synonym_judge

        return deepseek_class_synonym_judge
    return None


async def semantic_class_merge(router: Any, payload: dict[str, Any], *, embedder: Any = None,
                               log=print) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run the pass if enabled; returns (payload, merge records)."""
    try:
        spec = router.task_spec(TASK)
    except (AttributeError, KeyError):
        return payload, []
    if not spec.get("enabled"):
        return payload, []
    build = _judge_prompt(spec.get("prompt_set"))
    if build is None:
        log(f"[stage3b] class merge skipped: no {TASK} prompt for prompt_set={spec.get('prompt_set')!r}")
        return payload, []
    allitems = payload.get(CLASSES) or []
    valid = [k for k, it in enumerate(allitems) if isinstance(it, dict) and it.get("LABEL")]
    if len(valid) < 2:
        return payload, []
    items = [allitems[k] for k in valid]
    if embedder is None:
        from backend.app.services.embeddings import Embedder

        embedder = Embedder()
    texts = [f"{_words(it['LABEL'])}: {it.get('DESCRIPTION', '')}" for it in items]
    from backend.app.services.semantic_grouping import _embed_compact

    # float32 array in slices: nested lists cost ~32 bytes per float.
    vectors = await _embed_compact(embedder, texts)
    pairs = candidate_pairs(vectors, float(spec.get("threshold", 0.83)), int(spec.get("max_pairs", 150)))
    if not pairs:
        return payload, []
    batch = int(spec.get("pairs_per_call", 60))
    confirmed: list[tuple[int, int, int]] = []
    records: list[dict[str, Any]] = []
    for start in range(0, len(pairs), batch):
        chunk = pairs[start:start + batch]
        ask = [{"id": k, "a_label": items[i]["LABEL"], "a_desc": items[i].get("DESCRIPTION", ""),
                "b_label": items[j]["LABEL"], "b_desc": items[j].get("DESCRIPTION", "")}
               for k, (i, j, _) in enumerate(chunk)]
        system, user = build(ask)
        try:
            res = await router.chat("class_synonym_judge", system=system, user=user)
            decisions = json.loads(res.text).get("decisions") or []
        except Exception as exc:  # a failed judge call merges nothing
            log(f"[stage3b] judge call failed ({type(exc).__name__}); {len(chunk)} pair(s) left unmerged")
            continue
        for d in decisions:
            try:
                i, j, sim = chunk[int(d["id"])]
            except (KeyError, ValueError, IndexError, TypeError):
                continue
            if d.get("same") is True:
                keep = j if str(d.get("keep", "A")).upper() == "B" else i
                confirmed.append((i, j, keep))
                records.append({"a": items[i]["LABEL"], "b": items[j]["LABEL"], "sim": round(sim, 3),
                                "keep": items[keep]["LABEL"], "reason": d.get("reason", "")})
    local = merge_groups(len(items), confirmed)
    mapping = {valid[m]: valid[c] for m, c in local.items()}  # back to payload positions
    log(f"[stage3b] class merge: {len(pairs)} candidate pair(s) >= {spec.get('threshold', 0.83)}, "
        f"{len(confirmed)} confirmed same, {len(mapping)} class(es) merged away")
    for r in records[:40]:
        log(f"[stage3b]   {r['a']} ~ {r['b']} ({r['sim']}) -> keep {r['keep']}: {r['reason']}")
    return (apply_merges(payload, mapping) if mapping else payload), records
