#!/usr/bin/env python3
"""Compare class_proposal arms written by class_proposal_compare.py.

    uv run python scripts/bench/class_proposal_report.py benchmarks/<date>/classcmp \
        --merge output_ontologies/<merge> [--arms deepseek,qwen] [--corpora pharma12,websearch3x]

Per arm x corpus, from the audit log (parsed replies, after the code guard),
the run log and the cost dump:
  volume      replies, new classes, distinct labels, per-reply median/p90/max,
              replies cut by the guard, replies that hit max_tokens
  rule 1      proposals whose label already exists in the merged ontology
              (case/space/punctuation-insensitive, plural-folded)
  rule 2      named things proposed as classes: drugs, companies, places,
              trials/studies, people (pattern lists below; a lower bound)
  tie-in      parent is an existing class / NONE / another proposal; share of
              new classes in at least one relation that has an existing-class end
  lists       replies that declared LISTS, lists declared
  cost        output tokens, call seconds, USD
"""
from __future__ import annotations

import json
import re
import statistics as st
import sys
from collections import Counter
from pathlib import Path

DRUG = re.compile(r"(glutide|zepatide|pril$|sartan$|formin$|orlistat|phentermine|topiramate|naltrexone|bupropion|"
                  r"ozempic|wegovy|mounjaro|zepbound|trulicity|saxenda|qsymia|contrave|victoza|rybelsus|xenical|"
                  r"liraglutide|exenatide|insulinglargine|insulindegludec|hydrochlorothiazide|losartan|telmisartan)")
DRUG_KIND = re.compile(r"(class|agonist|inhibitor|blocker|therapy|receptor|diuretic|antagonist|drug|medication|"
                       r"antibod|secretagogue)", re.I)
NAMED = re.compile(r"^(china|india|japan|southkorea|korea|vietnam|indonesia|thailand|taiwan|iran|iraq|qatar|"
                   r"saudiarabia|uae|unitedarabemirates|russia|ukraine|unitedstates|usa|europe|asia|africa|"
                   r"tesla|nvidia|tsmc|samsung|intel|byd|yara|apple|micron|mosaic|sabic|reliance|elililly|lilly|"
                   r"novonordisk|pfizer|straitofhormuz|suezcanal|persiangulf|redsea|fda|who|imf|opec)$")
TRIAL = re.compile(r"(^nct\d|^surpass|^sustain|^award\d|^rewind|^hope(study)?$|^aire|study\d|trial\d)", re.I)


def key(label: str) -> str:
    k = re.sub(r"[^a-z0-9]", "", str(label).lower())
    return k[:-1] if k.endswith("s") and len(k) > 4 else k


def existing_index(merge: Path) -> tuple[set[str], set[str]]:
    m = json.loads((merge / "merged.json").read_text())
    classes = m["classes_dict"] if "classes_dict" in m else m
    keys, iris = set(), set(classes)
    for iri, rec in classes.items():
        for lab in (rec.get("labels") or []) + [rec.get("name") or "", iri.rsplit("#", 1)[-1].rsplit("/", 1)[-1]]:
            if lab:
                keys.add(key(lab))
    return keys, iris


def arm_stats(run: Path, log: Path, ex_keys: set[str], ex_iris: set[str]) -> dict:
    audits = list(run.glob("pe/*/llm_audit.jsonl"))
    rows = [json.loads(line) for a in audits for line in a.open() if '"class_proposal"' in line]
    rows = [r for r in rows if r.get("task") == "class_proposal"]
    text = log.read_text(errors="ignore") if log.exists() else ""
    per_reply, classes, lists_replies, lists_n = [], [], 0, 0
    parent = Counter()
    tied = 0
    for r in rows:
        p = r["payload"]
        nf = [x for x in (p.get("MATCH NOT FOUND") or []) if isinstance(x, dict) and x.get("LABEL")]
        per_reply.append(len(nf))
        own = {key(x["LABEL"]) for x in nf}
        if p.get("LISTS"):
            lists_replies += 1
            lists_n += len(p["LISTS"])
        rel_ends: dict[str, bool] = {}
        for rel in p.get("MATCH NOT FOUND RELATIONS") or []:
            if not isinstance(rel, dict):
                continue
            d, g = str(rel.get("DOMAIN", "")), str(rel.get("RANGE", ""))
            d_ex = d in ex_iris or (key(d) in ex_keys and key(d) not in own)
            g_ex = g in ex_iris or (key(g) in ex_keys and key(g) not in own)
            for end, other_ex in ((d, g_ex), (g, d_ex)):
                rel_ends[key(end)] = rel_ends.get(key(end), False) or other_ex
        for x in nf:
            lab = str(x["LABEL"])
            classes.append(lab)
            pl = str(x.get("PARENT_LABEL") or "NONE")
            if pl.upper() == "NONE":
                parent["none"] += 1
            elif pl in ex_iris or (key(pl) in ex_keys and key(pl) not in own):
                parent["existing"] += 1
            elif key(pl) in own:
                parent["new"] += 1
            else:
                parent["unresolved"] += 1
            if rel_ends.get(key(lab)):
                tied += 1
    n = len(classes) or 1
    keys = [key(c) for c in classes]
    dup = sorted({c for c in classes if key(c) in ex_keys})
    drugs = sorted({c for c in classes if DRUG.search(key(c)) and not DRUG_KIND.search(c)})
    named = sorted({c for c in classes if NAMED.match(key(c))})
    trials = sorted({c for c in classes if TRIAL.search(key(c))})
    costs = {}
    for f in run.glob("costs/*.json"):
        for t, v in (json.loads(f.read_text()).get("by_task") or {}).items():
            c = costs.setdefault(t, {"calls": 0, "out": 0, "sec": 0.0, "usd": 0.0})
            c["calls"] += v["calls"]; c["out"] += v["out_tokens"]; c["sec"] += v["call_seconds"]; c["usd"] += v["cost_usd"]
    cp = costs.get("class_proposal", {})
    srt = sorted(per_reply) or [0]
    ae_stats = adverse_event_stats(rows)
    return {
        "replies": len(rows), "new_classes": len(classes), "distinct": len(set(keys)),
        "per_reply_median": st.median(srt), "per_reply_p90": srt[int(0.9 * (len(srt) - 1))], "per_reply_max": srt[-1],
        "guard_cut": len(re.findall(r"over budget, kept the first", text)),
        "hit_max_tokens": len(re.findall(r"finish=length", text)),
        "dup_existing": len(dup), "dup_examples": dup[:12],
        "drug_classes": len(drugs), "drug_examples": drugs[:12],
        "named_classes": len(named), "named_examples": named[:12],
        "trial_classes": len(trials), "trial_examples": trials[:8],
        "parent_existing": round(parent["existing"] / n, 3), "parent_none": round(parent["none"] / n, 3),
        "parent_new": round(parent["new"] / n, 3), "parent_unresolved": round(parent["unresolved"] / n, 3),
        "in_relation_with_existing": round(tied / n, 3),
        "replies_with_lists": lists_replies, "lists_declared": lists_n,
        "out_tokens": cp.get("out"), "call_seconds": round(cp.get("sec", 0)), "usd": round(cp.get("usd", 0), 3),
        **ae_stats,
    }


def adverse_event_stats(rows: list[dict]) -> dict:
    """How the raw replies (BEFORE the Stage-3 normalizer) follow the
    adverse-event hierarchy: event classes under a canonical TYPE vs directly
    under AdverseEvent vs elsewhere, and variant names the normalizer would
    have to rewrite."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from backend.app.services import adverse_event_taxonomy as ae

    names = {ae.ROOT, *ae.TYPES}
    under_type = under_root = 0
    stray: Counter = Counter()
    rel_to_ae = 0
    for r in rows:
        p = r["payload"]
        for x in p.get("MATCH NOT FOUND") or []:
            if not isinstance(x, dict):
                continue
            lab, par = str(x.get("LABEL", "")), str(x.get("PARENT_LABEL", ""))
            for v in (lab, par):
                c = ae.canonical(v)
                if c and v not in names:
                    stray[v] += 1
            pc = par if par in names else ae.canonical(par)
            if lab in names or ae.canonical(lab):
                continue                       # the root/type itself
            if pc == ae.ROOT:
                under_root += 1
            elif pc in ae.TYPES:
                under_type += 1
        for rel in p.get("MATCH NOT FOUND RELATIONS") or []:
            if isinstance(rel, dict) and (str(rel.get("RANGE")) in names or ae.canonical(rel.get("RANGE"))):
                rel_to_ae += 1
    events = under_type + under_root
    return {"ae_event_classes": events, "ae_under_type": under_type, "ae_under_root_only": under_root,
            "ae_share_under_type": round(under_type / events, 3) if events else None,
            "ae_stray_names": sum(stray.values()), "ae_stray_examples": [k for k, _ in stray.most_common(10)],
            "ae_relations_to_adverse_event": rel_to_ae}


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    base = Path(argv[0])
    merge = Path(argv[argv.index("--merge") + 1])
    arms = (argv[argv.index("--arms") + 1] if "--arms" in argv else "deepseek,qwen").split(",")
    corpora = (argv[argv.index("--corpora") + 1] if "--corpora" in argv else "pharma12,websearch3x").split(",")
    ex_keys, ex_iris = existing_index(merge)
    report = {c: {a: arm_stats(base / a / c, base / f"{a}_{c}.log", ex_keys, ex_iris)
                  for a in arms if (base / a / c).exists()} for c in corpora}
    (base / "class_proposal_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
