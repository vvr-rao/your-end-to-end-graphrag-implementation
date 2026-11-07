"""Give qualified concept classes a generic parent (opt-in, Stage 3c).

Class proposals name concepts as the passage qualifies them, so one idea
arrives as many siblings with no shared parent. Measured on the Qwen
fin-gen build (2026-10-24): SemiconductorRevenue sat under Metric,
RelatedPartyRevenue and GeographicalRevenue under EconomicConcept, and no
`Revenue` class existed -- so "What was the revenue for LG Electronics?"
matched three unrelated classes and no single concept. The same happens in
any domain (ObjectiveResponseRate / OverallResponseRate, WeldDefect /
PaintDefect).

This pass, run on the deduplicated Stage-3 payload:

  1. groups the proposed classes by their HEAD noun (the last word of the
     label, or the last two when the last is too generic to stand alone:
     `Rate`, `Ratio`, `Value`...),
  2. asks a judge model, per group, what KIND of thing the head is and which
     members are not kinds of it; code maps the kind to a root, so the model
     never chooses a root and can answer that a head is not a concept at all,
  3. adds the head as a proposal under that root (reused by Stage 4's
     exact-label dedup when the ontology already has it) and re-parents
     confirmed members to it -- only members whose current parent is
     missing or a broad root, so a specific parent is never overwritten.

Runs whenever the `concept_head_judge` task is defined; `enabled: false` on
the task, or `--no-concept-heads` on the CLI (env `GRAPHRAG_NO_CONCEPT_HEADS`),
turns it off. The judge prompt follows `prompt_set` (default prompt, or `deepseek`); with no
prompt for the configured set the pass is skipped.
"""
from __future__ import annotations

import os
import re
from typing import Any

TASK = "concept_head_judge"
# Set by the CLI's --no-concept-heads (the stage sits deep inside prune-expand).
DISABLE_ENV = "GRAPHRAG_NO_CONCEPT_HEADS"
CLASSES = "MATCH NOT FOUND"

# Roots a head may go under: the domain-neutral layer of domain_concepts.owl
# plus its Quantity children. Anything else (Organization, Person, a place)
# is not a concept head, so the judge must reject it.
HEAD_ROOTS = (
    "Quantity", "Metric", "Measure", "Condition", "Process", "Method",
    "Intervention", "Outcome", "Material", "PolicyConcept", "EconomicConcept",
    "TechnologyConcept", "Industry", "Infrastructure", "SupplyChainConcept",
)
# The judge's only choices, and the root each maps to (None = not a concept
# head, so rejected). Choosing among 15 broad roots let it force nearly anything
# in: Nerve under Material, Temperature under Condition (replay, 2026-11-03).
HEAD_KINDS: tuple[tuple[str, str | None, str], ...] = (
    ("amount", "Measure", "a directly reported amount (revenue, dose, output volume, voltage)"),
    ("ratio", "Metric", "a derived ratio or rate (margin, response rate, defect rate)"),
    ("state", "Condition", "a state, disease or defect (diabetes, insolvency, corrosion, a risk)"),
    ("result", "Outcome", "a result or change (side effect, loss, growth, decline)"),
    ("substance", "Material", "a physical or biological substance or agent (drug, inhibitor, receptor, steel, resin)"),
    ("device", "TechnologyConcept", "an engineered device, system or software (battery, chip, syringe), never a drug or an accounting item"),
    ("network", "Infrastructure", "a physical network or utility (roads, grids, pipelines, ports), never financial or anatomical"),
    ("rule", "PolicyConcept", "a rule, law, standard or official decision (regulation, covenant, approval)"),
    ("activity", "Process", "an activity or sequence of steps (production, assembly, clinical study)"),
    ("technique", "Method", "a technique or way of doing something (assay, valuation method, lean)"),
    ("treatment", "Intervention", "a treatment or deliberate action taken on something (therapy, surgery, hedging)"),
    ("economic", "EconomicConcept", "an economic or financial item that is not an amount (loan, asset, subsidy, market)"),
    ("industry", "Industry", "an industry or sector (mining, healthcare)"),
    ("supply_chain", "SupplyChainConcept", "a supply-chain concept (procurement, logistics, supplier network)"),
    ("body_part", None, "a body part or anatomical structure (CoronaryArtery / RenalArtery under Artery)"),
    ("document_item", None, "part of a document, page or website, or a site category (AuthorByline under Byline, TableFootnote under Footnote)"),
    ("descriptor", None, "an adjective, participle or vague abstraction, not a concrete kind of thing (Reported, Pending, Essence)"),
    ("none", None, "none of the above, or the members only share a word"),
)
_KIND_ROOT = {k: r for k, r, _ in HEAD_KINDS}
# Almost never a useful parent in any domain; rejected without asking the model.
_BLOCKED_HEADS = frozenset({
    "date", "section", "page", "purpose", "chapter", "paragraph", "title",
    "author", "website", "link", "tag", "category",
})
# A member keeps a parent that is more specific than these.
_BROAD_PARENTS = frozenset(
    r.lower() for r in (*HEAD_ROOTS, "DomainConcept", "Concept", "Thing", "owl:Thing", "NONE", "")
)
# Last words too generic to be a concept on their own; the head then takes the
# last TWO words ("ResponseRate", "HazardRatio").
_WEAK_HEADS = frozenset({
    "rate", "ratio", "value", "level", "factor", "index", "type", "status",
    "data", "information", "system", "amount", "number", "total", "score",
    "measure", "metric", "concept", "process", "method", "condition",
    "outcome", "event", "item", "activity", "class", "category", "kind",
})


def _tokens(label: Any) -> list[str]:
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(label or ""))
    text = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", text)
    return [t for t in re.split(r"[^A-Za-z0-9]+", text) if t]


def _is_broad(parent: Any) -> bool:
    """True when `parent` (a label or a full IRI) is missing or a broad root."""
    name = re.split(r"[#/]", str(parent or "").strip())[-1]
    return name.lower() in _BROAD_PARENTS


def _key(label: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(label or "").lower())


def head_of(label: Any) -> str | None:
    """CamelCase head of a class label, or None when there is none.

    'SemiconductorRevenue' -> 'Revenue'; 'ObjectiveResponseRate' ->
    'ResponseRate'; 'Revenue' -> None (a one-word label IS a head);
    'Rate' -> None.
    """
    toks = _tokens(label)
    if len(toks) < 2:
        return None
    last = toks[-1]
    if last.lower() not in _WEAK_HEADS:
        head = [last]
    elif len(toks) >= 3:
        head = toks[-2:]
    else:
        return None
    # Digits mean a code or abbreviation split mid-token ('HbA1c' -> 'Hb', 'A1c').
    if any(c.isdigit() for t in head for c in t) or len(head[-1]) < 3:
        return None
    return "".join(t[:1].upper() + t[1:] for t in head)


def _is_negated(label: Any) -> bool:
    """A negative finding: NoPancreaticFinding is not a kind of Finding. NoSQL
    stays out of this, because the capital after `No` starts an acronym."""
    return bool(re.match(r"^No(?:[A-Z][a-z]|\s)", str(label or "")))


def _singular(head: str) -> str | None:
    if head.endswith("ies") and len(head) > 4:
        return head[:-3] + "y"
    if head.endswith("s") and not head.endswith(("ss", "us", "is")):
        return head[:-1]
    return None


def candidate_groups(items: list[dict[str, Any]], min_members: int = 2) -> dict[str, list[int]]:
    """{head: [item index, ...]} for heads shared by >= `min_members` proposals
    other than the head itself. A proposal whose label IS the head is not a
    member (it is the head); a negated label is never a member.

    A plural head joins its singular only when that singular is also a head or
    a proposed class, so Costs joins Cost while Diabetes never becomes Diabete."""
    groups: dict[str, list[int]] = {}
    for idx, it in enumerate(items):
        head = head_of(it.get("LABEL"))
        if head and head.lower() not in _BLOCKED_HEADS \
                and _key(head) != _key(it.get("LABEL")) and not _is_negated(it.get("LABEL")):
            groups.setdefault(head, []).append(idx)
    labels = {_key(it.get("LABEL")) for it in items}
    for head in list(groups):
        one = _singular(head)
        if one and (one in groups or _key(one) in labels):
            groups.setdefault(one, []).extend(groups.pop(head))
    return {h: m for h, m in groups.items() if len(m) >= min_members}


def apply_heads(payload: dict[str, Any], decisions: list[dict[str, Any]]) -> tuple[dict[str, Any], int]:
    """Add each accepted head as a proposal under its root and re-parent its
    confirmed members. `decisions`: [{head, root, description, members}],
    members as LABELS. Returns (payload, members re-parented)."""
    items = list(payload.get(CLASSES) or [])
    by_key = {_key(it.get("LABEL")): k for k, it in enumerate(items) if isinstance(it, dict)}
    moved = 0
    for d in decisions:
        head, root = d["head"], d["root"]
        hk = _key(head)
        if hk in by_key:
            existing = dict(items[by_key[hk]])
            if _is_broad(existing.get("PARENT_LABEL")):
                existing["PARENT_LABEL"] = root
            items[by_key[hk]] = existing
        else:
            items.append({"LABEL": head, "DESCRIPTION": d.get("description") or "",
                          "PARENT_LABEL": root, "SOURCE": "concept_head"})
            by_key[hk] = len(items) - 1
        for label in d["members"]:
            k = by_key.get(_key(label))
            if k is None:
                continue
            it = dict(items[k])
            if not _is_broad(it.get("PARENT_LABEL")):
                continue
            it["PARENT_LABEL"] = head
            items[k] = it
            moved += 1
    out = dict(payload)
    out[CLASSES] = items
    return out, moved


def _judge_prompt(prompt_set: str | None):
    if prompt_set == "deepseek":
        from backend.app.services.prompts_deepseek import deepseek_concept_head_judge

        return deepseek_concept_head_judge
    if prompt_set in (None, "", "default"):
        from backend.app.services.prompts import concept_head_judge

        return concept_head_judge
    return None


async def generalize_concept_heads(router: Any, payload: dict[str, Any], *, log=print
                                   ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run the pass if enabled; returns (payload, accepted head records)."""
    try:
        spec = router.task_spec(TASK)
    except (AttributeError, KeyError):
        return payload, []
    if not spec or spec.get("enabled") is False:
        return payload, []
    if os.environ.get(DISABLE_ENV):
        log("[stage3c] concept heads skipped (--no-concept-heads)")
        return payload, []
    build = _judge_prompt(spec.get("prompt_set"))
    if build is None:
        log(f"[stage3c] concept heads skipped: no {TASK} prompt for prompt_set={spec.get('prompt_set')!r}")
        return payload, []
    items = [it for it in (payload.get(CLASSES) or []) if isinstance(it, dict) and it.get("LABEL")]
    min_members = int(spec.get("min_members", 2))
    groups = candidate_groups(items, min_members)
    if not groups:
        log("[stage3c] concept heads: no shared heads among the proposals")
        return payload, []
    ordered = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    ordered = ordered[: int(spec.get("max_groups", 300))]
    max_members = int(spec.get("max_members_per_group", 30))
    batch = int(spec.get("groups_per_call", 25))
    accepted: list[dict[str, Any]] = []
    from backend.app.services.db_artifact_gen import _extract_json

    for start in range(0, len(ordered), batch):
        chunk = ordered[start:start + batch]
        ask = [{"id": k, "head": head,
                "members": [{"label": items[i]["LABEL"],
                             "description": str(items[i].get("DESCRIPTION") or "")[:160]}
                            for i in members[:max_members]]}
               for k, (head, members) in enumerate(chunk)]
        system, user = build(ask, [(k, d) for k, _, d in HEAD_KINDS])
        try:
            res = await router.chat(TASK, system=system, user=user)
            parsed = _extract_json(res.text)
            got = parsed.get("decisions") if isinstance(parsed, dict) else None
        except Exception as exc:  # a failed judge call generalizes nothing
            log(f"[stage3c] judge call failed ({type(exc).__name__}); {len(chunk)} group(s) left as-is")
            continue
        for d in got or []:
            try:
                head, members = chunk[int(d["id"])]
            except (KeyError, ValueError, IndexError, TypeError):
                continue
            root = _KIND_ROOT.get(str(d.get("kind") or "").strip().lower())
            if root is None:
                continue
            rejected = {_key(x) for x in d.get("rejected_members") or []}
            keep = [items[i]["LABEL"] for i in members[:max_members]
                    if _key(items[i]["LABEL"]) not in rejected]
            if len(keep) < min_members:
                continue
            accepted.append({"head": head, "root": root, "kind": str(d.get("kind")).strip().lower(),
                             "members": keep,
                             "description": str(d.get("description") or "").strip()})
    if not accepted:
        log(f"[stage3c] concept heads: {len(ordered)} candidate group(s), none accepted")
        return payload, []
    out, moved = apply_heads(payload, accepted)
    log(f"[stage3c] concept heads: {len(ordered)} candidate group(s), {len(accepted)} accepted, "
        f"{moved} class(es) re-parented")
    for r in accepted[:40]:
        log(f"[stage3c]   {r['head']} < {r['root']}: {', '.join(r['members'][:6])}"
            + (" ..." if len(r["members"]) > 6 else ""))
    return out, accepted

