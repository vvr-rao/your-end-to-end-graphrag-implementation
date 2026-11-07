"""Drop junk class proposals before dedup -- deterministic rules, per reply.

Junk is not a duplicate of anything, so no dedup removes it; it has to be
filtered out. Measured on the 27B web-search build (2026-10-17, 20,175 raw
proposals): replies that LOOPED produced families like 1,025 `Refinery*`
classes (`Refinery2024`, `Refinery25bn`, `RefineryAccueil`,
`RefineryAcquiredByInYear`), `H7 ... H0`, `MostNonet`/`LeastDecet`, and
`www.trulicity.com`. These rules are cheap, run on each reply as it arrives
(streaming-friendly), and log a reason for every drop; what they cannot
decide goes to the validity judge.

Rules (each was checked against OpenAI's mostly-clean proposals on the same
corpus as a false-positive control):
  number      a year (19xx/20xx), a number of 3+ digits, an amount (25bn,
              88000, 12%), a bare designator (H7, 5), or a word with a
              number tacked on (Refinery5) -- but NOT category names that
              carry a digit (Type2Diabetes, Phase3ClinicalTrial, CO2, PM2.5,
              GLP1ReceptorAgonist, SGLT2Inhibitor, COVID19, 5GNetwork)
  url         a URL, domain or e-mail address
  superlative Most*/Least* fragments (MostNonet, LeastDuodecet)
  phrase      more than 6 words -- a sentence fragment, not a class name
  placeholder a model's "nothing here" answer used as a name (NONE, N/A)
  verb        a bare inflected verb (Lowers, Reduces): what a relation endpoint
              becomes when the model splits "X lowers Y" into hasLowers
  family      within ONE reply, more than FAMILY_MAX labels sharing their
              first word: the loop signature. The first FAMILY_KEEP are kept.
              OFF by default (drop_family=False) since 2026-10-19: with the
              v5 class prompts 7 of 8 sampled family drops were real
              categories (RefineryConversion, RefineryClosure, ...; 850 drops
              on web-search), because those replies are genuine lists, not
              loops. The labels now stay and the validity judge reviews them;
              a real loop is still capped by class_proposal.max_new_classes.
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

FAMILY_MAX = 15
FAMILY_KEEP = 5
MAX_WORDS = 6

_YEAR = re.compile(r"(?<!\d)(19|20)\d{2}(?!\d)")
_BIG_NUMBER = re.compile(r"\d{3,}")
_AMOUNT = re.compile(r"\d+(?:\.\d+)?\s*(?:%|(?:bn|b|mn|m|k|mt|tn|t|percent|usd|eur|kg|mg|ml|bpd)\b)", re.I)
_DESIGNATOR = re.compile(r"^[A-Za-z]?\d+[A-Za-z]?$")
# A word of 4+ letters with digits stuck on the end ("Refinery5", "Refinery51bn").
# All-caps stems are acronyms that legitimately carry a number (COVID19, CO2).
_TRAILING_NUMBER = re.compile(r"^(?P<stem>[A-Za-z][a-z]{3,}(?:[A-Z][a-z]+)*)\d+[a-z]*$")
# ...unless the number qualifies a category: MultipleEndocrineNeoplasiaType2,
# ClinicalTrialPhase3, CancerStage4, Tier1, Industry40 (Industry 4.0).
_QUALIFIED_NUMBER = re.compile(r"(?:Type|Phase|Tier|Scope|Grade|Stage|Class|Level|Generation|Industry)\d+[a-z]?$")
_URL = re.compile(r"(https?://|www\.|\.(?:com|org|net|gov|edu|io|co|ai)\b|@[a-z0-9-]+\.)", re.I)
_SUPERLATIVE = re.compile(r"^(?:Most|Least)(?=[A-Z\s])")
# The WHOLE label is a value. A letter unit needs a currency sign, so 3M, 5G,
# 4K and 10K stay names; B12 and A1C never match because they start with a letter.
_VALUE = re.compile(
    r"^(?:[$\u20ac\u00a3\u00a5\u20a9]\s*\d[\d,.]*\s*(?:bn|b|mn|m|k|tn|t|billion|million|thousand|trillion)?"
    r"|\d[\d,.]*\s*(?:%|percent|bn|mn|billion|million|trillion|mg|kg|ml|bpd)?)$",
    re.I,
)
_PERIOD = re.compile(r"^(?:Q[1-4]\s*(?:FY)?\s*\d{2,4}|[1-4]Q\s*\d{2,4}|FY\s*\d{2,4})$", re.I)
_PLACEHOLDERS = frozenset({"NONE", "NULL", "N/A", "UNKNOWN", "TBD", "UNDEFINED"})
_BARE_VERBS = frozenset({
    "is", "are", "be", "has", "have", "lowers", "raises", "reduces", "increases",
    "decreases", "causes", "treats", "improves", "inhibits", "blocks",
})


def words(label: str) -> list[str]:
    """CamelCase / snake / spaced label -> words."""
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(label))
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", s)
    return [w for w in re.split(r"[\s_\-/]+", s) if w]


def non_class_reason(label: Any) -> str | None:
    """Why a label can never name a class, or None.

    The narrow subset of `junk_reason` that every provider mode applies at
    Stage 4, where a class also gets minted from a relation endpoint. It looks
    at the whole label only, so a real name that carries a number stays.
    """
    lab = str(label or "").strip()
    low = lab.lower()
    if lab in _PLACEHOLDERS:
        return "placeholder"
    if low in _BARE_VERBS:
        return "verb"
    if _VALUE.match(lab) or _PERIOD.match(lab):
        return "number"
    return None


def junk_reason(label: Any) -> str | None:
    """The rule a single label breaks, or None. Per label; no context."""
    lab = str(label or "").strip()
    if not lab:
        return "empty"
    narrow = non_class_reason(lab)
    if narrow:
        return narrow
    if _URL.search(lab):
        return "url"
    # Only "Most/Least + one word" (MostPopular, LeastPhysical, MostNonet):
    # MostFavoredNationClause is a real concept.
    if _SUPERLATIVE.match(lab) and len(words(lab)) <= 2:
        return "superlative"
    if len(words(lab)) > MAX_WORDS:
        return "phrase"
    if _DESIGNATOR.match(lab):
        return "number"
    if _YEAR.search(lab) or _BIG_NUMBER.search(lab) or _AMOUNT.search(lab):
        return "number"
    if _TRAILING_NUMBER.match(lab) and not _QUALIFIED_NUMBER.search(lab):
        return "number"
    return None


def first_word(label: Any) -> str:
    w = words(str(label or ""))
    return w[0].lower() if w else ""


def filter_reply(payload: dict[str, Any] | None, drop_family: bool = False
                 ) -> tuple[dict[str, Any] | None, list[dict[str, str]]]:
    """Remove junk MATCH NOT FOUND entries from one class_proposal reply.

    Returns (payload, drops) where each drop is {label, reason}. With
    `drop_family`, a family beyond FAMILY_MAX members in this reply keeps its
    first FAMILY_KEEP and the rest are dropped as `family` (off by default:
    see the module docstring). Relations and instances are left alone here.
    """
    if not payload:
        return payload, []
    items = [x for x in (payload.get("MATCH NOT FOUND") or []) if isinstance(x, dict)]
    drops: list[dict[str, str]] = []
    kept: list[dict[str, Any]] = []
    for x in items:
        r = junk_reason(x.get("LABEL"))
        if r:
            drops.append({"label": str(x.get("LABEL")), "reason": r})
        else:
            kept.append(x)
    if not drop_family:
        payload["MATCH NOT FOUND"] = kept
        return payload, drops
    fam: dict[str, list[int]] = defaultdict(list)
    for i, x in enumerate(kept):
        fam[first_word(x.get("LABEL"))].append(i)
    over = {i for members in fam.values() if len(members) > FAMILY_MAX for i in members[FAMILY_KEEP:]}
    if over:
        drops += [{"label": str(kept[i].get("LABEL")), "reason": "family"} for i in sorted(over)]
        kept = [x for i, x in enumerate(kept) if i not in over]
    payload["MATCH NOT FOUND"] = kept
    return payload, drops


# --------------------------------------------------------------------------- #
# Validity judge: what the rules above cannot decide. Runs on the DISTINCT
# kept labels in the staging store, `batch` per call -- cost grows with the
# vocabulary, not the corpus.
# --------------------------------------------------------------------------- #

VALIDITY_TASK = "class_validity_judge"


async def run_validity_judge(store: Any, router: Any, batch: int = 100, concurrency: int = 8,
                             log=print) -> dict[str, int]:
    """Judge every kept class label: `junk` -> status junk, `named_thing` ->
    status instance (demoted; it can still become a Phase-2 entity),
    `category` -> kept. A failed or unanswered batch leaves its labels KEPT:
    an error must never delete real classes."""
    import asyncio

    from backend.app.helpers.ontology_pruning import extract_json_from_output
    from backend.app.services.prompts_qwen import qwen_class_validity_judge

    stats = {"judged": 0, "category": 0, "named_thing": 0, "junk": 0, "unanswered": 0, "failed_batches": 0}
    sem = asyncio.Semaphore(concurrency)

    async def one(rows: list[dict[str, Any]]) -> None:
        items = [{"id": n, "label": r["label"], "parent": r.get("parent"), "desc": r.get("description")}
                 for n, r in enumerate(rows)]
        system, user = qwen_class_validity_judge(items)
        async with sem:
            try:
                out = await router.chat(VALIDITY_TASK, system=system, user=user)
            except Exception as exc:
                stats["failed_batches"] += 1
                log(f"[validity-judge] batch failed: {type(exc).__name__}: {exc}")
                return
        data = extract_json_from_output(out.text) or {}
        verdicts = {str(v.get("id")): v for v in (data.get("verdicts") or []) if isinstance(v, dict)}
        junk, named = [], []
        for n, r in enumerate(rows):
            v = verdicts.get(str(n))
            stats["judged"] += 1
            if not v:
                stats["unanswered"] += 1
                continue
            verdict = str(v.get("verdict", "")).strip().lower()
            if verdict == "junk":
                junk.append(r["id"])
            elif verdict == "named_thing":
                named.append(r["id"])
            stats[verdict if verdict in stats else "category"] += 1
        await store.set_status(junk, "junk", "validity_judge")
        await store.set_status(named, "instance", "validity_judge")

    # Collect ids first: statuses change while judging, and keyset pagination
    # over a shrinking 'kept' set would still be correct but harder to follow.
    batches = [rows async for rows in store.iter_rows("class", "kept", batch)]
    await asyncio.gather(*[one(b) for b in batches])
    log("[validity-judge] " + ", ".join(f"{k}={v}" for k, v in stats.items()))
    return stats
