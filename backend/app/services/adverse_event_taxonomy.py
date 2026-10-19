"""Canonical adverse-event class hierarchy for class proposal (user decisions
2026-10-18): only the CATEGORIES are classes,

    Event (existing)
      AdverseEvent                      -- adverse events in general
        <System>AdverseEvent            -- the event TYPE (GastrointestinalAdverseEvent, ...)
        GeneralAdverseEvent             -- events that fit no body system

and a specific reaction (nausea, pruritus) is an INSTANCE typed by its type.
(A first version made every reaction a class: pharma rose to 480-1,032
adverse-event classes per model, so it was reversed.)

Why fixed names: each chunk named its own types, so one pharma build held
AdverseEvent, AdverseReaction, AdverseReactions, DrugSideEffect,
GastrointestinalAdverseEvent AND GastrointestinalAdverseReaction side by side.
The class prompts list these names; `normalize` maps variants the model still
writes back onto them after dedup. Types follow MedDRA's system organ classes,
in plain names.
"""
from __future__ import annotations

import re
from typing import Any

ROOT = "AdverseEvent"
GENERAL = "GeneralAdverseEvent"
ROOT_PARENT_LABEL = "Event"

# canonical type label -> (one-line description, label aliases for its system)
TYPES: dict[str, tuple[str, tuple[str, ...]]] = {
    "GastrointestinalAdverseEvent": ("Adverse event of the digestive tract (nausea, diarrhea, pancreatitis).",
                                     ("gastrointestinal", "gi", "digestive", "gastric", "pancreatic", "bowel")),
    "CardiovascularAdverseEvent": ("Adverse event of the heart or blood vessels (hypotension, arrhythmia).",
                                   ("cardiovascular", "cardiac", "heart", "vascular", "cardio")),
    "NervousSystemAdverseEvent": ("Adverse event of the nervous system (headache, dizziness, seizure).",
                                  ("nervoussystem", "neurologic", "neurological", "neuro", "nervous", "cns")),
    "PsychiatricAdverseEvent": ("Psychiatric adverse event (depression, insomnia, suicidal thoughts).",
                                ("psychiatric", "mental", "psychological", "behavioral", "mood")),
    "SkinAdverseEvent": ("Adverse event of the skin (rash, pruritus, alopecia).",
                         ("skin", "dermatologic", "dermatological", "cutaneous", "derma")),
    "RenalAdverseEvent": ("Adverse event of the kidneys or urinary tract (acute kidney injury).",
                          ("renal", "kidney", "urinary", "nephro")),
    "HepatobiliaryAdverseEvent": ("Adverse event of the liver, gallbladder or bile ducts (cholelithiasis).",
                                  ("hepatobiliary", "hepatic", "liver", "gallbladder", "biliary")),
    "RespiratoryAdverseEvent": ("Adverse event of the airways or lungs (cough, dyspnea).",
                                ("respiratory", "pulmonary", "lung", "airway")),
    "MetabolicAdverseEvent": ("Metabolic or nutritional adverse event (hypoglycemia, hyperkalemia).",
                              ("metabolic", "metabolism", "nutritional", "electrolyte")),
    "MusculoskeletalAdverseEvent": ("Adverse event of muscles, bones or joints (muscle cramps, arthralgia).",
                                    ("musculoskeletal", "muscle", "muscular", "skeletal", "joint")),
    "ImmuneAdverseEvent": ("Allergic or immune adverse event (hypersensitivity, anaphylaxis, angioedema).",
                           ("immune", "immunologic", "allergic", "hypersensitivity", "allergy")),
    "HematologicAdverseEvent": ("Adverse event of blood or lymph (anemia, agranulocytosis).",
                                ("hematologic", "hematological", "blood", "haematologic", "lymphatic")),
    "EndocrineAdverseEvent": ("Endocrine adverse event (thyroid dysfunction).",
                              ("endocrine", "thyroid", "hormonal")),
    "NeoplasmAdverseEvent": ("Tumor or neoplasm reported as an adverse event (thyroid C-cell tumor).",
                             ("neoplasm", "neoplastic", "tumor", "tumour", "cancer", "oncologic")),
    "EyeAdverseEvent": ("Adverse event of the eye (diabetic retinopathy, blurred vision).",
                        ("eye", "ocular", "ophthalmic", "visual", "vision")),
    "EarAdverseEvent": ("Adverse event of the ear or balance (tinnitus, vertigo).",
                        ("ear", "auditory", "otic", "vestibular")),
    "ReproductiveAdverseEvent": ("Adverse event of reproduction, pregnancy or the fetus (fetal toxicity).",
                                 ("reproductive", "pregnancy", "fetal", "foetal", "fertility", "obstetric")),
    "InfectionAdverseEvent": ("Infection reported as an adverse event (urinary tract infection).",
                              ("infection", "infectious", "infective")),
    "AdministrationSiteAdverseEvent": ("Reaction where a medicine is given (injection site reaction).",
                                       ("administrationsite", "injectionsite", "injection", "infusion")),
    GENERAL: ("Adverse event that fits no single body system (fatigue, fever, malaise).",
              ("general", "systemic", "constitutional", "other", "miscellaneous")),
}
ROOT_DESCRIPTION = "Any undesired effect or outcome associated with a medicine, device or treatment."

_ADVERSE_WORD = r"(?:adverse(?:drug)?|side|undesirable|untoward|toxic)"
_KIND_WORD = r"(?:event|reaction|effect|sideeffect|experience|disorder|toxicity)"


def _key(label: Any) -> str:
    k = re.sub(r"[^a-z0-9]", "", str(label).lower())
    return k[:-1] if k.endswith("s") and len(k) > 5 else k


_ROOT_RE = re.compile(rf"^(?:{_ADVERSE_WORD}{_KIND_WORD}|sideeffect|{_ADVERSE_WORD}?drug(?:reaction|sideeffect)|"
                      r"adverse|adverseeffectofdrug)$")
_TYPE_RES = [
    (name, re.compile(rf"^(?:{'|'.join(map(re.escape, aliases))})(?:system)?"
                      # Not "toxicity": FetalToxicity / RenalToxicity are specific
                      # events, not the type itself.
                      rf"(?:{_ADVERSE_WORD}{_KIND_WORD}?|sideeffect|adversereaction|reaction)$"))
    for name, (_d, aliases) in TYPES.items()
]


def canonical(label: Any) -> str | None:
    """The canonical taxonomy label `label` is a variant of, else None.

    Only labels that SAY adverse / side effect / reaction / toxicity map: a
    plain 'CardiovascularEvent' (a heart attack need not be adverse) or
    'GastrointestinalDisorder' is left alone.
    """
    k = _key(label)
    if not k:
        return None
    if k == _key(ROOT) or _ROOT_RE.match(k):
        return ROOT
    for name, rx in _TYPE_RES:
        if k == _key(name) or rx.match(k):
            return name
    return None


def normalize(result: dict[str, Any], existing_labels: set[str]) -> dict[str, int]:
    """Rewrite a Stage-3 result in place onto the canonical hierarchy.

    1. A proposed class that is a variant of the root or a type is renamed to
       the canonical label (dedup at mint then folds same-label entries).
    2. Every PARENT_LABEL / TYPE_LABEL / DOMAIN / RANGE that is a variant is
       rewritten the same way.
    3. Each canonical type now referenced, and the root, is added as a
       proposal (type -> AdverseEvent -> Event) unless the ontology already
       has a class with that label (`existing_labels`, keyed by _key).
    Returns counts for the log.
    """
    stats = {"renamed": 0, "references": 0, "added": 0}
    classes = result.get("MATCH NOT FOUND") or []
    present = {_key(x.get("LABEL")) for x in classes if isinstance(x, dict)}
    used: set[str] = set()

    def fix(value: Any) -> Any:
        c = canonical(value) if isinstance(value, str) else None
        if c and c != value:
            stats["references"] += 1
            used.add(c)
            return c
        if c:
            used.add(c)
        return value

    for x in classes:
        if not isinstance(x, dict):
            continue
        c = canonical(x.get("LABEL"))
        if c:
            used.add(c)
            if c != x.get("LABEL"):
                x["LABEL"] = c
                stats["renamed"] += 1
        if "PARENT_LABEL" in x:
            x["PARENT_LABEL"] = fix(x["PARENT_LABEL"])
            if canonical(x.get("PARENT_LABEL")):
                used.add(ROOT)
    for x in result.get("MATCH NOT FOUND INSTANCES") or []:
        if isinstance(x, dict) and "TYPE_LABEL" in x:
            x["TYPE_LABEL"] = fix(x["TYPE_LABEL"])
    for x in result.get("MATCH NOT FOUND RELATIONS") or []:
        if isinstance(x, dict):
            for end in ("DOMAIN", "RANGE"):
                if end in x:
                    x[end] = fix(x[end])
    if not used:            # no adverse events in this result: add nothing
        return stats
    used.add(ROOT)
    have = present | {_key(lbl) for lbl in existing_labels}
    # The canonical entries themselves: root under Event, each type under the root.
    for name in [ROOT] + [n for n in TYPES if n in used]:
        desc = ROOT_DESCRIPTION if name == ROOT else TYPES[name][0]
        parent = ROOT_PARENT_LABEL if name == ROOT else ROOT
        hit = next((x for x in classes if isinstance(x, dict) and _key(x.get("LABEL")) == _key(name)), None)
        if hit is not None:
            hit["PARENT_LABEL"] = parent
            continue
        if _key(name) in have:
            continue
        classes.append({"LABEL": name, "DESCRIPTION": desc, "PARENT_LABEL": parent})
        stats["added"] += 1
    result["MATCH NOT FOUND"] = classes
    return stats
