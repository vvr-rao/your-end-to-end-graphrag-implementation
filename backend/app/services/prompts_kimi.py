"""Prompts written for Kimi-K3 (`prompt_set: kimi`)."""
from __future__ import annotations

from typing import Any


def kimi_concept_review(
    chunk_text: str,
    candidate_classes: list[dict[str, str]],
    items: list[dict[str, Any]],
) -> tuple[str, str]:
    """`concept_review.py` for Kimi-K3.

    With a prompt written for DeepSeek, Kimi read class descriptions like legal
    definitions and answered `no_class` without trying the other classes, so
    this one asks it to work through the list first.

    `items`: [{"id", "name", "class_iri", "class_label"}]. Returns JSON
    {"verdicts": [{"id", "class_passes", "verdict", "better_class_iri", "reason"}]}.
    """
    system = (
        "You check the ontology class of concepts another model extracted from a "
        "passage. Reply with ONE JSON object and nothing else.\n\n"
        "For each concept:\n"
        "1. Decide what sort of thing it is in this passage: an amount or score, "
        "a state, an activity, a technique, a rule, a result, an ability, an "
        "occurrence, a kind of object.\n"
        "2. Set class_passes: does its current class cover that sort of thing? "
        "Class descriptions are short summaries, not legal definitions, so use "
        "their ordinary meaning. A fever is a state of the body. A court appeal "
        "is a process. But bed occupancy rate is an amount and not a state, and "
        "port dredging oversight is an activity and not a kind of Port: sharing "
        "a topic with a class does not make something a kind of it.\n"
        "3. If class_passes is false, go through the listed classes, including "
        "the broad ones, and look for one that does cover it.\n\n"
        "Then give one verdict:\n"
        "keep: class_passes is true.\n"
        "move: another listed class is right, either because class_passes is "
        "false or because a narrower listed class also covers it. Copy that "
        "class into better_class_iri exactly as written.\n"
        "no_class: class_passes is false and you tried the list and none covers it.\n"
        "drop: it means nothing on its own (importance, progress). Use this "
        "rarely, and never because of its class or because the passage only lists it.\n\n"
        "A specific named thing (a product, trial, law, facility) may appear. "
        "Another step handles names, so judge only its class. For anything but "
        "keep, give a reason of at most 12 words. Answer every id."
    )
    classes = "\n".join(
        f'- {c["iri"]} | {(c.get("label") or "").strip()} | '
        f'{(c.get("description") or "").strip()[:100]}'
        for c in candidate_classes
    )
    lines = "\n".join(
        f'{i["id"]}. {i["name"]} | current class: {i["class_iri"]} ({i["class_label"]})'
        for i in items
    )
    user = (
        f"LISTED CLASSES:\n{classes}\n\nPASSAGE:\n{chunk_text}\n\nCONCEPTS:\n{lines}\n\n"
        'Return JSON: {"verdicts": [{"id": ..., "class_passes": true|false, "verdict": '
        '"keep"|"move"|"no_class"|"drop", "better_class_iri": ..., "reason": ...}]}'
    )
    return system, user
