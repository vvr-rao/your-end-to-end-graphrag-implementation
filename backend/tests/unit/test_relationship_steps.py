"""Per-chunk relationship steps: pass 1, the gap pass, the orphan check, then
verification -- every claim through the same gates, tagged with its step."""
from __future__ import annotations

import asyncio
import contextlib
import json
from types import SimpleNamespace

from backend.app.services import db_entity_extract as dee
from backend.app.services.prompts import relationship_extract, relationship_orphan_check

CLS = "https://x#Person"
ORG = "https://x#Org"
WORKS = "https://x#worksFor"
MENU = [{"iri": WORKS, "label": "worksFor", "domain_label": "Person",
         "range_label": "Org", "domain_iris": [CLS], "range_iris": [ORG]}]
TEXT = ("Ann Lee works for Acme. Bob Ray works for Acme too. "
        "Cy Dunn was mentioned in a list.")
KEPT = [{"canonical_name": n, "short_name": n, "class_iri": c} for n, c in
        (("Ann Lee", CLS), ("Bob Ray", CLS), ("Acme", ORG), ("Cy Dunn", CLS))]


def _claim(s, o, ev):
    return {"subject": s, "predicate_iri": WORKS, "object": o, "evidence": ev}


class FakeRouter:
    def __init__(self, replies):
        self.replies, self.calls = replies, []

    async def chat(self, task, *, system, user, **kw):
        self.calls.append((task, system, user))
        return SimpleNamespace(text=json.dumps(self.replies[task].pop(0)))


def _run(monkeypatch, replies, **kw):
    @contextlib.asynccontextmanager
    async def _scope():
        yield None

    async def _menu(session, iris):
        return MENU, {WORKS: (CLS, ORG)}, {CLS: {CLS}, ORG: {ORG}}

    monkeypatch.setattr(dee, "session_scope", _scope)
    monkeypatch.setattr(dee, "_candidate_predicates", _menu)
    router, stats, flags = FakeRouter(replies), {}, []
    drops = {k: 0 for k in ("unresolved", "bad_predicate", "domain_range",
             "self_loop", "no_evidence", "overlong_evidence",
             "one_sided_evidence", "unsupported", "reversed",
             "no_relation_phrase")}
    repairs = {"direction_swapped": 0, "rescued": 0,
               "predicate_recovered": 0, "type_swapped": 0}
    rels = asyncio.run(dee._relationships_for_chunk(
        router, TEXT, KEPT, {CLS: "Person", ORG: "Org"}, "chunk#1",
        rel_drops=drops, rel_repairs=repairs, pass_stats=stats,
        orphan_flags=flags, **kw))
    return rels, stats, flags, router


def test_gap_pass_and_orphan_check_add_claims_then_verify_sees_all(monkeypatch):
    replies = {
        "relationship_extract": [
            {"relationships": [_claim("Ann Lee", "Acme", "Ann Lee works for Acme")]},
            # gap pass: one new, one repeat of pass 1
            {"relationships": [_claim("Bob Ray", "Acme", "Bob Ray works for Acme too"),
                               _claim("Ann Lee", "Acme", "Ann Lee works for Acme")]},
        ],
        "relationship_orphan_check": [
            {"relationships": [],
             "no_relationship": [{"entity": "Cy Dunn", "reason": "only_listed"}]},
        ],
        "relationship_verify": [
            {"verdicts": [{"index": 0, "verdict": "supported"},
                          {"index": 1, "verdict": "supported"}]},
        ],
    }
    rels, stats, flags, router = _run(monkeypatch, replies, gap_pass=True,
                                      orphan_check=True, verify=True)
    assert [(r["subject"], r["found_by"]) for r in rels] == [
        ("Ann Lee", "pass1"), ("Bob Ray", "gap")]
    assert stats["gap_kept"] == 1 and stats["gap_duplicates"] == 1
    # Only Cy Dunn was still unlinked when the orphan check ran.
    assert stats["orphans_flagged"] == 1
    assert flags == [{"entity": "Cy Dunn", "reason": "only_listed", "chunk": "chunk#1"}]
    assert stats["orphan_reason_only_listed"] == 1
    assert stats["verified_pass1"] == 1 and stats["verified_gap"] == 1
    tasks = [c[0] for c in router.calls]
    assert tasks == ["relationship_extract", "relationship_extract",
                     "relationship_orphan_check", "relationship_verify"]
    # The gap call is shown what pass 1 accepted.
    assert "Ann Lee --worksFor--> Acme" in router.calls[1][2]
    # The orphan call lists only the unlinked entity.
    assert "NO RELATIONSHIP YET" in router.calls[2][2]
    assert "  - Cy Dunn" in router.calls[2][2]
    assert "  - Bob Ray\n" not in router.calls[2][2].split("NO RELATIONSHIP YET")[1]


def test_orphan_check_claims_face_the_same_gates(monkeypatch):
    replies = {
        "relationship_extract": [{"relationships": []}],
        "relationship_orphan_check": [
            {"relationships": [_claim("Cy Dunn", "Acme", "Cy Dunn works for Acme")],
             "no_relationship": [{"entity": "Ann Lee", "reason": "made_up"}]},
        ],
    }
    rels, stats, _flags, _ = _run(monkeypatch, replies, orphan_check=True,
                                 verify=False)
    assert rels == []                                # quote not in the passage
    assert stats["orphan_reason_unspecified"] == 1   # unknown reason normalised


def test_steps_are_off_unless_asked(monkeypatch):
    replies = {"relationship_extract": [
        {"relationships": [_claim("Ann Lee", "Acme", "Ann Lee works for Acme")]}]}
    rels, _stats, _, router = _run(monkeypatch, replies, verify=False)
    assert [c[0] for c in router.calls] == ["relationship_extract"]
    assert len(rels) == 1


def test_default_extract_prompt_unchanged_without_already_found():
    ents = [{"canonical_name": "A", "class_label": "X"}]
    s1, u1 = relationship_extract("t", ents, MENU)
    s2, u2 = relationship_extract("t", ents, MENU, already_found=None)
    assert (s1, u1) == (s2, u2)
    assert "ALREADY FOUND" not in u1 and "FIRST PASS" not in s1


def test_orphan_prompt_requires_every_orphan_accounted_for():
    ents = [{"canonical_name": "A", "class_label": "X"}]
    s, u = relationship_orphan_check("t", ents, MENU, orphans=["A"], found=[])
    assert "exactly once" in s and "Flagging is a normal answer" in s
    for reason in ("only_listed", "no_partner", "implied_only", "not_an_entity"):
        assert reason in s and reason in u
