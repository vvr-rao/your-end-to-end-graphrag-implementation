"""Stage-3 dedup batches its payload and fails loudly.

The pharma-docs-3 run produced:
  [stage3] merging+dedup: 131 matches, 3702 new class proposals, 8153 new relation proposals
  [stage3] dedup failed: 400 - This model's maximum context length is 1047576 tokens.
           However, your messages resulted in 1348538 tokens -- returning merged results unchanged

The 400 was caught and the input returned unchanged, so the run "succeeded"
while emitting proposals that had never been deduplicated: 757 -> 4,942
classes, 3,007 orphans, and entity extraction reusing 48 of 4,422 entities.
"""

from __future__ import annotations

import json

import pytest

from backend.app.services.pipeline_llm import (
    DedupFailedError,
    _dedup,
    _existing_concept_names,
    _merge_dedup_batches,
)


class _StubRouter:
    """Records every dedup prompt; echoes the batch back by default."""

    def __init__(self, *, max_tokens: int = 32768, handler=None) -> None:
        self._max_tokens = max_tokens
        self.prompts: list[str] = []
        self.total_cost_usd = 0.0
        self._handler = handler

    def task_spec(self, task: str) -> dict:
        return {"max_tokens": self._max_tokens}

    async def chat(self, task: str, *, system: str, user: str, **kw):
        self.prompts.append(user)
        payload = json.loads(user.split("INPUT:\n", 1)[1].split("\n\nReturn", 1)[0])
        if self._handler is not None:
            return self._handler(len(self.prompts) - 1, payload)
        return _Result(json.dumps(payload))


class _Result:
    def __init__(self, text: str, finish_reason: str = "stop") -> None:
        self.text = text
        # _dedup reads this to tell a reply truncated at max_tokens from a
        # model that returned prose -- opposite causes, opposite remedies.
        self.finish_reason = finish_reason


def _proposals(n: int, prefix: str = "Concept") -> list[dict]:
    return [
        {
            "LABEL": f"{prefix}{i}",
            "DESCRIPTION": "A proposed class. " * 20,
            "PARENT_LABEL": "NONE",
        }
        for i in range(n)
    ]


@pytest.mark.asyncio
async def test_large_proposal_set_is_split_into_several_requests() -> None:
    merged = {
        "MATCHES FOUND": [{"IRI": "http://e#A", "TEXT_SNIPPET": "s"}],
        "MATCH NOT FOUND": _proposals(3702),
        "MATCH NOT FOUND RELATIONS": [
            {"LABEL": f"rel{i}", "DOMAIN": "A", "RANGE": "B", "DESCRIPTION": "d " * 20}
            for i in range(8153)
        ],
        "MATCH NOT FOUND INSTANCES": [],
    }
    router = _StubRouter()
    out = await _dedup(router, merged)

    assert len(router.prompts) > 1, "the failing payload must be split"
    # Every request must fit the budget the batcher was given.
    from backend.app.services.token_budget import count_tokens

    for p in router.prompts:
        assert count_tokens(p) < 32768, "a batch exceeded its own output budget"
    assert out["MATCH NOT FOUND"], "proposals survived the round trip"


@pytest.mark.asyncio
async def test_matches_found_is_returned_untouched() -> None:
    matches = [
        {"IRI": "http://e#Aspirin", "TEXT_SNIPPET": "aspirin is a drug"},
        {"IRI": "http://e#Ibuprofen", "TEXT_SNIPPET": "ibuprofen too"},
    ]
    merged = {"MATCHES FOUND": matches, "MATCH NOT FOUND": _proposals(5)}
    out = await _dedup(_StubRouter(), merged)
    assert out["MATCHES FOUND"] == matches


@pytest.mark.asyncio
async def test_all_four_keys_present_for_downstream_consumers() -> None:
    # merge_suggestions_into_results indexes deduped["MATCH NOT FOUND"]
    # unguarded, and _apply_expand reads all three proposal lists.
    out = await _dedup(_StubRouter(), {"MATCHES FOUND": [], "MATCH NOT FOUND": _proposals(3)})
    for key in (
        "MATCHES FOUND",
        "MATCH NOT FOUND",
        "MATCH NOT FOUND RELATIONS",
        "MATCH NOT FOUND INSTANCES",
    ):
        assert key in out
        assert isinstance(out[key], list)


@pytest.mark.asyncio
async def test_a_failing_batch_raises_instead_of_passing_input_through() -> None:
    """The whole point: a run that cannot dedup must stop, not emit 6.5x
    the classes and report success."""

    def handler(idx, payload):
        raise RuntimeError("400 - context_length_exceeded")

    merged = {"MATCHES FOUND": [], "MATCH NOT FOUND": _proposals(50)}
    with pytest.raises(DedupFailedError) as ei:
        await _dedup(_StubRouter(handler=handler), merged)
    assert "un-deduplicated" in str(ei.value)


@pytest.mark.asyncio
async def test_unparseable_response_also_raises() -> None:
    """A response truncated at max_tokens used to be indistinguishable from
    'nothing to do' -- both silently returned the input unchanged."""

    def handler(idx, payload):
        # cut mid-JSON: this is what hitting max_tokens looks like
        return _Result('{"MATCH NOT FOUND": [{"LABEL": "trunca', finish_reason="length")

    merged = {"MATCHES FOUND": [], "MATCH NOT FOUND": _proposals(5)}
    with pytest.raises(DedupFailedError):
        await _dedup(_StubRouter(handler=handler), merged)


@pytest.mark.asyncio
async def test_nothing_to_dedup_skips_the_llm_entirely() -> None:
    router = _StubRouter()
    merged = {"MATCHES FOUND": [{"IRI": "http://e#A"}], "MATCH NOT FOUND": []}
    out = await _dedup(router, merged)
    assert router.prompts == []
    assert out is merged


def test_cross_batch_duplicates_collapse_deterministically() -> None:
    """Batching splits duplicate pairs across requests, so no single call
    sees both halves. The post-merge catches the exact-match subset."""
    per_batch = [
        {"MATCH NOT FOUND": [{"LABEL": "Aspirin", "DESCRIPTION": "first"}]},
        {"MATCH NOT FOUND": [{"LABEL": "  aspirin ", "DESCRIPTION": "second"}]},
    ]
    merged = _merge_dedup_batches(per_batch, existing=set())
    assert len(merged["MATCH NOT FOUND"]) == 1
    assert merged["MATCH NOT FOUND"][0]["DESCRIPTION"] == "first", "first wins"


def test_relations_with_different_endpoints_are_not_duplicates() -> None:
    per_batch = [
        {
            "MATCH NOT FOUND RELATIONS": [
                {"LABEL": "treats", "DOMAIN": "Drug", "RANGE": "Disease"},
                {"LABEL": "treats", "DOMAIN": "Therapy", "RANGE": "Symptom"},
                {"LABEL": "treats", "DOMAIN": "Drug", "RANGE": "Disease"},  # dup
            ]
        }
    ]
    merged = _merge_dedup_batches(per_batch, existing=set())
    assert len(merged["MATCH NOT FOUND RELATIONS"]) == 2


def test_instances_collapse_on_canonical_form() -> None:
    per_batch = [
        {
            "MATCH NOT FOUND INSTANCES": [
                {"LABEL": "Jan 2004", "CANONICAL_FORM": "January 2004"},
                {"LABEL": "Jan04", "CANONICAL_FORM": "january 2004"},
            ]
        }
    ]
    merged = _merge_dedup_batches(per_batch, existing=set())
    assert len(merged["MATCH NOT FOUND INSTANCES"]) == 1


def test_existing_classes_cannot_re_enter_as_new_proposals() -> None:
    per_batch = [{"MATCH NOT FOUND": [{"LABEL": "Aspirin"}, {"LABEL": "Ibuprofen"}]}]
    merged = _merge_dedup_batches(per_batch, existing={"aspirin"})
    assert [e["LABEL"] for e in merged["MATCH NOT FOUND"]] == ["Ibuprofen"]


def test_existing_concept_names_are_local_names_not_snippets() -> None:
    names = _existing_concept_names(
        [
            {"IRI": "http://ex.org/onto#Aspirin", "TEXT_SNIPPET": "long text"},
            {"IRI": "http://ex.org/onto/Ibuprofen", "TEXT_SNIPPET": "more text"},
            {"IRI": "http://ex.org/onto#aspirin"},  # case-insensitive dup
        ]
    )
    assert names == ["Aspirin", "Ibuprofen"]
