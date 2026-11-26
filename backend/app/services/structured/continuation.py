"""Id-driven continuation for LLM stages whose output can exceed one reply.

A stage has a list of WORK IDS (source columns, candidate quantities). Each
call asks for a batch of ids and must return one record per id. The engine:

* keeps every VALIDATED, COMPLETE record of a reply (a truncated reply is
  salvaged entry by entry; a partial record is never published);
* asks again only for the ids still missing or invalid;
* sizes batches from observed throughput: after repeated truncation the
  batch shrinks, after comfortable replies it may grow back;
* stops on a per-stage call budget, an output-token budget, or when calls
  stop making progress -- and says which, per id;
* checkpoints after every call (``on_progress``), so a resumed run only does
  what is left (``done`` seeds the engine).

Statuses of ids not completed (see :data:`INCOMPLETE`):
``incomplete_due_to_budget`` (call or token budget spent),
``incomplete_due_to_truncation`` (replies kept being cut off, even at the
smallest batch), ``invalid_reply`` (the model kept returning unusable
records for it). Nothing here knows what the records mean.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from typing import Any

INCOMPLETE = ("incomplete_due_to_budget", "incomplete_due_to_truncation", "invalid_reply")


@dataclass
class StageStats:
    """Per-stage accounting (one per stage per table)."""

    stage: str
    calls: int = 0
    retries: int = 0                     # provider-level retried attempts
    requested_cap: int | None = None
    effective_cap: int | None = None
    output_tokens: int = 0               # provider completion tokens (reasoning INCLUDED)
    reasoning_tokens: int = 0            # subset of output_tokens, when reported
    finish_reasons: dict[str, int] = field(default_factory=dict)
    truncated_calls: int = 0
    batch_sizes: list[int] = field(default_factory=list)
    batches: list[list[str]] = field(default_factory=list)   # the ids each call asked for
    resumed_ids: int = 0                 # completed before this run (checkpoint)
    ids_total: int = 0
    ids_done: int = 0
    remaining: int = 0
    stop_reason: str | None = None
    seconds: float = 0.0

    def add_call(self, res: Any, batch: int) -> None:
        self.calls += 1
        self.batch_sizes.append(batch)
        self.output_tokens += int(getattr(res, "completion_tokens", 0) or 0)
        self.reasoning_tokens += int(getattr(res, "reasoning_tokens", 0) or 0)
        fr = str(getattr(res, "finish_reason", None))
        self.finish_reasons[fr] = self.finish_reasons.get(fr, 0) + 1
        att = getattr(res, "attempts", None)
        if isinstance(att, int) and att > 1:
            self.retries += att - 1
        for k, attr in (("requested_cap", "requested_max_tokens"), ("effective_cap", "effective_max_tokens")):
            v = getattr(res, attr, None)
            if isinstance(v, int):
                setattr(self, k, max(getattr(self, k) or 0, v))

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Budget:
    max_calls: int = 40                  # per table and stage
    max_output_tokens: int = 1_500_000   # per table and stage (reasoning included)
    no_progress_calls: int = 2           # consecutive calls that complete nothing new
    max_attempts_per_id: int = 3         # an id the model keeps getting wrong
    min_batch: int = 1
    max_batch: int | None = None


def truncated(res: Any) -> bool:
    return str(getattr(res, "finish_reason", "") or "").lower() in {"length", "max_tokens"}


def parse_reply(text: str) -> tuple[dict, bool]:
    """(object, salvaged): the reply as JSON; a cut-off reply is salvaged to its
    complete entries (``json_salvage``), flagged so callers can tell."""
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", (text or "").strip())
    try:
        obj = json.loads(t)
        return (obj if isinstance(obj, dict) else {}), False
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", t, re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj, False
        except json.JSONDecodeError:
            pass
    from backend.app.helpers.json_salvage import salvage_truncated_json

    return (salvage_truncated_json(t) or {}), True


def continuation_settings(router, task: str, default_batch: int) -> tuple[int, Any]:
    """(batch size, Budget) from the task's ``continuation:`` block in
    models.yaml (``batch``, ``max_calls``, ``max_output_tokens``,
    ``no_progress_calls``); defaults otherwise."""
    spec = None
    try:
        spec = router.task_spec(task) if hasattr(router, "task_spec") else None
    except Exception:
        spec = None
    c = (spec or {}).get("continuation") if isinstance(spec, dict) else getattr(spec, "continuation", None)
    c = c if isinstance(c, dict) else {}
    b = Budget()
    for k in ("max_calls", "max_output_tokens", "no_progress_calls", "max_attempts_per_id", "max_batch"):
        if isinstance(c.get(k), int) and c[k] > 0:
            setattr(b, k, c[k])
    return int(c.get("batch") or default_batch), b


CallFn = Callable[[list[str], int], Awaitable[tuple[list[dict], Any]]]
ValidateFn = Callable[[str, dict], str | None]


async def run_by_ids(stage: str, ids: list[str], call: CallFn, *, validate: ValidateFn,
                     batch_size: int, budget: Budget | None = None,
                     done: dict[str, dict] | None = None,
                     on_progress: Callable[[dict[str, dict]], None] | None = None,
                     key: str = "id",
                     log: Callable[[str], None] = lambda *_: None) -> tuple[dict[str, dict], dict[str, str], StageStats]:
    """Complete one record per id. ``call(batch_ids, attempt_no)`` returns
    (records, chat result); ``validate(id, record)`` returns an error or None.

    Returns (records by id, {id: incomplete status} for ids not completed,
    stats). ``done`` (a checkpoint) is not asked again."""
    b = budget or Budget()
    t0 = time.monotonic()
    out: dict[str, dict] = dict(done or {})
    order = list(dict.fromkeys(ids))
    queue = [i for i in order if i not in out]
    st = StageStats(stage=stage, ids_total=len(order), resumed_ids=sum(1 for i in order if i in out))
    if st.resumed_ids:
        log(f"[structured] {stage}: resuming -- {st.resumed_ids}/{len(order)} id(s) done in the checkpoint, "
            f"{len(queue)} left")
    attempts: dict[str, int] = {}
    failed: dict[str, str] = {}
    size = max(b.min_batch, batch_size)
    cap_size = b.max_batch or max(batch_size, 1) * 2
    trunc_streak = no_progress = 0
    while queue:
        if st.calls >= b.max_calls or st.output_tokens >= b.max_output_tokens:
            st.stop_reason = "budget"
            for i in queue:
                failed[i] = "incomplete_due_to_budget"
            break
        batch = queue[:size]
        records, res = await call(batch, st.calls + 1)
        st.add_call(res, len(batch))
        st.batches.append(list(batch))
        cut = truncated(res)
        got = 0
        wanted = set(batch)
        for r in records or []:
            if not isinstance(r, dict):
                continue
            rid = str(r.get(key) or "")
            if rid not in wanted or rid in out:
                continue
            err = validate(rid, r)
            if err:
                continue
            out[rid] = r
            got += 1
        missing = [i for i in batch if i not in out]
        # a cut-off reply only reached the record it was writing: ids after it
        # were never attempted and are not charged an attempt
        for i in (missing[:1] if cut else missing):
            attempts[i] = attempts.get(i, 0) + 1
        hopeless = [i for i in missing if attempts.get(i, 0) >= b.max_attempts_per_id]
        for i in hopeless:
            failed[i] = "incomplete_due_to_truncation" if cut else "invalid_reply"
        queue = [i for i in missing if i not in hopeless] + queue[len(batch):]
        if cut:
            st.truncated_calls += 1
            trunc_streak += 1
            if trunc_streak >= 2 and size > b.min_batch:
                size = max(b.min_batch, size // 2)          # shrink after repeated truncation
                log(f"[structured] {stage}: replies cut off twice -> batch {size}")
                trunc_streak = 0
        else:
            trunc_streak = 0
            eff = getattr(res, "effective_max_tokens", None) or 0
            used = getattr(res, "completion_tokens", None) or 0
            if got == len(batch) and eff and used < 0.4 * eff and size < cap_size:
                size = min(cap_size, size + max(1, size // 2))   # comfortable: grow
        log(f"[structured] {stage}: call {st.calls} asked {len(batch)} id(s) [{batch[0]} .. {batch[-1]}] -> "
            f"{got} complete, finish={getattr(res, 'finish_reason', None)}, "
            f"out={getattr(res, 'completion_tokens', None)} tok (reasoning "
            f"{getattr(res, 'reasoning_tokens', None)}), cap={getattr(res, 'effective_max_tokens', None)}; "
            f"{len(queue)} id(s) left")
        no_progress = 0 if got else no_progress + 1
        if on_progress is not None and got:
            on_progress(out)
        if no_progress >= b.no_progress_calls:
            st.stop_reason = "no_progress"
            for i in queue:
                failed[i] = "incomplete_due_to_truncation" if cut else "invalid_reply"
            break
    st.ids_done = sum(1 for i in order if i in out)
    st.remaining = len(order) - st.ids_done
    st.stop_reason = st.stop_reason or ("complete" if not st.remaining else "attempts")
    st.seconds = round(time.monotonic() - t0, 1)
    return {i: out[i] for i in order if i in out}, failed, st
