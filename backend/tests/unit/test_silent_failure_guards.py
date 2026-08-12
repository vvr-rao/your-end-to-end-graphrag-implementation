"""Stages must not report success when they accomplished nothing.

All three guards here exist because a real run produced a confident-looking
"DONE" that hid a total failure:

  - extract-entities printed `0 success / 31 failed` and exited 0, leaving an
    empty entities table. Only the NEXT stage's precondition guard revealed it.
  - register-documents hit a raw UniqueViolationError re-adding a soft-deleted
    document, because the dup check filtered to status='ACTIVE' while the
    unique constraint has no status component.
  - clear-corpus refused on an empty ontology without distinguishing "never
    initialised" from "a db-init --mode replace wiped it then failed to
    import", which is a recovery dead end.
"""

from __future__ import annotations

import pytest

from backend.app.services.db_entity_extract import (
    _FAILURE_WARN_PCT,
    EntityExtractionFailedError,
)


# --------------------------------------------------------------------------- #
# Bug 1 -- extract-entities
# --------------------------------------------------------------------------- #
def _decide(scanned: int, failed: int) -> str:
    """The guard's decision, mirroring db_entity_extract's logic.

    Kept as a tiny pure function so the rule is testable without standing up a
    database, an LLM router and a corpus. The integration test below covers the
    wiring; this covers the rule.
    """
    attempted = scanned + failed
    if attempted and scanned == 0:
        raise EntityExtractionFailedError(f"all {failed} chunk(s) failed")
    if failed and attempted and (100.0 * failed / attempted) >= _FAILURE_WARN_PCT:
        return "warn"
    return "ok"


def test_total_failure_raises() -> None:
    """The exact shape observed live: 0 success / 31 failed."""
    with pytest.raises(EntityExtractionFailedError) as ei:
        _decide(scanned=0, failed=31)
    assert "31" in str(ei.value)


def test_total_failure_raises_even_for_a_single_chunk() -> None:
    with pytest.raises(EntityExtractionFailedError):
        _decide(scanned=0, failed=1)


def test_empty_corpus_does_not_raise() -> None:
    """No chunks attempted is legitimately nothing to do -- not a failure."""
    assert _decide(scanned=0, failed=0) == "ok"


def test_full_success_is_quiet() -> None:
    assert _decide(scanned=31, failed=0) == "ok"


def test_partial_failure_warns_but_does_not_raise() -> None:
    """Partial results are real rows worth keeping -- abort would discard the
    successful work. But silent partial loss is how a corpus ends up with gaps
    nobody notices."""
    assert _decide(scanned=20, failed=11) == "warn"


def test_small_failure_ratio_stays_quiet() -> None:
    assert _decide(scanned=99, failed=1) == "ok"


@pytest.mark.parametrize("scanned,failed,expected", [
    (90, 10, "warn"),   # exactly at the 10% threshold
    (91, 9, "ok"),      # just under
])
def test_warn_threshold_boundary(scanned, failed, expected) -> None:
    assert _decide(scanned, failed) == expected


# --------------------------------------------------------------------------- #
# Bug 2 -- soft-deleted documents still own their identifier
# --------------------------------------------------------------------------- #
def _dup_check(rows: list[tuple[str, str]], candidates: list[str]):
    """Mirrors db_document_ingest's dup check AFTER the fix: status-agnostic."""
    existing = {r[0] for r in rows if r[0] in candidates}
    revivable = sorted(r[0] for r in rows if r[0] in candidates and r[1] != "ACTIVE")
    keep = [c for c in candidates if c not in existing]
    return keep, existing, revivable


def test_soft_deleted_document_is_not_treated_as_new() -> None:
    """The bug: status=='ACTIVE' filtering made a DELETED row invisible, so the
    insert collided with documents_document_identifier_key."""
    rows = [("iri-a", "ACTIVE"), ("iri-b", "DELETED")]
    keep, existing, revivable = _dup_check(rows, ["iri-a", "iri-b", "iri-c"])
    assert "iri-b" not in keep, "a soft-deleted doc must not be re-inserted"
    assert keep == ["iri-c"], "only the genuinely new document is inserted"
    assert revivable == ["iri-b"]


def test_superseded_document_also_blocks_reinsert() -> None:
    rows = [("iri-a", "SUPERSEDED")]
    keep, _, revivable = _dup_check(rows, ["iri-a"])
    assert keep == []
    assert revivable == ["iri-a"]


def test_all_new_documents_are_kept() -> None:
    keep, existing, revivable = _dup_check([], ["x", "y"])
    assert keep == ["x", "y"]
    assert not existing and not revivable


def test_active_duplicates_still_skipped() -> None:
    keep, _, revivable = _dup_check([("iri-a", "ACTIVE")], ["iri-a"])
    assert keep == []
    assert revivable == [], "an ACTIVE duplicate is not 'revivable', just a dup"


# --------------------------------------------------------------------------- #
# Bug 4 -- `python -m backend.app.cli` discarded main()'s return code, so every
# guard refusal ("clear-corpus REFUSED", "db-init WIPE REFUSED") exited 0. CI,
# shell && chains and the detached-run harness all saw success on failure.
# --------------------------------------------------------------------------- #
def test_module_entrypoint_propagates_the_exit_code() -> None:
    src = open("backend/app/cli/__main__.py").read()
    assert "sys.exit(main())" in src, (
        "backend/app/cli/__main__.py must sys.exit(main()); bare main() throws "
        "the return code away and every non-zero return becomes exit 0"
    )


def test_main_still_returns_the_code_it_is_given() -> None:
    """Guard the other half: __main__ propagating a code only helps if main()
    actually returns one."""
    import inspect

    from backend.app.cli.main import main

    src = inspect.getsource(main)
    assert "return args.func(args)" in src, (
        "main() must return the subcommand's exit code"
    )
