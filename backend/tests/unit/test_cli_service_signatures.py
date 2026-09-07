"""CLI call sites must match the service signatures they invoke.

`conversation turn` was DEAD: the parser advertised --rerank, the CLI passed
`rerank=` to `add_turn`, and `add_turn` had no such parameter. Every
invocation raised

    TypeError: add_turn() got an unexpected keyword argument 'rerank'

before doing any work, so follow-up resolution -- the whole point of the
conversation surface -- could never have run. The flag had been wired at both
ends and not in the middle, and nothing caught it because no test and no
earlier manual run had ever invoked the command.

These tests compare each CLI call site's keyword arguments against the
service function's real signature, so a parameter added to one side and not
the other fails here instead of at the user's first attempt.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

_CLI = Path(__file__).resolve().parents[2] / "app" / "cli" / "main.py"


def _kwargs_passed_to(func_name: str) -> set[str]:
    """Keyword names the CLI passes to `func_name`, found by parsing the AST
    rather than by importing -- the call sites sit inside command handlers
    that would need a live DB to execute."""
    tree = ast.parse(_CLI.read_text())
    found: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = getattr(fn, "id", None) or getattr(fn, "attr", None)
        if name != func_name:
            continue
        found |= {kw.arg for kw in node.keywords if kw.arg}
    return found


@pytest.mark.parametrize(
    "module_path,func_name",
    [
        ("backend.app.services.db_conversation", "add_turn"),
        ("backend.app.services.db_entity_extract", "extract_entities"),
        ("backend.app.services.retrieval", "retrieve_and_answer"),
    ],
)
def test_cli_passes_only_arguments_the_service_accepts(
    module_path: str, func_name: str
) -> None:
    import importlib

    fn = getattr(importlib.import_module(module_path), func_name)
    accepted = set(inspect.signature(fn).parameters)
    passed = _kwargs_passed_to(func_name)
    if not passed:
        pytest.skip(f"no CLI call site for {func_name} found")
    unknown = passed - accepted
    assert not unknown, (
        f"{_CLI.name} passes {sorted(unknown)} to {func_name}(), which does "
        f"not accept them. The command will raise TypeError before doing any "
        f"work. Accepted: {sorted(accepted)}"
    )


def test_add_turn_forwards_rerank_to_retrieval() -> None:
    """Accepting the argument is not enough -- it has to reach retrieval, or
    --rerank silently does nothing."""
    from backend.app.services import db_conversation

    src = inspect.getsource(db_conversation.add_turn)
    assert "rerank=rerank" in src, (
        "add_turn accepts `rerank` but never forwards it to "
        "retrieve_and_answer; the flag would be silently ignored"
    )
