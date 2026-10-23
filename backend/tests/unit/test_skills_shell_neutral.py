"""Every command in the Claude Code skills must run unchanged in bash, zsh and
PowerShell (Claude Code on Windows uses Git Bash or its PowerShell tool).

This is the 2026-08-01 shell-neutral audit (commit 2424ef2) turned into a
test: it was a one-off then, and by 2026-10-22 three bash-only commands had
crept back in (a `grep` over a log, a `VAR=value cmd` prefix, a `\\`
continuation). Checked: fenced code blocks and inline `code` spans of every
`.claude/skills/*/SKILL.md`. SQL blocks are exempt.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SKILLS = sorted((REPO / ".claude" / "skills").glob("*/SKILL.md"))

RULES = {
    "$(...) capture": r"\$\(",
    "| tail / | head": r"\|\s*(tail|head)\b",
    "backslash line continuation": r"\\\s*$",
    "[ -f ] test": r"\[\s+-[fdex]\s",
    "$VAR expansion": r"\$[A-Z_]{2,}\b",
    "VAR=value command prefix": r"(^|\s)[A-Z][A-Z0-9_]{2,}=(<[^>]+>|\S+)\s+(uv|python|python3)\b",
    "bash-only tool": r"(^|[\s|;(])(grep|sed|awk|cat|wc|xargs)\s",
    "&& chaining": r"&&",
}


def _commands(text: str):
    """(line number, command text) for code-block lines and inline spans."""
    lang = None
    for n, line in enumerate(text.splitlines(), 1):
        fence = re.match(r"\s*```(\w*)", line)
        if fence:
            lang = None if lang is not None else fence.group(1).lower()
            continue
        if lang is not None:
            if lang != "sql":
                yield n, line
            continue
        for span in re.findall(r"`([^`]+)`", line):
            yield n, span


@pytest.mark.parametrize("skill", SKILLS, ids=lambda p: p.parent.name)
def test_skill_commands_are_shell_neutral(skill):
    bad = [
        f"{skill.parent.name}:{n} [{name}] {cmd.strip()[:100]}"
        for n, cmd in _commands(skill.read_text(encoding="utf-8"))
        for name, rx in RULES.items()
        if re.search(rx, cmd)
    ]
    assert not bad, "bash-only syntax in a skill command:\n" + "\n".join(bad)


def test_the_rules_catch_what_they_are_for():
    samples = {
        "$(...) capture": "X=$(uv run python a.py)",
        "| tail / | head": "uv run python a.py | tail -1",
        "backslash line continuation": "uv run python a.py \\",
        "VAR=value command prefix": "MODELS_CONFIG_PATH=<that file> uv run python -m x",
        "bash-only tool": 'grep -n "alias refresh failed" log',
        "&& chaining": "uv run a && uv run b",
    }
    for name, cmd in samples.items():
        assert re.search(RULES[name], cmd), name
    assert SKILLS, "no skills found"
