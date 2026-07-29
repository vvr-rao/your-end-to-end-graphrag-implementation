#!/usr/bin/env python3
"""Select an LLM provider mode for the app: scaffold configs from the tracked
examples, copy the matching models preset into config/models.yaml, presence-check
that mode's required keys, and record the step. Cross-platform.

    uv run python scripts/select_llm_mode.py groq-openai
    uv run python scripts/select_llm_mode.py openai
    uv run python scripts/select_llm_mode.py anthropic

Never reads or prints key VALUES — only whether each required key is present in
.env (which the user fills in themselves).
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _skillutil import ROOT, env_present  # noqa: E402
import build_state  # noqa: E402

# mode -> (models preset relative to config/, required .env keys)
MODES: dict[str, tuple[str, list[str]]] = {
    "groq-openai": ("models.example.yaml", ["OPENAI_API_KEY", "GROQ_API_KEY"]),
    "openai": ("models.openai.example.yaml", ["OPENAI_API_KEY"]),
    "anthropic": ("models.anthropic.example.yaml", ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]),
}


def _scaffold(src_rel: str, dst_rel: str) -> None:
    src, dst = ROOT / src_rel, ROOT / dst_rel
    if not dst.exists() and src.exists():
        shutil.copyfile(src, dst)
        print(f"  scaffolded {dst_rel} from {src_rel}")


def main(argv: list[str]) -> int:
    if len(argv) != 1 or argv[0] not in MODES:
        print(f"usage: select_llm_mode.py <{' | '.join(MODES)}>", file=sys.stderr)
        return 2
    mode = argv[0]
    preset, required = MODES[mode]

    # 1. Scaffold .env + config from the tracked templates (no secrets in templates).
    _scaffold(".env.example", ".env")
    _scaffold("config/config.example.yaml", "config/config.yaml")

    # 2. Activate the preset.
    src = ROOT / "config" / preset
    if not src.exists():
        print(f"ERROR: preset not found: config/{preset}", file=sys.stderr)
        return 1
    shutil.copyfile(src, ROOT / "config" / "models.yaml")
    print(f"  activated LLM mode '{mode}' -> config/models.yaml (from {preset})")

    # 3. Presence-check required keys (values never shown). Embeddings always need OpenAI.
    missing = [k for k in required if not env_present(k)]
    for k in required:
        print(f"  {k}: {'present' if k not in missing else 'MISSING'}")

    # 4. Record for the cross-session tracker.
    build_state.record_step("llm-mode", {"mode": mode})

    if missing:
        print(f"\n  Add these to .env before running paid steps: {', '.join(missing)} "
              f"(edit .env yourself; never paste keys in chat).")
        return 1
    print("  all required keys present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
