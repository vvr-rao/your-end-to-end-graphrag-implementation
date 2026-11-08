"""Provider modules. Importing one registers it in ``structured._base.SOURCES``."""

from __future__ import annotations

import importlib

MODULES = (
    "sec", "fred", "gleif", "ffiec",
    "clinicaltrials", "drugsfda", "orangebook", "chembl", "opentargets",
)


def load_all() -> None:
    for name in MODULES:
        importlib.import_module(f"{__name__}.{name}")
