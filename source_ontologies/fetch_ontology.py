"""Fetch a supported domain ontology into source_ontologies/downloaded/<domain>/.

Supported domains: pharma (OCRe), finance (FIBO), manufacturing (OntoCAPE).

Upstream availability was verified live on 2026-07-21 — and it is uneven, which
is exactly why this goes through a registry instead of hardcoded URLs:

  - FIBO (finance): a real, direct, unauthenticated download works —
    https://spec.edmcouncil.org/fibo/ontology/master/latest/prod.rdf.zip
    (verified 200 OK, application/zip, ~1.5 MB of FIBO production RDF/XML).

  - OCRe (pharma): the original Google Code download is OFFLINE; the PURL
    (purl.org/net/OCRe) redirects to a BioPortal *page*, and the actual file
    download needs a free BioPortal API key. With BIOPORTAL_API_KEY set we fetch
    it for real; otherwise we fall back to the vendored zip.

  - OntoCAPE (manufacturing): RWTH Aachen gates the download behind a
    questionnaire form — there is no direct URL. It is NOT bundled in the repo
    (GPL + form-gated), so the user must supply it themselves; this script prints
    where to place the zip.

OCRe and FIBO are vendored in the repo (source_ontologies/{pharma,finance}_*/),
so a clean clone can always build those two domains even when an upstream is down
or gated. Downloaded files land under source_ontologies/downloaded/<domain>/
(gitignored).

Usage:
    python source_ontologies/fetch_ontology.py finance
    python source_ontologies/fetch_ontology.py pharma        # vendored unless BIOPORTAL_API_KEY
    python source_ontologies/fetch_ontology.py manufacturing  # vendored (form-gated upstream)

Prints the path to the ontology file ready to pass to `merge --ontology`.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from download_ontology import download_ontology  # noqa: E402

HERE = Path(__file__).parent
DOWNLOADED = HERE / "downloaded"

REGISTRY: dict[str, dict] = {
    "finance": {
        "name": "FIBO",
        "method": "direct",
        # Verified 2026-07-21: 200 OK, application/zip. merge accepts .zip directly.
        "url": "https://spec.edmcouncil.org/fibo/ontology/master/latest/prod.rdf.zip",
        "vendored": "finance_ontologies/prod.rdf.zip",
    },
    "pharma": {
        "name": "OCRe",
        "method": "bioportal",
        "bioportal_acronym": "OCRE",
        "vendored": "pharma_ontologies/OCRe.zip",
        "note": ("Upstream Google Code is offline; the OCRe PURL redirects to a "
                 "BioPortal page. A real download needs a free BioPortal API key "
                 "(set BIOPORTAL_API_KEY in .env)."),
    },
    "manufacturing": {
        "name": "OntoCAPE",
        "method": "gated",
        "form_url": "https://www.avt.rwth-aachen.de/cms/avt/forschung/sonstiges/software/~ipts/ontocape/?lidx=1",
        "vendored": "manufacturing_supplychain_ontologies/OntoCAPE_domain+ontology.zip",
        "note": ("RWTH Aachen requires filling out a questionnaire before download; "
                 "there is no direct URL."),
    },
}

# Friendly aliases so callers can say what they mean.
ALIASES = {
    "fibo": "finance", "financial": "finance",
    "ocre": "pharma", "clinical": "pharma", "pharmaceutical": "pharma",
    "ontocape": "manufacturing", "manufacturing_supplychain": "manufacturing",
    "supplychain": "manufacturing", "supply_chain": "manufacturing",
}


def _use_vendored(cfg: dict, dest: Path) -> Path:
    src = HERE / cfg["vendored"]
    if not src.exists():
        # No live download AND no local copy. Give the user an actionable path
        # instead of a bare traceback -- this is the expected state for OntoCAPE,
        # which we deliberately do not ship (GPL + RWTH form-gated).
        if cfg.get("form_url"):
            hint = (f"It is not bundled in this repo. Request it from "
                    f"{cfg['form_url']}, then place the zip at:\n    {src}\nand re-run.")
        elif cfg.get("bioportal_acronym"):
            hint = (f"It is not bundled here. Set BIOPORTAL_API_KEY in .env to "
                    f"download it, or place a copy at:\n    {src}\nand re-run.")
        else:
            hint = f"Place a copy at:\n    {src}\nand re-run."
        raise SystemExit(
            f"[fetch] {cfg['name']}: no source available -- no live download and "
            f"no local copy.\n{hint}")
    out = dest / src.name
    shutil.copy2(src, out)
    print(f"[fetch] {cfg['name']}: using vendored copy -> {out}")
    return out


def fetch(domain: str, dest_root: Path | None = None, apikey: str | None = None) -> Path:
    """Fetch the ontology for `domain`; return the path to the ready-to-merge file.

    Downloads for real where a live source exists (FIBO always; OCRe when a
    BioPortal key is available), otherwise copies the vendored zip. Never raises
    on a dead/gated upstream as long as the vendored fallback is present.
    """
    key = ALIASES.get(domain, domain)
    if key not in REGISTRY:
        raise KeyError(f"unknown domain '{domain}'. Known: {', '.join(REGISTRY)} "
                       f"(aliases: {', '.join(ALIASES)})")
    cfg = REGISTRY[key]
    dest = (dest_root or DOWNLOADED) / key
    dest.mkdir(parents=True, exist_ok=True)

    method = cfg["method"]
    if method == "direct":
        try:
            out = download_ontology(cfg["url"], str(dest))
            print(f"[fetch] {cfg['name']}: downloaded {cfg['url']} -> {out}")
            return out
        except Exception as exc:  # network down / URL moved -> vendored
            print(f"[fetch] {cfg['name']}: live download failed ({exc}); falling back.")
            return _use_vendored(cfg, dest)

    if method == "bioportal":
        key_val = apikey or os.environ.get("BIOPORTAL_API_KEY")
        if key_val:
            # Never print the key.
            url = (f"https://data.bioontology.org/ontologies/{cfg['bioportal_acronym']}"
                   f"/download?apikey={key_val}&download_format=rdf")
            try:
                out = download_ontology(url, str(dest), filename=f"{cfg['name']}.rdf")
                print(f"[fetch] {cfg['name']}: downloaded from BioPortal -> {out}")
                return out
            except Exception as exc:
                print(f"[fetch] {cfg['name']}: BioPortal download failed ({exc}); falling back.")
                return _use_vendored(cfg, dest)
        print(f"[fetch] {cfg['name']}: no BIOPORTAL_API_KEY set. {cfg['note']}")
        return _use_vendored(cfg, dest)

    if method == "gated":
        print(f"[fetch] {cfg['name']}: {cfg['note']}")
        print(f"[fetch] {cfg['name']}: to refresh, request it at {cfg['form_url']}")
        return _use_vendored(cfg, dest)

    raise ValueError(f"unknown method '{method}' for {cfg['name']}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch a supported domain ontology.")
    ap.add_argument("domain", help="pharma | finance | manufacturing (aliases accepted)")
    ap.add_argument("--dest-root", type=Path, default=None,
                    help="Override the download root (default: source_ontologies/downloaded/).")
    args = ap.parse_args()
    path = fetch(args.domain, dest_root=args.dest_root)
    # Final line is JUST the path, so callers can capture it.
    print(path)


if __name__ == "__main__":
    main()
