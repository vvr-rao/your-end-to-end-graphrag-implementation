"""Enumerate ontology inputs (.owl / .rdf / .ttl / .zip) into a flat list
of (source_label, file_path, working_dir) entries that the merger can consume.

ZIPs are extracted into per-process temp directories whose lifetimes are tied
to the OntologyInputs context manager. Cross-file owl:imports inside a zip are
resolved by registering a local IRI map (existing helper) so owlready2 follows
the local copies instead of fetching from the network.
"""

from __future__ import annotations

import re
import shutil
import tempfile
import zipfile
from collections.abc import Iterator
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path

from backend.app.helpers.ontology_parsing import (
    build_local_iri_map_from_folder,
    import_ontologies,
    register_local_iri_map,
)

# Patterns used to discover hardcoded import URLs inside .owl/.rdf source files.
# We sniff these to register filename-based aliases so cross-file imports
# resolve to the local extracted copies (handles e.g. OntoCAPE's
# `file:/C:/OntoCAPE/...` imports baked into the OWL XML on a Windows machine).
_ENTITY_DECL_RE = re.compile(r'<!ENTITY\s+([\w-]+)\s+"([^"]+)">', re.IGNORECASE)
_ENTITY_REF_RE = re.compile(r'&([\w-]+);')
_IMPORTS_DIRECT_RE = re.compile(
    r'<owl:imports\s+rdf:resource\s*=\s*"([^"]+)"', re.IGNORECASE
)


def _resolve_entities(text: str) -> dict[str, str]:
    """Parse `<!ENTITY name "value">` declarations and resolve `&ref;` chains.

    Returns a name -> fully-resolved-value map. Values that still contain
    unresolved references after a few passes are dropped (cycles or external
    references we can't handle).
    """
    raw: dict[str, str] = {}
    for m in _ENTITY_DECL_RE.finditer(text):
        raw[m.group(1)] = m.group(2)
    resolved: dict[str, str] = {}
    # Up to 5 passes is plenty for any sane entity chain.
    for _ in range(5):
        progress = False
        for name, value in raw.items():
            if name in resolved:
                continue

            def _sub(match: re.Match[str]) -> str:
                ref = match.group(1)
                return resolved.get(ref, match.group(0))

            new_value = _ENTITY_REF_RE.sub(_sub, value)
            if _ENTITY_REF_RE.search(new_value) is None:
                resolved[name] = new_value
                progress = True
        if not progress:
            break
    return resolved


def _discover_imports_in_text(text: str) -> list[str]:
    """Return every fully-expanded owl:imports URL in a single ontology file."""
    entities = _resolve_entities(text)
    imports: list[str] = []
    for m in _IMPORTS_DIRECT_RE.finditer(text):
        raw_url = m.group(1)

        def _sub(match: re.Match[str]) -> str:
            ref = match.group(1)
            return entities.get(ref, match.group(0))

        expanded = _ENTITY_REF_RE.sub(_sub, raw_url)
        if _ENTITY_REF_RE.search(expanded) is None:
            imports.append(expanded)
    return imports


ONTOLOGY_SUFFIXES = (".owl", ".rdf", ".ttl", ".xml")


@dataclass
class OntologySource:
    """One ontology file ready to be parsed.

    label is a human-readable identifier (e.g. 'OCRe.zip::statistics.owl' or
    'skos.rdf') used in the manifest's provenance.
    working_dir is the directory containing the file, used to build a local
    IRI map for cross-file imports.
    """

    label: str
    path: Path
    working_dir: Path
    is_from_zip: bool = False


@dataclass
class OntologyInputs:
    """All sources produced from a list of input arguments, plus the temp dirs
    that need to live until the merge is done.

    Use as a context manager so temp dirs are cleaned up on exit.
    """

    sources: list[OntologySource] = field(default_factory=list)
    _exit_stack: ExitStack = field(default_factory=ExitStack)

    def __enter__(self) -> OntologyInputs:
        return self

    def __exit__(self, *exc) -> None:
        self._exit_stack.close()


def enumerate_inputs(inputs: list[Path]) -> OntologyInputs:
    """Walk inputs, expanding zips into temp dirs and listing flat files.

    Returns an OntologyInputs context manager — the caller MUST use it as
    `with enumerate_inputs(paths) as bundle: ...` so the temp dirs survive
    long enough for owlready2 to read them.
    """
    bundle = OntologyInputs()
    for path in inputs:
        if not path.exists():
            raise FileNotFoundError(f"Input not found: {path}")
        if path.is_dir():
            for f in sorted(path.iterdir()):
                if f.is_file() and f.suffix.lower() in ONTOLOGY_SUFFIXES:
                    bundle.sources.append(
                        OntologySource(label=str(f), path=f, working_dir=f.parent)
                    )
        elif path.suffix.lower() == ".zip":
            tmp = bundle._exit_stack.enter_context(tempfile.TemporaryDirectory(prefix="onto_zip_"))
            tmp_path = Path(tmp)
            with zipfile.ZipFile(path) as zf:
                zf.extractall(tmp_path)
            for f in sorted(tmp_path.rglob("*")):
                if f.is_file() and f.suffix.lower() in ONTOLOGY_SUFFIXES:
                    bundle.sources.append(
                        OntologySource(
                            label=f"{path.name}::{f.relative_to(tmp_path)}",
                            path=f,
                            working_dir=f.parent,
                            is_from_zip=True,
                        )
                    )
        elif path.suffix.lower() in ONTOLOGY_SUFFIXES:
            bundle.sources.append(
                OntologySource(label=path.name, path=path, working_dir=path.parent)
            )
        else:
            raise ValueError(
                f"Unsupported input type: {path} (expected .owl/.rdf/.ttl/.xml/.zip or a directory)"
            )
    if not bundle.sources:
        raise ValueError("No ontology files found in the given inputs")
    return bundle


def _basename_index(paths: list[Path]) -> dict[str, Path]:
    """Map lowercase basename -> a representative local path. If multiple files
    share a basename, the first wins (rare in practice; would log if ambiguous)."""
    out: dict[str, Path] = {}
    for p in paths:
        key = p.name.lower()
        out.setdefault(key, p)
    return out


def _build_import_url_map(working_files: list[Path]) -> dict[str, str]:
    """Sniff every ontology file for fully-expanded import URLs and map each
    one to a local file with a matching basename.

    Handles OntoCAPE-style ontologies that bake absolute Windows paths into
    their owl:imports via XML entity declarations, plus simpler ontologies
    that use HTTP IRIs directly.
    """
    basename_to_path = _basename_index(working_files)
    discovered: dict[str, str] = {}
    for f in working_files:
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for url in _discover_imports_in_text(text):
            basename = url.rsplit("/", 1)[-1].lower()
            local = basename_to_path.get(basename)
            if local is not None and url not in discovered:
                discovered[url] = str(local)
    return discovered


def _strip_unresolved_imports(working_files: list[Path], resolvable_urls: set[str]) -> int:
    """Rewrite each .owl/.rdf file in place to remove `<owl:imports>` elements
    whose targets are `file:` URLs that don't exist locally.

    Conservative scope:
      - Only `file:` URLs are eligible for stripping. HTTP(S) imports are
        left alone — owlready2 typically handles those gracefully (or our
        IRI map already redirects them).
      - Real-world ontologies often reference sibling-package files we don't
        ship (e.g. OntoCAPE's main package imports `meta_model.owl` from a
        separate package). owlready2 hard-fails on those `file:/...` paths;
        stripping them lets the rest of the load succeed.

    Returns the number of imports stripped across all files.
    """
    import_block_re = re.compile(
        r'<owl:imports\s+rdf:resource\s*=\s*"([^"]+)"\s*/>',
        re.IGNORECASE,
    )
    stripped = 0
    for f in working_files:
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        entities = _resolve_entities(text)

        def _maybe_strip(match: re.Match[str], _entities: dict[str, str] = entities) -> str:
            nonlocal stripped
            raw_url = match.group(1)

            def _sub(m: re.Match[str]) -> str:
                ref = m.group(1)
                return _entities.get(ref, m.group(0))

            expanded = _ENTITY_REF_RE.sub(_sub, raw_url)
            # Only consider stripping local file: URLs. Leave HTTP(S) alone.
            if not expanded.lower().startswith("file:"):
                return match.group(0)
            if expanded in resolvable_urls or raw_url in resolvable_urls:
                return match.group(0)
            stripped += 1
            return "<!-- import stripped (unresolvable): " + raw_url + " -->"

        new_text = import_block_re.sub(_maybe_strip, text)
        if new_text != text:
            try:
                f.write_text(new_text, encoding="utf-8")
            except OSError:
                continue
    return stripped


def _find_root_files(working_files: list[Path]) -> list[Path]:
    """Identify entry-point files: ontology files that aren't imported by any
    other file in the same set.

    Loading each file individually is O(N²) when imports nest deeply — for
    OntoCAPE (64 files with extensive cross-imports) this is prohibitive.
    Instead, find files that nothing else imports and load just those;
    owlready2's import walker pulls in the rest in one pass per root.
    """
    basename_to_path = _basename_index(working_files)
    imported_basenames: set[str] = set()
    for f in working_files:
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for url in _discover_imports_in_text(text):
            basename = url.rsplit("/", 1)[-1].lower()
            if basename in basename_to_path:
                imported_basenames.add(basename)
    roots = [p for p in working_files if p.name.lower() not in imported_basenames]
    # If sniffing found nothing (e.g. .ttl files we don't parse, or imports
    # we missed), fall back to loading every file. Better slow than wrong.
    if not roots:
        return list(working_files)
    return roots


def load_ontology_files(sources: list[OntologySource]) -> dict[str, dict]:
    """Run the existing extract_ontology_to_dicts() against each source,
    registering local IRI maps so cross-file imports resolve to the unzipped
    siblings rather than network URLs or hardcoded Windows paths.

    Returns the canonical loaded ontology dict-of-dicts:
        {classes_dict, object_properties_dict, data_properties_dict, instances_dict}
    """
    if not sources:
        raise ValueError("No sources to load")

    # Base IRI map: filename-based aliases inside each working dir for HTTP-style
    # imports (covers OCRe-style http://purl.org/net/OCRe/<filename>.owl).
    working_dirs = {s.working_dir for s in sources}
    full_iri_map: dict[str, str] = {}
    for d in working_dirs:
        try:
            full_iri_map.update(build_local_iri_map_from_folder(str(d)))
        except FileNotFoundError:
            pass

    # Extra: sniff hardcoded file:/... import URLs (OntoCAPE-style) and add to
    # the map. These won't be discovered by build_local_iri_map_from_folder.
    all_files = [s.path for s in sources]
    full_iri_map.update(_build_import_url_map(all_files))

    if full_iri_map:
        register_local_iri_map(full_iri_map)

    # Strip imports that reference files outside the local set (sibling
    # packages we don't have copies of). owlready2 hard-fails on missing
    # imports; this makes the load resilient. Only modifies temp-extracted
    # files — never touches user sources.
    files_in_temp = [
        s.path for s in sources if s.is_from_zip or str(s.path).startswith(tempfile.gettempdir())
    ]
    if files_in_temp:
        n_stripped = _strip_unresolved_imports(files_in_temp, set(full_iri_map.keys()))
        if n_stripped:
            print(f"[ontology_io] stripped {n_stripped} unresolvable import(s) from extracted files")

    # When sources come from a zip with many cross-imports (e.g. OntoCAPE's 64
    # files), loading every file individually is O(N²) because each load
    # transitively re-parses imports. Identify root files (not imported by any
    # other source in the same set) and only load those — owlready2's import
    # walker pulls in the rest in a single pass per root.
    root_paths = set(_find_root_files(all_files))
    effective_sources = [s for s in sources if s.path in root_paths]
    if len(effective_sources) < len(sources):
        print(
            f"[ontology_io] loading {len(effective_sources)} root files "
            f"(out of {len(sources)} total — others pulled in transitively)"
        )

    # Compose the dict that import_ontologies() expects.
    owl_dict = {}
    for s in effective_sources:
        owl_dict[s.label] = {
            "filename": str(s.path),
            "load_imported": True,
            "local_only": True,
            "local_ontology_dir": str(s.working_dir),
            "iri_map": full_iri_map,
        }
    return import_ontologies(owl_dict)


def iter_documents(documents_dir: Path) -> Iterator[Path]:
    """Walk a documents folder, returning paths of .pdf and .txt files."""
    if not documents_dir.exists():
        raise FileNotFoundError(f"Documents folder not found: {documents_dir}")
    if not documents_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {documents_dir}")
    for f in sorted(documents_dir.rglob("*")):
        if f.is_file() and f.suffix.lower() in (".pdf", ".txt"):
            yield f


def copy_to_version(src: Path, dst_dir: Path) -> Path:
    """Copy a file into a version dir (used to attach inputs to a snapshot)."""
    dst = dst_dir / src.name
    shutil.copy2(src, dst)
    return dst
