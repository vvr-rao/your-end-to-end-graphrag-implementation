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

from owlready2 import PREDEFINED_ONTOLOGIES

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


ONTOLOGY_SUFFIXES = (".owl", ".rdf", ".ttl")


def _convert_ttl_to_rdfxml_bytes(ttl_path: Path) -> bytes:
    """Read a Turtle (.ttl) ontology file and return its RDF/XML
    serialization as bytes via rdflib.

    Why: owlready2's built-in Turtle reader is fragile. On the W3C
    `time.ttl` (OWL Time ontology) it raises `OwlReadyOntologyParsingError:
    NTriples parsing error (or unrecognized file format)`. rdflib's
    Turtle parser handles the same file cleanly; converting to RDF/XML
    up-front lets owlready2 consume the file via its much more robust
    RDF/XML backend.

    We ALSO strip rdfs:subClassOf / owl:equivalentClass triples whose
    object is a blank node (anonymous owl:Restriction or
    owl:intersectionOf expressions). owlready2's `_find_base_classes`
    machinery raises `TypeError: issubclass() arg 1 must be a class`
    when one of the "base classes" is an anonymous expression. The
    W3C ORG ontology (`org.ttl`) trips this on every load. Stripping
    the offending axioms preserves the NAMED class declarations +
    labels + comments + named parent IRIs (everything downstream Stage
    2 needs as a slice anchor) while dropping the equivalence /
    restriction detail. Acceptable trade-off: the LLM doesn't consume
    OWL axioms anyway.

    rdflib.Graph holds onto significant RAM that doesn't release back
    to the OS automatically; on tight memory budgets (the user's 2.7 GiB
    dev box) every MB matters, so we explicitly drop the Graph and
    force a GC pass before returning.
    """
    import gc

    import rdflib
    from rdflib import BNode, OWL, RDFS

    g = rdflib.Graph()
    g.parse(str(ttl_path), format="turtle")

    # Drop axiom patterns that owlready2's class loader can't model:
    #   - rdfs:subClassOf / owl:equivalentClass pointing at a blank
    #     node (anonymous Restriction / intersection / union).
    #   - owl:equivalentClass pointing at a NAMED class outside this
    #     file's namespace (owlready2 can't resolve the cross-onto
    #     reference at load time and fills it with a non-class
    #     placeholder, then issubclass() blows up).
    #   - rdfs:domain / rdfs:range pointing at a blank node (anonymous
    #     union/intersection of class IRIs).
    #   - owl:hasKey / owl:propertyChainAxiom (collection-based axioms
    #     that owlready2 sometimes mishandles).
    # We preserve the NAMED class declarations + labels + comments +
    # NAMED-class parent IRIs. The downstream Stage 2 LLM only consumes
    # those fields for its ontology slice anchor.
    stripped = 0
    own_ns = None
    # Try to identify the file's own namespace from owl:Ontology
    for s in g.subjects(rdflib.RDF.type, OWL.Ontology):
        own_ns = str(s).rstrip("#/")
        break

    for s, p, o in list(g.triples((None, RDFS.subClassOf, None))):
        if isinstance(o, BNode):
            g.remove((s, p, o))
            stripped += 1
    for s, p, o in list(g.triples((None, OWL.equivalentClass, None))):
        # Drop ALL equivalentClass triples -- blank-node ones break owlready2
        # outright, and named-class ones pointing to external vocabularies
        # (e.g. foaf:Organization) also break it because the placeholder
        # isn't a class.
        g.remove((s, p, o))
        stripped += 1
    for s, p, o in list(g.triples((None, RDFS.domain, None))):
        if isinstance(o, BNode):
            g.remove((s, p, o))
            stripped += 1
    for s, p, o in list(g.triples((None, RDFS.range, None))):
        if isinstance(o, BNode):
            g.remove((s, p, o))
            stripped += 1
    for s, p, o in list(g.triples((None, OWL.hasKey, None))):
        g.remove((s, p, o))
        stripped += 1
    for s, p, o in list(g.triples((None, OWL.propertyChainAxiom, None))):
        g.remove((s, p, o))
        stripped += 1
    if stripped:
        print(
            f"[ontology_io] {ttl_path.name}: stripped {stripped} "
            f"owlready2-incompatible axiom(s) "
            f"(blank-node restrictions / cross-namespace equivalents / "
            f"hasKey / propertyChainAxiom)"
        )

    out = g.serialize(format="xml", encoding="utf-8")
    del g
    gc.collect()
    return out


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
            # Copy user-supplied source files into a per-bundle temp dir so
            # _strip_unresolved_imports can safely rewrite them without
            # touching the user's tree on disk. Symmetric with how
            # zip-extracted files already get handled in temp.
            tmp = bundle._exit_stack.enter_context(
                tempfile.TemporaryDirectory(prefix="onto_src_")
            )
            if path.suffix.lower() == ".ttl":
                # owlready2's built-in Turtle reader is fragile (it raises
                # `NTriples parsing error` on the W3C `time` ontology even
                # though the file is valid Turtle). rdflib's parser is
                # robust; convert to RDF/XML up-front and let owlready2
                # consume the .rdf instead.
                tmp_copy = Path(tmp) / (path.stem + ".rdf")
                tmp_copy.write_bytes(_convert_ttl_to_rdfxml_bytes(path))
            else:
                tmp_copy = Path(tmp) / path.name
                shutil.copy2(path, tmp_copy)
            bundle.sources.append(
                OntologySource(
                    label=path.name,
                    path=tmp_copy,
                    working_dir=tmp_copy.parent,
                )
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


def _read_file_header(path: Path, max_bytes: int = 262144) -> str:
    """Read at most `max_bytes` from the start of a file. owl:imports and
    XML <!ENTITY> declarations always live in the file header (before the
    class definitions), so a few hundred KB is enough. Avoids OOM-killing
    the process when sniffing imports across huge files like DRON (670MB)
    or HP (73MB)."""
    try:
        with open(path, "rb") as fh:
            data = fh.read(max_bytes)
    except OSError:
        return ""
    return data.decode("utf-8", errors="replace")


def _build_import_url_map(working_files: list[Path]) -> dict[str, str]:
    """Sniff every ontology file for fully-expanded import URLs and map each
    one to a local file with a matching basename.

    Handles OntoCAPE-style ontologies that bake absolute Windows paths into
    their owl:imports via XML entity declarations, plus simpler ontologies
    that use HTTP IRIs directly.

    Only the file header is read (256 KB) -- imports are always declared
    near the top, and reading multi-hundred-MB files like DRON in full
    would OOM-kill the parser.
    """
    basename_to_path = _basename_index(working_files)
    discovered: dict[str, str] = {}
    for f in working_files:
        text = _read_file_header(f)
        if not text:
            continue
        for url in _discover_imports_in_text(text):
            basename = url.rsplit("/", 1)[-1].lower()
            local = basename_to_path.get(basename)
            if local is not None and url not in discovered:
                discovered[url] = str(local)
    return discovered


def _strip_unresolved_imports(working_files: list[Path], resolvable_urls: set[str]) -> int:
    """Rewrite each .owl/.rdf file in place to remove `<owl:imports>` elements
    that would trigger a network fetch by owlready2.

    Why this matters: owlready2's `Ontology.load()` ALWAYS recursively loads
    imported ontologies (namespace.py line 1053), and the `only_local` flag
    is NOT propagated to those nested loads. So any `<owl:imports>` element
    pointing at an HTTP(S) URL not in PREDEFINED_ONTOLOGIES will hang the
    parse on a TCP SYN to the IRI's host (FIBO -> omg.org / edmcouncil.org).

    Stripping rules:
      - file:/// URLs that aren't in `resolvable_urls`: stripped (sibling
        packages we don't ship, e.g. OntoCAPE's meta_model.owl).
      - HTTP(S) URLs that aren't in PREDEFINED_ONTOLOGIES AND aren't in
        `resolvable_urls`: stripped (FIBO/external imports).
      - Anything found in PREDEFINED_ONTOLOGIES or `resolvable_urls`: kept
        (owlready2's built-in W3C schemas, our OASIS catalog mappings,
        our XML-entity-derived local mappings).

    Real-world cases this handles:
      - OntoCAPE imports `file:/C:/.../meta_model.owl` we don't ship: dropped.
      - OCRe imports `http://www.w3.org/2002/07/owl`: kept (W3C built-in).
      - FIBO imports `https://www.omg.org/spec/Commons/...`: dropped (avoids
        network hang on Cloudflare SYN).

    Only modifies files passed in; never touches user sources outside temp dirs.
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
            # Kept if we can resolve it ourselves (local IRI map / catalog).
            if expanded in resolvable_urls or raw_url in resolvable_urls:
                return match.group(0)
            lower = expanded.lower()
            # file:// imports without a local resolution: strip.
            if lower.startswith("file:"):
                stripped += 1
                return "<!-- import stripped (file: unresolvable): " + raw_url + " -->"
            # HTTP(S) imports: strip unless owlready2 has a built-in mapping
            # (PREDEFINED_ONTOLOGIES is populated with W3C schemas + whatever
            # we registered via register_local_iri_map). This avoids hanging
            # on TCP SYN to external hosts like omg.org / edmcouncil.org.
            if lower.startswith(("http:", "https:")):
                stripped_iri = expanded.rstrip("#").rstrip("/")
                if (
                    expanded in PREDEFINED_ONTOLOGIES
                    or stripped_iri in PREDEFINED_ONTOLOGIES
                    or (stripped_iri + "#") in PREDEFINED_ONTOLOGIES
                    or (stripped_iri + "/") in PREDEFINED_ONTOLOGIES
                ):
                    return match.group(0)
                stripped += 1
                return "<!-- import stripped (http: unresolvable): " + raw_url + " -->"
            # Other URI schemes: leave alone (shouldn't occur in practice).
            return match.group(0)

        new_text = import_block_re.sub(_maybe_strip, text)
        if new_text != text:
            try:
                f.write_text(new_text, encoding="utf-8")
            except OSError:
                continue
    return stripped


# OASIS XML catalog parser. Many ontologies (FIBO is the big one) ship with
# `catalog-v001.xml` files that explicitly map ontology IRIs to relative
# local-file paths. Parsing them gives a precise IRI map without guessing.
_CATALOG_URI_RE = re.compile(
    r'<uri\s+name\s*=\s*"([^"]+)"\s+uri\s*=\s*"([^"]+)"\s*/>',
    re.IGNORECASE,
)


def _build_iri_map_from_oasis_catalogs(temp_root: Path) -> dict[str, str]:
    """Walk `temp_root` for OASIS `catalog-v*.xml` files and convert each
    `<uri name=... uri=.../>` entry into an IRI -> absolute-local-path
    mapping. Relative `uri=` values are resolved against the catalog file's
    directory.
    """
    mapping: dict[str, str] = {}
    for catalog in temp_root.rglob("catalog-v*.xml"):
        try:
            text = catalog.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        catalog_dir = catalog.parent
        for match in _CATALOG_URI_RE.finditer(text):
            iri, rel_path = match.group(1).strip(), match.group(2).strip()
            if not iri or not rel_path:
                continue
            target = (catalog_dir / rel_path).resolve()
            if target.exists():
                mapping[iri] = str(target)
                # Also map IRI variant with trailing slash dropped, since
                # owlready2 looks up both forms.
                mapping[iri.rstrip("/")] = str(target)
    return mapping


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


def load_ontology_files(
    sources: list[OntologySource],
    stamp_provenance: bool = False,
) -> dict[str, dict]:
    """Run the existing extract_ontology_to_dicts() against each source,
    registering local IRI maps so cross-file imports resolve to the unzipped
    siblings rather than network URLs or hardcoded Windows paths.

    Returns the canonical loaded ontology dict-of-dicts:
        {classes_dict, object_properties_dict, data_properties_dict, instances_dict}
    """
    import time

    if not sources:
        raise ValueError("No sources to load")

    print(f"[ontology_io] load_ontology_files: {len(sources)} source file(s)")
    t0 = time.monotonic()

    # Base IRI map: filename-based aliases inside each working dir for HTTP-style
    # imports (covers OCRe-style http://purl.org/net/OCRe/<filename>.owl).
    working_dirs = {s.working_dir for s in sources}
    full_iri_map: dict[str, str] = {}
    for d in working_dirs:
        try:
            full_iri_map.update(build_local_iri_map_from_folder(str(d)))
        except FileNotFoundError:
            pass
    print(
        f"[ontology_io] folder-based IRI map: {len(full_iri_map)} entry(ies) "
        f"({time.monotonic() - t0:.1f}s)"
    )

    # Extra: sniff hardcoded file:/... import URLs (OntoCAPE-style) and add to
    # the map. These won't be discovered by build_local_iri_map_from_folder.
    t1 = time.monotonic()
    all_files = [s.path for s in sources]
    before = len(full_iri_map)
    full_iri_map.update(_build_import_url_map(all_files))
    print(
        f"[ontology_io] import-URL sniff added {len(full_iri_map) - before} entry(ies) "
        f"({time.monotonic() - t1:.1f}s)"
    )

    # Extra: OASIS catalog files (FIBO ships catalog-v001.xml in each subdir
    # that maps every IRI -> local file). Walking these gives an exact map
    # for ontology packages that use catalogs.
    t2 = time.monotonic()
    before = len(full_iri_map)
    catalog_roots = {s.working_dir for s in sources if s.is_from_zip}
    # Walk one level up from each file's dir to catch catalogs that live in
    # ancestor directories (FIBO's catalogs sit at FBC/, IND/, etc. — siblings
    # of the .rdf files we extracted).
    for d in list(catalog_roots):
        catalog_roots.add(d.parent)
    for catalog_root in catalog_roots:
        if catalog_root.exists():
            full_iri_map.update(_build_iri_map_from_oasis_catalogs(catalog_root))
    print(
        f"[ontology_io] OASIS catalog parse added {len(full_iri_map) - before} entry(ies) "
        f"({time.monotonic() - t2:.1f}s)"
    )

    if full_iri_map:
        t3 = time.monotonic()
        register_local_iri_map(full_iri_map)
        print(f"[ontology_io] global IRI-map registration ({time.monotonic() - t3:.1f}s)")

    # Strip imports that reference files outside the local set (sibling
    # packages we don't have copies of). owlready2 hard-fails on missing
    # imports; this makes the load resilient. Only modifies temp-extracted
    # files — never touches user sources.
    t4 = time.monotonic()
    files_in_temp = [
        s.path for s in sources if s.is_from_zip or str(s.path).startswith(tempfile.gettempdir())
    ]
    if files_in_temp:
        n_stripped = _strip_unresolved_imports(files_in_temp, set(full_iri_map.keys()))
        print(
            f"[ontology_io] stripped {n_stripped} unresolvable import(s) "
            f"from {len(files_in_temp)} extracted file(s) ({time.monotonic() - t4:.1f}s)"
        )

    # Decide how aggressively owlready2 should follow imports.
    # - Multi-source case (zip with many .owl files, or user passed several
    #   --ontology flags): we already enumerate every sibling, so do NOT have
    #   owlready2 re-walk the import graph during each load. With 234 files
    #   in FIBO and dense cross-imports, walking would re-parse the same
    #   low-level files dozens of times (effectively O(N²)). Loading each
    #   file standalone and merging the dicts is O(N) and produces the same
    #   result because every class definition is visited at least once.
    # - Single-source case (one .owl on disk, no siblings to load
    #   explicitly): keep load_imported=True so transitive deps still get
    #   followed via the IRI map / HTTP.
    has_siblings = any(s.is_from_zip for s in sources) or len(sources) > 1
    follow_imports = not has_siblings

    if follow_imports:
        # Single isolated file: keep the root-files heuristic (no-op here
        # since there's only one source) and let owlready2 walk imports.
        effective_sources = list(sources)
    else:
        # Multi-source: load EVERY file. `_find_root_files` is no longer a
        # win because each load is now cheap; dropping non-roots would miss
        # classes that are only declared in imported-only files.
        effective_sources = list(sources)
        print(
            f"[ontology_io] multi-source load: {len(effective_sources)} file(s), "
            f"load_imported=False (each file parsed standalone, then merged)"
        )

    # Compose the dict that import_ontologies() expects. The iri_map is
    # intentionally NOT passed per file: it has already been registered
    # globally above via register_local_iri_map(full_iri_map). Re-registering
    # the same thousands of FIBO catalog entries on each of 100+ files
    # dominates wall time (the print statements alone) for no benefit.
    owl_dict = {}
    for s in effective_sources:
        owl_dict[s.label] = {
            "filename": str(s.path),
            "load_imported": follow_imports,
            "local_only": True,
            "local_ontology_dir": str(s.working_dir),
            "iri_map": None,
        }
    return import_ontologies(owl_dict, stamp_provenance=stamp_provenance)


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
