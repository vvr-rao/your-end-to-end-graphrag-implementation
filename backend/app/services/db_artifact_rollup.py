"""Hierarchical clustered-rollup artifact generation.

Consolidates semantically-similar intelligence artifacts into new, LOSSLESS
"rollup" artifacts, layer by layer, giving retrieval a coarse-to-fine surface
(a query can hit one consolidated rollup instead of scattering across N
near-duplicate leaves).

Pipeline (per layer L in 1..`layers`, per `artifact_type`):
  1. Candidate set = ACTIVE artifacts of that type at level L-1 (level 0 =
     original leaves; level k = rollups with extra_metadata.layer == k) that are
     NOT already referenced by a rollup (resume-safe / idempotent).
  2. Cluster within the type: fetch all same-type similarity edges within
     `threshold` in one batched query (retrieval_sql.same_type_neighbor_edges)
     and union them (union-find). Connected components with >= `min_cluster` members
     are clusters; a component larger than `max_merge_inputs` is split into
     consecutive batches (each batch -> its own rollup; siblings consolidate
     further at the next layer).
  3. Merge each batch with one gpt-4.1 `artifact_merge` call -- STRICTLY LOSSLESS
     (duplicate removal only; every distinct fact survives). Mint a new artifact
     of the SAME type, tagged extra_metadata {rollup, layer, child_ids,
     child_count, source_types}.
  4. Embed + insert (batched); wire one viao:referencesArtifact edge rollup->child
     per child. Originals are NEVER modified -> nothing is lost, the rollup is
     purely additive.
  5. Bump graph_version once at the end.

Generic + corpus-agnostic. No new artifact_type (rollups reuse the child type),
no VIAO change (referencesArtifact already exists), no migration.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select
from sqlalchemy import text as sql_text
from sqlalchemy.dialects.postgresql import insert as pg_insert

from backend.app.db.graph_version import bump_version, current_version
from backend.app.db.models.artifacts import IntelligenceArtifact
from backend.app.db.models.graph import GraphRelationship
from backend.app.db.session import session_scope
from backend.app.services import retrieval_sql
from backend.app.services.db_artifact_gen import _artifact_iri, _extract_json
from backend.app.services.embeddings import Embedder
from backend.app.services.evaluated_summarizer import check_section_coverage
from backend.app.services.llm_router import LLMRouter
from backend.app.services.predicates import VIAO_REFERENCES_ARTIFACT
from backend.app.services.prompts import PROMPTS

# Artifact types eligible for the `--rollup` flag by default. StructuredTable is
# excluded (tabular JSON-LD where free-text merging is inappropriate). Summary IS
# included: it is rolled up automatically during generate-artifacts --type Summary,
# and --rollup then ADDITIVELY adds more layers on top (see generate_rollups).
ALL_ROLLUP_TYPES: tuple[str, ...] = (
    "Claim", "Finding", "Observation", "Event",
    "Summary", "Insight", "Recommendation",
)


@dataclass
class RollupGenSummary:
    layers_run: int = 0
    candidates_scanned: int = 0
    clusters_found: int = 0
    rollups_inserted: int = 0
    references_edges: int = 0
    inherited_edges: int = 0
    clusters_revised: int = 0
    clusters_incomplete_after_loop: int = 0
    by_layer: dict[int, int] = field(default_factory=dict)
    by_type: dict[str, int] = field(default_factory=dict)
    llm_cost_usd: float = 0.0
    embedding_cost_usd: float = 0.0
    total_cost_usd: float = 0.0
    wall_seconds: float = 0.0
    new_graph_version: int = 0
    samples: list[dict[str, Any]] = field(default_factory=list)


class _UnionFind:
    """Minimal union-find over a fixed set of UUIDs (path compression +
    union by size). Used to turn pairwise similarity edges into clusters."""

    def __init__(self, items: list[uuid.UUID]) -> None:
        self._parent: dict[uuid.UUID, uuid.UUID] = {x: x for x in items}
        self._size: dict[uuid.UUID, int] = {x: 1 for x in items}

    def find(self, x: uuid.UUID) -> uuid.UUID:
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        # path compression
        while self._parent[x] != root:
            self._parent[x], x = root, self._parent[x]
        return root

    def union(self, a: uuid.UUID, b: uuid.UUID) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self._size[ra] < self._size[rb]:
            ra, rb = rb, ra
        self._parent[rb] = ra
        self._size[ra] += self._size[rb]

    def components(self) -> dict[uuid.UUID, list[uuid.UUID]]:
        comps: dict[uuid.UUID, list[uuid.UUID]] = {}
        for x in self._parent:
            comps.setdefault(self.find(x), []).append(x)
        return comps


def _level_predicate(level: int) -> str:
    """SQL fragment selecting artifacts at a given rollup level.
    level 0 = original leaves (no rollup flag); level k = rollups whose
    extra_metadata.layer == k."""
    if level == 0:
        return "(a.extra_metadata->>'rollup') IS DISTINCT FROM 'true'"
    return (
        "(a.extra_metadata->>'rollup') = 'true' "
        f"AND (a.extra_metadata->>'layer') = '{level}'"
    )


async def _max_rollup_layer(session, artifact_type: str) -> int:
    """Highest existing rollup layer for `artifact_type` (0 if none). Used to
    resume additively so a new run adds layers on TOP of what exists."""
    r = await session.execute(
        sql_text("""
        SELECT coalesce(max((extra_metadata->>'layer')::int), 0)
          FROM graphrag.intelligence_artifacts
         WHERE status = 'ACTIVE' AND artifact_type = :atype
           AND (extra_metadata->>'rollup') = 'true'
        """),
        {"atype": artifact_type},
    )
    return int(r.scalar() or 0)


async def _fetch_candidates(
    artifact_type: str, level: int, *, scope_document_iri: str | None,
) -> list[tuple[uuid.UUID, str, float | None]]:
    """ACTIVE artifacts of `artifact_type` at `level`, embedded, and NOT already
    referenced by a rollup. Returns (id, text, confidence)."""
    scope_join = ""
    scope_where = ""
    params: dict[str, Any] = {"atype": artifact_type}
    if scope_document_iri is not None:
        # Restrict to artifacts derived from chunks of one document (leaves only;
        # rollups have no chunk provenance, so scoping only applies at level 0).
        scope_join = (
            " JOIN graphrag.artifact_sources asrc ON asrc.artifact_id = a.id "
            " JOIN graphrag.chunks ch ON ch.id = asrc.chunk_id "
            " JOIN graphrag.documents d ON d.id = ch.document_id "
        )
        scope_where = " AND d.document_identifier = :diri "
        params["diri"] = scope_document_iri
    q = sql_text(f"""
        SELECT DISTINCT a.id, a.text, a.confidence
          FROM graphrag.intelligence_artifacts a
          {scope_join}
         WHERE a.status = 'ACTIVE'
           AND a.artifact_type = :atype
           AND a.embedding IS NOT NULL
           AND {_level_predicate(level)}
           {scope_where}
           AND NOT EXISTS (
               SELECT 1 FROM graphrag.graph_relationships gr
                WHERE gr.predicate_label = 'viao:referencesArtifact'
                  AND gr.target_node_type = 'intelligence_artifact'
                  AND gr.target_node_id = a.id
           )
    """)
    async with session_scope() as session:
        r = await session.execute(q, params)
        return [
            (rid, rtext, float(conf) if conf is not None else None)
            for rid, rtext, conf in r.all()
        ]


async def _cluster(
    candidates: list[tuple[uuid.UUID, str, float | None]],
    artifact_type: str,
    *,
    threshold: float,
    max_neighbors: int,
    min_cluster: int,
    max_merge_inputs: int,
) -> list[list[uuid.UUID]]:
    """Union-find clustering of candidates via same-type pgvector neighbors.
    Returns a list of clusters (each a list of ids); components larger than
    `max_merge_inputs` are split into consecutive batches. All similarity edges
    are fetched in ONE batched query (retrieval_sql.same_type_neighbor_edges) --
    per-candidate round-trips do not scale on a latency-bound pooler."""
    ids = [c[0] for c in candidates]
    id_set = set(ids)
    conf_by_id = {c[0]: (c[2] or 0.0) for c in candidates}
    uf = _UnionFind(ids)
    # One batched edge query for the whole candidate set (not one round-trip per
    # candidate -- that times out at corpus scale on a latency-bound pooler).
    async with session_scope() as session:
        edges = await retrieval_sql.same_type_neighbor_edges(
            session, ids, artifact_type,
            threshold=threshold, max_neighbors=max_neighbors,
        )
    for a, b in edges:
        if a in id_set and b in id_set:
            uf.union(a, b)
    clusters: list[list[uuid.UUID]] = []
    for members in uf.components().values():
        if len(members) < min_cluster:
            continue
        # Deterministic order (highest-confidence first) so batch splits are
        # stable across resumes.
        members.sort(key=lambda x: conf_by_id.get(x, 0.0), reverse=True)
        for i in range(0, len(members), max_merge_inputs):
            batch = members[i : i + max_merge_inputs]
            if len(batch) >= min_cluster:
                clusters.append(batch)
    return clusters


async def generate_rollups(
    *,
    types: tuple[str, ...] = ALL_ROLLUP_TYPES,
    layers: int = 2,
    threshold: float = 0.35,
    min_cluster: int = 2,
    max_neighbors: int = 25,
    max_merge_inputs: int = 30,
    concurrency: int = 4,
    max_cost_usd: float = 5.0,
    scope_document_iri: str | None = None,
    eval_rounds: int = 1,
    verbose: bool = False,
) -> RollupGenSummary:
    """Build `layers` layers of clustered rollup artifacts across `types`.

    `eval_rounds`: after each merge, an LLM loss-check lists any dropped facts and
    a reviser adds them back (up to `eval_rounds` passes) so the merge is lossless
    (dedup only). 0 disables the loop."""
    t0 = time.time()
    summary = RollupGenSummary()
    router = LLMRouter()
    cost_before = router.total_cost_usd
    embedder = Embedder()

    async with session_scope() as session:
        gv = await current_version(session)
        # Additive: each type resumes ABOVE its current top rollup level, so a
        # later --rollup (or --rollup on top of the auto Summary rollup) adds MORE
        # layers rather than re-clustering level 0. base=0 means only leaves exist.
        base_level: dict[str, int] = {}
        for atype in types:
            base_level[atype] = await _max_rollup_layer(session, atype)

    all_payloads: list[dict[str, Any]] = []
    all_iris: list[str] = []
    # rollup IRI -> child ids (for edge building after we learn DB ids)
    rollup_children: dict[str, list[uuid.UUID]] = {}
    cost_limit_hit = asyncio.Event()

    for step in range(1, layers + 1):
        # (atype, batch, text_map, conf_map, new_level)
        layer_clusters: list[tuple[str, list[uuid.UUID], dict[uuid.UUID, str],
                                   dict[uuid.UUID, float | None], int]] = []
        for atype in types:
            src_level = base_level[atype] + step - 1
            new_level = base_level[atype] + step
            candidates = await _fetch_candidates(
                atype, src_level, scope_document_iri=scope_document_iri,
            )
            summary.candidates_scanned += len(candidates)
            if len(candidates) < min_cluster:
                continue
            text_map = {c[0]: c[1] for c in candidates}
            conf_map = {c[0]: c[2] for c in candidates}
            clusters = await _cluster(
                candidates, atype, threshold=threshold,
                max_neighbors=max_neighbors, min_cluster=min_cluster,
                max_merge_inputs=max_merge_inputs,
            )
            for batch in clusters:
                layer_clusters.append((atype, batch, text_map, conf_map, new_level))

        summary.clusters_found += len(layer_clusters)
        if not layer_clusters:
            if verbose:
                print(f"[rollup] step {step}: no clusters met min_cluster={min_cluster}")
            # A step with no clusters means the top level is all-distinct; no
            # higher layers can form, so stop early.
            break
        print(
            f"[rollup] step {step}: {len(layer_clusters)} cluster(s) to merge "
            f"across {len(types)} type(s) (concurrency={concurrency})"
        )

        sem = asyncio.Semaphore(concurrency)

        # `_sem` is bound as a default so the closure captures THIS step's
        # semaphore (not a later loop reassignment); gather runs before the next
        # iteration, so it is already safe, and this also satisfies ruff B023.
        async def _merge_one(atype, batch, text_map, conf_map, new_level, _sem=sem):
            if cost_limit_hit.is_set():
                return None
            async with _sem:
                if cost_limit_hit.is_set():
                    return None
                child_texts = [text_map[i] for i in batch]
                # Type-aware model: the section-structured Summary rollup uses the
                # more capable `summary_merge*` tasks (gpt-5.4 -- measurably higher
                # retention); short prose artifacts (Claim/Finding/Observation/Event)
                # use the cheaper `artifact_merge*` (gpt-4.1), which merges them
                # perfectly, so paying more there is waste. Prompt builder is shared
                # (already Summary-aware); only the router task/model differs.
                _fam = "summary_merge" if atype == "Summary" else "artifact_merge"
                merge_task, eval_task, revise_task = _fam, f"{_fam}_evaluate", f"{_fam}_revise"
                model_name = router.task_spec(merge_task).get("model")
                system, user = PROMPTS["artifact_merge"](atype, child_texts)
                try:
                    out = await router.chat(merge_task, system=system, user=user)
                except Exception as exc:
                    print(f"[rollup] merge failed ({atype}, {len(batch)} items): {exc}")
                    return None
                parsed = _extract_json(out.text) or {}
                text = (parsed.get("text") or "").strip()
                if not text:
                    return None
                try:
                    conf = (
                        float(parsed.get("confidence"))
                        if parsed.get("confidence") is not None else None
                    )
                except (TypeError, ValueError):
                    conf = None
                if conf is None:
                    child_confs = [conf_map.get(i) or 0.0 for i in batch]
                    conf = max(child_confs) if child_confs else None

                # Lossless eval->revise loop: the LLM merge over-compresses distinct
                # inputs and drops facts, so verify the merged text preserves every
                # distinct fact from the children and add back anything missing.
                # +1 extra evaluation beyond the revise budget verifies the last revise.
                revised = False
                final_complete = None
                if eval_rounds > 0:
                    for round_idx in range(1, eval_rounds + 2):
                        if cost_limit_hit.is_set():
                            break
                        esys, euser = PROMPTS["artifact_merge_evaluate"](
                            atype, child_texts, text)
                        try:
                            eout = await router.chat(
                                eval_task, system=esys, user=euser)
                            ev = _extract_json(eout.text) or {}
                        except Exception:
                            break
                        missing = list(ev.get("missing_items") or [])
                        # Deterministic guard: a merged SUMMARY must retain all 9
                        # section headers. Missing headers are non-negotiable and
                        # override the LLM's 'complete' verdict, forcing a revise.
                        missing_secs = (
                            [c for c, ok in check_section_coverage(text).items() if not ok]
                            if atype == "Summary" else []
                        )
                        if (ev.get("complete") or not missing) and not missing_secs:
                            final_complete = True
                            break
                        if round_idx > eval_rounds:
                            final_complete = False  # out of budget, gaps remain
                            break
                        missing = missing + [
                            f"the '{s}' section header is missing -- emit it "
                            "(write 'None identified.' if the section is empty)"
                            for s in missing_secs
                        ]
                        rsys, ruser = PROMPTS["artifact_merge_revise"](
                            atype, child_texts, text, missing)
                        try:
                            rout = await router.chat(
                                revise_task, system=rsys, user=ruser)
                            rtext = ((_extract_json(rout.text) or {}).get("text") or "").strip()
                            if rtext:
                                text = rtext
                                revised = True
                        except Exception:
                            break

                if router.total_cost_usd - cost_before > max_cost_usd:
                    if not cost_limit_hit.is_set():
                        cost_limit_hit.set()
                        print(f"[rollup] HALT: cost cap ${max_cost_usd:.2f} reached")
                return {"atype": atype, "batch": batch, "text": text,
                        "conf": conf, "new_level": new_level, "model_name": model_name,
                        "revised": revised, "incomplete": final_complete is False}

        results = await asyncio.gather(*[
            _merge_one(a, b, tm, cm, nl) for (a, b, tm, cm, nl) in layer_clusters
        ])
        results = [r for r in results if r is not None]
        if not results:
            continue
        summary.clusters_revised += sum(1 for r in results if r.get("revised"))
        summary.clusters_incomplete_after_loop += sum(
            1 for r in results if r.get("incomplete"))

        # Stage this step's rollup payloads. Insert per step so the NEXT step can
        # cluster over them.
        layer_payloads: list[dict[str, Any]] = []
        layer_iris: list[str] = []
        embed_texts: list[str] = []
        for res in results:
            airi = _artifact_iri(res["atype"])
            layer_iris.append(airi)
            rollup_children[airi] = list(res["batch"])
            layer_payloads.append({
                "artifact_identifier": airi,
                "artifact_type": res["atype"],
                "title": None,
                "text": res["text"],
                "confidence": res["conf"],
                "model_name": res["model_name"],
                "prompt_version": "artifact_merge@v1",
                "status": "ACTIVE",
                "graph_version": gv,
                "extra_metadata": {
                    "rollup": True,
                    "layer": res["new_level"],
                    "child_count": len(res["batch"]),
                    "child_ids": [str(x) for x in res["batch"]],
                    "source_types": [res["atype"]],
                },
            })
            embed_texts.append(res["text"])
            summary.by_type[res["atype"]] = summary.by_type.get(res["atype"], 0) + 1
            summary.by_layer[res["new_level"]] = (
                summary.by_layer.get(res["new_level"], 0) + 1
            )

        embeds = await embedder.embed(embed_texts) if embed_texts else []
        for p, vec in zip(layer_payloads, embeds, strict=False):
            p["embedding"] = vec

        # Insert this step's rollups NOW so the next step can cluster over them.
        async with session_scope() as session:
            for i in range(0, len(layer_payloads), 100):
                await session.execute(
                    pg_insert(IntelligenceArtifact).values(layer_payloads[i : i + 100])
                )
        summary.rollups_inserted += len(layer_payloads)
        summary.layers_run = step
        all_payloads.extend(layer_payloads)
        all_iris.extend(layer_iris)
        if verbose:
            print(f"[rollup] step {step}: inserted {len(layer_payloads)} rollup(s)")

    summary.llm_cost_usd = router.total_cost_usd - cost_before
    summary.embedding_cost_usd = embedder.total_cost_usd

    if not all_iris:
        summary.total_cost_usd = summary.llm_cost_usd + summary.embedding_cost_usd
        summary.wall_seconds = time.time() - t0
        print("[rollup] no rollups produced")
        return summary

    # Resolve DB ids for every rollup, then wire referencesArtifact edges.
    async with session_scope() as session:
        r = await session.execute(
            select(
                IntelligenceArtifact.id,
                IntelligenceArtifact.artifact_identifier,
            ).where(IntelligenceArtifact.artifact_identifier.in_(all_iris))
        )
        iri_to_id = {iri: aid for aid, iri in r.all()}

    edge_rows: list[dict[str, Any]] = []
    for airi, child_ids in rollup_children.items():
        parent_id = iri_to_id.get(airi)
        if not parent_id:
            continue
        for child_id in child_ids:
            edge_rows.append({
                "source_node_type": "intelligence_artifact",
                "source_node_id": parent_id,
                "target_node_type": "intelligence_artifact",
                "target_node_id": child_id,
                "predicate_iri": VIAO_REFERENCES_ARTIFACT,
                "predicate_label": "viao:referencesArtifact",
                "relationship_type": "referencesArtifact",
                "relationship_source": "LLM_INFERENCE",
                "is_authoritative": True,
                "source_chunk_id": None,
                "source_document_id": None,
                "source_artifact_id": parent_id,
                "graph_version": gv,
                "extra_metadata": {},
            })
        if len(summary.samples) < 5:
            summary.samples.append({
                "type": next(
                    (p["artifact_type"] for p in all_payloads
                     if p["artifact_identifier"] == airi), ""),
                "child_count": len(child_ids),
                "text": next(
                    (p["text"] for p in all_payloads
                     if p["artifact_identifier"] == airi), "")[:160],
            })

    # Inherit the children's "aboutness" edges so a rollup is reachable via the
    # SAME entity/graph BFS as its leaves: a rollup of Claims about Honda itself
    # assertsAbout Honda; a Summary rollup inherits its children's summarizes ->
    # document links. (Higher layers inherit transitively, since layer-1 rollups
    # already carry the inherited edges.)
    child_to_parent: dict[uuid.UUID, uuid.UUID] = {}
    for airi, child_ids in rollup_children.items():
        pid = iri_to_id.get(airi)
        if pid:
            for cid in child_ids:
                child_to_parent[cid] = pid
    if child_to_parent:
        async with session_scope() as session:
            r = await session.execute(
                sql_text("""
                SELECT DISTINCT source_node_id, target_node_type, target_node_id,
                       predicate_iri, predicate_label
                  FROM graphrag.graph_relationships
                 WHERE source_node_id = ANY(CAST(:cids AS uuid[]))
                   AND predicate_label IN ('viao:assertsAbout','viao:summarizes')
                """),
                {"cids": [str(c) for c in child_to_parent]},
            )
            inherited = r.all()
        seen_inh: set[tuple[uuid.UUID, str, uuid.UUID]] = set()
        for src, tnode_type, tnode_id, pred_iri, pred_label in inherited:
            parent_id = child_to_parent.get(src)
            if not parent_id:
                continue
            key = (parent_id, pred_label, tnode_id)
            if key in seen_inh:
                continue
            seen_inh.add(key)
            edge_rows.append({
                "source_node_type": "intelligence_artifact",
                "source_node_id": parent_id,
                "target_node_type": tnode_type,
                "target_node_id": tnode_id,
                "predicate_iri": pred_iri,
                "predicate_label": pred_label,
                "relationship_type": pred_label.split(":", 1)[-1],
                "relationship_source": "LLM_INFERENCE",
                "is_authoritative": True,
                "source_chunk_id": None,
                "source_document_id": tnode_id if tnode_type == "document" else None,
                "source_artifact_id": parent_id,
                "graph_version": gv,
                "extra_metadata": {"inherited_from_children": True},
            })
        summary.inherited_edges = len(seen_inh)

    async with session_scope() as session:
        for i in range(0, len(edge_rows), 500):
            await session.execute(
                pg_insert(GraphRelationship).values(edge_rows[i : i + 500])
            )
        summary.new_graph_version = await bump_version(session)
    summary.references_edges = len(edge_rows) - summary.inherited_edges

    summary.total_cost_usd = summary.llm_cost_usd + summary.embedding_cost_usd
    summary.wall_seconds = time.time() - t0
    print(
        f"[rollup] DONE: rollups={summary.rollups_inserted} "
        f"(by_layer={summary.by_layer}, by_type={summary.by_type}), "
        f"referencesArtifact_edges={summary.references_edges}, "
        f"inherited_entity/doc_edges={summary.inherited_edges}, "
        f"loss-loop(revised={summary.clusters_revised}, "
        f"still_incomplete={summary.clusters_incomplete_after_loop}), "
        f"cost=${summary.total_cost_usd:.4f}, wall={summary.wall_seconds:.1f}s, "
        f"graph_version -> {summary.new_graph_version}"
    )
    return summary
