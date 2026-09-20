"""Embed extracted relationships so retrieval can match on what an edge SAYS.

Every entity->entity edge carries the sentence that supports it
(`extra_metadata.evidence`), and a `graphrag:relatedTo` edge also carries
the free-text `relation` phrase the passage used. Until 0008 neither was
embedded, so the graph walk could only use an edge as adjacency: every
edge out of a node was worth the same 0.7^hop, whatever it asserted.

One vector per edge over

    "<source name> <relation> <target name>. <evidence>"

turns that into something a question can be matched against -- both for
relationship-aware hopping and for ranking the members of a class by how
well their edges fit a relation phrase ("who was accused of fraud").

Only DOCUMENT_EXTRACTION / LLM_INFERENCE entity->entity edges are
embedded. Ontology TBox rows (subClassOf, rdf:type) assert nothing a
question would phrase as a relationship, and on a large import they are
the bulk of the table.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from sqlalchemy import text as sql_text

from backend.app.db.session import session_scope
from backend.app.services.embeddings import Embedder

# One UPDATE round-trip per batch, and one embeddings request per batch.
_BATCH = 200


@dataclass
class RelationshipEmbedSummary:
    considered: int = 0
    embedded: int = 0
    skipped_no_text: int = 0
    cost_usd: float = 0.0
    samples: list[str] = field(default_factory=list)


def build_edge_text(
    source_name: str | None,
    target_name: str | None,
    relation: str | None,
    predicate_label: str | None,
    evidence: str | None = None,          # unused; see the docstring
) -> str:
    """The embedded string for one edge: `<source> <relation> <target>`.

    `relation` (the passage's own words, present on relatedTo edges) wins over
    the predicate label, which is a camel-cased ontology term that matches
    question vocabulary far less well.

    THE EVIDENCE SENTENCE IS DELIBERATELY EXCLUDED, against the obvious
    intuition that more context embeds better. Measured on the 40-document
    news graph, appending it moved every probe the wrong way:

        probe                            triple only   + evidence
        "court is located in US"            0.575         0.548
        "someone was accused of fraud"      0.571         0.542
        "founder founded FTX"               0.716         0.668

    A question is a short relation; an evidence sentence is long, specific
    document prose. Concatenating them pulls the edge vector toward the
    document's wording and away from the relation, and the effect was enough
    to push two correct matches below the 0.55 hop threshold. Truncating the
    quote did not help -- 120 characters cost as much as 700.

    The parameter is kept so callers need not care, and because the trade-off
    is not free: a question phrased in the document's words rather than as a
    relation ("who lost money when the exchange collapsed") would match the
    evidence better. Relation matching is what this vector is FOR; the quote
    is still stored in `extra_metadata` and still cited.
    """
    rel = (relation or "").strip() or (predicate_label or "").strip()
    return " ".join(
        p for p in ((source_name or "").strip(), rel, (target_name or "").strip())
        if p
    )


_SELECT_SQL = """
SELECT g.id,
       se.name,
       oe.name,
       g.extra_metadata->>'relation',
       g.predicate_label,
       g.extra_metadata->>'evidence'
  FROM graphrag.graph_relationships g
  JOIN graphrag.entities se ON se.id = g.source_node_id
  JOIN graphrag.entities oe ON oe.id = g.target_node_id
 WHERE g.source_node_type = 'entity'
   AND g.target_node_type = 'entity'
   AND g.relationship_source IN ('DOCUMENT_EXTRACTION', 'LLM_INFERENCE')
   {missing}
 ORDER BY g.created_at
 {limit}
"""


async def embed_relationships(
    *,
    only_missing: bool = True,
    limit: int | None = None,
    dry_run: bool = False,
    verbose: bool = False,
    embedder: Embedder | None = None,
) -> RelationshipEmbedSummary:
    """Write `graph_relationships.embedding` for extracted entity edges.

    Idempotent: with `only_missing` (the default) an edge that already has
    a vector is left alone, so this is safe to run after every ingest.
    """
    summary = RelationshipEmbedSummary()
    emb = embedder or Embedder()
    sql = _SELECT_SQL.format(
        missing="AND g.embedding IS NULL" if only_missing else "",
        limit=f"LIMIT {int(limit)}" if limit else "",
    )
    async with session_scope() as session:
        rows = (await session.execute(sql_text(sql))).all()
    summary.considered = len(rows)
    if not rows:
        return summary

    pending: list[tuple[object, str]] = []
    for rid, sname, oname, relation, plabel, evidence in rows:
        t = build_edge_text(sname, oname, relation, plabel, evidence)
        if not t:
            summary.skipped_no_text += 1
            continue
        pending.append((rid, t))
    summary.samples = [t for _, t in pending[:5]]
    if dry_run or not pending:
        return summary

    for i in range(0, len(pending), _BATCH):
        batch = pending[i : i + _BATCH]
        vectors = await emb.embed([t for _, t in batch])
        async with session_scope() as session:
            await session.execute(
                sql_text("""
                UPDATE graphrag.graph_relationships AS g
                   SET embedding = cast(v.emb AS vector)
                  FROM (
                    SELECT cast(x->>0 AS uuid) AS id, x->>1 AS emb
                      FROM jsonb_array_elements(cast(:payload AS jsonb)) AS x
                  ) AS v
                 WHERE g.id = v.id
                """),
                {
                    "payload": json.dumps([
                        [str(rid), "[" + ",".join(repr(float(f)) for f in vec) + "]"]
                        for (rid, _), vec in zip(batch, vectors, strict=False)
                    ])
                },
            )
        summary.embedded += len(batch)
        if verbose:
            print(
                f"[embed-relationships] {summary.embedded}/{len(pending)} "
                f"edge(s) embedded"
            )
    summary.cost_usd = emb.total_cost_usd
    return summary
