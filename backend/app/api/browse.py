"""Browse endpoints: documents, entities, classes, artifacts.

Read-only. Each returns the row plus a few neighbor links for UI use.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query, status
from sqlalchemy import text as sql_text

from backend.app.db.session import session_scope


router = APIRouter(tags=["browse"])


# ----- documents -----

@router.get("/documents/{iri:path}", operation_id="document_get")
async def get_document(iri: str) -> dict[str, Any]:
    async with session_scope() as session:
        r = await session.execute(
            sql_text("""
            SELECT id, document_identifier, title, file_name, file_path,
                   file_type, document_hash, text_summary, status, version,
                   created_at
              FROM graphrag.documents WHERE document_identifier = :iri
            """),
            {"iri": iri},
        )
        row = r.first()
        if row is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"document not found: {iri}",
            )
        did, diri, title, fname, fpath, ftype, dhash, summ, sts, ver, ts = row
        # Chunk count.
        cr = await session.execute(
            sql_text(
                "SELECT count(*) FROM graphrag.chunks "
                "WHERE document_id = :id AND status = 'ACTIVE'"
            ),
            {"id": did},
        )
        chunk_count = int(cr.scalar_one())
    return {
        "iri": diri, "title": title, "file_name": fname, "file_path": fpath,
        "file_type": ftype, "document_hash": dhash,
        "text_summary": summ, "status": sts, "version": ver,
        "chunk_count": chunk_count, "created_at": ts.isoformat(),
    }


@router.get("/documents", operation_id="document_list")
async def list_documents(
    status_filter: str | None = Query(default=None, alias="status"),
    limit: int = Query(default=50, ge=1, le=500),
) -> list[dict[str, Any]]:
    params: dict[str, Any] = {"limit": limit}
    where = ""
    if status_filter is not None:
        where = "WHERE status = :status"
        params["status"] = status_filter
    async with session_scope() as session:
        r = await session.execute(
            sql_text(f"""
            SELECT document_identifier, title, status, version, created_at,
                   (SELECT count(*) FROM graphrag.chunks
                     WHERE document_id = d.id AND status='ACTIVE') AS chunks
              FROM graphrag.documents d {where}
             ORDER BY created_at DESC LIMIT :limit
            """),
            params,
        )
        return [
            {"iri": iri, "title": t, "status": s, "version": v,
             "chunks": ch, "created_at": ts.isoformat()}
            for iri, t, s, v, ts, ch in r.all()
        ]


# ----- entities -----

@router.get("/entities/{iri:path}", operation_id="entity_get")
async def get_entity(iri: str) -> dict[str, Any]:
    async with session_scope() as session:
        r = await session.execute(
            sql_text("""
            SELECT e.id, e.entity_identifier, e.name, e.normalized_name,
                   e.status, c.label AS class_label, c.iri AS class_iri
              FROM graphrag.entities e
              JOIN graphrag.ontology_classes c ON c.id = e.class_id
             WHERE e.entity_identifier = :iri
            """),
            {"iri": iri},
        )
        row = r.first()
        if row is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"entity not found: {iri}",
            )
        eid, eiri, name, nname, sts, clabel, ciri = row
        # Chunks asserting about this entity.
        cr = await session.execute(
            sql_text("""
            SELECT c.chunk_identifier, left(c.text, 200), d.document_identifier, d.title
              FROM graphrag.graph_relationships gr
              JOIN graphrag.chunks c ON c.id = gr.source_chunk_id
              JOIN graphrag.documents d ON d.id = c.document_id
             WHERE gr.target_node_id = :eid
               AND gr.target_node_type = 'entity'
               AND gr.predicate_label = 'viao:assertsAbout'
               AND gr.relationship_source = 'DOCUMENT_EXTRACTION'
             LIMIT 20
            """),
            {"eid": eid},
        )
        chunks = [
            {"chunk_iri": ciri_, "snippet": txt + "...",
             "document_iri": diri, "document_title": dtitle}
            for ciri_, txt, diri, dtitle in cr.all()
        ]
    return {
        "iri": eiri, "name": name, "normalized_name": nname,
        "status": sts, "class_label": clabel, "class_iri": ciri,
        "chunk_count": len(chunks),
        "chunks": chunks,
    }


# ----- classes -----

@router.get("/classes/{iri:path}", operation_id="class_get")
async def get_class(iri: str) -> dict[str, Any]:
    async with session_scope() as session:
        r = await session.execute(
            sql_text("""
            SELECT id, iri, label, description, namespace, source_ontology,
                   is_viao_class
              FROM graphrag.ontology_classes WHERE iri = :iri
            """),
            {"iri": iri},
        )
        row = r.first()
        if row is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"class not found: {iri}",
            )
        cid, ciri, label, descr, ns, src, viao = row
        # Direct subclasses.
        sr = await session.execute(
            sql_text("""
            SELECT child.iri, child.label
              FROM graphrag.graph_relationships gr
              JOIN graphrag.ontology_classes child ON child.id = gr.source_node_id
             WHERE gr.target_node_id = :id
               AND gr.target_node_type = 'ontology_class'
               AND gr.source_node_type = 'ontology_class'
               AND gr.predicate_label = 'rdfs:subClassOf'
             LIMIT 30
            """),
            {"id": cid},
        )
        children = [{"iri": ci, "label": cl} for ci, cl in sr.all()]
        # Entity instances of this class.
        er = await session.execute(
            sql_text(
                "SELECT entity_identifier, name FROM graphrag.entities "
                "WHERE class_id = :id LIMIT 20"
            ),
            {"id": cid},
        )
        entities = [{"iri": eiri, "name": en} for eiri, en in er.all()]
    return {
        "iri": ciri, "label": label, "description": descr,
        "namespace": ns, "source_ontology": src, "is_viao_class": viao,
        "subclass_count": len(children), "subclasses": children,
        "entity_count": len(entities), "entities": entities,
    }


# ----- artifacts -----

@router.get("/artifacts/{iri:path}", operation_id="artifact_get")
async def get_artifact(iri: str) -> dict[str, Any]:
    async with session_scope() as session:
        r = await session.execute(
            sql_text("""
            SELECT id, artifact_identifier, artifact_type, title, text,
                   confidence, model_name, prompt_version, status,
                   graph_version, created_at
              FROM graphrag.intelligence_artifacts
             WHERE artifact_identifier = :iri
            """),
            {"iri": iri},
        )
        row = r.first()
        if row is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"artifact not found: {iri}",
            )
        aid, airi, atype, title, text, conf, mname, pv, sts, gv, ts = row
        # Source chunks.
        sr = await session.execute(
            sql_text("""
            SELECT c.chunk_identifier, left(c.text, 200), d.title
              FROM graphrag.artifact_sources asrc
              JOIN graphrag.chunks c ON c.id = asrc.chunk_id
              JOIN graphrag.documents d ON d.id = c.document_id
             WHERE asrc.artifact_id = :id LIMIT 20
            """),
            {"id": aid},
        )
        sources = [
            {"chunk_iri": ci, "snippet": tx + "...", "document_title": dt}
            for ci, tx, dt in sr.all()
        ]
        # Entities mentioned by this artifact.
        er = await session.execute(
            sql_text("""
            SELECT e.entity_identifier, e.name, oc.label AS class_label
              FROM graphrag.graph_relationships gr
              JOIN graphrag.entities e ON e.id = gr.target_node_id
              JOIN graphrag.ontology_classes oc ON oc.id = e.class_id
             WHERE gr.source_node_id = :id
               AND gr.source_node_type = 'intelligence_artifact'
               AND gr.predicate_label = 'viao:assertsAbout'
            """),
            {"id": aid},
        )
        entities = [
            {"iri": eiri, "name": en, "class_label": cl}
            for eiri, en, cl in er.all()
        ]
    return {
        "iri": airi, "type": atype, "title": title, "text": text,
        "confidence": float(conf) if conf is not None else None,
        "model_name": mname, "prompt_version": pv,
        "status": sts, "graph_version": gv,
        "created_at": ts.isoformat(),
        "source_count": len(sources), "sources": sources,
        "entity_count": len(entities), "entities": entities,
    }
