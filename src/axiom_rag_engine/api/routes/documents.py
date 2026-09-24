"""Corpus document management: /v1/documents (+ /upload)."""

from __future__ import annotations

import asyncio
import logging
import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response

from axiom_rag_engine.api.auth import verify_api_key
from axiom_rag_engine.api.deps import Services
from axiom_rag_engine.config.settings import use_settings
from axiom_rag_engine.corpus.ingest import IngestionError, extract_text, ingest_text
from axiom_rag_engine.corpus.store import CorpusStore, DocumentMeta
from axiom_rag_engine.models import (
    DocumentIngestRequest,
    DocumentListResponse,
    DocumentResponse,
)
from axiom_rag_engine.services import AppServices

logger = logging.getLogger("axiom_rag_engine")

router = APIRouter()

_CORPUS_DISABLED_DETAIL = (
    "Corpus is not enabled. Set AXIOM_CORPUS_DB_PATH to enable document ingestion."
)
_EMBEDDING_REQUIRED_DETAIL = (
    "Document ingestion requires AXIOM_EMBEDDING_MODEL — chunks are embedded at "
    "ingest and matched by cosine at query time."
)


def _meta_to_response(meta: DocumentMeta) -> DocumentResponse:
    return DocumentResponse(
        doc_id=meta.doc_id,
        title=meta.title,
        source=meta.source,
        embedding_model=meta.embedding_model,
        content_sha=meta.content_sha,
        chunk_count=meta.chunk_count,
        char_count=meta.char_count,
        created_at=meta.created_at,
    )


def _corpus_store(services: AppServices) -> CorpusStore:
    """The corpus store, or 404 when ingestion is not enabled."""
    if services.corpus_store is None:
        raise HTTPException(status_code=404, detail=_CORPUS_DISABLED_DETAIL)
    return services.corpus_store


def _embedding_model(services: AppServices) -> str:
    model = services.settings.embedding_model
    if not model:
        raise HTTPException(status_code=409, detail=_EMBEDDING_REQUIRED_DETAIL)
    return model


async def _extract_text_or_422(
    data: str | bytes,
    *,
    filename: str | None = None,
    content_type: str | None = None,
) -> str:
    """Extract document text, mapping a format/parse failure to a clean 422.

    Parsing (pypdf, trafilatura) is CPU-bound, so it runs in a worker thread
    rather than stalling every other request on the event loop.
    """
    try:
        return await asyncio.to_thread(
            extract_text, data, filename=filename, content_type=content_type
        )
    except IngestionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


async def _ingest_and_respond(
    services: AppServices,
    *,
    doc_id: str,
    text: str,
    title: str,
    source: str,
) -> Response:
    """Shared ingest path for the JSON and file-upload endpoints."""
    use_settings(services.settings)
    store = _corpus_store(services)
    embedding_model = _embedding_model(services)
    if not text.strip():
        raise HTTPException(status_code=422, detail="Document produced no text to ingest.")
    try:
        meta = await ingest_text(
            store,
            doc_id=doc_id,
            text=text,
            embedding_model=embedding_model,
            title=title,
            source=source,
            max_chunks=services.settings.corpus_max_chunks_per_document,
        )
    except IngestionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # embedding backend failure, etc.
        # Full detail goes to the server log only: provider errors can carry
        # keys, internal URLs, or account details.
        logger.exception("Document ingestion failed for %s", doc_id)
        raise HTTPException(
            status_code=502,
            detail=f"Ingestion backend error ({type(exc).__name__}) — see server logs.",
        ) from exc
    return JSONResponse(status_code=201, content=_meta_to_response(meta).model_dump())


@router.post("/v1/documents", summary="Ingest a document into the corpus from raw text.")
async def ingest_document(
    services: Services,
    payload: DocumentIngestRequest,
    _api_key: str | None = Depends(verify_api_key),
) -> Response:
    """Chunk, embed, and store a document. Re-ingesting an existing ``doc_id``
    replaces it. Returns the stored document's metadata (201)."""
    _corpus_store(services)
    _embedding_model(services)
    return await _ingest_and_respond(
        services,
        doc_id=payload.doc_id or uuid.uuid4().hex,
        text=await _extract_text_or_422(payload.text),
        title=payload.title,
        source=payload.source,
    )


@router.post("/v1/documents/upload", summary="Ingest a document from an uploaded file.")
async def upload_document(
    services: Services,
    file: UploadFile = File(...),
    title: str = Form("", max_length=500),
    source: str = Form("", max_length=2000),
    doc_id: str | None = Form(None, max_length=200),
    _api_key: str | None = Depends(verify_api_key),
) -> Response:
    """Ingest an uploaded file (text / markdown / HTML / PDF). The filename becomes
    the default ``source`` and content type guides extraction."""
    _corpus_store(services)
    _embedding_model(services)
    data = await file.read()
    text = await _extract_text_or_422(data, filename=file.filename, content_type=file.content_type)
    return await _ingest_and_respond(
        services,
        doc_id=doc_id or uuid.uuid4().hex,
        text=text,
        title=title or (file.filename or ""),
        source=source or (file.filename or ""),
    )


@router.get(
    "/v1/documents", summary="List ingested documents.", response_model=DocumentListResponse
)
async def list_documents(
    services: Services,
    _api_key: str | None = Depends(verify_api_key),
) -> DocumentListResponse:
    store = _corpus_store(services)
    stats = await asyncio.to_thread(store.stats)
    documents = await asyncio.to_thread(store.list_documents)
    return DocumentListResponse(
        documents=[_meta_to_response(m) for m in documents],
        total_documents=stats.documents,
        total_chunks=stats.chunks,
        embedding_models=stats.embedding_models,
    )


@router.get(
    "/v1/documents/{doc_id}",
    summary="Fetch one ingested document's metadata.",
    response_model=DocumentResponse,
)
async def get_document(
    services: Services,
    doc_id: str,
    _api_key: str | None = Depends(verify_api_key),
) -> DocumentResponse:
    store = _corpus_store(services)
    meta = await asyncio.to_thread(store.get_document, doc_id)
    if meta is None:
        raise HTTPException(status_code=404, detail=f"No document with id {doc_id!r}.")
    return _meta_to_response(meta)


@router.delete("/v1/documents/{doc_id}", summary="Delete an ingested document.")
async def delete_document(
    services: Services,
    doc_id: str,
    _api_key: str | None = Depends(verify_api_key),
) -> Response:
    store = _corpus_store(services)
    if not await asyncio.to_thread(store.delete_document, doc_id):
        raise HTTPException(status_code=404, detail=f"No document with id {doc_id!r}.")
    return JSONResponse(content={"deleted": True, "doc_id": doc_id})
