"""Bring-your-own-corpus document ingestion and retrieval.

This package lets operators ingest their own documents (text / markdown / HTML,
with PDF as a follow-up) into a local, single-node store and retrieve over them
through the same verified-RAG pipeline used for web search. It is dependency-light
by design — SQLite plus the stdlib, no vector-DB server and no numpy — matching
the engine's existing pure-Python retrieval philosophy.

Public surface:
    CorpusStore   — SQLite-backed persistence + brute-force cosine search.
    DocumentMeta  — metadata for one ingested document.
    ScoredChunk   — a search hit (chunk text + similarity + provenance).
    ingest_text   — parse → chunk → embed → persist one document.
    extract_text  — format-aware raw-bytes/str → clean text.
"""

from __future__ import annotations

from axiom_rag_engine.corpus.ingest import (
    IngestionError,
    extract_text,
    ingest_text,
)
from axiom_rag_engine.corpus.store import (
    CorpusStore,
    DocumentMeta,
    ScoredChunk,
)

__all__ = [
    "CorpusStore",
    "DocumentMeta",
    "IngestionError",
    "ScoredChunk",
    "extract_text",
    "ingest_text",
]
