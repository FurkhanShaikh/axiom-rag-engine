"""Search backends for the ingested corpus.

Corpus retrieval plugs in behind the same ``SearchBackend`` protocol the web
retriever uses, so an ingested document flows through the identical
chunk → score → rank → verify pipeline as a web result — corpus and web answers
are held to one standard, and no new retrieval path has to be maintained.

``CorpusSearchBackend`` embeds the query with the corpus's embedding model and
returns the top-k chunks by cosine similarity, shaped as retriever result dicts.
``CompositeSearchBackend`` fans one query out to several backends (e.g. web +
corpus) and concatenates their results; the retriever already deduplicates by URL
and content hash, so overlap is harmless.

Async-in-sync
-------------
The retriever calls ``search`` off the event loop (``asyncio.to_thread``), so
these backends are synchronous. Query embedding is async (LiteLLM), so it runs
under ``asyncio.run`` inside that worker thread — safe because no event loop is
running there.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from urllib.parse import quote

from axiom_rag_engine.corpus.store import CorpusStore
from axiom_rag_engine.embeddings import embed_query
from axiom_rag_engine.nodes.retriever import SearchBackend

logger = logging.getLogger("axiom_rag_engine.search.corpus")

# Synthetic host for citations back to ingested documents. It is a real,
# dot-bearing hostname so the retriever's SSRF/URL filter accepts it, while
# clearly marking the source as local corpus rather than a live web page.
_CORPUS_HOST = "corpus.local"


def corpus_chunk_url(doc_id: str, chunk_index: int) -> str:
    """Stable, filter-passing citation URL for one corpus chunk.

    The fragment keeps per-chunk URLs distinct so the retriever treats each hit
    as its own source instead of deduplicating them by URL.
    """
    return f"https://{_CORPUS_HOST}/doc/{quote(doc_id, safe='')}#chunk-{chunk_index}"


class CorpusSearchBackend:
    """SearchBackend over an ingested document corpus (dense retrieval).

    Args:
        store: The corpus store to query.
        embedding_model: Model used to embed the query. Must match the model the
            documents were embedded with — the store only scores same-model
            chunks, so a mismatch simply returns nothing.
        max_results: Top-k chunks returned per query.
    """

    def __init__(
        self,
        store: CorpusStore,
        embedding_model: str,
        max_results: int = 5,
    ) -> None:
        self._store = store
        self._model = embedding_model
        self._max_results = max_results

    def search(self, query: str) -> list[dict[str, Any]]:
        if not query.strip():
            return []
        try:
            query_vec = asyncio.run(embed_query(self._model, query))
        except Exception:
            # Fail soft: a dead embedder must not abort retrieval. With 'both',
            # web results still flow; with 'corpus' the retriever reports empty.
            logger.exception("Corpus query embedding failed for %r", query)
            return []

        hits = self._store.search(query_vec, embedding_model=self._model, k=self._max_results)
        return [
            {
                "url": corpus_chunk_url(hit.doc_id, hit.chunk_index),
                "content": hit.text,
                "title": hit.title,
                # Stored chunks are full extracted text, not search snippets, so
                # citations are verified against the real source content.
                "content_mode": "raw",
            }
            for hit in hits
        ]


class CompositeSearchBackend:
    """Fan a query out to several backends and concatenate their results.

    Order is preserved (earlier backends first). The retriever deduplicates the
    merged stream by URL and content hash, so backends returning the same source
    do not double-count. A single backend raising does not sink the others.
    """

    def __init__(self, backends: list[SearchBackend]) -> None:
        self._backends = backends

    def search(self, query: str) -> list[dict[str, Any]]:
        merged: list[dict[str, Any]] = []
        for backend in self._backends:
            try:
                merged.extend(backend.search(query))
            except Exception:
                logger.exception(
                    "Backend %s failed for %r; continuing with the rest.",
                    type(backend).__name__,
                    query,
                )
        return merged
