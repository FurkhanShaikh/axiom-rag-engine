"""Ingest one document into the corpus store: extract → chunk → embed → persist.

Text extraction is format-aware but deliberately small: plain text and markdown
pass through as-is; HTML is run through the retriever's ``strip_html`` (the same
boilerplate removal used for web results); PDF is parsed with ``pypdf``. PDF is
the only format needing a heavy parser, so its import is lazy — it costs nothing
until a PDF is actually ingested.

Chunking and embedding reuse the production pipeline verbatim: the same
``chunk_into_paragraphs`` the web retriever uses, and the same LiteLLM embedder
the ranker uses. That means an ingested chunk is scored, ranked, and verified by
exactly the code that handles a web chunk — corpus and web answers are held to
one standard.
"""

from __future__ import annotations

import io
import re
from collections.abc import Awaitable, Callable

from axiom_rag_engine.corpus.store import CorpusStore, DocumentMeta
from axiom_rag_engine.embeddings import embed_documents
from axiom_rag_engine.nodes.retriever import chunk_into_paragraphs, strip_html

# An async ``(model, texts) -> vectors`` embedder. Injectable so ingestion is
# unit-testable without a live embedding backend.
Embedder = Callable[[str, list[str]], Awaitable[list[list[float]]]]


class IngestionError(Exception):
    """Raised when a document cannot be turned into usable, embedded chunks."""


_HTML_EXTENSIONS = (".html", ".htm", ".xhtml")
_HTML_DOC_SIGNATURE = re.compile(r"<\s*html[\s>]|<!doctype\s+html", re.IGNORECASE)
_PDF_MAGIC = b"%PDF-"


def _looks_like_pdf(data: str | bytes, filename: str | None, content_type: str | None) -> bool:
    if isinstance(data, bytes) and data[:5] == _PDF_MAGIC:
        return True
    if filename and filename.lower().endswith(".pdf"):
        return True
    return bool(content_type and "pdf" in content_type.lower())


def _looks_like_html(text: str, filename: str | None, content_type: str | None) -> bool:
    if filename and filename.lower().endswith(_HTML_EXTENSIONS):
        return True
    if content_type and "html" in content_type.lower():
        return True
    return bool(_HTML_DOC_SIGNATURE.search(text[:1000]))


def _extract_pdf(data: bytes) -> str:
    """Extract text from a PDF's pages, joined by blank lines.

    ``pypdf`` is imported lazily so it is a cost only for PDF ingestion. A
    corrupt or encrypted PDF raises :class:`IngestionError` rather than a raw
    parser error, so the API can turn it into a clean 422.
    """
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(io.BytesIO(data))
        pages = [(page.extract_text() or "").strip() for page in reader.pages]
    except (PdfReadError, ValueError, OSError) as exc:
        raise IngestionError(f"could not read PDF: {exc}") from exc
    return "\n\n".join(p for p in pages if p)


def extract_text(
    data: str | bytes,
    *,
    filename: str | None = None,
    content_type: str | None = None,
) -> str:
    """Turn raw document bytes/str into clean text ready for chunking.

    Format is resolved before decoding, so binary formats are never UTF-8
    mangled: PDF (magic bytes / ``.pdf`` / content type) is parsed with pypdf;
    HTML (signature / ``.htm*`` / content type) is boilerplate-stripped;
    everything else (txt, md, rst) is treated as UTF-8 plain text (decoded with
    replacement so one bad byte never fails an ingest).

    Raises:
        IngestionError: a PDF was detected but the input is text, or unreadable.
    """
    if _looks_like_pdf(data, filename, content_type):
        if not isinstance(data, bytes):
            raise IngestionError("PDF input must be raw bytes, not text.")
        return _extract_pdf(data)

    text = data.decode("utf-8", errors="replace") if isinstance(data, bytes) else data
    if _looks_like_html(text, filename, content_type):
        return strip_html(text)
    return text.strip()


async def ingest_text(
    store: CorpusStore,
    *,
    doc_id: str,
    text: str,
    embedding_model: str,
    title: str = "",
    source: str = "",
    embedder: Embedder | None = None,
    max_chunks: int | None = None,
) -> DocumentMeta:
    """Chunk, embed, and persist ``text`` as document ``doc_id``.

    Re-ingesting an existing ``doc_id`` replaces it (see
    :meth:`CorpusStore.add_document`). Returns the stored document's metadata.

    Args:
        text: Already-extracted plain text (call :func:`extract_text` first for
            raw uploads).
        embedding_model: LiteLLM embedding model; recorded on the document so
            retrieval only compares same-model vectors.
        embedder: Override the embedding backend (tests inject a fake). Defaults
            to :func:`embed_documents`.
        max_chunks: Optional cap on chunks per document (guards a pathological
            upload from dominating the corpus).

    Raises:
        IngestionError: the text yields no usable chunks, or the embedder
            returns the wrong number of vectors.
    """
    embedder = embedder or embed_documents

    chunks = chunk_into_paragraphs(text)
    if max_chunks is not None and max_chunks > 0:
        chunks = chunks[:max_chunks]
    if not chunks:
        raise IngestionError(
            f"document {doc_id!r} produced no usable chunks (empty, too short, or all boilerplate)"
        )

    vectors = await embedder(embedding_model, chunks)
    if len(vectors) != len(chunks):
        raise IngestionError(f"embedder returned {len(vectors)} vectors for {len(chunks)} chunks")

    return store.add_document(
        doc_id=doc_id,
        title=title,
        source=source,
        embedding_model=embedding_model,
        chunks=list(zip(chunks, vectors, strict=True)),
    )
