"""
Axiom Engine v2.3 — Retrieval & Indexing Node (Module 2)

Responsibilities:
  - Generates search queries from the user query (original + reformulations).
  - Executes searches via a pluggable search backend (Tavily, Exa, or mock).
  - Deduplicates results by URL across multiple queries.
  - Strips HTML from retrieved content.
  - Chunks content into paragraphs.
  - Deduplicates chunks by content hash to prevent redundant indexing.
  - Assigns strict unique alphanumeric IDs: doc_X_chunk_Y (e.g. doc_1_chunk_A).
  - Filters out results from banned_domains.
  - Updates GraphState keys: search_queries, indexed_chunks, audit_trail.
"""

from __future__ import annotations

import hashlib
import logging
import re
import string
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, Protocol
from urllib.parse import urlparse

from tenacity import retry, stop_after_attempt, wait_exponential

from axiom_engine.state import GraphState
from axiom_engine.utils.audit import make_audit_event

_audit = partial(make_audit_event, "retriever")
logger = logging.getLogger("axiom_engine.retriever")

# ---------------------------------------------------------------------------
# Chunk ID generation — sequential alphanumeric labels (A, B, C, ..., Z, AA, AB, ...)
# ---------------------------------------------------------------------------


def _chunk_label(index: int) -> str:
    """Convert a 0-based index into an alphanumeric label: 0→A, 25→Z, 26→AA."""
    label = ""
    i = index
    while True:
        label = string.ascii_uppercase[i % 26] + label
        i = i // 26 - 1
        if i < 0:
            break
    return label


# ---------------------------------------------------------------------------
# HTML stripping
# ---------------------------------------------------------------------------

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_COLLAPSE_RE = re.compile(r"\n{3,}")


def strip_html(raw: str) -> str:
    """Remove HTML tags and collapse excessive newlines."""
    text = _HTML_TAG_RE.sub("", raw)
    text = _WHITESPACE_COLLAPSE_RE.sub("\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Paragraph-based chunking (architecture §Module 2)
# ---------------------------------------------------------------------------

_MIN_CHUNK_LENGTH = 40  # Skip trivially short paragraphs (nav bars, footers)


def chunk_into_paragraphs(text: str) -> list[str]:
    """
    Split text into paragraphs on double-newline boundaries.
    Filters out chunks shorter than _MIN_CHUNK_LENGTH.
    """
    raw_paragraphs = re.split(r"\n\s*\n", text)
    return [p.strip() for p in raw_paragraphs if len(p.strip()) >= _MIN_CHUNK_LENGTH]


# ---------------------------------------------------------------------------
# Search backend protocol
# ---------------------------------------------------------------------------


class SearchBackend(Protocol):
    """Interface for pluggable search providers (Tavily, Exa, mock)."""

    def search(self, query: str) -> list[dict[str, Any]]:
        """
        Execute a search and return a list of result dicts.
        Each dict must have at minimum:
          - "url": str
          - "content": str (may contain HTML)
          - "title": str (optional)
        """
        ...


class MockSearchBackend:
    """
    Default mock backend used when no real search provider is configured.
    Returns pre-loaded results for testing.
    """

    def __init__(self, results: list[dict[str, Any]] | None = None) -> None:
        self._results = results or []

    def search(self, query: str) -> list[dict[str, Any]]:
        return self._results


# ---------------------------------------------------------------------------
# Query expansion
# ---------------------------------------------------------------------------


def generate_search_queries(user_query: str) -> list[str]:
    """
    Generate search queries from the user query.
    Returns the original query plus simple reformulations.
    """
    queries = [user_query]

    # Add a "what is" reformulation if the query doesn't already start with it.
    lower = user_query.lower().strip()
    if not lower.startswith("what is") and not lower.startswith("what are"):
        queries.append(f"What is {user_query}")

    # Add an "explain" reformulation.
    if not lower.startswith("explain"):
        queries.append(f"Explain {user_query}")

    return queries


# ---------------------------------------------------------------------------
# Domain filtering
# ---------------------------------------------------------------------------


def extract_domain(url: str) -> str:
    """Extract the domain from a URL, stripping www. prefix."""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        if domain.startswith("www."):
            domain = domain[4:]
        return domain
    except Exception:
        return ""


def is_banned(url: str, banned_domains: list[str]) -> bool:
    """Check if a URL's domain is in the banned list."""
    domain = extract_domain(url)
    return any(domain == bd.lower() or domain.endswith("." + bd.lower()) for bd in banned_domains)


# ---------------------------------------------------------------------------
# Content hashing for deduplication
# ---------------------------------------------------------------------------


def _content_hash(text: str) -> str:
    """Return a short SHA-256 hex digest for deduplication."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

# Module-level search backend — replaced via set_search_backend() for testing.
_search_backend: SearchBackend = MockSearchBackend()


def set_search_backend(backend: SearchBackend) -> None:
    """Replace the module-level search backend (for DI/testing)."""
    global _search_backend
    _search_backend = backend


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=4),
    reraise=True,
)
def _search_with_retry(query: str) -> list[dict[str, Any]]:
    """Execute a search with exponential-backoff retry (3 attempts)."""
    return _search_backend.search(query)


def _safe_search(
    query: str,
) -> tuple[str, list[dict[str, Any]], Exception | None]:
    """
    Execute one search query, capturing any exception so a single-query
    failure does not abort the other parallel searches.

    Returns (query, results, error_or_None).
    """
    try:
        return query, _search_with_retry(query), None
    except Exception as exc:
        return query, [], exc


def retriever_node(state: GraphState) -> dict[str, Any]:
    """
    LangGraph node — Retrieval & Indexing.

    Generates search queries, executes searches, deduplicates by URL,
    strips HTML, chunks into paragraphs, deduplicates chunks by content
    hash, assigns doc_X_chunk_Y IDs, and filters banned domains.

    Returns keys: search_queries, indexed_chunks, audit_trail
    """
    audit: list[dict[str, Any]] = []
    app_cfg: dict = state.get("app_config") or {}
    banned: list[str] = app_cfg.get("banned_domains") or []

    user_query: str = state["user_query"]
    queries = generate_search_queries(user_query)

    audit.append(
        _audit(
            "retriever_start",
            {"query_count": len(queries), "banned_domains": banned},
        )
    )

    indexed_chunks: list[dict[str, Any]] = []
    doc_counter = 1
    total_results = 0
    total_banned = 0
    total_duplicate_urls = 0
    total_duplicate_chunks = 0

    seen_urls: set[str] = set()
    seen_chunk_hashes: set[str] = set()

    # Run all search queries in parallel; results arrive in submission order
    # so doc_counter assignment is deterministic across calls.
    with ThreadPoolExecutor(max_workers=len(queries)) as executor:
        search_outcomes = list(executor.map(_safe_search, queries))

    for query, results, exc in search_outcomes:
        if exc is not None:
            audit.append(
                _audit(
                    "retriever_search_error",
                    {"query": query, "error": str(exc)},
                )
            )
            continue

        for result in results:
            url: str = result.get("url", "")
            total_results += 1

            # Filter banned domains.
            if is_banned(url, banned):
                total_banned += 1
                audit.append(
                    _audit(
                        "retriever_banned_domain",
                        {"url": url, "domain": extract_domain(url)},
                    )
                )
                continue

            # Deduplicate by URL across queries.
            normalized_url = url.lower().rstrip("/")
            if normalized_url in seen_urls:
                total_duplicate_urls += 1
                continue
            seen_urls.add(normalized_url)

            raw_content: str = result.get("content", "")
            clean_text = strip_html(raw_content)

            if not clean_text:
                continue

            paragraphs = chunk_into_paragraphs(clean_text)
            title: str = result.get("title", "")

            for chunk_idx, paragraph in enumerate(paragraphs):
                # Deduplicate by content hash across all chunks.
                h = _content_hash(paragraph)
                if h in seen_chunk_hashes:
                    total_duplicate_chunks += 1
                    continue
                seen_chunk_hashes.add(h)

                chunk_id = f"doc_{doc_counter}_chunk_{_chunk_label(chunk_idx)}"
                indexed_chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "text": paragraph,
                        "source_url": url,
                        "domain": extract_domain(url),
                        "title": title,
                        "doc_index": doc_counter,
                        "chunk_index": chunk_idx,
                    }
                )

            doc_counter += 1

    audit.append(
        _audit(
            "retriever_complete",
            {
                "total_search_results": total_results,
                "banned_filtered": total_banned,
                "duplicate_urls_skipped": total_duplicate_urls,
                "duplicate_chunks_skipped": total_duplicate_chunks,
                "total_chunks": len(indexed_chunks),
            },
        )
    )

    return {
        "search_queries": queries,
        "indexed_chunks": indexed_chunks,
        "audit_trail": audit,
    }
