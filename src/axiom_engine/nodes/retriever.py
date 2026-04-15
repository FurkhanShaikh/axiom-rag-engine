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

import asyncio
import hashlib
import html as _html_stdlib
import ipaddress
import logging
import re
import string
from functools import partial
from typing import Any, Protocol
from urllib.parse import urlparse

import pysbd
import trafilatura
from tenacity import retry, stop_after_attempt, wait_exponential

from axiom_engine.config.observability import get_tracer
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
# HTML / content extraction
# ---------------------------------------------------------------------------

_SCRIPT_STYLE_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", flags=re.DOTALL | re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_COLLAPSE_RE = re.compile(r"\n{3,}")
# Signature that indicates a full HTML document rather than a fragment.
_HTML_DOC_SIGNATURE = re.compile(r"<\s*html[\s>]|<!doctype\s+html", re.IGNORECASE)


def strip_html(raw: str) -> str:
    """
    Extract clean plain text from raw content.

    For full HTML documents (detected via <html>/<DOCTYPE> signature) trafilatura
    is used — it removes boilerplate, navigation, and ads, handles malformed
    markup, and decodes entities natively.

    For HTML fragments and already-extracted text (the typical case from Tavily)
    a fast regex path is used, followed by html.unescape() so that residual
    entities (&amp;, &nbsp;, &#x2019;, etc.) are decoded rather than left as
    literal character sequences.
    """
    if not raw:
        return ""
    if _HTML_DOC_SIGNATURE.search(raw[:1000]):
        extracted = trafilatura.extract(
            raw, include_tables=False, include_comments=False, no_fallback=False
        )
        if extracted:
            return extracted.strip()
    # Fragment / plain-text path.
    text = _SCRIPT_STYLE_RE.sub("", raw)
    text = _HTML_TAG_RE.sub("", text)
    text = _html_stdlib.unescape(text)
    text = _WHITESPACE_COLLAPSE_RE.sub("\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Sentence-aware chunking with overlap (architecture §Module 2)
# ---------------------------------------------------------------------------

_MIN_CHUNK_LENGTH = 40  # Skip trivially short paragraphs (nav bars, footers)
_MAX_CHUNK_LENGTH = 1500  # Prevent unbounded context overflow
_OVERLAP_SENTENCES = 1  # Sentences carried forward to bridge chunk boundaries
_MAX_CHUNKS_PER_REQUEST = 200  # Hard cap — configurable via app_config

# Module-level segmenter; pySBD is stateless so this is safe for concurrent use.
_SEGMENTER = pysbd.Segmenter(language="en", clean=False)


def _split_sentences(text: str) -> list[str]:
    """Return a list of non-empty sentences using pySBD."""
    return [s.strip() for s in _SEGMENTER.segment(text) if s.strip()]


def chunk_into_paragraphs(text: str) -> list[str]:
    """
    Split text into bounded, overlapping chunks.

    Strategy:
      1. Split on paragraph boundaries (double newline).
      2. Paragraphs within _MAX_CHUNK_LENGTH are emitted as-is.
      3. Longer paragraphs are sentence-segmented via pySBD and windowed:
         each window accumulates sentences up to _MAX_CHUNK_LENGTH, then the
         last _OVERLAP_SENTENCES are carried into the next window so that
         claims whose evidence spans a split point remain mechanically verifiable.
    Chunks shorter than _MIN_CHUNK_LENGTH are discarded (nav bars, footers).
    """
    raw_paragraphs = re.split(r"\n\s*\n", text)
    chunks: list[str] = []

    for para in raw_paragraphs:
        para = para.strip()
        if len(para) < _MIN_CHUNK_LENGTH:
            continue

        if len(para) <= _MAX_CHUNK_LENGTH:
            chunks.append(para)
            continue

        # Long paragraph: sentence-window with overlap.
        sentences = _split_sentences(para)
        if not sentences:
            # pySBD returned nothing (unusual) — fall back to hard truncation.
            start = 0
            while start < len(para):
                part = para[start : start + _MAX_CHUNK_LENGTH].strip()
                if len(part) >= _MIN_CHUNK_LENGTH:
                    chunks.append(part)
                start += _MAX_CHUNK_LENGTH
            continue

        window: list[str] = []
        window_len = 0

        for sent in sentences:
            # Guard against pathologically long individual sentences.
            if len(sent) > _MAX_CHUNK_LENGTH:
                sent = sent[:_MAX_CHUNK_LENGTH]
            slen = len(sent)

            if window and window_len + slen + 1 > _MAX_CHUNK_LENGTH:
                candidate = " ".join(window)
                if len(candidate) >= _MIN_CHUNK_LENGTH:
                    chunks.append(candidate)
                # Carry the tail of the current window into the next.
                overlap = window[-_OVERLAP_SENTENCES:]
                window = [*overlap, sent]
                window_len = sum(len(s) + 1 for s in window)
            else:
                window.append(sent)
                window_len += slen + 1

        if window:
            candidate = " ".join(window)
            if len(candidate) >= _MIN_CHUNK_LENGTH:
                chunks.append(candidate)

    return chunks


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


def generate_search_queries(
    user_query: str,
    rewrite_requests: list[str] | None = None,
) -> list[str]:
    """
    Generate search queries from the user query.

    On first retrieval (rewrite_requests is None or empty) returns the original
    query plus simple reformulations.

    On re-retrieval after a failed rewrite loop (rewrite_requests is non-empty),
    the failure context is injected as targeted search terms so the fresh
    retrieval is more likely to surface chunks that can fix the specific
    verification failures rather than returning the same results again (C5 fix).
    """
    queries = [user_query]
    lower = user_query.lower().strip()

    if rewrite_requests:
        # Generate varied formulation to force the search engine to return fresh
        # sources without polluting the query with internal stack traces.
        if f"{user_query} details" not in queries:
            queries.append(f"{user_query} details")
        if f"{user_query} evidence facts" not in queries:
            queries.append(f"{user_query} evidence facts")
    else:
        # First-pass reformulations.
        if not lower.startswith("what is") and not lower.startswith("what are"):
            queries.append(f"What is {user_query}")
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
    except ValueError:
        return ""
    domain = (parsed.netloc or "").lower()
    if domain.startswith("www."):
        domain = domain[4:]
    # Strip any :port suffix.
    return domain.split(":", 1)[0]


def is_banned(url: str, banned_domains: list[str]) -> bool:
    """Check if a URL's domain is in the banned list."""
    domain = extract_domain(url)
    return any(domain == bd.lower() or domain.endswith("." + bd.lower()) for bd in banned_domains)


# Only http(s) URLs with public hostnames are accepted from the search backend.
# This is a defense-in-depth filter: even if a backend returns a poisoned result
# pointing at file://, gopher://, internal IPs, or link-local addresses, the
# retriever will drop the URL before its content is chunked and fed to the LLM.
_ALLOWED_SCHEMES = frozenset({"http", "https"})


def is_safe_public_url(url: str) -> tuple[bool, str]:
    """
    Return (ok, reason). ``ok`` is False when the URL must be rejected.

    Rejects:
      - non-http(s) schemes (file://, gopher://, ftp://, data://, javascript:)
      - empty/unparseable hosts
      - raw IP literals inside private / loopback / link-local / reserved ranges
      - ``localhost`` and bare hostnames with no dot (likely internal)
    """
    if not url:
        return False, "empty_url"
    try:
        parsed = urlparse(url)
    except ValueError as exc:
        return False, f"unparseable:{exc}"

    scheme = (parsed.scheme or "").lower()
    if scheme not in _ALLOWED_SCHEMES:
        return False, f"disallowed_scheme:{scheme or 'none'}"

    host = (parsed.hostname or "").lower()
    if not host:
        return False, "empty_host"

    if host in {"localhost", "localhost.localdomain"}:
        return False, "localhost"

    # If the host parses as an IP literal, block anything that isn't global.
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        ip = None
    if ip is not None:
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            return False, f"non_global_ip:{host}"
        return True, "ok"

    # Hostnames without a dot are almost always internal (e.g. "intranet").
    if "." not in host:
        return False, "bare_hostname"

    return True, "ok"


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


async def _safe_search(
    query: str,
) -> tuple[str, list[dict[str, Any]], Exception | None]:
    """
    Execute one search query asynchronously (Tavily client is synchronous, so
    it runs in a thread), capturing any exception so a single-query failure
    does not abort the other parallel searches.

    Returns (query, results, error_or_None).
    """
    try:
        results = await asyncio.to_thread(_search_with_retry, query)
        return query, results, None
    except Exception as exc:  # intentional: isolate per-query failure
        logger.warning("Search query %r failed: %s", query, exc)
        return query, [], exc


async def retriever_node(state: GraphState) -> dict[str, Any]:
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
    rewrite_requests: list[str] = list(state.get("rewrite_requests") or [])
    queries = generate_search_queries(user_query, rewrite_requests=rewrite_requests or None)

    audit.append(
        _audit(
            "retriever_start",
            {"query_count": len(queries), "banned_domains": banned},
        )
    )

    max_chunks = int(app_cfg.get("max_chunks_per_request") or _MAX_CHUNKS_PER_REQUEST)

    indexed_chunks: list[dict[str, Any]] = []
    doc_counter = int(state.get("next_doc_index", 1))
    total_results = 0
    total_banned = 0
    total_unsafe_urls = 0
    total_duplicate_urls = 0
    total_duplicate_chunks = 0
    failed_queries = 0
    cap_hit = False

    # On re-retrieval, seed seen_urls with URLs already mapped globally from past cycles.
    past_urls: list[str] = list(state.get("past_seen_urls") or [])
    seen_urls: set[str] = {url for url in past_urls if url}
    seen_chunk_hashes: set[str] = set()
    new_urls: list[str] = []

    # Run all search queries concurrently; asyncio.gather preserves submission
    # order so doc_counter assignment remains deterministic across calls.
    # Tavily is a sync client, so _safe_search wraps it in asyncio.to_thread.
    tracer = get_tracer()
    with tracer.start_as_current_span("retriever.search", attributes={"query_count": len(queries)}):
        search_outcomes = list(await asyncio.gather(*[_safe_search(q) for q in queries]))

    for query, results, exc in search_outcomes:
        if cap_hit:
            break
        if exc is not None:
            failed_queries += 1
            audit.append(
                _audit(
                    "retriever_search_error",
                    {"query": query, "error": str(exc)},
                )
            )
            continue

        for result in results:
            if cap_hit:
                break
            url: str = result.get("url", "")
            total_results += 1

            # SSRF defense-in-depth: drop non-http(s) schemes, internal IPs,
            # link-local / loopback / private ranges, and bare hostnames before
            # any content is ingested into the LLM context.
            url_ok, url_reason = is_safe_public_url(url)
            if not url_ok:
                total_unsafe_urls += 1
                audit.append(
                    _audit(
                        "retriever_unsafe_url",
                        {"url": url, "reason": url_reason},
                    )
                )
                continue

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
            new_urls.append(normalized_url)

            raw_content: str = result.get("content", "")
            clean_text = strip_html(raw_content)

            if not clean_text:
                audit.append(_audit("retriever_empty_content", {"url": url}))
                continue

            paragraphs = chunk_into_paragraphs(clean_text)
            title: str = result.get("title", "")

            if not paragraphs:
                audit.append(_audit("retriever_no_chunks", {"url": url}))
                doc_counter += 1
                continue

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
            if len(indexed_chunks) >= max_chunks:
                cap_hit = True
                audit.append(
                    _audit(
                        "retriever_chunk_cap_reached",
                        {"cap": max_chunks, "total_chunks": len(indexed_chunks)},
                    )
                )
                break

    audit.append(
        _audit(
            "retriever_complete",
            {
                "total_search_results": total_results,
                "failed_queries": failed_queries,
                "banned_filtered": total_banned,
                "unsafe_urls_filtered": total_unsafe_urls,
                "duplicate_urls_skipped": total_duplicate_urls,
                "duplicate_chunks_skipped": total_duplicate_chunks,
                "total_chunks": len(indexed_chunks),
            },
        )
    )

    if failed_queries == len(queries) and queries:
        raise RuntimeError("Retriever stage failed: all search queries errored.")

    return {
        "search_queries": queries,
        "indexed_chunks": indexed_chunks,
        "next_doc_index": doc_counter,
        "past_seen_urls": new_urls,
        "audit_trail": audit,
    }
