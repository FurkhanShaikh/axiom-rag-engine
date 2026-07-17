"""
Axiom Engine — Tavily search backend.

Implements the SearchBackend protocol using the Tavily API.
Activated automatically at startup when TAVILY_API_KEY is set.

Content modes
-------------
Tavily returns two bodies per result: ``content`` (a short, query-biased
snippet) and — when ``include_raw_content`` is requested — ``raw_content``
(the extracted page text).

Verification runs against whatever this backend returns, so the choice is
load-bearing. A snippet is a *summary* of the page: a quote the synthesizer
copied verbatim from the real source can be absent from the snippet and get
marked Tier 5 (Hallucinated) even though the source fully supports it. Raw
content is therefore the default — it makes "verified against the source"
mean the source rather than a search-result excerpt.

Each result reports which body was used via ``content_mode`` so the retriever
can record it in the audit trail and operators can tell the two apart.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from tavily import TavilyClient

logger = logging.getLogger("axiom_rag_engine.search.tavily")

ContentMode = Literal["raw", "snippet"]

# Tavily accepts True | "markdown" | "text". "text" yields prose closest to what
# the mechanical verifier normalizes, without markdown syntax that would have to
# survive normalization to keep a quote matchable.
_RAW_CONTENT_FORMAT = "text"


class TavilySearchBackend:
    """
    Production search backend backed by the Tavily Search API.

    Returns up to ``max_results`` results per query, normalised to the shape
    expected by the retriever node::

        {"url": str, "content": str, "title": str, "content_mode": "raw" | "snippet"}

    Args:
        api_key: Tavily API key.
        max_results: Results requested per query.
        fetch_full_pages: Request full page text. When False, or when Tavily
            returns no raw content for a result, the snippet is used instead.
        max_raw_content_chars: Per-document truncation cap on raw page text.
    """

    def __init__(
        self,
        api_key: str,
        max_results: int = 5,
        fetch_full_pages: bool = True,
        max_raw_content_chars: int = 200_000,
    ) -> None:
        self._client = TavilyClient(api_key=api_key)
        self._max_results = max_results
        self._fetch_full_pages = fetch_full_pages
        self._max_raw_content_chars = max_raw_content_chars

    def _pick_content(self, result: dict[str, Any]) -> tuple[str, ContentMode]:
        """Return (text, mode), preferring full page text over the snippet.

        Falls back to the snippet whenever raw content is disabled, absent, or
        blank — some pages (paywalls, JS-only, robots-blocked) yield no
        extractable text, and a snippet is strictly better than nothing.
        """
        snippet = str(result.get("content") or "")
        if not self._fetch_full_pages:
            return snippet, "snippet"

        raw = str(result.get("raw_content") or "").strip()
        if not raw:
            return snippet, "snippet"

        if len(raw) > self._max_raw_content_chars:
            raw = raw[: self._max_raw_content_chars]
        return raw, "raw"

    def search(self, query: str) -> list[dict[str, Any]]:
        response = self._client.search(
            query,
            max_results=self._max_results,
            include_raw_content=_RAW_CONTENT_FORMAT if self._fetch_full_pages else None,
        )
        results: list[dict[str, Any]] = []
        snippet_fallbacks = 0
        for r in response.get("results", []):
            text, mode = self._pick_content(r)
            if mode == "snippet" and self._fetch_full_pages:
                snippet_fallbacks += 1
            results.append(
                {
                    "url": r.get("url", ""),
                    "content": text,
                    "title": r.get("title", ""),
                    "content_mode": mode,
                }
            )

        if snippet_fallbacks:
            # Not an error — but a quote can fail mechanical verification purely
            # because only a snippet was available, so make it visible.
            logger.info(
                "Tavily returned no page text for %d/%d results for %r; "
                "those fall back to snippet verification.",
                snippet_fallbacks,
                len(results),
                query,
            )
        return results
