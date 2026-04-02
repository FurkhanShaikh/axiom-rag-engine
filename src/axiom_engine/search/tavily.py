"""
Axiom Engine — Tavily search backend.

Implements the SearchBackend protocol using the Tavily API.
Activated automatically at startup when TAVILY_API_KEY is set.
"""

from __future__ import annotations

from typing import Any

from tavily import TavilyClient


class TavilySearchBackend:
    """
    Production search backend backed by the Tavily Search API.

    Returns up to `max_results` results per query, normalised to the
    shape expected by the retriever node:
        {"url": str, "content": str, "title": str}
    """

    def __init__(self, api_key: str, max_results: int = 5) -> None:
        self._client = TavilyClient(api_key=api_key)
        self._max_results = max_results

    def search(self, query: str) -> list[dict[str, Any]]:
        response = self._client.search(query, max_results=self._max_results)
        results: list[dict[str, Any]] = []
        for r in response.get("results", []):
            results.append(
                {
                    "url": r.get("url", ""),
                    "content": r.get("content", ""),
                    "title": r.get("title", ""),
                }
            )
        return results
