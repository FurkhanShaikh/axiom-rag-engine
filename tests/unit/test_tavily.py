"""TavilySearchBackend — content-mode selection.

Verification runs against whatever this backend returns, so which body it picks
(full page text vs. search snippet) directly determines whether a genuine quote
can be marked Tier 5. These tests pin that selection.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from axiom_rag_engine.search.tavily import TavilySearchBackend

_SNIPPET = "Vaccines train the immune system."
_RAW_PAGE = (
    "Vaccination protects against infectious disease by training the immune system "
    "to recognize and fight specific pathogens. Vaccines contain weakened or inactive "
    "parts of a particular organism that trigger an immune response within the body."
)


def _backend(**kwargs: Any) -> tuple[TavilySearchBackend, MagicMock]:
    """Build a backend with a stubbed TavilyClient; return (backend, mock_client)."""
    with patch("axiom_rag_engine.search.tavily.TavilyClient") as client_cls:
        backend = TavilySearchBackend(api_key="tvly-test", **kwargs)
    return backend, client_cls.return_value


def _result(**overrides: Any) -> dict[str, Any]:
    base = {
        "url": "https://www.who.int/vaccines",
        "title": "Vaccines",
        "content": _SNIPPET,
        "raw_content": _RAW_PAGE,
    }
    base.update(overrides)
    return base


class TestContentModeSelection:
    def test_prefers_full_page_over_snippet(self) -> None:
        backend, client = _backend()
        client.search.return_value = {"results": [_result()]}

        results = backend.search("how do vaccines work")

        assert results[0]["content"] == _RAW_PAGE
        assert results[0]["content_mode"] == "raw"

    def test_requests_raw_content_from_api(self) -> None:
        """The page text has to actually be asked for — it is not returned by default."""
        backend, client = _backend()
        client.search.return_value = {"results": []}

        backend.search("q")

        assert client.search.call_args.kwargs["include_raw_content"] == "text"

    def test_falls_back_to_snippet_when_page_text_unavailable(self) -> None:
        """Paywalled / JS-only pages yield no raw text; a snippet beats nothing."""
        backend, client = _backend()
        client.search.return_value = {"results": [_result(raw_content=None)]}

        results = backend.search("q")

        assert results[0]["content"] == _SNIPPET
        assert results[0]["content_mode"] == "snippet"

    def test_falls_back_to_snippet_when_page_text_is_blank(self) -> None:
        backend, client = _backend()
        client.search.return_value = {"results": [_result(raw_content="   \n  ")]}

        results = backend.search("q")

        assert results[0]["content"] == _SNIPPET
        assert results[0]["content_mode"] == "snippet"

    def test_disabled_uses_snippet_and_does_not_request_raw(self) -> None:
        backend, client = _backend(fetch_full_pages=False)
        client.search.return_value = {"results": [_result()]}

        results = backend.search("q")

        assert client.search.call_args.kwargs["include_raw_content"] is None
        assert results[0]["content"] == _SNIPPET
        assert results[0]["content_mode"] == "snippet"


class TestRawContentCap:
    def test_oversized_page_is_truncated_not_dropped(self) -> None:
        backend, client = _backend(max_raw_content_chars=50)
        client.search.return_value = {"results": [_result(raw_content="x" * 5_000)]}

        results = backend.search("q")

        assert len(results[0]["content"]) == 50
        # Still "raw": it is page text, just bounded. Calling it a snippet would
        # misreport why a downstream Tier 5 happened.
        assert results[0]["content_mode"] == "raw"

    def test_page_within_cap_is_untouched(self) -> None:
        backend, client = _backend(max_raw_content_chars=10_000)
        client.search.return_value = {"results": [_result()]}

        results = backend.search("q")

        assert results[0]["content"] == _RAW_PAGE
