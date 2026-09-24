"""
Search backends run concurrently; Tavily calls carry an explicit timeout.

``retrieval_source=both`` queried the web and then the corpus, so their
latencies added up, and the Tavily call had no timeout of its own (the client
default is 60 s, retried three times). Backends now run in parallel with a
deterministic merge order, and the timeout is configurable (RET-3).
"""

from __future__ import annotations

import contextvars
import time
from typing import Any
from unittest.mock import MagicMock

from axiom_rag_engine.config.settings import Settings, current_settings, use_settings
from axiom_rag_engine.search.corpus_backend import CompositeSearchBackend
from axiom_rag_engine.search.tavily import TavilySearchBackend


class _Slow:
    def __init__(self, name: str, delay: float) -> None:
        self.name = name
        self.delay = delay
        self.seen_limit: str | None = None

    def search(self, query: str) -> list[dict[str, Any]]:
        time.sleep(self.delay)
        self.seen_limit = current_settings().rate_limit
        return [{"url": f"https://{self.name}.example/{query}", "content": self.name}]


class _Broken:
    def search(self, query: str) -> list[dict[str, Any]]:
        raise RuntimeError("backend down")


def test_backends_run_concurrently() -> None:
    backends = [_Slow("web", 0.3), _Slow("corpus", 0.3)]
    start = time.monotonic()
    CompositeSearchBackend(backends).search("q")  # type: ignore[arg-type]
    assert time.monotonic() - start < 0.5  # not 0.6: the slower one, not the sum


def test_merge_order_follows_backend_order_not_finish_order() -> None:
    composite = CompositeSearchBackend([_Slow("web", 0.2), _Slow("corpus", 0.0)])  # type: ignore[list-item]
    assert [r["content"] for r in composite.search("q")] == ["web", "corpus"]


def test_one_failing_backend_does_not_sink_the_others() -> None:
    composite = CompositeSearchBackend([_Broken(), _Slow("corpus", 0.0)])  # type: ignore[list-item]
    assert [r["content"] for r in composite.search("q")] == ["corpus"]


def test_backends_see_the_request_settings() -> None:
    backends = [_Slow("web", 0.0), _Slow("corpus", 0.0)]

    def request() -> None:  # in its own context, so the setting does not leak
        use_settings(Settings(env="test", rate_limit="7/minute"))
        CompositeSearchBackend(backends).search("q")  # type: ignore[arg-type]

    contextvars.copy_context().run(request)
    assert [b.seen_limit for b in backends] == ["7/minute", "7/minute"]


def test_tavily_call_carries_the_configured_timeout() -> None:
    backend = TavilySearchBackend(api_key="tvly-test", timeout_seconds=7.5)
    backend._client = MagicMock()
    backend._client.search.return_value = {"results": []}
    backend.search("q")
    assert backend._client.search.call_args.kwargs["timeout"] == 7.5


def test_timeout_is_a_setting() -> None:
    assert Settings(env="test").search_timeout_seconds == 20.0
    assert Settings(env="test", search_timeout_seconds=5).search_timeout_seconds == 5
