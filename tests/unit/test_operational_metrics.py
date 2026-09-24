"""
Operational metrics (OBS-3): each one moves when its event happens.

Rewrite and re-retrieval passes, early halts, LLM calls per request, budget
exhaustion, synthesizer parse failures, search failures, snippet-only sources,
rate-limit rejections, cache errors and embedding usage were invisible to
Prometheus (some only logged).
"""

from __future__ import annotations

import contextvars
from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

from axiom_rag_engine.api.routes.synthesize import _record_outcome_metrics
from axiom_rag_engine.cache import RedisCacheBackend
from axiom_rag_engine.config.settings import Settings
from axiom_rag_engine.graph import route_post_verification
from axiom_rag_engine.main import create_app
from axiom_rag_engine.marshalling import marshal_response
from axiom_rag_engine.nodes.retriever import MockSearchBackend, _safe_search
from axiom_rag_engine.search.corpus_backend import CompositeSearchBackend
from axiom_rag_engine.utils.llm import (
    LLMBudgetExceededError,
    call_embedding,
    consume_llm_budget,
    reset_llm_budget,
)

_TEXT = "Alpha batteries use lithium iron phosphate chemistry for a long cycle life."


def _value(name: str, **labels: str) -> float:
    return REGISTRY.get_sample_value(name, labels) or 0.0


def _delta(name: str, action: Callable[[], Any], **labels: str) -> float:
    before = _value(name, **labels)
    action()
    return _value(name, **labels) - before


def _pending(loop_count: int, retries: int) -> dict[str, Any]:
    return {
        "is_answerable": True,
        "pending_rewrite_count": 1,
        "loop_count": loop_count,
        "retrieval_retry_count": retries,
        "pipeline_config": {"stages": {"max_rewrite_loops": 1, "max_retrieval_retries": 1}},
    }


def test_rewrite_and_re_retrieval_are_counted() -> None:
    assert (
        _delta("axiom_rewrite_passes_total", lambda: route_post_verification(_pending(1, 0))) == 1
    )
    assert _delta("axiom_re_retrievals_total", lambda: route_post_verification(_pending(2, 0))) == 1
    # Exhausted: neither moves.
    assert (
        _delta("axiom_rewrite_passes_total", lambda: route_post_verification(_pending(2, 1))) == 0
    )


def test_halts_and_calls_per_request_are_recorded() -> None:
    result = {"is_answerable": True, "final_sentences": [], "halt_reason": "deadline"}
    snapshot = {"calls": 5, "prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}
    response = marshal_response("r", result, usage_snapshot=snapshot)
    calls_before = _value("axiom_llm_calls_per_request_count")
    assert (
        _delta(
            "axiom_pipeline_halts_total",
            lambda: _record_outcome_metrics(response, result),
            reason="deadline",
        )
        == 1
    )
    assert _value("axiom_llm_calls_per_request_count") - calls_before == 1


def test_budget_exhaustion_is_counted_by_cap() -> None:
    def exhaust() -> None:
        reset_llm_budget(max_calls=0)
        with pytest.raises(LLMBudgetExceededError):
            consume_llm_budget("synthesizer")

    run = lambda: contextvars.copy_context().run(exhaust)  # noqa: E731 - isolate the budget
    assert _delta("axiom_llm_budget_exhausted_total", run, cap="calls") == 1


def test_search_failures_are_counted_by_backend() -> None:
    class Broken:
        def search(self, query: str) -> list[dict[str, Any]]:
            raise RuntimeError("down")

    composite = CompositeSearchBackend([Broken()])  # type: ignore[list-item]
    assert (
        _delta("axiom_search_failures_total", lambda: composite.search("q"), backend="Broken") == 1
    )


async def test_failed_search_query_is_counted() -> None:
    class Broken:
        def search(self, query: str) -> list[dict[str, Any]]:
            raise RuntimeError("down")

    before = _value("axiom_search_failures_total", backend="Broken")
    with patch("axiom_rag_engine.nodes.retriever._search_with_retry.retry.sleep", lambda _s: None):
        await _safe_search("q", Broken())  # type: ignore[arg-type]
    assert _value("axiom_search_failures_total", backend="Broken") - before == 1


def _reply(content: str) -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content=content))]
    return response


async def _not_json(**kwargs: Any) -> MagicMock:
    return _reply("not json")


def test_parse_failures_and_source_modes_are_counted() -> None:
    backend = MockSearchBackend(
        [
            {
                "url": "https://example.com/a",
                "title": "A",
                "content": _TEXT,
                "content_mode": "snippet",
            }
        ]
    )
    app = create_app(Settings(env="test", max_llm_calls_per_request=2), search_backend=backend)
    body = {
        "request_id": "m",
        "user_query": "alpha batteries chemistry",
        "models": {"synthesizer": "mock/s", "verifier": "mock/v"},
    }
    parse_before = _value("axiom_synthesizer_parse_failures_total")
    snippet_before = _value("axiom_sources_by_content_mode_total", mode="snippet")
    with TestClient(app) as client, patch("litellm.acompletion", side_effect=_not_json):
        client.post("/v1/synthesize", json=body)
    assert _value("axiom_synthesizer_parse_failures_total") - parse_before >= 1
    assert _value("axiom_sources_by_content_mode_total", mode="snippet") - snippet_before == 1


def test_rate_limit_rejections_are_counted() -> None:
    app = create_app(
        Settings(env="test", rate_limit="1/minute"), search_backend=MockSearchBackend([])
    )
    before = _value("axiom_rate_limit_rejections_total")
    with TestClient(app) as client:
        codes = [client.post("/v1/synthesize", json={}).status_code for _ in range(3)]
    assert codes.count(429) == 2
    assert _value("axiom_rate_limit_rejections_total") - before == 2


async def test_cache_errors_are_counted() -> None:
    class FailingRedis:
        async def get(self, key: str) -> None:
            raise OSError("redis down")

    before = _value("axiom_cache_errors_total", op="get")
    assert await RedisCacheBackend(client=FailingRedis()).get("k") is None
    assert _value("axiom_cache_errors_total", op="get") - before == 1


async def test_embedding_inputs_are_counted() -> None:
    async def fake_embedding(**kwargs: Any) -> MagicMock:
        return MagicMock(usage=None)

    before = _value("axiom_embedding_inputs_total", model="ollama/…")
    with patch("litellm.aembedding", side_effect=fake_embedding):
        await call_embedding("embedding", "ollama/nomic-embed-text", {"input": ["a", "b", "c"]})
    assert _value("axiom_embedding_inputs_total", model="ollama/…") - before == 3
