"""
Embedding calls follow the same policy as chat calls.

Embeddings went straight to ``litellm.aembedding`` / ``litellm.embedding``,
skipping the per-request budget, the concurrency limit, retries, and usage and
cost accounting — hybrid ranking (query + up to 200 chunks per request) and
ingestion (up to 2,000 chunks per document) never showed up in ``usage`` or in
``axiom_llm_cost_usd_total``.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import litellm
import pytest
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

from axiom_rag_engine.config.settings import Settings, get_settings
from axiom_rag_engine.embeddings import embed_documents, embed_query_sync
from axiom_rag_engine.main import create_app
from axiom_rag_engine.nodes.retriever import MockSearchBackend
from axiom_rag_engine.utils import llm as llm_mod
from axiom_rag_engine.utils.llm import LLMBudgetExceededError, reset_llm_budget

_TEXT = "Alpha batteries use lithium iron phosphate chemistry for a long cycle life."


class _EmbeddingResponse(dict):
    """LiteLLM embedding response shape: ``resp["data"]`` plus ``resp.usage``."""

    def __init__(self, n: int, prompt_tokens: int = 10) -> None:
        super().__init__(data=[{"embedding": [1.0, 0.0]} for _ in range(n)])
        self.usage = MagicMock(prompt_tokens=prompt_tokens, completion_tokens=0, total_tokens=0)


async def _aembed(**kwargs: Any) -> _EmbeddingResponse:
    return _EmbeddingResponse(len(kwargs["input"]))


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Zero retry backoff, and no request budget shared with other tests."""
    monkeypatch.setenv("AXIOM_LLM_RETRY_MAX_WAIT_SECONDS", "0")
    get_settings.cache_clear()
    llm_mod._llm_budget_ctx.set(None)
    yield
    llm_mod._llm_budget_ctx.set(None)


def test_hybrid_ranking_embeddings_appear_in_usage() -> None:
    synth = json.dumps(
        {
            "is_answerable": True,
            "sentences": [
                {
                    "sentence_id": "s_01",
                    "text": "Alpha batteries use LFP.",
                    "is_cited": True,
                    "citations": [
                        {
                            "citation_id": "cite_1",
                            "chunk_id": "doc_1_chunk_A",
                            "exact_source_quote": "Alpha batteries use lithium iron phosphate",
                        }
                    ],
                }
            ],
        }
    )
    reply = MagicMock()
    reply.choices = [MagicMock(message=MagicMock(content=synth))]
    reply.usage = MagicMock(prompt_tokens=5, completion_tokens=5, total_tokens=10)

    app = create_app(
        Settings(env="test", embedding_model="mock/embed", semantic_verification_enabled=False),
        search_backend=MockSearchBackend(
            [
                {"url": "https://a.example.com/x", "title": "A", "content": _TEXT},
                {"url": "https://b.example.org/y", "title": "B", "content": _TEXT + " More."},
            ]
        ),
    )
    with (
        TestClient(app) as client,
        patch("litellm.acompletion", AsyncMock(return_value=reply)),
        patch("litellm.aembedding", side_effect=_aembed),
    ):
        body = client.post(
            "/v1/synthesize",
            json={
                "request_id": "hybrid",
                "user_query": "alpha batteries chemistry",
                "models": {"synthesizer": "mock/s", "verifier": "mock/v"},
            },
        ).json()
    assert body["usage"]["by_model"]["mock/embed"]["calls"] == 1
    assert body["usage"]["by_model"]["mock/embed"]["prompt_tokens"] == 10


async def test_embedding_calls_consume_the_request_budget() -> None:
    reset_llm_budget(max_calls=0)
    with patch("litellm.aembedding", side_effect=_aembed), pytest.raises(LLMBudgetExceededError):
        await embed_documents("mock/embed", ["a passage"])


def test_sync_embedding_consumes_the_request_budget() -> None:
    reset_llm_budget(max_calls=0)
    with patch("litellm.embedding", MagicMock()), pytest.raises(LLMBudgetExceededError):
        embed_query_sync("mock/embed", "a query")


async def test_transient_embedding_errors_are_retried() -> None:
    flaky = AsyncMock(
        side_effect=[
            litellm.RateLimitError("slow down", llm_provider="openai", model="m"),
            _EmbeddingResponse(1),
        ]
    )
    with patch("litellm.aembedding", flaky):
        vectors = await embed_documents("mock/embed", ["a passage"])
    assert flaky.await_count == 2
    assert len(vectors) == 1


async def test_ingestion_tokens_reach_metrics_without_a_request_budget() -> None:
    # Ingestion runs outside any request budget; its tokens still count.
    before = (
        REGISTRY.get_sample_value("axiom_llm_tokens_total", {"model": "other", "kind": "prompt"})
        or 0.0
    )
    with patch("litellm.aembedding", side_effect=_aembed):
        await embed_documents("mock/embed", ["one", "two"])
    after = REGISTRY.get_sample_value(
        "axiom_llm_tokens_total", {"model": "other", "kind": "prompt"}
    )
    assert after == before + 10
