"""
App factory — everything a running app depends on is per-app, not per-process.

``create_app(settings, search_backend=...)`` builds an isolated app: its own
response cache, rate limiter, audit store, body-size limits, and search backend.
Previously these were module globals captured at import time, so two
configurations could not coexist in one process and tests had to reach into
``main._response_cache`` / ``set_search_backend`` to reset shared state.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from axiom_rag_engine.config.settings import Settings
from axiom_rag_engine.main import create_app
from axiom_rag_engine.nodes.retriever import MockSearchBackend, set_search_backend


def _settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {"env": "test", "semantic_verification_enabled": False}
    base.update(overrides)
    return Settings(**base)


def _backend(url: str, text: str) -> MockSearchBackend:
    return MockSearchBackend([{"url": url, "title": "T", "content": text}])


_TEXT_A = "Alpha batteries use lithium iron phosphate chemistry for a long cycle life."
_TEXT_B = "Beta batteries favor energy density over longevity in portable electronics."


def _synth_for(text: str) -> str:
    quote = " ".join(text.split()[:6])
    return json.dumps(
        {
            "is_answerable": True,
            "sentences": [
                {
                    "sentence_id": "s_01",
                    "text": "A claim.",
                    "is_cited": True,
                    "citations": [
                        {
                            "citation_id": "cite_1",
                            "chunk_id": "doc_1_chunk_A",
                            "exact_source_quote": quote,
                        }
                    ],
                }
            ],
        }
    )


def _reply(content: str) -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content=content))]
    return response


def _post(client: TestClient, request_id: str = "r1", query: str = "batteries chemistry") -> Any:
    return client.post(
        "/v1/synthesize",
        json={
            "request_id": request_id,
            "user_query": query,
            "models": {"synthesizer": "mock/s", "verifier": "mock/v"},
        },
    )


class TestFactory:
    def test_module_level_app_still_exists(self) -> None:
        from axiom_rag_engine.main import app

        assert isinstance(app, FastAPI)

    def test_apps_have_independent_state(self) -> None:
        app_a = create_app(
            _settings(audit_retention=5), search_backend=_backend("https://a.example.com", _TEXT_A)
        )
        app_b = create_app(
            _settings(audit_retention=0), search_backend=_backend("https://b.example.com", _TEXT_B)
        )
        with TestClient(app_a) as a, TestClient(app_b) as b:
            assert a.app.state.services.cache is not b.app.state.services.cache
            assert a.app.state.services.audit_store.capacity == 5
            assert b.app.state.services.audit_store.capacity == 0


class TestInjectedSearchBackend:
    def test_injected_backend_wins_over_the_module_global(self) -> None:
        set_search_backend(_backend("https://global.example.org", _TEXT_B))
        app = create_app(
            _settings(), search_backend=_backend("https://injected.example.com", _TEXT_A)
        )
        with (
            TestClient(app) as client,
            patch("litellm.acompletion", return_value=_reply(_synth_for(_TEXT_A))),
        ):
            body = _post(client).json()
        source = body["final_response"][0]["citations"][0]["source"]
        assert source["url"] == "https://injected.example.com"

    def test_stream_endpoint_uses_the_injected_backend(self) -> None:
        set_search_backend(_backend("https://global.example.org", _TEXT_B))
        app = create_app(
            _settings(), search_backend=_backend("https://injected.example.com", _TEXT_A)
        )
        with (
            TestClient(app) as client,
            patch("litellm.acompletion", return_value=_reply(_synth_for(_TEXT_A))),
        ):
            resp = client.post(
                "/v1/synthesize/stream",
                json={
                    "request_id": "s1",
                    "user_query": "batteries chemistry",
                    "models": {"synthesizer": "mock/s", "verifier": "mock/v"},
                },
            )
        complete = [
            json.loads(line[5:])
            for line in resp.text.splitlines()
            if line.startswith("data:") and '"complete"' in line
        ][-1]
        source = complete["response"]["final_response"][0]["citations"][0]["source"]
        assert source["url"] == "https://injected.example.com"

    def test_two_apps_search_different_backends(self) -> None:
        app_a = create_app(_settings(), search_backend=_backend("https://a.example.com", _TEXT_A))
        app_b = create_app(_settings(), search_backend=_backend("https://b.example.org", _TEXT_B))

        async def fake_llm(**kwargs: Any) -> MagicMock:
            prompt = kwargs["messages"][1]["content"]
            return _reply(_synth_for(_TEXT_A if "Alpha" in prompt else _TEXT_B))

        with (
            TestClient(app_a) as a,
            TestClient(app_b) as b,
            patch("litellm.acompletion", side_effect=fake_llm),
        ):
            url_a = _post(a).json()["final_response"][0]["citations"][0]["source"]["url"]
            url_b = _post(b).json()["final_response"][0]["citations"][0]["source"]["url"]
        assert (url_a, url_b) == ("https://a.example.com", "https://b.example.org")


class TestPerAppLimits:
    def test_body_limit_is_per_app(self) -> None:
        small = create_app(_settings(max_body_bytes=300), search_backend=MockSearchBackend([]))
        with TestClient(small) as client:
            resp = client.post(
                "/v1/synthesize",
                json={"request_id": "r", "user_query": "x" * 1000},
            )
        assert resp.status_code == 413

    def test_rate_limit_is_per_app(self) -> None:
        strict = create_app(_settings(rate_limit="1/minute"), search_backend=MockSearchBackend([]))
        relaxed = create_app(
            _settings(rate_limit="100/minute"), search_backend=MockSearchBackend([])
        )
        with TestClient(strict) as s, TestClient(relaxed) as r:
            assert _post(s, "s1").status_code == 200
            assert _post(s, "s2").status_code == 429
            assert _post(r, "r1").status_code == 200
            assert _post(r, "r2").status_code == 200

    def test_cache_is_per_app(self) -> None:
        app_a = create_app(_settings(), search_backend=_backend("https://a.example.com", _TEXT_A))
        app_b = create_app(_settings(), search_backend=_backend("https://a.example.com", _TEXT_A))
        with (
            TestClient(app_a) as a,
            TestClient(app_b) as b,
            patch("litellm.acompletion", return_value=_reply(_synth_for(_TEXT_A))) as mock_llm,
        ):
            _post(a, "first")
            _post(a, "second")  # served from app A's cache
            calls_after_a = mock_llm.call_count
            _post(b, "third")  # app B has its own (empty) cache
        assert calls_after_a == 1
        assert mock_llm.call_count == 2
