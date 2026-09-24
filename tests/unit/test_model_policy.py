"""
Model choice is server policy when auth is required.

The verifier grants the confidence tiers, so a caller must not be able to pick a
lenient judge for its own answers; and an arbitrary caller-chosen synthesizer
bills any model reachable with the operator's provider keys. Domain trust and
semantic-verification on/off were already server-enforced — the models were the
gap. With auth disabled (development/test) the caller is the operator, and its
choices are honoured as before.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import axiom_rag_engine.bootstrap as bootstrap_module
from axiom_rag_engine.config.settings import Settings
from axiom_rag_engine.main import create_app
from axiom_rag_engine.nodes.retriever import MockSearchBackend

KEY = "tenant-key-0123456789"
_TEXT = "Alpha batteries use lithium iron phosphate chemistry for a long cycle life."
_SYNTH = json.dumps(
    {
        "is_answerable": True,
        "sentences": [
            {
                "sentence_id": "s_01",
                "text": "Alpha batteries use LFP chemistry.",
                "is_cited": True,
                "citations": [
                    {
                        "citation_id": "cite_1",
                        "chunk_id": "doc_1_chunk_A",
                        "exact_source_quote": "Alpha batteries use lithium iron phosphate chemistry",
                    }
                ],
            }
        ],
    }
)
_SEMANTIC = '{"semantic_check": "passed", "failure_reason": null, "reasoning": "ok"}'


def _reply(content: str) -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content=content))]
    return response


class _Recorder:
    """Fake litellm.acompletion that records which model each role used."""

    def __init__(self) -> None:
        self.models: dict[str, set[str]] = {"synthesizer": set(), "verifier": set()}

    async def __call__(self, **kwargs: Any) -> MagicMock:
        if "Cognitive Synthesizer" in kwargs["messages"][0]["content"]:
            self.models["synthesizer"].add(kwargs["model"])
            return _reply(_SYNTH)
        self.models["verifier"].add(kwargs["model"])
        return _reply(_SEMANTIC)


def _app(env: str, keys: tuple[str, ...] = (KEY,), **overrides: Any) -> Any:
    settings = Settings(
        env=env,
        api_keys=list(keys),
        allow_mock_search=True,
        default_synthesizer_model="server/synth",
        default_verifier_model="server/verifier",
        **overrides,
    )
    backend = MockSearchBackend([{"url": "https://example.com/a", "title": "A", "content": _TEXT}])
    return create_app(settings, search_backend=backend)


@pytest.fixture(autouse=True)
def _no_ollama(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bootstrap_module, "_list_ollama_models", lambda _base: [])


@pytest.fixture
def recorder() -> Iterator[_Recorder]:
    rec = _Recorder()
    with patch("litellm.acompletion", side_effect=rec.__call__):
        yield rec


def _post(client: TestClient, models: dict[str, str], path: str = "/v1/synthesize") -> Any:
    return client.post(
        path,
        headers={"X-API-Key": KEY},
        json={"request_id": "r1", "user_query": "alpha batteries chemistry", "models": models},
    )


class TestProduction:
    def test_caller_verifier_is_ignored(self, recorder: _Recorder) -> None:
        with TestClient(_app("production")) as client:
            resp = _post(client, {"verifier": "lenient/judge"})
        assert resp.status_code == 200, resp.text
        assert recorder.models["verifier"] == {"server/verifier"}

    def test_unlisted_synthesizer_is_rejected(self, recorder: _Recorder) -> None:
        with TestClient(_app("production")) as client:
            resp = _post(client, {"synthesizer": "expensive/model"})
        assert resp.status_code == 422
        assert "not allowed" in resp.json()["detail"]
        assert recorder.models["synthesizer"] == set()  # nothing was billed

    def test_unlisted_synthesizer_is_rejected_on_stream(self, recorder: _Recorder) -> None:
        with TestClient(_app("production")) as client:
            resp = _post(client, {"synthesizer": "expensive/model"}, "/v1/synthesize/stream")
        assert resp.status_code == 422

    def test_allowlisted_synthesizer_is_honoured(self, recorder: _Recorder) -> None:
        app = _app("production", allowed_synthesizer_models=["fast/model"])
        with TestClient(app) as client:
            resp = _post(client, {"synthesizer": "fast/model"})
        assert resp.status_code == 200, resp.text
        assert recorder.models["synthesizer"] == {"fast/model"}

    def test_server_default_synthesizer_is_always_allowed(self, recorder: _Recorder) -> None:
        with TestClient(_app("production")) as client:
            resp = _post(client, {"synthesizer": "server/synth"})
        assert resp.status_code == 200, resp.text


class TestAuthDisabled:
    def test_caller_models_are_honoured(self, recorder: _Recorder) -> None:
        # Auth is off only without keys: configured keys are always enforced.
        with TestClient(_app("test", keys=())) as client:
            resp = _post(client, {"synthesizer": "any/synth", "verifier": "any/verifier"})
        assert resp.status_code == 200, resp.text
        assert recorder.models == {"synthesizer": {"any/synth"}, "verifier": {"any/verifier"}}


class TestCacheKey:
    def test_ignored_verifier_override_shares_the_cache_entry(self, recorder: _Recorder) -> None:
        with TestClient(_app("production")) as client:
            first = _post(client, {"verifier": "lenient/judge"})
            second = _post(client, {})
        assert first.status_code == second.status_code == 200
        # One pipeline run: the second request hit the cache the first one wrote.
        assert second.json()["usage"] is None
