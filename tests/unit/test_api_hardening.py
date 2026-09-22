"""
API hardening — tenant isolation, authenticated status, no leaked internals,
bounded upload fields, and startup that respects explicitly configured models.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import axiom_rag_engine.main as main_module
from axiom_rag_engine.config.settings import get_settings
from axiom_rag_engine.main import app
from axiom_rag_engine.nodes.retriever import MockSearchBackend, set_search_backend

KEY_A = "tenant-a-key"
KEY_B = "tenant-b-key"

_CHUNK_TEXT = (
    "Solid-state batteries replace liquid electrolytes with solid ceramics. "
    "This substitution significantly improves thermal stability and energy density."
)
_SYNTH = json.dumps(
    {
        "is_answerable": True,
        "sentences": [
            {
                "sentence_id": "s_01",
                "text": "Solid-state batteries use solid ceramic electrolytes.",
                "is_cited": True,
                "citations": [
                    {
                        "citation_id": "cite_1",
                        "chunk_id": "doc_1_chunk_A",
                        "exact_source_quote": (
                            "Solid-state batteries replace liquid electrolytes with solid ceramics"
                        ),
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


async def _fake_llm(**kwargs: Any) -> MagicMock:
    if "Cognitive Synthesizer" in kwargs["messages"][0]["content"]:
        return _reply(_SYNTH)
    return _reply('{"semantic_check": "passed", "failure_reason": null, "reasoning": "ok"}')


@pytest.fixture
def prod_client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """Production-mode client with two tenants and audit retention on."""
    monkeypatch.setenv("AXIOM_ENV", "production")
    monkeypatch.setenv("AXIOM_API_KEYS", f"{KEY_A},{KEY_B}")
    monkeypatch.setenv("AXIOM_ALLOW_MOCK_SEARCH", "true")
    monkeypatch.setenv("AXIOM_AUDIT_RETENTION", "20")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setattr(main_module, "_list_ollama_models", lambda _base: [])
    get_settings.cache_clear()
    main_module._response_cache.clear()
    set_search_backend(
        MockSearchBackend([{"url": "https://example.com/a", "title": "A", "content": _CHUNK_TEXT}])
    )
    with TestClient(app) as c, patch("litellm.acompletion", side_effect=_fake_llm):
        yield c
    get_settings.cache_clear()


def _synthesize(client: TestClient, key: str, request_id: str, query: str) -> dict:
    main_module._response_cache.clear()
    resp = client.post(
        "/v1/synthesize",
        headers={"X-API-Key": key},
        json={"request_id": request_id, "user_query": query},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


# ---------------------------------------------------------------------------
# Audit trails are scoped to the API key that produced them
# ---------------------------------------------------------------------------


class TestAuditTenantIsolation:
    def test_tenant_cannot_list_another_tenants_audits(self, prod_client: TestClient) -> None:
        _synthesize(prod_client, KEY_A, "a-private", "solid-state batteries")
        listed = prod_client.get("/v1/audits", headers={"X-API-Key": KEY_B}).json()
        assert "a-private" not in listed["request_ids"]
        own = prod_client.get("/v1/audits", headers={"X-API-Key": KEY_A}).json()
        assert "a-private" in own["request_ids"]

    def test_tenant_cannot_read_another_tenants_audit(self, prod_client: TestClient) -> None:
        _synthesize(prod_client, KEY_A, "a-private", "solid-state batteries")
        resp = prod_client.get("/v1/audits/a-private", headers={"X-API-Key": KEY_B})
        assert resp.status_code == 404
        assert (
            prod_client.get("/v1/audits/a-private", headers={"X-API-Key": KEY_A}).status_code == 200
        )

    def test_reusing_a_request_id_cannot_overwrite_another_tenant(
        self, prod_client: TestClient
    ) -> None:
        _synthesize(prod_client, KEY_A, "shared-id", "solid-state batteries")
        before = prod_client.get("/v1/audits/shared-id", headers={"X-API-Key": KEY_A}).json()
        _synthesize(prod_client, KEY_B, "shared-id", "thermal stability of ceramics")
        after = prod_client.get("/v1/audits/shared-id", headers={"X-API-Key": KEY_A}).json()
        assert after["recorded_at"] == before["recorded_at"]
        # ...and tenant B sees its own entry under the same id.
        mine = prod_client.get("/v1/audits/shared-id", headers={"X-API-Key": KEY_B})
        assert mine.status_code == 200
        assert mine.json()["recorded_at"] != before["recorded_at"]


# ---------------------------------------------------------------------------
# /v1/status is an operator endpoint — authenticated like the rest of /v1
# ---------------------------------------------------------------------------


class TestStatusRequiresAuth:
    def test_status_rejects_anonymous_callers(self, prod_client: TestClient) -> None:
        assert prod_client.get("/v1/status").status_code == 401

    def test_status_accepts_a_valid_key(self, prod_client: TestClient) -> None:
        resp = prod_client.get("/v1/status", headers={"X-API-Key": KEY_A})
        assert resp.status_code == 200
        assert resp.json()["service"] == "axiom-rag-engine"

    def test_health_probes_stay_open(self, prod_client: TestClient) -> None:
        assert prod_client.get("/health/live").status_code == 200
        assert prod_client.get("/health/ready").status_code == 200


# ---------------------------------------------------------------------------
# Document ingestion never echoes backend errors; form fields are bounded
# ---------------------------------------------------------------------------


@pytest.fixture
def corpus_client(tmp_path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setenv("AXIOM_ENV", "test")
    monkeypatch.delenv("AXIOM_API_KEYS", raising=False)
    monkeypatch.setenv("AXIOM_CORPUS_DB_PATH", str(tmp_path / "corpus.db"))
    monkeypatch.setenv("AXIOM_EMBEDDING_MODEL", "fake/model")
    set_search_backend(MockSearchBackend([]))
    get_settings.cache_clear()
    with TestClient(app) as c:
        yield c
    get_settings.cache_clear()


_DOC = (
    "Alpha cells use lithium iron phosphate chemistry with a long cycle life.\n\n"
    "Beta cells favor energy density over longevity for portable electronics."
)


class TestIngestionErrors:
    def test_embedding_backend_error_is_not_echoed(
        self, corpus_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def _boom(model: str, texts: list[str]) -> list[list[float]]:
            raise RuntimeError("upstream said: invalid key sk-live-SECRET123 at http://10.0.0.5")

        monkeypatch.setattr("axiom_rag_engine.corpus.ingest.embed_documents", _boom)
        resp = corpus_client.post("/v1/documents", json={"text": _DOC})
        assert resp.status_code == 502
        detail = resp.json()["detail"]
        assert "SECRET123" not in detail
        assert "10.0.0.5" not in detail


class TestUploadFieldLimits:
    @pytest.mark.parametrize(
        ("field", "limit"),
        [("title", 500), ("source", 2000), ("doc_id", 200)],
    )
    def test_oversized_form_field_is_rejected(
        self, corpus_client: TestClient, field: str, limit: int
    ) -> None:
        resp = corpus_client.post(
            "/v1/documents/upload",
            files={"file": ("doc.txt", _DOC.encode(), "text/plain")},
            data={field: "x" * (limit + 1)},
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Startup: explicitly configured models need no Anthropic/OpenAI/Ollama
# ---------------------------------------------------------------------------


class TestStartupWithExplicitModels:
    def _settings(self, monkeypatch: pytest.MonkeyPatch, **env: str):
        monkeypatch.setenv("AXIOM_ENV", "production")
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        monkeypatch.setattr(main_module, "_list_ollama_models", lambda _base: [])
        get_settings.cache_clear()
        return get_settings()

    def test_both_models_explicit_boots_without_known_provider(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        settings = self._settings(
            monkeypatch,
            AXIOM_DEFAULT_SYNTHESIZER_MODEL="gemini/gemini-2.5-pro",
            AXIOM_DEFAULT_VERIFIER_MODEL="gemini/gemini-2.5-flash",
        )
        assert main_module._resolve_llm_defaults(settings) == (
            "gemini/gemini-2.5-pro",
            "gemini/gemini-2.5-flash",
        )

    def test_one_implicit_model_still_needs_a_provider(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        settings = self._settings(
            monkeypatch, AXIOM_DEFAULT_SYNTHESIZER_MODEL="gemini/gemini-2.5-pro"
        )
        with pytest.raises(RuntimeError, match="No LLM provider"):
            main_module._resolve_llm_defaults(settings)
