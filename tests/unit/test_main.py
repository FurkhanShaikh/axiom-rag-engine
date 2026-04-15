"""
Phase 5 — TDD: FastAPI gateway tests.

Test categories:
  A. Confidence scoring — tier breakdown and overall score computation
  B. Status determination — success / partial / unanswerable
  C. Response marshalling — GraphState → AxiomResponse
  D. Endpoint integration — POST /v1/synthesize with mocked graph
  E. Error handling — unhandled exceptions return structured AxiomResponse
  F. Validation errors — malformed payloads return 422
"""

from __future__ import annotations

import asyncio
import json
from collections import deque
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import axiom_engine.main as _main_module
from axiom_engine.main import (
    app,
    compute_confidence_summary,
    determine_status,
    make_error_response,
    marshal_response,
)
from axiom_engine.models import AxiomResponse
from axiom_engine.nodes.retriever import MockSearchBackend, set_search_backend
from tests.conftest import make_final_sentence_dict, mock_litellm_response

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SYNTH_MODEL = "claude-3-5-sonnet-20241022"
_VERIFIER_MODEL = "gpt-4o-mini"


def _make_model_router(
    synth_responses: list[str],
    semantic_responses: list[str],
) -> Any:
    synth_q: deque[str] = deque(synth_responses)
    semantic_q: deque[str] = deque(semantic_responses)

    async def _router(*args: Any, **kwargs: Any) -> MagicMock:
        model = kwargs.get("model") or (args[0] if args else "")
        if model == _SYNTH_MODEL:
            if not synth_q:
                raise RuntimeError("Unexpected extra synthesizer call")
            return mock_litellm_response(synth_q.popleft())
        elif model == _VERIFIER_MODEL:
            if not semantic_q:
                raise RuntimeError("Unexpected extra semantic call")
            return mock_litellm_response(semantic_q.popleft())
        else:
            raise RuntimeError(f"Unexpected model: {model!r}")

    return _router


_VALID_REQUEST = {
    "request_id": "req_001",
    "user_query": "What is a solid-state battery?",
    "app_config": {"expertise_level": "intermediate", "banned_domains": []},
    "models": {"synthesizer": _SYNTH_MODEL, "verifier": _VERIFIER_MODEL},
    "pipeline_config": {
        "stages": {
            "semantic_verification_enabled": True,
            "max_ranked_chunks": 10,
            "max_rewrite_loops": 3,
        }
    },
}

_SAMPLE_CHUNKS = [
    {
        "chunk_id": "doc_1_chunk_A",
        "source_url": "https://nih.gov/article",
        "text": (
            "Solid-state batteries replace liquid electrolytes with solid ceramics. "
            "This substitution significantly improves thermal stability and energy density."
        ),
        "domain": "nih.gov",
    },
]

_SEARCH_RESULTS = [
    {
        "url": "https://nih.gov/article",
        "content": _SAMPLE_CHUNKS[0]["text"],
        "title": "Solid-State Battery Review",
    }
]


@pytest.fixture()
def client(monkeypatch):
    # Clear the response cache before each test to prevent cross-test interference.
    if hasattr(_main_module, "_response_cache") and hasattr(_main_module._response_cache, "clear"):
        _main_module._response_cache.clear()
    # Remove TAVILY_API_KEY so the lifespan uses MockSearchBackend, not real Tavily.
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setenv("AXIOM_ENV", "test")
    monkeypatch.delenv("AXIOM_API_KEYS", raising=False)
    set_search_backend(MockSearchBackend([]))
    with TestClient(app) as c:
        yield c


# ===========================================================================
# A. Confidence scoring
# ===========================================================================


class TestConfidenceScoring:
    def test_all_tier_1(self) -> None:
        sentences = [make_final_sentence_dict(tier=1) for _ in range(3)]
        summary = compute_confidence_summary(sentences)
        assert summary.overall_score == 1.0
        assert summary.tier_breakdown.tier_1_claims == 3

    def test_all_tier_3(self) -> None:
        sentences = [make_final_sentence_dict(tier=3) for _ in range(4)]
        summary = compute_confidence_summary(sentences)
        assert summary.overall_score == 0.6
        assert summary.tier_breakdown.tier_3_claims == 4

    def test_mixed_tiers(self) -> None:
        sentences = [
            make_final_sentence_dict(sentence_id="s_01", tier=1),
            make_final_sentence_dict(sentence_id="s_02", tier=2),
            make_final_sentence_dict(sentence_id="s_03", tier=3),
        ]
        summary = compute_confidence_summary(sentences)
        expected = round((1.0 + 0.85 + 0.60) / 3, 4)
        assert summary.overall_score == expected
        assert summary.tier_breakdown.tier_1_claims == 1
        assert summary.tier_breakdown.tier_2_claims == 1
        assert summary.tier_breakdown.tier_3_claims == 1

    def test_empty_sentences_returns_zero(self) -> None:
        summary = compute_confidence_summary([])
        assert summary.overall_score == 0.0

    def test_tier_5_contributes_zero(self) -> None:
        sentences = [
            make_final_sentence_dict(sentence_id="s_01", tier=1),
            make_final_sentence_dict(sentence_id="s_02", tier=5),
        ]
        summary = compute_confidence_summary(sentences)
        assert summary.overall_score == round((1.0 + 0.0) / 2, 4)
        assert summary.tier_breakdown.tier_5_claims == 1

    def test_tier_6_weighted_at_040(self) -> None:
        sentences = [make_final_sentence_dict(tier=6)]
        summary = compute_confidence_summary(sentences)
        assert summary.overall_score == 0.4
        assert summary.tier_breakdown.tier_6_claims == 1


# ===========================================================================
# B. Status determination
# ===========================================================================


class TestDetermineStatus:
    def test_unanswerable(self) -> None:
        assert determine_status(False, []) == "unanswerable"

    def test_success_all_tier_1(self) -> None:
        sentences = [make_final_sentence_dict(tier=1)]
        assert determine_status(True, sentences) == "success"

    def test_success_all_tier_2_and_3(self) -> None:
        sentences = [
            make_final_sentence_dict(sentence_id="s_01", tier=2),
            make_final_sentence_dict(sentence_id="s_02", tier=3),
        ]
        assert determine_status(True, sentences) == "success"

    def test_partial_with_tier_4(self) -> None:
        sentences = [
            make_final_sentence_dict(sentence_id="s_01", tier=1),
            make_final_sentence_dict(sentence_id="s_02", tier=4),
        ]
        assert determine_status(True, sentences) == "partial"

    def test_partial_with_tier_6(self) -> None:
        sentences = [make_final_sentence_dict(tier=6)]
        assert determine_status(True, sentences) == "partial"

    def test_empty_sentences_are_unanswerable(self) -> None:
        # M8 fix: empty sentences with is_answerable=True means the pipeline could
        # not ground any output — should be "unanswerable", not "partial".
        assert determine_status(True, []) == "unanswerable"


# ===========================================================================
# C. Response marshalling
# ===========================================================================


class TestMarshalResponse:
    def test_success_response_structure(self) -> None:
        graph_result = {
            "is_answerable": True,
            "final_sentences": [make_final_sentence_dict(tier=1)],
        }
        resp = marshal_response("req_001", graph_result)
        assert isinstance(resp, AxiomResponse)
        assert resp.request_id == "req_001"
        assert resp.status == "success"
        assert resp.is_answerable is True
        assert len(resp.final_response) == 1
        assert resp.confidence_summary.overall_score == 1.0

    def test_unanswerable_response(self) -> None:
        graph_result = {"is_answerable": False, "final_sentences": []}
        resp = marshal_response("req_002", graph_result)
        assert resp.status == "unanswerable"
        assert resp.is_answerable is False
        assert resp.final_response == []
        assert resp.confidence_summary.overall_score == 0.0

    def test_partial_response_with_tier_4(self) -> None:
        graph_result = {
            "is_answerable": True,
            "final_sentences": [
                make_final_sentence_dict(sentence_id="s_01", tier=1),
                make_final_sentence_dict(sentence_id="s_02", tier=4),
            ],
        }
        resp = marshal_response("req_003", graph_result)
        assert resp.status == "partial"

    def test_error_response_structure(self) -> None:
        resp = make_error_response("req_err", RuntimeError("boom"))
        assert resp.status == "error"
        assert resp.is_answerable is False
        assert resp.confidence_summary.overall_score == 0.0
        assert resp.final_response == []
        assert resp.error_message is not None
        assert "req_err" in resp.error_message
        assert "Internal pipeline error" in resp.error_message

    def test_success_response_has_no_error_message(self) -> None:
        graph_result = {
            "is_answerable": True,
            "final_sentences": [make_final_sentence_dict(tier=1)],
        }
        resp = marshal_response("req_ok", graph_result)
        assert resp.error_message is None


# ===========================================================================
# D. Endpoint integration — POST /v1/synthesize
# ===========================================================================


class TestEndpoint:
    @patch("litellm.acompletion", new_callable=AsyncMock)
    def test_full_pipeline_success(self, mock_llm: AsyncMock, client: TestClient) -> None:
        synth_json = json.dumps(
            {
                "is_answerable": True,
                "sentences": [
                    {
                        "sentence_id": "s_01",
                        "text": "Solid-state batteries replace liquid electrolytes with solid ceramics.",
                        "is_cited": True,
                        "citations": [
                            {
                                "citation_id": "cite_1",
                                "chunk_id": "doc_1_chunk_A",
                                "exact_source_quote": (
                                    "Solid-state batteries replace liquid electrolytes with solid ceramics."
                                ),
                            }
                        ],
                    }
                ],
            }
        )
        semantic_json = json.dumps(
            {
                "semantic_check": "passed",
                "failure_reason": None,
                "reasoning": "Faithfully represents the source.",
            }
        )
        mock_llm.side_effect = _make_model_router([synth_json], [semantic_json])
        set_search_backend(MockSearchBackend(_SEARCH_RESULTS))

        resp = client.post("/v1/synthesize", json=_VALID_REQUEST)

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["is_answerable"] is True
        assert data["request_id"] == "req_001"
        assert len(data["final_response"]) == 1
        assert data["confidence_summary"]["overall_score"] == 1.0
        assert data["confidence_summary"]["tier_breakdown"]["tier_1_claims"] == 1

    @patch("litellm.acompletion", new_callable=AsyncMock)
    def test_unanswerable_pipeline(self, mock_llm: AsyncMock, client: TestClient) -> None:
        unanswerable_json = json.dumps(
            {
                "is_answerable": False,
                "sentences": [],
            }
        )
        mock_llm.side_effect = _make_model_router([unanswerable_json], [])
        set_search_backend(MockSearchBackend(_SEARCH_RESULTS))
        resp = client.post("/v1/synthesize", json=_VALID_REQUEST)

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "unanswerable"
        assert data["is_answerable"] is False

    def test_response_matches_axiom_response_schema(self, client: TestClient) -> None:
        """The response should be parseable as AxiomResponse."""
        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            synth_json = json.dumps({"is_answerable": False, "sentences": []})
            mock_llm.side_effect = _make_model_router([synth_json], [])
            set_search_backend(MockSearchBackend(_SEARCH_RESULTS))
            resp = client.post("/v1/synthesize", json=_VALID_REQUEST)

        data = resp.json()
        parsed = AxiomResponse.model_validate(data)
        assert parsed.request_id == "req_001"


# ===========================================================================
# E. Error handling — unhandled exceptions
# ===========================================================================


class TestErrorHandling:
    def test_graph_exception_returns_error_status(self, client: TestClient) -> None:
        """If the graph itself raises, the endpoint returns HTTP 500 with a
        structured AxiomResponse body (H4 fix: was incorrectly 200 before)."""
        with patch.object(app.state, "engine", create=True) as mock_engine:
            mock_engine.ainvoke = AsyncMock(side_effect=RuntimeError("LangGraph internal failure"))
            resp = client.post("/v1/synthesize", json=_VALID_REQUEST)

        # H4: pipeline errors now return 500, not 200.
        assert resp.status_code == 500
        data = resp.json()
        assert data["status"] == "error"
        assert data["is_answerable"] is False
        assert data["request_id"] == "req_001"
        assert data["confidence_summary"]["overall_score"] == 0.0
        assert data["error_message"] is not None
        assert "req_001" in data["error_message"]
        assert "Internal pipeline error" in data["error_message"]

    def test_error_response_is_valid_axiom_response(self, client: TestClient) -> None:
        with patch.object(app.state, "engine", create=True) as mock_engine:
            mock_engine.ainvoke = AsyncMock(side_effect=RuntimeError("crash"))
            resp = client.post("/v1/synthesize", json=_VALID_REQUEST)

        # Body must still be a valid AxiomResponse regardless of HTTP status.
        parsed = AxiomResponse.model_validate(resp.json())
        assert parsed.status == "error"


# ===========================================================================
# F. Validation errors — malformed payloads
# ===========================================================================


class TestValidationErrors:
    def test_missing_request_id_returns_422(self, client: TestClient) -> None:
        bad_payload = {"user_query": "What is a battery?"}
        resp = client.post("/v1/synthesize", json=bad_payload)
        assert resp.status_code == 422

    def test_empty_request_id_returns_422(self, client: TestClient) -> None:
        bad_payload = {"request_id": "", "user_query": "What is a battery?"}
        resp = client.post("/v1/synthesize", json=bad_payload)
        assert resp.status_code == 422

    def test_empty_user_query_returns_422(self, client: TestClient) -> None:
        bad_payload = {"request_id": "req_001", "user_query": ""}
        resp = client.post("/v1/synthesize", json=bad_payload)
        assert resp.status_code == 422

    def test_invalid_expertise_level_returns_422(self, client: TestClient) -> None:
        bad_payload = {
            "request_id": "req_001",
            "user_query": "What is a battery?",
            "app_config": {"expertise_level": "genius"},
        }
        resp = client.post("/v1/synthesize", json=bad_payload)
        assert resp.status_code == 422

    def test_empty_body_returns_422(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/synthesize", content=b"", headers={"content-type": "application/json"}
        )
        assert resp.status_code == 422

    def test_valid_minimal_payload_accepted(self, client: TestClient) -> None:
        """Minimal payload with only required fields should be accepted."""
        minimal = {"request_id": "req_min", "user_query": "Test query"}
        with patch.object(app.state, "engine", create=True) as mock_engine:
            mock_engine.ainvoke = AsyncMock(
                return_value={"is_answerable": False, "final_sentences": []}
            )
            resp = client.post("/v1/synthesize", json=minimal)
        assert resp.status_code == 200
        assert resp.json()["request_id"] == "req_min"

    def test_request_trust_policy_overrides_are_ignored(
        self, client: TestClient, monkeypatch
    ) -> None:
        monkeypatch.delenv("AXIOM_AUTHORITATIVE_DOMAINS", raising=False)
        monkeypatch.delenv("AXIOM_LOW_QUALITY_DOMAINS", raising=False)
        monkeypatch.delenv("AXIOM_EXCLUDE_DEFAULT_DOMAINS", raising=False)
        payload = {
            "request_id": "req_cfg",
            "user_query": "What is a battery?",
            "app_config": {
                "expertise_level": "expert",
                "banned_domains": ["spam.com"],
                "authoritative_domains": ["internal.example.com"],
                "low_quality_domains": ["blogs.example.com"],
                "exclude_default_domains": ["reddit.com"],
            },
        }
        with patch.object(app.state, "engine", create=True) as mock_engine:
            mock_engine.ainvoke = AsyncMock(
                return_value={"is_answerable": False, "final_sentences": []}
            )
            resp = client.post("/v1/synthesize", json=payload)
        assert resp.status_code == 200
        initial_state = mock_engine.ainvoke.call_args.args[0]
        assert initial_state["app_config"]["banned_domains"] == ["spam.com"]
        assert initial_state["app_config"]["authoritative_domains"] == []
        assert initial_state["app_config"]["low_quality_domains"] == []
        assert initial_state["app_config"]["exclude_default_domains"] == []

    def test_request_cannot_disable_semantic_verification(
        self, client: TestClient, monkeypatch
    ) -> None:
        monkeypatch.delenv("AXIOM_SEMANTIC_VERIFICATION_ENABLED", raising=False)
        payload = {
            "request_id": "req_semantic",
            "user_query": "What is a battery?",
            "pipeline_config": {"stages": {"semantic_verification_enabled": False}},
        }
        with patch.object(app.state, "engine", create=True) as mock_engine:
            mock_engine.ainvoke = AsyncMock(
                return_value={"is_answerable": False, "final_sentences": []}
            )
            resp = client.post("/v1/synthesize", json=payload)
        assert resp.status_code == 200
        initial_state = mock_engine.ainvoke.call_args.args[0]
        assert initial_state["pipeline_config"]["stages"]["semantic_verification_enabled"] is True


# ===========================================================================
# G. Health endpoint
# ===========================================================================


class TestHealthEndpoint:
    def test_health_returns_ok(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestCacheIsolation:
    def test_cached_response_uses_current_request_id(self, client: TestClient) -> None:
        graph_result = {
            "is_answerable": True,
            "final_sentences": [make_final_sentence_dict(tier=1)],
        }
        with patch.object(app.state, "engine", create=True) as mock_engine:
            mock_engine.ainvoke = AsyncMock(return_value=graph_result)
            first = client.post("/v1/synthesize", json={**_VALID_REQUEST, "request_id": "req_a"})
            second = client.post("/v1/synthesize", json={**_VALID_REQUEST, "request_id": "req_b"})

        assert first.status_code == 200
        assert second.status_code == 200
        assert first.json()["request_id"] == "req_a"
        assert second.json()["request_id"] == "req_b"
        assert mock_engine.ainvoke.call_count == 1

    def test_include_debug_isolated_from_non_debug_cache_entries(self, client: TestClient) -> None:
        graph_result = {
            "is_answerable": True,
            "final_sentences": [make_final_sentence_dict(tier=1)],
            "audit_trail": [],
            "indexed_chunks": _SAMPLE_CHUNKS,
            "ranked_chunks": _SAMPLE_CHUNKS,
        }
        with patch.object(app.state, "engine", create=True) as mock_engine:
            mock_engine.ainvoke = AsyncMock(return_value=graph_result)
            no_debug = client.post(
                "/v1/synthesize", json={**_VALID_REQUEST, "include_debug": False}
            )
            with_debug = client.post(
                "/v1/synthesize", json={**_VALID_REQUEST, "include_debug": True}
            )

        assert no_debug.status_code == 200
        assert with_debug.status_code == 200
        assert no_debug.json()["debug"] is None
        assert with_debug.json()["debug"] is not None
        assert mock_engine.ainvoke.call_count == 2


class TestAuthMode:
    def test_verify_api_key_fails_closed_when_server_is_misconfigured(self, monkeypatch) -> None:
        monkeypatch.setenv("AXIOM_ENV", "production")
        monkeypatch.delenv("AXIOM_API_KEYS", raising=False)

        async def _run() -> None:
            with pytest.raises(_main_module.HTTPException) as exc_info:
                await _main_module.verify_api_key(None)
            # 503 (Service Unavailable): auth is required but no keys are configured.
            # Orchestrators that retry on 503 will recover once keys are supplied,
            # while 500 would be misread as a permanent crash.
            assert exc_info.value.status_code == 503

        asyncio.run(_run())

    def test_lifespan_requires_api_keys_outside_dev(self, monkeypatch) -> None:
        monkeypatch.setenv("AXIOM_ENV", "production")
        monkeypatch.delenv("AXIOM_API_KEYS", raising=False)

        async def _run() -> None:
            async with _main_module.lifespan(app):
                pass

        with pytest.raises(RuntimeError, match="AXIOM_API_KEYS"):
            asyncio.run(_run())

    def test_lifespan_requires_live_search_outside_dev(self, monkeypatch) -> None:
        monkeypatch.setenv("AXIOM_ENV", "production")
        monkeypatch.setenv("AXIOM_API_KEYS", "key-1")
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        monkeypatch.delenv("AXIOM_ALLOW_MOCK_SEARCH", raising=False)

        async def _run() -> None:
            async with _main_module.lifespan(app):
                pass

        with pytest.raises(RuntimeError, match="TAVILY_API_KEY"):
            asyncio.run(_run())
