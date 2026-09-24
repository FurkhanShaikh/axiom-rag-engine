"""
Per-call LLM timeout and the overall request deadline.

The per-call timeout was a hard-coded 600 s and nothing bounded a request as a
whole, so a few hung calls could hold a request — and the process-wide LLM
concurrency slots — for many minutes. The timeout is now configurable
(default 120 s) and a request deadline (default 300 s) ends the run: after a
verified pass it returns that pass; before one it fails with 504.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from axiom_rag_engine.api.sse import stream_pipeline
from axiom_rag_engine.config.settings import Settings, get_settings
from axiom_rag_engine.graph import PipelineDeadlineError, build_axiom_graph, run_pipeline
from axiom_rag_engine.main import create_app
from axiom_rag_engine.models import AxiomRequest
from axiom_rag_engine.nodes.retriever import MockSearchBackend
from axiom_rag_engine.state import make_initial_state
from axiom_rag_engine.utils.llm import build_completion_kwargs

_TEXT = (
    "Solid-state batteries replace liquid electrolytes with solid ceramics. "
    "This substitution significantly improves thermal stability and energy density of cells."
)
_DRAFT = json.dumps(
    {
        "is_answerable": True,
        "sentences": [
            {
                "sentence_id": "s_01",
                "text": "Solid-state batteries use solid ceramics.",
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
            },
            {
                "sentence_id": "s_02",
                "text": "They run on nuclear power.",
                "is_cited": True,
                "citations": [
                    {
                        "citation_id": "cite_2",
                        "chunk_id": "doc_1_chunk_A",
                        "exact_source_quote": "powered by tiny nuclear reactors inside every cell",
                    }
                ],
            },
        ],
    }
)
_PASS = json.dumps({"semantic_check": "passed", "failure_reason": None})
_BACKEND = {
    "configurable": {
        "search_backend": MockSearchBackend(
            [{"url": "https://example.com/a", "title": "A", "content": _TEXT}]
        )
    }
}


def _fake_llm(hang_on_synth_call: int) -> Any:
    """Synthesizer hangs on its Nth call (1-based); semantic checks pass."""
    calls = {"synth": 0}

    async def _fake(node: str, model: str, messages: Any, **_: Any) -> str:
        if node == "synthesizer":
            calls["synth"] += 1
            if calls["synth"] == hang_on_synth_call:
                await asyncio.sleep(30)
            return _DRAFT
        return _PASS

    return _fake


def _state() -> Any:
    return make_initial_state(
        request_id="req_deadline",
        user_query="What are solid-state batteries?",
        app_config={},
        models_config={"synthesizer": "mock/s", "verifier": "mock/v"},
        pipeline_config={"stages": {"semantic_verification_enabled": True}},
    )


def _patched(fake: Any) -> Any:
    return (
        patch("axiom_rag_engine.nodes.synthesizer.call_llm", fake),
        patch("axiom_rag_engine.nodes.semantic.call_llm", fake),
    )


class TestPerCallTimeout:
    def test_default_is_configurable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert build_completion_kwargs("gpt-4o-mini", [])["timeout"] == 120.0
        monkeypatch.setenv("AXIOM_LLM_TIMEOUT_SECONDS", "900")
        get_settings.cache_clear()
        assert build_completion_kwargs("gpt-4o-mini", [])["timeout"] == 900.0
        assert build_completion_kwargs("gpt-4o-mini", [], timeout=5)["timeout"] == 5


class TestRunPipeline:
    async def test_deadline_after_a_verified_pass_returns_it(self) -> None:
        a, b = _patched(_fake_llm(hang_on_synth_call=2))  # the rewrite hangs
        with a, b:
            result = await run_pipeline(build_axiom_graph(), _state(), _BACKEND, 0.5)

        assert result["halt_reason"] == "deadline"
        assert [s["sentence_id"] for s in result["final_sentences"]] == ["s_01", "s_02"]
        assert any(
            e["event_type"] == "pipeline_halted_best_pass_returned" for e in result["audit_trail"]
        )

    async def test_deadline_before_any_pass_raises(self) -> None:
        a, b = _patched(_fake_llm(hang_on_synth_call=1))
        with a, b, pytest.raises(PipelineDeadlineError):
            await run_pipeline(build_axiom_graph(), _state(), _BACKEND, 0.3)

    async def test_zero_disables_the_deadline(self) -> None:
        a, b = _patched(_fake_llm(hang_on_synth_call=0))
        with a, b:
            result = await run_pipeline(build_axiom_graph(), _state(), _BACKEND, 0)
        assert result["halt_reason"] is None

    async def test_inner_timeout_error_is_not_mistaken_for_the_deadline(self) -> None:
        async def _fake(node: str, model: str, messages: Any, **_: Any) -> str:
            raise TimeoutError("provider socket timeout")

        a, b = _patched(_fake)
        with a, b, pytest.raises(RuntimeError, match="Synthesizer stage failed"):
            await run_pipeline(build_axiom_graph(), _state(), _BACKEND, 30)


class TestEndpoints:
    def test_json_endpoint_returns_504_before_any_pass(self) -> None:
        app = create_app(
            Settings(env="test", request_deadline_seconds=0.3),
            search_backend=MockSearchBackend(
                [{"url": "https://example.com/a", "title": "A", "content": _TEXT}]
            ),
        )
        a, b = _patched(_fake_llm(hang_on_synth_call=1))
        with TestClient(app) as client, a, b:
            resp = client.post(
                "/v1/synthesize",
                json={"request_id": "slow", "user_query": "solid-state batteries"},
            )
        assert resp.status_code == 504
        assert resp.json()["status"] == "error"

    async def test_stream_returns_best_pass_on_deadline(self) -> None:
        payload = AxiomRequest(request_id="req_deadline", user_query="solid-state batteries")
        a, b = _patched(_fake_llm(hang_on_synth_call=2))
        with a, b:
            frames = [
                frame
                async for frame in stream_pipeline(
                    payload=payload,
                    engine=build_axiom_graph(),
                    initial_state=_state(),
                    run_config=_BACKEND,
                    deadline_seconds=0.5,
                )
            ]
        complete = [f for f in frames if f.startswith("event: complete")]
        assert complete, frames[-1]
        response = json.loads(complete[0].split("data: ", 1)[1])["response"]
        assert response["status"] == "partial"
        assert response["debug"] is None
        assert [s["sentence_id"] for s in response["final_response"]] == ["s_01", "s_02"]

    async def test_stream_reports_deadline_before_any_pass(self) -> None:
        payload = AxiomRequest(request_id="req_deadline", user_query="solid-state batteries")
        a, b = _patched(_fake_llm(hang_on_synth_call=1))
        with a, b:
            frames = [
                frame
                async for frame in stream_pipeline(
                    payload=payload,
                    engine=build_axiom_graph(),
                    initial_state=_state(),
                    run_config=_BACKEND,
                    deadline_seconds=0.3,
                )
            ]
        assert frames[-1].startswith("event: error")
        assert "deadline_exceeded" in frames[-1]
