"""
Failures and early stops are visible to the client, not only in server logs.

* A run that halts early (a later pass failed, the synthesizer gave up, or the
  deadline hit) still returns its best verified pass — but callers could only
  learn *that it was cut short* from the debug block. ``halt_reason`` is now a
  top-level response field.
* A failure before any verified pass returns ``status="error"`` with an
  ``error_type`` in the vocabulary the SSE endpoint already uses, and an upstream
  LLM failure is HTTP 502 (the gateway's upstream failed) instead of a 500 that
  looks like an engine bug.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import litellm
import pytest

from axiom_rag_engine.marshalling import marshal_response
from axiom_rag_engine.nodes.retriever import MockSearchBackend, set_search_backend

_TEXT = (
    "Solid-state batteries replace liquid electrolytes with solid ceramics. "
    "This substitution significantly improves thermal stability and energy density."
)
_GOOD = "Solid-state batteries replace liquid electrolytes with solid ceramics"
_BAD = "Solid-state batteries are powered by tiny nuclear reactors inside"


def _sentence(i: int, quote: str) -> dict[str, Any]:
    return {
        "sentence_id": f"s_{i:02d}",
        "text": f"Claim number {i}.",
        "is_cited": True,
        "citations": [
            {"citation_id": f"cite_{i}", "chunk_id": "doc_1_chunk_A", "exact_source_quote": quote}
        ],
    }


def _reply(content: str) -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content=content))]
    return response


@pytest.fixture
def no_llm_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    from axiom_rag_engine.config.settings import get_settings

    monkeypatch.setenv("AXIOM_LLM_MAX_RETRIES", "0")
    get_settings.cache_clear()


@pytest.fixture
def one_source() -> None:
    set_search_backend(
        MockSearchBackend([{"url": "https://example.com/a", "title": "A", "content": _TEXT}])
    )


def _post(client: Any, path: str = "/v1/synthesize") -> Any:
    return client.post(
        path,
        json={"request_id": "vis", "user_query": "solid-state batteries"},
    )


class TestHaltReasonIsTopLevel:
    def test_marshalled_response_carries_the_halt_reason(self) -> None:
        response = marshal_response("r", {"is_answerable": True, "halt_reason": "deadline"})
        assert response.halt_reason == "deadline"

    def test_normal_run_has_no_halt_reason(self) -> None:
        assert marshal_response("r", {"is_answerable": True}).halt_reason is None

    def test_failed_rewrite_pass_is_reported(
        self, client: Any, one_source: None, no_llm_retries: None
    ) -> None:
        first = json.dumps(
            {"is_answerable": True, "sentences": [_sentence(1, _GOOD), _sentence(2, _BAD)]}
        )
        calls = {"n": 0}

        async def llm(**kwargs: Any) -> MagicMock:
            system = kwargs["messages"][0]["content"]
            if "Cognitive Synthesizer" not in system:
                return _reply('{"semantic_check": "passed", "failure_reason": ""}')
            calls["n"] += 1
            if calls["n"] == 1:
                return _reply(first)
            raise RuntimeError("synthesizer blew up on the rewrite")

        with patch("litellm.acompletion", side_effect=llm):
            resp = _post(client)
        body = resp.json()
        assert resp.status_code == 200
        assert body["halt_reason"] == "node_error"
        assert len(body["final_response"]) == 2


class TestFirstPassFailuresAreTyped:
    def test_provider_outage_is_502_llm_unavailable(
        self, client: Any, one_source: None, no_llm_retries: None
    ) -> None:
        outage = litellm.APIConnectionError(
            message="connection refused", llm_provider="openai", model="m"
        )
        with patch("litellm.acompletion", new_callable=AsyncMock, side_effect=outage):
            resp = _post(client)
        assert resp.status_code == 502
        assert resp.json()["error_type"] == "llm_unavailable"

    def test_unusable_model_output_is_502(
        self, client: Any, one_source: None, no_llm_retries: None
    ) -> None:
        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=_reply("nope")):
            resp = _post(client)
        assert resp.status_code == 502
        assert resp.json()["error_type"] == "llm_output_unusable"

    def test_engine_bug_stays_500_internal(self, client: Any) -> None:
        class _BrokenGraph:
            async def astream_events(self, state: dict, **_: object):  # type: ignore[no-untyped-def]
                raise KeyError("bug")
                yield  # pragma: no cover - makes this an async generator

        with patch.object(client.app.state.services, "engine", _BrokenGraph()):
            resp = _post(client)
        assert resp.status_code == 500
        assert resp.json()["error_type"] == "internal"

    def test_stream_uses_the_same_vocabulary(
        self, client: Any, one_source: None, no_llm_retries: None
    ) -> None:
        outage = litellm.APIConnectionError(
            message="connection refused", llm_provider="openai", model="m"
        )
        with patch("litellm.acompletion", new_callable=AsyncMock, side_effect=outage):
            resp = _post(client, "/v1/synthesize/stream")
        frames = [
            json.loads(line[5:]) for line in resp.text.splitlines() if line.startswith("data:")
        ]
        errors = [f for f in frames if f.get("type") == "error"]
        assert errors and errors[-1]["error_type"] == "llm_unavailable"
