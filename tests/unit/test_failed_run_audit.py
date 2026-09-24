"""
Failed requests keep their audit trail.

The trail was persisted only on success: when the graph raised (HTTP 429/500,
or a stream error frame), every event from the run was lost and one log line
remained — so the requests most in need of debugging had no record. The trail
up to the failure is now kept, with a terminal ``pipeline_failed`` event naming
the node that failed and the error type.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from axiom_rag_engine.config.settings import Settings
from axiom_rag_engine.main import create_app
from axiom_rag_engine.nodes.retriever import MockSearchBackend

_TEXT = "Alpha batteries use lithium iron phosphate chemistry for a long cycle life."


def _app(**overrides: Any) -> Any:
    return create_app(
        Settings(env="test", audit_retention=10, **overrides),
        search_backend=MockSearchBackend(
            [{"url": "https://example.com/a", "title": "A", "content": _TEXT}]
        ),
    )


def _reply(content: str) -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content=content))]
    return response


async def _provider_down(**kwargs: Any) -> MagicMock:
    raise RuntimeError("provider down")


async def _not_json(**kwargs: Any) -> MagicMock:
    return _reply("not json")  # parse retry -> second call -> over a budget of 1


def _body(request_id: str) -> dict[str, Any]:
    return {
        "request_id": request_id,
        "user_query": "alpha batteries chemistry",
        "models": {"synthesizer": "mock/s", "verifier": "mock/v"},
    }


def _trail(client: TestClient, request_id: str) -> dict[str, Any]:
    resp = client.get(f"/v1/audits/{request_id}")
    assert resp.status_code == 200, resp.text
    return resp.json()


def _failure(entry: dict[str, Any]) -> dict[str, Any]:
    events = [e for e in entry["audit_trail"] if e["event_type"] == "pipeline_failed"]
    assert len(events) == 1
    return events[0]["payload"]


def test_pipeline_error_keeps_the_trail() -> None:
    with TestClient(_app()) as client, patch("litellm.acompletion", side_effect=_provider_down):
        assert client.post("/v1/synthesize", json=_body("boom")).status_code == 500
        entry = _trail(client, "boom")

    assert entry["status"] == "error"
    types = [e["event_type"] for e in entry["audit_trail"]]
    # Everything that completed before the failure is kept.
    assert "retriever_complete" in types
    assert "ranker_complete" in types
    assert _failure(entry) == {"failed_node": "synthesizer", "error_type": "RuntimeError"}


def test_budget_exhaustion_keeps_the_trail() -> None:
    app = _app(max_llm_calls_per_request=1)
    with TestClient(app) as client, patch("litellm.acompletion", side_effect=_not_json):
        assert client.post("/v1/synthesize", json=_body("broke")).status_code == 429
        entry = _trail(client, "broke")
    assert _failure(entry)["error_type"] == "LLMBudgetExceededError"


def test_stream_error_keeps_the_trail() -> None:
    with TestClient(_app()) as client, patch("litellm.acompletion", side_effect=_provider_down):
        resp = client.post("/v1/synthesize/stream", json=_body("stream-boom"))
        assert "event: error" in resp.text
        entry = _trail(client, "stream-boom")
    assert entry["status"] == "error"
    assert _failure(entry) == {"failed_node": "synthesizer", "error_type": "RuntimeError"}
