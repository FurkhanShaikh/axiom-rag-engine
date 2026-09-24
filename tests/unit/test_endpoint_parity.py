"""
The JSON and streaming endpoints share one run path (API-5).

They used to orchestrate separately and had drifted: the stream re-derived the
final state with a hand-written reducer (``past_seen_urls`` was replaced, not
appended), and the two recorded different metrics and audit trails. Both now run
through ``graph.run_events`` — final state straight from LangGraph — and end
through the same bookkeeping.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

from axiom_rag_engine.api.routes.synthesize import _Run, _tracked
from axiom_rag_engine.config.settings import Settings
from axiom_rag_engine.graph import RunFinished, build_axiom_graph, run_events
from axiom_rag_engine.main import create_app
from axiom_rag_engine.models import AxiomRequest
from axiom_rag_engine.nodes.retriever import MockSearchBackend
from axiom_rag_engine.state import make_initial_state

_TEXT = "Alpha batteries use lithium iron phosphate chemistry for a long cycle life."
_RESULTS = [{"url": "https://example.com/a", "title": "A", "content": _TEXT}]
_UNANSWERABLE = '{"is_answerable": false, "sentences": []}'


def _reply(content: str) -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content=content))]
    return response


def _llm(content: str) -> Any:
    async def call(**kwargs: Any) -> MagicMock:
        return _reply(content)

    return call


def _sample(name: str, **labels: str) -> float:
    return REGISTRY.get_sample_value(name, labels) or 0.0


def _run_both(content: str) -> dict[str, dict[str, Any]]:
    """Send the same request to both endpoints; return each one's metric deltas
    and audit trail."""
    app = create_app(
        Settings(env="test", audit_retention=10, max_llm_calls_per_request=2),
        search_backend=MockSearchBackend(_RESULTS),
    )
    outcomes: dict[str, dict[str, Any]] = {}
    with TestClient(app) as client, patch("litellm.acompletion", side_effect=_llm(content)):
        for path in ("/v1/synthesize", "/v1/synthesize/stream"):
            rid = f"parity-{path.rsplit('/', 1)[-1]}"
            before = {
                s: _sample("axiom_requests_by_status_total", status=s)
                for s in ("error", "unanswerable")
            }
            durations = _sample("axiom_pipeline_duration_seconds_count")
            client.post(
                path,
                json={
                    "request_id": rid,
                    "user_query": "alpha batteries chemistry",
                    "models": {"synthesizer": "mock/s", "verifier": "mock/v"},
                },
            )
            trail = client.get(f"/v1/audits/{rid}").json()
            outcomes[path] = {
                "status": trail["status"],
                "events": [e["event_type"] for e in trail["audit_trail"]],
                "durations": _sample("axiom_pipeline_duration_seconds_count") - durations,
                "by_status": {
                    s: _sample("axiom_requests_by_status_total", status=s) - v
                    for s, v in before.items()
                },
            }
    return outcomes


def test_completed_runs_are_recorded_identically() -> None:
    json_run, stream_run = _run_both(_UNANSWERABLE).values()
    assert json_run == stream_run
    assert json_run["status"] == "unanswerable"
    assert json_run["durations"] == 1


def test_failed_runs_are_recorded_identically() -> None:
    json_run, stream_run = _run_both("not json").values()
    assert json_run == stream_run
    assert json_run["status"] == "error"
    assert "pipeline_failed" in json_run["events"]
    assert json_run["by_status"]["error"] == 1


async def test_run_events_final_state_is_langgraphs() -> None:
    state = make_initial_state(
        "rs", "alpha batteries chemistry", {}, {"synthesizer": "m/s", "verifier": "m/v"}, {}
    )
    config = {"configurable": {"search_backend": MockSearchBackend(_RESULTS)}}
    with patch("litellm.acompletion", side_effect=_llm(_UNANSWERABLE)):
        invoked = await build_axiom_graph().ainvoke(state, config=config)
        finals = [
            e.state
            async for e in run_events(build_axiom_graph(), state, config, 0)
            if isinstance(e, RunFinished)
        ]
    assert len(finals) == 1
    streamed = finals[0]
    # Reducer fields append, exactly as LangGraph merges them.
    assert list(streamed["past_seen_urls"]) == list(invoked["past_seen_urls"])
    assert streamed["past_seen_urls"]
    assert [e["event_type"] for e in streamed["audit_trail"]] == [
        e["event_type"] for e in invoked["audit_trail"]
    ]
    assert streamed["is_answerable"] == invoked["is_answerable"]


def test_a_stream_closed_early_is_a_cancelled_run() -> None:
    app = create_app(Settings(env="test"), search_backend=MockSearchBackend([]))

    async def frames() -> Any:
        yield "event: accepted\n\n"
        yield "event: stage\n\n"

    async def close_after_first_frame(run: _Run) -> None:
        stream = _tracked(frames(), run)
        assert await stream.__anext__() == "event: accepted\n\n"
        await stream.aclose()  # the client went away

    before = _sample("axiom_requests_by_status_total", status="cancelled")
    with TestClient(app) as client:
        run = _Run(app.state.services, AxiomRequest(request_id="c", user_query="q"), None, "k")
        client.portal.call(close_after_first_frame, run)
    assert run.ended
    assert _sample("axiom_requests_by_status_total", status="cancelled") - before == 1


@pytest.mark.parametrize("path", ["/v1/synthesize", "/v1/synthesize/stream"])
def test_both_endpoints_share_the_runner(path: str) -> None:
    app = create_app(Settings(env="test"), search_backend=MockSearchBackend(_RESULTS))
    with (
        TestClient(app) as client,
        patch("axiom_rag_engine.api.sse.run_events", wraps=run_events) as sse_runner,
        patch("axiom_rag_engine.graph.run_events", wraps=run_events) as json_runner,
        patch("litellm.acompletion", side_effect=_llm(_UNANSWERABLE)),
    ):
        client.post(path, json={"request_id": "r", "user_query": "alpha batteries chemistry"})
    assert sse_runner.call_count + json_runner.call_count == 1
