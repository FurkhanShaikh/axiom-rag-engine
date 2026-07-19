"""Tests for the SSE streaming generator and /v1/synthesize/stream endpoint."""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from axiom_rag_engine.api.sse import _apply_node_update, _sse, _stage_metadata, stream_pipeline
from axiom_rag_engine.models import AxiomRequest

# ---------------------------------------------------------------------------
# Unit helpers
# ---------------------------------------------------------------------------


def test_sse_frame_format() -> None:
    frame = _sse("stage", {"type": "stage", "stage": "retriever"}, event_id=3)
    assert frame.startswith("event: stage\n")
    assert "id: 3\n" in frame
    assert frame.endswith("\n\n")
    data_line = next(line for line in frame.splitlines() if line.startswith("data:"))
    parsed = json.loads(data_line[len("data:") :])
    assert parsed["stage"] == "retriever"


def test_sse_frame_without_id() -> None:
    frame = _sse("accepted", {"type": "accepted"})
    assert "id:" not in frame


def test_stage_metadata_start_is_empty() -> None:
    assert _stage_metadata("retriever", "start", {"anything": 1}) == {}


def test_stage_metadata_retriever_complete() -> None:
    meta = _stage_metadata("retriever", "complete", {"indexed_chunks": [1, 2, 3]})
    assert meta == {"chunks_retrieved": 3}


def test_stage_metadata_verifier_complete() -> None:
    meta = _stage_metadata(
        "verifier",
        "complete",
        {"final_sentences": [1], "pending_rewrite_count": 1, "loop_count": 2},
    )
    assert meta["final_sentences"] == 1
    assert meta["pending_rewrite_count"] == 1
    assert meta["loop_count"] == 2


def test_apply_node_update_appends_audit_trail() -> None:
    state: dict[str, Any] = {"audit_trail": [{"e": 1}], "foo": "old"}
    _apply_node_update(state, {"audit_trail": [{"e": 2}], "foo": "new"})
    assert state["audit_trail"] == [{"e": 1}, {"e": 2}]
    assert state["foo"] == "new"


def test_apply_node_update_handles_missing_audit_trail() -> None:
    state: dict[str, Any] = {}
    _apply_node_update(state, {"audit_trail": [{"e": 1}]})
    assert state["audit_trail"] == [{"e": 1}]


# ---------------------------------------------------------------------------
# stream_pipeline integration (mock engine)
# ---------------------------------------------------------------------------


def _make_payload(request_id: str = "test-001") -> AxiomRequest:
    return AxiomRequest(request_id=request_id, user_query="What is RAG?")


def _langgraph_event(event: str, name: str, output: dict | None = None) -> dict:
    """Build a minimal LangGraph astream_events dict."""
    return {"event": event, "name": name, "data": {"output": output or {}}}


async def _collect(gen) -> list[dict]:
    """Drain an SSE generator and parse each data frame."""
    frames = []
    async for raw in gen:
        if raw.startswith(":"):
            continue  # keepalive comment
        data_line = next((line for line in raw.splitlines() if line.startswith("data:")), None)
        if data_line:
            frames.append(json.loads(data_line[5:]))
    return frames


def _mock_engine(events: list[dict]) -> Any:
    """Build a mock LangGraph engine whose astream_events yields the given events."""

    async def _astream_events(state, *, version):
        for e in events:
            yield e

    engine = MagicMock()
    engine.astream_events = _astream_events
    return engine


@pytest.mark.asyncio
async def test_stream_cache_hit_emits_accepted_then_complete() -> None:
    from axiom_rag_engine.models import AxiomResponse, ConfidenceSummary, TierBreakdown

    cached = AxiomResponse(
        request_id="test-001",
        status="success",
        is_answerable=True,
        confidence_summary=ConfidenceSummary(overall_score=0.8, tier_breakdown=TierBreakdown()),
    )
    payload = _make_payload()
    initial_state = {"request_id": "test-001", "audit_trail": []}  # type: ignore[arg-type]

    frames = await _collect(
        stream_pipeline(payload, engine=None, initial_state=initial_state, cached_response=cached)
    )
    assert frames[0]["type"] == "accepted"
    assert frames[0]["cached"] is True
    assert frames[-1]["type"] == "complete"
    assert len(frames) == 2


@pytest.mark.asyncio
async def test_stream_pipeline_emits_stage_events_in_order() -> None:
    node_seq = ["retriever", "scorer", "ranker", "synthesizer", "verifier"]
    events = []
    for node in node_seq:
        events.append(_langgraph_event("on_chain_start", node))
        events.append(_langgraph_event("on_chain_end", node, output={}))

    payload = _make_payload()
    initial_state = {
        "request_id": "test-001",
        "user_query": "q",
        "audit_trail": [],
        "is_answerable": True,
        "final_sentences": [],
        "loop_count": 0,
        "retrieval_retry_count": 0,
        "pending_rewrite_count": 0,
    }

    # Patch marshal_response so we don't need a real graph result
    with patch("axiom_rag_engine.api.sse.marshal_response") as mock_marshal:
        from axiom_rag_engine.models import AxiomResponse, ConfidenceSummary, TierBreakdown

        mock_marshal.return_value = AxiomResponse(
            request_id="test-001",
            status="unanswerable",
            is_answerable=False,
            confidence_summary=ConfidenceSummary(overall_score=0.0, tier_breakdown=TierBreakdown()),
        )
        frames = await _collect(
            stream_pipeline(
                payload,
                engine=_mock_engine(events),
                initial_state=initial_state,  # type: ignore[arg-type]
            )
        )

    types = [f["type"] for f in frames]
    assert types[0] == "accepted"
    assert types[-1] == "complete"
    stage_frames = [f for f in frames if f["type"] == "stage"]
    stages_in_order = [f["stage"] for f in stage_frames if f["phase"] == "start"]
    assert stages_in_order == node_seq


@pytest.mark.asyncio
async def test_client_disconnect_cancels_inflight_pipeline() -> None:
    """A dropped client must cancel the in-flight engine iteration.

    Starlette tears down the response by cancelling the task that iterates the
    generator and then closing it. The generator's cleanup must propagate that
    cancellation into the pending ``astream_events`` step — otherwise the
    abandoned pipeline task leaks and keeps burning LLM budget with no reader.
    """
    import contextlib

    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def _astream_events(state, *, version):
        yield _langgraph_event("on_chain_start", "retriever")
        started.set()
        try:
            await asyncio.sleep(3600)  # simulate a long in-flight LLM call
        except asyncio.CancelledError:
            cancelled.set()
            raise
        yield _langgraph_event("on_chain_end", "retriever", {})

    engine = MagicMock()
    engine.astream_events = _astream_events

    gen = stream_pipeline(
        _make_payload(),
        engine=engine,
        initial_state={"request_id": "test-001", "audit_trail": []},  # type: ignore[arg-type]
    )
    await gen.__anext__()  # accepted frame
    await gen.__anext__()  # retriever stage-start frame

    # Resume the generator in a task; it parks on the blocked engine step.
    consumer = asyncio.ensure_future(gen.__anext__())
    await started.wait()
    # Simulate the disconnect: Starlette cancels the consuming task, then
    # closes the generator.
    consumer.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await consumer
    await gen.aclose()

    assert cancelled.is_set(), "engine iteration was not cancelled on client disconnect"


@pytest.mark.asyncio
async def test_stream_pipeline_emits_loop_event_on_rewrite() -> None:
    events = [
        _langgraph_event("on_chain_start", "verifier"),
        _langgraph_event(
            "on_chain_end",
            "verifier",
            output={"pending_rewrite_count": 1, "loop_count": 1, "retrieval_retry_count": 0},
        ),
        # Second verifier pass after rewrite
        _langgraph_event("on_chain_start", "verifier"),
        _langgraph_event(
            "on_chain_end",
            "verifier",
            output={"pending_rewrite_count": 0, "loop_count": 1, "retrieval_retry_count": 0},
        ),
    ]
    payload = _make_payload()
    initial_state = {"request_id": "test-001", "audit_trail": [], "is_answerable": True}

    with patch("axiom_rag_engine.api.sse.marshal_response") as mock_marshal:
        from axiom_rag_engine.models import AxiomResponse, ConfidenceSummary, TierBreakdown

        mock_marshal.return_value = AxiomResponse(
            request_id="test-001",
            status="unanswerable",
            is_answerable=False,
            confidence_summary=ConfidenceSummary(overall_score=0.0, tier_breakdown=TierBreakdown()),
        )
        frames = await _collect(
            stream_pipeline(payload, engine=_mock_engine(events), initial_state=initial_state)  # type: ignore[arg-type]
        )

    loop_frames = [f for f in frames if f["type"] == "loop"]
    assert len(loop_frames) == 1
    assert loop_frames[0]["reason"] == "rewrite"
    assert loop_frames[0]["loop_count"] == 1


@pytest.mark.asyncio
async def test_stream_pipeline_emits_error_on_exception() -> None:
    async def _bad_stream(state, *, version):
        raise RuntimeError("synthetic failure")
        yield  # make it an async generator

    engine = MagicMock()
    engine.astream_events = _bad_stream
    payload = _make_payload()
    initial_state = {"request_id": "test-001", "audit_trail": []}

    frames = await _collect(
        stream_pipeline(payload, engine=engine, initial_state=initial_state)  # type: ignore[arg-type]
    )
    assert frames[0]["type"] == "accepted"
    assert frames[-1]["type"] == "error"
    assert frames[-1]["error_type"] == "RuntimeError"


# ---------------------------------------------------------------------------
# HTTP endpoint smoke test
# ---------------------------------------------------------------------------


def test_stream_endpoint_exists(client) -> None:
    """Verify the route is registered and returns text/event-stream."""
    # POST with a minimal payload; pipeline will error on the mock backend
    # but the endpoint itself must respond with the correct content-type.
    resp = client.post(
        "/v1/synthesize/stream",
        json={"request_id": "smoke-1", "user_query": "test"},
        headers={"Accept": "text/event-stream"},
    )
    assert resp.headers["content-type"].startswith("text/event-stream")
