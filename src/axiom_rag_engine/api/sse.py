"""SSE streaming generator for ``POST /v1/synthesize/stream``.

Yields one SSE frame per pipeline event so the caller can show live
progress while the graph runs. Answer text is withheld until the pipeline
finishes verifying: ``sentence`` frames are emitted only after the final
verification pass, each carrying its verification result. That includes
sentences that did NOT verify (Tier 4/5, or tier_label "unverified") — they are
labelled, not hidden, exactly as in the non-streaming response. Draft text from
intermediate passes never crosses the wire.

``loop`` frames are emitted when a rewrite pass or a re-retrieval actually
starts (not when the verifier merely reports pending failures, which may end
the run instead).

Event ordering guarantee:
  accepted → stage*(start|complete) / loop* → sentence* → complete | error
"""

from __future__ import annotations

import contextlib
import json
import logging
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

from axiom_rag_engine.graph import (
    Keepalive,
    NodeFinished,
    NodeStarted,
    PipelineDeadlineError,
    PipelineProgress,
    RunFinished,
    run_events,
    with_failure_event,
)
from axiom_rag_engine.marshalling import marshal_response
from axiom_rag_engine.utils.llm import LLMBudgetExceededError, get_llm_usage_snapshot

logger = logging.getLogger("axiom_rag_engine.api.sse")

if TYPE_CHECKING:
    from axiom_rag_engine.models import AxiomRequest, AxiomResponse
    from axiom_rag_engine.state import GraphState

# Seconds between keepalive comment frames; prevents proxy idle-connection drops
# during the synthesizer's long LLM call.
_KEEPALIVE_INTERVAL: float = 15.0


# ---------------------------------------------------------------------------
# SSE frame helpers
# ---------------------------------------------------------------------------


def _sse(event_type: str, data: Any, event_id: int | None = None) -> str:
    """Encode one SSE frame as a string ready for the wire."""
    parts = [f"event: {event_type}"]
    if event_id is not None:
        parts.append(f"id: {event_id}")
    parts.append(f"data: {json.dumps(data, default=str)}")
    return "\n".join(parts) + "\n\n"


def _stage_metadata(node: str, phase: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Extract a minimal, bounded metadata dict from a node's input/output."""
    if phase == "start":
        return {}
    # phase == "complete": payload is the node's state-update dict
    if node in ("retriever", "re_retriever"):
        return {"chunks_retrieved": len(payload.get("indexed_chunks") or [])}
    if node == "scorer":
        return {"chunks_scored": len(payload.get("scored_chunks") or [])}
    if node == "ranker":
        return {"chunks_ranked": len(payload.get("ranked_chunks") or [])}
    if node == "synthesizer":
        return {"draft_sentences": len(payload.get("draft_sentences") or [])}
    if node == "verifier":
        return {
            "final_sentences": len(payload.get("final_sentences") or []),
            "pending_rewrite_count": int(payload.get("pending_rewrite_count") or 0),
            "loop_count": int(payload.get("loop_count") or 0),
        }
    return {}


# ---------------------------------------------------------------------------
# Loop detection
# ---------------------------------------------------------------------------


def _loop_reason(node: str, state: dict[str, Any]) -> str | None:
    """Classify a node start as the beginning of a loop iteration, if it is one.

    ``state`` is the state LangGraph hands the node. A ``synthesizer`` start is
    a rewrite when the previous verification pass left correction requests; a
    ``re_retriever`` start is always a re-retrieval.
    """
    if node == "re_retriever":
        return "re_retrieve"
    if node == "synthesizer" and state.get("rewrite_requests"):
        return "rewrite"
    return None


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------


async def stream_pipeline(
    payload: AxiomRequest,
    engine: Any,
    initial_state: GraphState,
    cached_response: AxiomResponse | None = None,
    on_complete: Any | None = None,
    run_config: dict[str, Any] | None = None,
    deadline_seconds: float = 0.0,
    on_error: Any | None = None,
) -> AsyncGenerator[str, None]:
    """Async generator that yields SSE frames for one pipeline execution.

    The run itself is ``graph.run_events`` — the same runner the JSON endpoint
    uses — so both see the same final state, deadline and failure trail.

    ``on_complete`` is awaited with ``(AxiomResponse, final_state)`` immediately
    before the ``complete`` frame — use it for cache writes, Prometheus updates,
    and audit persistence. ``run_config`` is forwarded to LangGraph (it carries
    the app's search backend). ``deadline_seconds`` (0 = none) bounds the run:
    on expiry the best verified pass is returned, or an ``error`` frame
    (``deadline_exceeded``) if no pass was verified yet. ``on_error`` is awaited
    with the state reached so far (plus a ``pipeline_failed`` audit event)
    before any ``error`` frame, so a failed run's audit trail is kept.
    """
    event_id = 0

    def _next_id() -> int:
        nonlocal event_id
        event_id += 1
        return event_id

    # -- accepted (always first) --
    yield _sse(
        "accepted",
        {
            "type": "accepted",
            "request_id": payload.request_id,
            "cached": cached_response is not None,
        },
        _next_id(),
    )

    # -- cache-hit fast path --
    if cached_response is not None:
        yield _sse(
            "complete", {"type": "complete", "response": cached_response.model_dump()}, _next_id()
        )
        return

    # -- live pipeline --
    progress = PipelineProgress(initial_state)
    final_state: dict[str, Any] = dict(initial_state)

    async def _report_failure(exc: BaseException) -> None:
        if on_error is None:
            return
        try:
            await on_error(with_failure_event(dict(progress.state), progress.node, exc))
        except Exception:
            logger.exception("on_error hook failed for request %s", payload.request_id)

    try:
        async for event in run_events(
            engine,
            initial_state,
            run_config,
            deadline_seconds,
            progress=progress,
            keepalive_seconds=_KEEPALIVE_INTERVAL,
        ):
            if isinstance(event, Keepalive):
                yield ": keepalive\n\n"
            elif isinstance(event, NodeStarted):
                loop_reason = _loop_reason(event.node, event.state)
                if loop_reason is not None:
                    retry = int(event.state.get("retrieval_retry_count") or 0)
                    yield _sse(
                        "loop",
                        {
                            "type": "loop",
                            "loop_count": int(event.state.get("loop_count") or 0),
                            "retrieval_retry_count": retry + 1
                            if loop_reason == "re_retrieve"
                            else retry,
                            "reason": loop_reason,
                        },
                        _next_id(),
                    )
                yield _sse(
                    "stage",
                    {
                        "type": "stage",
                        "stage": event.node,
                        "phase": "start",
                        "elapsed_ms": 0,
                        "metadata": {},
                    },
                    _next_id(),
                )
            elif isinstance(event, NodeFinished):
                yield _sse(
                    "stage",
                    {
                        "type": "stage",
                        "stage": event.node,
                        "phase": "complete",
                        "elapsed_ms": event.elapsed_ms,
                        "metadata": _stage_metadata(event.node, "complete", event.update),
                    },
                    _next_id(),
                )
            elif isinstance(event, RunFinished):
                final_state = event.state
    except LLMBudgetExceededError as exc:
        await _report_failure(exc)
        yield _sse(
            "error",
            {
                "type": "error",
                "error_type": "budget_exceeded",
                "message": str(exc),
                "request_id": payload.request_id,
                "usage": get_llm_usage_snapshot(),
            },
            _next_id(),
        )
        return
    except PipelineDeadlineError as exc:
        await _report_failure(exc)
        yield _sse(
            "error",
            {
                "type": "error",
                "error_type": "deadline_exceeded",
                "message": "Request deadline expired before any verified pass.",
                "request_id": payload.request_id,
            },
            _next_id(),
        )
        return
    except Exception as exc:
        await _report_failure(exc)
        with contextlib.suppress(Exception):
            logger.exception(
                "Unhandled pipeline error for request %s: %s",
                payload.request_id,
                ascii(str(exc)),
            )
        yield _sse(
            "error",
            {
                "type": "error",
                "error_type": type(exc).__name__,
                "message": f"Pipeline error — see server logs for {payload.request_id}.",
                "request_id": payload.request_id,
            },
            _next_id(),
        )
        return

    # -- marshal final response --
    response = marshal_response(
        payload.request_id,
        final_state,
        payload.include_debug,
        get_llm_usage_snapshot(),
    )

    # -- sentence events (verified sentences only) --
    for sentence in response.final_response:
        yield _sse(
            "sentence",
            {"type": "sentence", "sentence": sentence.model_dump()},
            _next_id(),
        )

    # -- post-complete hook (cache, metrics, audit) before terminal frame --
    if on_complete is not None:
        try:
            await on_complete(response, final_state)
        except Exception:
            # Housekeeping (cache, metrics, audit) must not break the stream,
            # but its failures must be visible.
            logger.exception("on_complete hook failed for request %s", payload.request_id)

    # -- complete (terminal) --
    yield _sse("complete", {"type": "complete", "response": response.model_dump()}, _next_id())
