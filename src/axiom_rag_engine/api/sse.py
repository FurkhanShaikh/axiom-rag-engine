"""SSE streaming generator for ``POST /v1/synthesize/stream``.

Yields one SSE frame per pipeline event so the caller can show live
progress while the graph runs.  The verification contract is preserved:
``sentence`` frames are only emitted after a citation clears both
mechanical and semantic verification — unverified text never crosses the wire.

Event ordering guarantee:
  accepted → stage*(start|complete) / loop* → sentence* → complete | error
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

from axiom_rag_engine.marshalling import marshal_response
from axiom_rag_engine.utils.llm import LLMBudgetExceededError, get_llm_usage_snapshot

logger = logging.getLogger("axiom_rag_engine.api.sse")

if TYPE_CHECKING:
    from axiom_rag_engine.models import AxiomRequest, AxiomResponse
    from axiom_rag_engine.state import GraphState

# Node names registered in graph.py — used to filter LangGraph event stream.
_NODE_NAMES = frozenset(
    {"retriever", "re_retriever", "scorer", "ranker", "synthesizer", "verifier"}
)

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
# State accumulator
# ---------------------------------------------------------------------------


def _apply_node_update(state: dict[str, Any], update: dict[str, Any]) -> None:
    """Merge a node's state-update dict into the accumulated state.

    ``audit_trail`` uses an ``operator.add`` reducer in LangGraph — new events
    are appended, never replaced.  All other fields are replaced wholesale.
    """
    for key, value in update.items():
        if key == "audit_trail":
            existing = list(state.get("audit_trail") or [])
            state["audit_trail"] = existing + list(value or [])
        else:
            state[key] = value


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------


async def stream_pipeline(
    payload: AxiomRequest,
    engine: Any,
    initial_state: GraphState,
    cached_response: AxiomResponse | None = None,
    on_complete: Any | None = None,
) -> AsyncGenerator[str, None]:
    """Async generator that yields SSE frames for one pipeline execution.

    ``on_complete`` is awaited with ``(AxiomResponse, accumulated_state)``
    immediately before the ``complete`` frame — use it for cache writes,
    Prometheus updates, and audit persistence.
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
    node_start_times: dict[str, float] = {}
    last_loop_count = 0
    last_retry_count = 0

    # Accumulate state updates so we can build the final result without a
    # second ainvoke call.
    accumulated: dict[str, Any] = dict(initial_state)
    accumulated["audit_trail"] = list(initial_state.get("audit_trail") or [])

    async def _next(iterator: Any) -> Any:
        return await iterator.__anext__()

    try:
        it = engine.astream_events(initial_state, version="v2").__aiter__()
        pending_event: asyncio.Task[Any] | None = None
        while True:
            if pending_event is None:
                pending_event = asyncio.ensure_future(_next(it))
            # Race the next pipeline event against a keepalive timer so we
            # don't cancel __anext__() when we want to emit a keepalive.
            timeout_task = asyncio.ensure_future(asyncio.sleep(_KEEPALIVE_INTERVAL))
            done, _pending_set = await asyncio.wait(
                {pending_event, timeout_task}, return_when=asyncio.FIRST_COMPLETED
            )
            if pending_event not in done:
                timeout_task.cancel()
                yield ": keepalive\n\n"
                continue
            timeout_task.cancel()
            try:
                event = pending_event.result()
            except StopAsyncIteration:
                pending_event = None
                break
            pending_event = None

            evt_type: str = event.get("event", "")
            name: str = event.get("name", "")
            data: dict[str, Any] = event.get("data") or {}

            if name not in _NODE_NAMES:
                continue

            if evt_type == "on_chain_start":
                node_start_times[name] = time.monotonic()
                yield _sse(
                    "stage",
                    {
                        "type": "stage",
                        "stage": name,
                        "phase": "start",
                        "elapsed_ms": 0,
                        "metadata": {},
                    },
                    _next_id(),
                )

            elif evt_type == "on_chain_end":
                elapsed_ms = round(
                    (time.monotonic() - node_start_times.get(name, time.monotonic())) * 1000
                )
                output: dict[str, Any] = data.get("output") or {}
                if isinstance(output, dict):
                    _apply_node_update(accumulated, output)
                if name == "verifier":
                    logger.debug(
                        "[%s] verifier on_chain_end: output type=%s keys=%s final_sentences=%d draft_sentences_in_accum=%d",
                        payload.request_id,
                        type(data.get("output")).__name__,
                        list(output.keys()) if isinstance(output, dict) else "N/A",
                        len(output.get("final_sentences") or [])
                        if isinstance(output, dict)
                        else -1,
                        len(accumulated.get("draft_sentences") or []),
                    )

                metadata = _stage_metadata(
                    name, "complete", output if isinstance(output, dict) else {}
                )
                yield _sse(
                    "stage",
                    {
                        "type": "stage",
                        "stage": name,
                        "phase": "complete",
                        "elapsed_ms": elapsed_ms,
                        "metadata": metadata,
                    },
                    _next_id(),
                )

                # Emit loop events from verifier output so the UI can show
                # "rewriting..." or "fetching more sources..." badges.
                if name == "verifier" and isinstance(output, dict):
                    new_loop = int(output.get("loop_count") or 0)
                    new_retry = int(output.get("retrieval_retry_count") or 0)
                    pending = int(output.get("pending_rewrite_count") or 0)
                    if pending > 0 and new_loop > last_loop_count:
                        yield _sse(
                            "loop",
                            {
                                "type": "loop",
                                "loop_count": new_loop,
                                "retrieval_retry_count": new_retry,
                                "reason": "rewrite",
                            },
                            _next_id(),
                        )
                        last_loop_count = new_loop
                    elif pending > 0 and new_retry > last_retry_count:
                        yield _sse(
                            "loop",
                            {
                                "type": "loop",
                                "loop_count": new_loop,
                                "retrieval_retry_count": new_retry,
                                "reason": "re_retrieve",
                            },
                            _next_id(),
                        )
                        last_retry_count = new_retry

    except LLMBudgetExceededError as exc:
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
    except Exception as exc:
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
    final_sentences = accumulated.get("final_sentences") or []
    logger.debug(
        "Stream complete for %s — final_sentences=%d is_answerable=%s",
        payload.request_id,
        len(final_sentences),
        accumulated.get("is_answerable"),
    )
    if final_sentences:
        first = final_sentences[0].get("verification", {})
        logger.debug(
            "First sentence tier=%s mech=%s", first.get("tier"), first.get("mechanical_check")
        )
    usage_snapshot = get_llm_usage_snapshot()
    response = marshal_response(
        payload.request_id,
        accumulated,
        payload.include_debug,
        usage_snapshot,
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
        with contextlib.suppress(Exception):
            await on_complete(response, accumulated)

    # -- complete (terminal) --
    yield _sse("complete", {"type": "complete", "response": response.model_dump()}, _next_id())
