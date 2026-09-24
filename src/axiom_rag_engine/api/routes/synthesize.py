"""The pipeline endpoints: POST /v1/synthesize and its SSE twin."""

from __future__ import annotations

import asyncio
import contextlib
import copy
import hashlib
import json
import logging
import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from slowapi import Limiter

from axiom_rag_engine.api.auth import verify_api_key
from axiom_rag_engine.api.deps import Services
from axiom_rag_engine.api.routes.audits import audit_owner, persist_and_emit_audit
from axiom_rag_engine.api.sse import stream_pipeline
from axiom_rag_engine.config.logging import request_id_ctx
from axiom_rag_engine.config.observability import (
    CACHE_HITS,
    CACHE_MISSES,
    LLM_CALLS_PER_REQUEST,
    PIPELINE_DURATION,
    PIPELINE_HALTS,
    REQUESTS_BY_STATUS,
    TIER_ASSIGNMENTS,
    tag_current_span,
)
from axiom_rag_engine.config.settings import Settings, use_settings
from axiom_rag_engine.graph import (
    PipelineDeadlineError,
    PipelineProgress,
    run_pipeline,
    with_failure_event,
)
from axiom_rag_engine.marshalling import make_error_response, marshal_response
from axiom_rag_engine.models import AxiomRequest, AxiomResponse
from axiom_rag_engine.services import AppServices
from axiom_rag_engine.state import GraphState, make_initial_state
from axiom_rag_engine.utils.llm import (
    LLMBudgetExceededError,
    get_llm_usage_snapshot,
    reset_llm_budget,
)

logger = logging.getLogger("axiom_rag_engine")

# ---------------------------------------------------------------------------
# Server-enforced request policy
# ---------------------------------------------------------------------------


def effective_app_config(payload: AxiomRequest, settings: Settings) -> dict[str, Any]:
    """The caller's app_config with trust policy replaced by server settings."""
    effective = payload.app_config.model_dump()
    ignored_fields = [
        field
        for field in ("authoritative_domains", "low_quality_domains", "exclude_default_domains")
        if effective.get(field)
    ]
    if ignored_fields:
        logger.warning(
            "Ignoring caller trust-policy overrides for request %s: %s",
            payload.request_id,
            ", ".join(ignored_fields),
        )
    effective["authoritative_domains"] = list(settings.authoritative_domains)
    effective["low_quality_domains"] = list(settings.low_quality_domains)
    effective["exclude_default_domains"] = list(settings.exclude_default_domains)
    return effective


def effective_pipeline_config(payload: AxiomRequest, settings: Settings) -> dict[str, Any]:
    """The caller's pipeline_config with semantic verification set by server policy."""
    effective = payload.pipeline_config.model_dump()
    server_semantic = settings.semantic_verification_enabled
    requested = bool(effective["stages"].get("semantic_verification_enabled", True))
    if requested != server_semantic:
        logger.warning(
            "Ignoring caller semantic_verification_enabled=%s for request %s; server policy is %s.",
            requested,
            payload.request_id,
            server_semantic,
        )
    effective["stages"]["semantic_verification_enabled"] = server_semantic
    return effective


def effective_models_config(payload: AxiomRequest, services: AppServices) -> dict[str, Any]:
    """The models this request runs on, under server model policy.

    When auth is required, the verifier is server policy — it grants the tiers,
    so a caller must not pick a lenient judge for its own answers — and a caller
    may choose only a synthesizer the operator allows (the server default or
    AXIOM_ALLOWED_SYNTHESIZER_MODELS). With auth disabled the caller is the
    operator, so its choices are honoured. Omitted models fall back to the
    startup-detected defaults (which account for available API keys).

    Raises:
        HTTPException: 422 when the requested synthesizer is not allowed.
    """
    settings = services.settings
    synthesizer = payload.models.synthesizer or services.default_synthesizer_model
    verifier = payload.models.verifier or services.default_verifier_model
    if not settings.auth_required():
        return {"synthesizer": synthesizer, "verifier": verifier}

    if verifier != services.default_verifier_model:
        logger.warning(
            "Ignoring caller verifier=%s for request %s; the verifier is server policy.",
            verifier,
            payload.request_id,
        )
    allowed = {services.default_synthesizer_model, *settings.allowed_synthesizer_models}
    if synthesizer not in allowed:
        raise HTTPException(
            status_code=422,
            detail=f"Synthesizer model {synthesizer!r} is not allowed on this server.",
        )
    return {"synthesizer": synthesizer, "verifier": services.default_verifier_model}


def _initial_state(payload: AxiomRequest, services: AppServices) -> GraphState:
    """Build the graph input from the request and server policy (both endpoints)."""
    return make_initial_state(
        request_id=payload.request_id,
        user_query=payload.user_query,
        app_config=effective_app_config(payload, services.settings),
        models_config=effective_models_config(payload, services),
        pipeline_config=effective_pipeline_config(payload, services.settings),
    )


# ---------------------------------------------------------------------------
# Response cache
# ---------------------------------------------------------------------------


def cache_key(
    payload: AxiomRequest,
    api_key: str | None,
    app_config: dict[str, Any],
    pipeline_config: dict[str, Any],
    models_config: dict[str, Any],
    corpus_version: int | None = None,
) -> str:
    """
    SHA-256 of the request fields that shape the response body, namespaced by
    a hash of the caller's API key. Configs are the *effective* ones (after
    server policy), so an ignored override never splits or aliases entries.

    Namespacing prevents cross-tenant cache poisoning: two callers with different
    API keys cannot serve each other's cached results even when all other fields
    match. The full 64-hex digest keeps collisions out of reach at any scale.

    ``corpus_version`` (when a corpus store is configured) changes on every
    ingest and delete, so answers built from a deleted or replaced document are
    never served from cache.
    """
    key_namespace = hashlib.sha256((api_key or "anonymous").encode()).hexdigest()
    raw = json.dumps(
        {
            "ns": key_namespace,
            "query": payload.user_query,
            "models": models_config,
            "pipeline": pipeline_config,
            "app": app_config,
            "include_debug": payload.include_debug,
            "corpus_version": corpus_version,
        },
        sort_keys=True,
    )
    # Prefix with the namespace so backends that scan (Redis) always see the
    # tenant boundary.
    body_hash = hashlib.sha256(raw.encode()).hexdigest()
    return f"{key_namespace}:{body_hash}"


def _response_to_cache_value(response: AxiomResponse) -> dict[str, Any]:
    """Serialize a response without request-scoped identifiers for safe reuse."""
    data = response.model_dump()
    data.pop("request_id", None)
    return data


def _hydrate_cached_response(request_id: str, cached: dict[str, Any]) -> AxiomResponse:
    """Rebuild a response for the current request from cached template data.

    Strips the stored ``usage`` block: a cache hit consumes zero tokens and zero
    cost, so the caller's billing view reflects *this* request.
    """
    data = copy.deepcopy(cached)
    data["usage"] = None
    return AxiomResponse.model_validate({"request_id": request_id, **data})


async def _request_cache_key(
    services: AppServices,
    payload: AxiomRequest,
    api_key: str | None,
    initial_state: GraphState,
) -> str:
    """The cache key for this request under its effective configuration."""
    corpus_version = (
        await asyncio.to_thread(services.corpus_store.version)
        if services.corpus_store is not None
        else None
    )
    return cache_key(
        payload,
        api_key,
        initial_state["app_config"],
        initial_state["pipeline_config"],
        initial_state["models_config"],
        corpus_version=corpus_version,
    )


async def _get_cached(services: AppServices, key: str, request_id: str) -> AxiomResponse | None:
    cached = await services.cache.get(key)
    return None if cached is None else _hydrate_cached_response(request_id, cached)


async def _set_cached(services: AppServices, key: str, response: AxiomResponse) -> None:
    # Cache successful and partial responses only — not errors or unanswerable.
    if response.status in ("success", "partial"):
        await services.cache.set(key, _response_to_cache_value(response))


def _record_outcome_metrics(response: AxiomResponse, graph_result: dict[str, Any]) -> None:
    REQUESTS_BY_STATUS.labels(status=response.status).inc()
    if graph_result.get("halt_reason"):
        PIPELINE_HALTS.labels(reason=str(graph_result["halt_reason"])).inc()
    if response.usage is not None:
        LLM_CALLS_PER_REQUEST.observe(response.usage.calls)
    # Claims only, like the response's tier_breakdown: uncited sentences carry
    # no checked quote and used to be counted here as tier 3.
    for sentence in response.final_response:
        if sentence.is_cited:
            verification = sentence.verification
            TIER_ASSIGNMENTS.labels(
                tier=str(verification.tier), label=verification.tier_label
            ).inc()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


class ClientDisconnectedError(Exception):
    """The client went away before the pipeline finished."""


# How often a running request checks whether its client is still connected.
_DISCONNECT_POLL_SECONDS = 0.5


async def _wait_for_disconnect(request: Request) -> None:
    while not await request.is_disconnected():
        await asyncio.sleep(_DISCONNECT_POLL_SECONDS)


async def run_unless_disconnected(request: Request, awaitable: Any) -> Any:
    """Await ``awaitable``, cancelling it if the client disconnects first.

    Starlette does not cancel a handler when its client goes away, so without
    this a JSON request kept running — and spending LLM budget — for a response
    nobody would read. (The SSE endpoint gets the same effect from generator
    teardown.) The task inherits this context, so the per-request LLM budget
    and usage counters are shared.

    Raises:
        ClientDisconnectedError: the client disconnected; the work was cancelled.
    """
    task = asyncio.ensure_future(awaitable)
    watcher = asyncio.ensure_future(_wait_for_disconnect(request))
    try:
        done, _ = await asyncio.wait({task, watcher}, return_when=asyncio.FIRST_COMPLETED)
    except BaseException:
        task.cancel()
        watcher.cancel()
        raise
    watcher.cancel()
    if task in done:
        return task.result()
    task.cancel()
    with contextlib.suppress(Exception, asyncio.CancelledError):
        await task
    raise ClientDisconnectedError


async def synthesize(
    services: Services,
    request: Request,
    payload: AxiomRequest,
    _api_key: str | None = Depends(verify_api_key),
) -> Response:
    """
    Accept an AxiomRequest, execute the LangGraph DAG, and return a fully
    validated AxiomResponse with tier breakdown and confidence score.

    Pipeline errors return HTTP 500, budget exhaustion before any verified pass
    HTTP 422 (not retryable: the same request would exhaust it again), and a request
    deadline that expires before any verified pass HTTP 504 (one that expires
    later returns the best verified pass); successful, partial, and unanswerable
    results return HTTP 200. If the client
    disconnects first, the pipeline is cancelled and no further LLM budget is
    spent.
    """
    request_id_ctx.set(payload.request_id)
    use_settings(services.settings)
    tag_current_span(payload.request_id)
    initial_state = _initial_state(payload, services)

    key = await _request_cache_key(services, payload, _api_key, initial_state)
    cached = await _get_cached(services, key, payload.request_id)
    if cached is not None:
        CACHE_HITS.inc()
        logger.info("Cache hit for request %s", payload.request_id)
        REQUESTS_BY_STATUS.labels(status=cached.status).inc()
        return JSONResponse(content=cached.model_dump())
    CACHE_MISSES.inc()

    # Initialize the per-request LLM call budget. The mutable dict stored in the
    # ContextVar is shared by all asyncio tasks spawned from this coroutine.
    reset_llm_budget()
    progress = PipelineProgress(initial_state)

    def _failed(
        status_code: int, exc: Exception, public_message: str | None = None
    ) -> JSONResponse:
        """Error response for a failed run, keeping the trail of how far it got."""
        REQUESTS_BY_STATUS.labels(status="error").inc()
        usage = get_llm_usage_snapshot()
        persist_and_emit_audit(
            services,
            payload.request_id,
            "error",
            with_failure_event(dict(progress.state), progress.node, exc),
            usage_snapshot=usage,
            owner=audit_owner(_api_key),
        )
        error_resp = make_error_response(payload.request_id, exc, usage, public_message)
        return JSONResponse(status_code=status_code, content=error_resp.model_dump())

    try:
        with PIPELINE_DURATION.time():
            graph_result = await run_unless_disconnected(
                request,
                run_pipeline(
                    services.engine,
                    initial_state,
                    services.run_config(),
                    services.settings.request_deadline_seconds,
                    progress=progress,
                ),
            )
    except ClientDisconnectedError:
        REQUESTS_BY_STATUS.labels(status="cancelled").inc()
        logger.info("Client disconnected; cancelled request %s", payload.request_id)
        # 499 (client closed request): nobody reads it, but logs show why.
        return Response(status_code=499)
    except LLMBudgetExceededError as exc:
        # 422, not 429: the same request would exhaust the same budget again,
        # and clients and proxies retry 429s. (After a verified pass the run
        # already returns that pass with 200.)
        return _failed(
            422,
            exc,
            "The request exhausted its LLM budget before any answer was verified "
            "(AXIOM_MAX_LLM_CALLS_PER_REQUEST / AXIOM_MAX_TOKENS_PER_REQUEST).",
        )
    except PipelineDeadlineError as exc:
        return _failed(504, exc)
    except Exception as exc:
        return _failed(500, exc)

    response = marshal_response(
        payload.request_id,
        graph_result,
        payload.include_debug,
        get_llm_usage_snapshot(),
    )
    _record_outcome_metrics(response, graph_result)
    persist_and_emit_audit(
        services,
        payload.request_id,
        response.status,
        graph_result,
        usage_snapshot=response.usage.model_dump() if response.usage else None,
        owner=audit_owner(_api_key),
    )
    await _set_cached(services, key, response)
    return JSONResponse(content=response.model_dump())


async def synthesize_stream(
    services: Services,
    request: Request,
    payload: AxiomRequest,
    _api_key: str | None = Depends(verify_api_key),
) -> Response:
    """Stream pipeline progress as Server-Sent Events.

    Same request body as ``POST /v1/synthesize``. Emits one SSE frame per
    pipeline stage plus a ``complete`` frame carrying the full AxiomResponse.
    Sentences appear in ``sentence`` frames only after the final verification
    pass, each with its verification result — including sentences that failed
    or could not be verified, which are labelled (as in the JSON response), not
    hidden. Draft text from intermediate passes never reaches the client.

    Disconnect behavior: if the client drops mid-stream the pipeline is
    cancelled — in-flight LLM calls are unwound and no further budget is
    consumed. Audit trails, metrics, and cache writes happen only for runs
    that stream to completion.
    """
    request_id_ctx.set(payload.request_id)
    use_settings(services.settings)
    tag_current_span(payload.request_id)
    initial_state = _initial_state(payload, services)

    key = await _request_cache_key(services, payload, _api_key, initial_state)
    cached = await _get_cached(services, key, payload.request_id)
    if cached is not None:
        CACHE_HITS.inc()
        REQUESTS_BY_STATUS.labels(status=cached.status).inc()
    else:
        CACHE_MISSES.inc()
        reset_llm_budget()

    started = time.monotonic()

    async def _on_complete(response: AxiomResponse, graph_result: dict[str, Any]) -> None:
        """Post-pipeline housekeeping: metrics, audit, cache."""
        PIPELINE_DURATION.observe(time.monotonic() - started)
        _record_outcome_metrics(response, graph_result)
        persist_and_emit_audit(
            services,
            payload.request_id,
            response.status,
            graph_result,
            usage_snapshot=response.usage.model_dump() if response.usage else None,
            owner=audit_owner(_api_key),
        )
        await _set_cached(services, key, response)

    async def _on_error(failed_state: dict[str, Any]) -> None:
        """Failed run: count it and keep the trail of how far it got."""
        PIPELINE_DURATION.observe(time.monotonic() - started)
        REQUESTS_BY_STATUS.labels(status="error").inc()
        persist_and_emit_audit(
            services,
            payload.request_id,
            "error",
            failed_state,
            usage_snapshot=get_llm_usage_snapshot(),
            owner=audit_owner(_api_key),
        )

    return StreamingResponse(
        stream_pipeline(
            payload=payload,
            engine=services.engine,
            initial_state=initial_state,
            cached_response=cached,
            on_complete=_on_complete,
            run_config=services.run_config(),
            deadline_seconds=services.settings.request_deadline_seconds,
            on_error=_on_error,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


def build_router(limiter: Limiter, stream_rate_limit: str) -> APIRouter:
    """The pipeline routes, with the stream endpoint bound to ``limiter``.

    Built per app because slowapi binds a route-specific limit to a limiter
    instance at decoration time.
    """
    router = APIRouter()
    router.add_api_route(
        "/v1/synthesize",
        synthesize,
        methods=["POST"],
        response_model=AxiomResponse,
        summary="Run the Axiom Engine verification pipeline.",
    )
    router.add_api_route(
        "/v1/synthesize/stream",
        limiter.limit(stream_rate_limit)(synthesize_stream),
        methods=["POST"],
        summary="Run the Axiom Engine pipeline with SSE progress events.",
    )
    return router
