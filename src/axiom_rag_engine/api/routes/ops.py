"""Health probes and the operator status snapshot."""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, Response

from axiom_rag_engine.api.auth import _api_keys, _auth_required, verify_api_key
from axiom_rag_engine.api.deps import Services
from axiom_rag_engine.services import AppServices

router = APIRouter()


@router.get("/health", summary="Combined liveness + readiness probe (backward compat).")
async def health() -> dict[str, str]:
    """Legacy combined probe — prefer /health/live and /health/ready."""
    return {"status": "ok"}


@router.get("/health/live", summary="Liveness probe — is the process alive?")
async def health_live() -> dict[str, str]:
    return {"status": "ok"}


# Probes can arrive every few seconds from several sources; dependency checks
# are reused for this long.
_READINESS_TTL_SECONDS = 5.0


async def _dependency_checks(services: AppServices) -> dict[str, str]:
    """Reachability of the app's dependencies, cached for a few seconds."""
    now = time.monotonic()
    if services.readiness_checks and now - services.readiness_checked_at < _READINESS_TTL_SECONDS:
        return services.readiness_checks
    checks = {"cache": "ok" if await services.cache.ping() else "unavailable"}
    if services.corpus_store is not None:
        try:
            await asyncio.to_thread(services.corpus_store.count_documents)
            checks["corpus"] = "ok"
        except Exception:
            checks["corpus"] = "unavailable"
    services.readiness_checks, services.readiness_checked_at = checks, now
    return checks


@router.get("/health/ready", summary="Readiness probe — is the engine ready to serve?")
async def health_ready(request: Request) -> Response:
    """200 when the engine can serve, 503 when it cannot.

    Beyond configuration, the probe checks the app's dependencies (cached for
    a few seconds). A configured corpus database that cannot be read makes the
    app not ready. The response cache is optional by design — its failures
    degrade to cache misses — so an unreachable Redis reports
    ``"status": "degraded"`` but stays ready, rather than pulling every
    replica out of rotation.
    """
    services: AppServices | None = getattr(request.app.state, "services", None)
    if services is None or services.engine is None:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "detail": "Graph engine not yet compiled."},
        )
    settings = services.settings
    if _auth_required(settings) and not _api_keys(settings):
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "detail": "API keys are not configured."},
        )
    if (
        settings.is_production()
        and services.search_backend_mode == "mock"
        and not settings.allow_mock_search
    ):
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "detail": "Live search backend is not configured."},
        )
    checks = await _dependency_checks(services)
    if checks.get("corpus") == "unavailable":
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "detail": "Corpus database is unavailable.",
                "checks": checks,
            },
        )
    degraded = any(state != "ok" for state in checks.values())
    return JSONResponse(content={"status": "degraded" if degraded else "ok", "checks": checks})


@router.get("/v1/status", summary="Operator-oriented runtime status snapshot.")
async def get_status(
    services: Services,
    _api_key: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Summarise the process: version, uptime, policy, and configured backends.

    Intended for ops dashboards and smoke tests. Authenticated like the rest of
    ``/v1`` (it reveals models, limits, and corpus contents); the unauthenticated
    probes are ``/health/live`` and ``/health/ready``. Does not expose secrets —
    API keys and Redis URLs are reported as booleans only.
    """
    settings = services.settings
    store = services.audit_store
    corpus_stats = (
        (await asyncio.to_thread(services.corpus_store.stats)).as_dict()
        if services.corpus_store is not None
        else None
    )

    return {
        "service": "axiom-rag-engine",
        "version": services.version,
        "env": settings.env,
        "uptime_seconds": round(time.time() - services.started_at, 3),
        "engine_ready": services.engine is not None,
        "search_backend": services.search_backend_mode,
        "auth_required": _auth_required(settings),
        "api_keys_configured": bool(_api_keys(settings)),
        "cache": {
            "backend": type(services.cache).__name__,
            "ttl_seconds": settings.cache_ttl_seconds,
            "max_size": settings.cache_max_size,
            "redis_configured": bool(settings.redis_url),
        },
        "audit_retention": {
            "enabled": store.enabled,
            "capacity": store.capacity,
            "retained": len(store),
        },
        "limits": {
            "rate_limit": settings.rate_limit,
            "stream_rate_limit": settings.stream_rate_limit,
            "max_body_bytes": settings.max_body_bytes,
            "max_llm_calls_per_request": settings.max_llm_calls_per_request,
            "max_tokens_per_request": settings.max_tokens_per_request,
            "max_concurrent_llm": settings.max_concurrent_llm,
            "max_concurrent_verifier_llm": (
                settings.max_concurrent_verifier_llm or settings.max_concurrent_llm
            ),
        },
        "models": {
            "synthesizer_default": services.default_synthesizer_model,
            "verifier_default": services.default_verifier_model,
        },
        "retrieval": {
            "ranking_mode": "hybrid" if settings.embedding_model else "bm25",
            "embedding_model": settings.embedding_model,
            "rrf_k": settings.rrf_k,
            "reranker_model": settings.reranker_model,
            "rerank_top_k": settings.rerank_top_k if settings.reranker_model else None,
            "source": settings.retrieval_source,
            "corpus": corpus_stats,
        },
        "observability": {
            "log_format": settings.log_format,
            "log_audit_events": settings.log_audit_events,
            "tracing_configured": bool(os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")),
        },
    }


# Probes and the status snapshot are never rate-limited (orchestrators poll them).
RATE_LIMIT_EXEMPT = (health, health_live, health_ready, get_status)
