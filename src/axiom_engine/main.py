"""
Axiom Engine v2.3 — FastAPI Gateway

This module is the application entry point.  Domain logic has been extracted
into focused modules; this file handles only:
  - App factory & lifespan
  - CORS / rate-limiting / Prometheus middleware wiring
  - Global exception handler
  - Endpoint definitions
  - Backward-compatible re-exports for existing callers

Extracted modules:
  - axiom_engine.api.auth       — API key authentication
  - axiom_engine.scoring        — confidence scoring & status determination
  - axiom_engine.marshalling    — GraphState → AxiomResponse conversion
"""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Any

from dotenv import load_dotenv
from fastapi import (  # noqa: F401 — HTTPException re-exported for test compat
    Depends,
    FastAPI,
    HTTPException,
    Request,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

# --- Internal imports --------------------------------------------------------
from axiom_engine.api.auth import (
    _api_keys,
    _auth_required,
    verify_api_key,
)
from axiom_engine.cache import CacheBackend, MemoryCacheBackend, RedisCacheBackend
from axiom_engine.config.logging import configure_logging, request_id_ctx
from axiom_engine.config.observability import (
    CACHE_HITS,
    CACHE_MISSES,
    PIPELINE_DURATION,
    REQUESTS_BY_STATUS,
    TIER_ASSIGNMENTS,
    setup_prometheus,
    setup_tracing,
)
from axiom_engine.graph import build_axiom_graph
from axiom_engine.marshalling import make_error_response, marshal_response
from axiom_engine.models import AxiomRequest, AxiomResponse
from axiom_engine.nodes.retriever import set_search_backend
from axiom_engine.scoring import (  # noqa: F401 — re-exported
    compute_confidence_summary,
    determine_status,
)
from axiom_engine.state import make_initial_state
from axiom_engine.utils.llm import LLMBudgetExceededError, reset_llm_budget

load_dotenv()  # Load .env from project root; no-op if absent.


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("axiom_engine")


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _csv_env(name: str) -> list[str]:
    return [part.strip() for part in os.environ.get(name, "").split(",") if part.strip()]


def _allow_mock_search() -> bool:
    return _env_bool("AXIOM_ALLOW_MOCK_SEARCH", default=False)


def _semantic_verification_enabled() -> bool:
    return _env_bool("AXIOM_SEMANTIC_VERIFICATION_ENABLED", default=True)


def _effective_app_config(payload: AxiomRequest) -> dict[str, Any]:
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

    effective["authoritative_domains"] = _csv_env("AXIOM_AUTHORITATIVE_DOMAINS")
    effective["low_quality_domains"] = _csv_env("AXIOM_LOW_QUALITY_DOMAINS")
    effective["exclude_default_domains"] = _csv_env("AXIOM_EXCLUDE_DEFAULT_DOMAINS")
    return effective


def _effective_pipeline_config(payload: AxiomRequest) -> dict[str, Any]:
    effective = payload.pipeline_config.model_dump()
    server_semantic = _semantic_verification_enabled()
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


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


def get_real_ip(request: Request) -> str:
    """Extract client IP handling proxy X-Forwarded-For headers.

    Note: X-Forwarded-For can be spoofed by clients. In production, configure
    a reverse proxy (nginx, cloud LB) and trust only its header additions.
    """
    forwarded = request.headers.get("X-Forwarded-For")
    trusted_proxies = set(_csv_env("AXIOM_TRUSTED_PROXY_IPS"))
    if forwarded and trusted_proxies:
        remote_ip = get_remote_address(request)
        if "*" in trusted_proxies or remote_ip in trusted_proxies:
            # Some proxies add spaces after commas — strip all entries.
            return forwarded.split(",")[0].strip()
    return get_remote_address(request)


def rate_limit_key(request: Request) -> str:
    """
    Rate-limit bucket.

    Prefer a hashed API key so one key shared across many IPs still hits a
    single bucket (previously the IP-based key let an attacker bypass limits
    by spraying source addresses). Fall back to the real client IP for
    unauthenticated callers.
    """
    api_key = request.headers.get("X-API-Key")
    if api_key:
        return "key:" + hashlib.sha256(api_key.encode()).hexdigest()[:32]
    return "ip:" + get_real_ip(request)


# Configurable via env var; default is 20 requests/minute per IP.
# Set AXIOM_RATE_LIMIT=100/minute in production if needed.
_RATE_LIMIT = os.environ.get("AXIOM_RATE_LIMIT", "20/minute")

limiter = Limiter(key_func=rate_limit_key, default_limits=[_RATE_LIMIT])


# ---------------------------------------------------------------------------
# Response cache (bounded, TTL-based)
# ---------------------------------------------------------------------------

_CACHE_TTL_SECONDS: int = int(os.environ.get("AXIOM_CACHE_TTL_SECONDS", "300"))
_CACHE_MAX_SIZE: int = int(os.environ.get("AXIOM_CACHE_MAX_SIZE", "256"))

# Module-level cache backend — initialized during lifespan
_response_cache: CacheBackend = MemoryCacheBackend(
    maxsize=_CACHE_MAX_SIZE, ttl_seconds=_CACHE_TTL_SECONDS
)


def _cache_key(
    payload: AxiomRequest,
    api_key: str | None,
    app_config: dict[str, Any],
    pipeline_config: dict[str, Any],
) -> str:
    """
    SHA-256 of the request fields that shape the response body, namespaced by
    a hash of the caller's API key.

    Namespacing prevents cross-tenant cache poisoning: two callers with different
    API keys (and potentially different app_configs) cannot serve each other's
    cached results even when all other fields match.
    """
    # Use a stable hash of the key itself so the raw key never enters the cache.
    # Full 64-hex digest — 16 hex was far inside the birthday bound at any scale.
    key_namespace = hashlib.sha256((api_key or "anonymous").encode()).hexdigest()
    raw = json.dumps(
        {
            "ns": key_namespace,
            "query": payload.user_query,
            "models": payload.models.model_dump(),
            "pipeline": pipeline_config,
            "app": app_config,
            "include_debug": payload.include_debug,
        },
        sort_keys=True,
    )
    # Prefix the final key with the namespace so backends that scan (Redis) or
    # inspect keys always see the tenant boundary, and cross-tenant collisions
    # are impossible even if the hashed body happens to match.
    body_hash = hashlib.sha256(raw.encode()).hexdigest()
    return f"{key_namespace}:{body_hash}"


def _response_to_cache_value(response: AxiomResponse) -> dict[str, Any]:
    """Serialize a response without request-scoped identifiers for safe reuse."""
    data = response.model_dump()
    data.pop("request_id", None)
    return data


def _hydrate_cached_response(request_id: str, cached: dict[str, Any]) -> AxiomResponse:
    """Rebuild a response for the current request from cached template data."""
    return AxiomResponse.model_validate({"request_id": request_id, **copy.deepcopy(cached)})


def _get_cached(key: str, request_id: str) -> AxiomResponse | None:
    cached = _response_cache.get(key)
    if cached is None:
        return None
    return _hydrate_cached_response(request_id, cached)


def _set_cached(key: str, response: AxiomResponse) -> None:
    _response_cache.set(key, _response_to_cache_value(response))


# ---------------------------------------------------------------------------
# App factory & lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Compile the graph once at startup, attach to app state."""
    configure_logging()

    global _response_cache
    redis_url = os.environ.get("AXIOM_REDIS_URL")
    if redis_url:
        try:
            _response_cache = RedisCacheBackend(redis_url=redis_url, ttl_seconds=_CACHE_TTL_SECONDS)
            logger.info("Response cache: Redis backing layer initialized.")
        except ImportError:
            logger.warning(
                "AXIOM_REDIS_URL is set but 'redis' package is not installed. Falling back to MemoryCacheBackend. (Run `uv add redis`)"
            )
        except Exception as exc:
            logger.error(
                "Failed to initialize Redis cache: %s. Falling back to MemoryCacheBackend.", exc
            )

    if _auth_required() and not _api_keys():
        raise RuntimeError("AXIOM_API_KEYS must be configured when AXIOM_ENV is not development.")

    # Wire search backend — Tavily if key present, else MockSearchBackend.
    tavily_key = os.environ.get("TAVILY_API_KEY")
    if tavily_key:
        from axiom_engine.search.tavily import TavilySearchBackend

        set_search_backend(TavilySearchBackend(api_key=tavily_key))
        app.state.search_backend_mode = "tavily"
        logger.info("Search backend: Tavily (live web search enabled).")
    elif _auth_required() and not _allow_mock_search():
        raise RuntimeError(
            "TAVILY_API_KEY must be configured in non-development environments unless "
            "AXIOM_ALLOW_MOCK_SEARCH=true."
        )
    else:
        app.state.search_backend_mode = "mock"
        logger.warning(
            "TAVILY_API_KEY not set — using MockSearchBackend. "
            "Set TAVILY_API_KEY in .env for live web search."
        )

    setup_tracing(app, "axiom-engine", _VERSION)

    app.state.engine = build_axiom_graph()
    logger.info("Axiom Engine graph compiled and ready.")
    yield
    logger.info("Axiom Engine shutting down.")


try:
    _VERSION = importlib.metadata.version("axiom-engine")
except importlib.metadata.PackageNotFoundError:
    _VERSION = "2.3.0-dev"

# Disable interactive API docs in production (AXIOM_DOCS_ENABLED=false).
_docs_enabled = os.environ.get("AXIOM_DOCS_ENABLED", "true").lower() == "true"

app = FastAPI(
    title="Axiom Engine",
    version=_VERSION,
    description="Configuration-driven Agentic RAG with 6-tier verification.",
    lifespan=lifespan,
    docs_url="/docs" if _docs_enabled else None,
    redoc_url="/redoc" if _docs_enabled else None,
)

# CORS — locked down by default; set AXIOM_CORS_ORIGINS to allow specific origins.
# Wildcard ("*") is refused outright: browser clients should always name the
# origins they trust, and a wildcard disables credentialed requests anyway.
_cors_origins = [
    o.strip() for o in os.environ.get("AXIOM_CORS_ORIGINS", "").split(",") if o.strip()
]
if "*" in _cors_origins:
    logger.warning(
        "AXIOM_CORS_ORIGINS contained '*'; dropping and refusing to enable wildcard CORS."
    )
    _cors_origins = [o for o in _cors_origins if o != "*"]
if not _cors_origins:
    logger.info(
        "AXIOM_CORS_ORIGINS is unset; browser clients will be blocked. "
        "Set AXIOM_CORS_ORIGINS=https://example.com to enable.",
    )
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["X-API-Key", "Content-Type"],
)

app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]


# Hard cap on raw request-body size. Defaults to 128 KiB — far larger than any
# legitimate AxiomRequest (user_query is capped at 10k chars + small configs)
# and small enough to stop trivial OOM / slow-parse DoS with oversized bodies.
_MAX_BODY_BYTES = int(os.environ.get("AXIOM_MAX_BODY_BYTES", str(128 * 1024)))


@app.middleware("http")
async def _enforce_body_size(request: Request, call_next):
    if request.method in ("POST", "PUT", "PATCH"):
        declared = request.headers.get("content-length")
        if declared is not None:
            try:
                if int(declared) > _MAX_BODY_BYTES:
                    return JSONResponse(
                        status_code=413,
                        content={"detail": f"Request body exceeds {_MAX_BODY_BYTES} bytes."},
                    )
            except ValueError:
                return JSONResponse(
                    status_code=400,
                    content={"detail": "Malformed Content-Length header."},
                )
        else:
            # No Content-Length (chunked transfer-encoding): buffer the stream
            # with a hard byte cap so oversized bodies cannot bypass the limit.
            chunks: list[bytes] = []
            total = 0
            async for chunk in request.stream():
                total += len(chunk)
                if total > _MAX_BODY_BYTES:
                    return JSONResponse(
                        status_code=413,
                        content={"detail": f"Request body exceeds {_MAX_BODY_BYTES} bytes."},
                    )
                chunks.append(chunk)
            # Cache the buffered body so downstream handlers can still read it.
            request._body = b"".join(chunks)
    return await call_next(request)


# Prometheus instrumentation must happen at module level (before the app starts),
# because it adds middleware which Starlette forbids after startup.
setup_prometheus(app)


# ---------------------------------------------------------------------------
# Global exception handler (architecture §7, Category 1)
# ---------------------------------------------------------------------------


@app.exception_handler(Exception)
async def unhandled_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """
    Catch-all for any unhandled exception that escapes the endpoint.
    Returns a structured AxiomResponse with status="error".
    """
    # Use request_id from context var if available, avoid re-reading request stream
    request_id = request_id_ctx.get() or "unknown"

    logger.exception("Unhandled exception for request %s", request_id)
    error_response = make_error_response(request_id, exc)
    return JSONResponse(
        status_code=500,
        content=error_response.model_dump(),
    )


# ---------------------------------------------------------------------------
# Health checks
# ---------------------------------------------------------------------------


@app.get("/health", summary="Combined liveness + readiness probe (backward compat).")
@limiter.exempt
async def health() -> dict[str, str]:
    """Legacy combined probe — prefer /health/live and /health/ready."""
    return {"status": "ok"}


@app.get("/health/live", summary="Liveness probe — is the process alive?")
@limiter.exempt
async def health_live() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health/ready", summary="Readiness probe — is the engine ready to serve?")
@limiter.exempt
async def health_ready() -> Response:
    """Returns 200 if the graph engine is compiled and ready, 503 otherwise."""
    if not hasattr(app.state, "engine") or app.state.engine is None:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "detail": "Graph engine not yet compiled."},
        )
    if _auth_required() and not _api_keys():
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "detail": "API keys are not configured."},
        )
    if (
        _auth_required()
        and getattr(app.state, "search_backend_mode", "mock") == "mock"
        and not _allow_mock_search()
    ):
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "detail": "Live search backend is not configured."},
        )
    return JSONResponse(content={"status": "ok"})


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@app.post(
    "/v1/synthesize",
    response_model=AxiomResponse,
    summary="Run the Axiom Engine verification pipeline.",
)
async def synthesize(
    payload: AxiomRequest,
    _api_key: str | None = Depends(verify_api_key),
) -> Response:
    """
    Accept an AxiomRequest, execute the LangGraph DAG, and return
    a fully validated AxiomResponse with tier breakdown and confidence score.

    H4: Pipeline errors now return HTTP 500 (not 200).  Successful, partial,
    and unanswerable results still return HTTP 200.
    """
    request_id_ctx.set(payload.request_id)
    effective_app_config = _effective_app_config(payload)
    effective_pipeline_config = _effective_pipeline_config(payload)

    # Cache lookup — keyed by (api_key_hash, payload_hash) to prevent
    # cross-tenant cache sharing (H3).
    key = _cache_key(payload, _api_key, effective_app_config, effective_pipeline_config)
    cached = _get_cached(key, payload.request_id)
    if cached is not None:
        CACHE_HITS.inc()
        logger.info("Cache hit for request %s", payload.request_id)
        REQUESTS_BY_STATUS.labels(status=cached.status).inc()
        return JSONResponse(content=cached.model_dump())
    CACHE_MISSES.inc()

    initial_state = make_initial_state(
        request_id=payload.request_id,
        user_query=payload.user_query,
        app_config=effective_app_config,
        models_config=payload.models.model_dump(),
        pipeline_config=effective_pipeline_config,
    )

    # Initialize the per-request LLM call budget. The mutable dict stored in the
    # ContextVar is shared by all asyncio tasks spawned from this coroutine
    # (gather tasks copy the ContextVar snapshot but reference the same dict).
    reset_llm_budget()

    try:
        engine = app.state.engine
        with PIPELINE_DURATION.time():
            # OTel context and request_id ContextVar are already set on the
            # event-loop coroutine — no thread-boundary propagation needed now
            # that we use ainvoke directly instead of asyncio.to_thread.
            graph_result = await engine.ainvoke(initial_state)
    except LLMBudgetExceededError as exc:
        REQUESTS_BY_STATUS.labels(status="error").inc()
        error_resp = make_error_response(payload.request_id, exc)
        return JSONResponse(status_code=429, content=error_resp.model_dump())
    except Exception as exc:
        REQUESTS_BY_STATUS.labels(status="error").inc()
        error_resp = make_error_response(payload.request_id, exc)
        # H4: unrecoverable pipeline failures are HTTP 500.
        return JSONResponse(status_code=500, content=error_resp.model_dump())

    response = marshal_response(payload.request_id, graph_result, payload.include_debug)

    # Update Prometheus metrics.
    REQUESTS_BY_STATUS.labels(status=response.status).inc()
    for sentence in graph_result.get("final_sentences", []):
        tier = sentence.get("verification", {}).get("tier", 3)
        TIER_ASSIGNMENTS.labels(tier=str(tier)).inc()

    # Cache successful and partial responses only — not errors or unanswerable.
    if response.status in ("success", "partial"):
        _set_cached(key, response)

    return JSONResponse(content=response.model_dump())
