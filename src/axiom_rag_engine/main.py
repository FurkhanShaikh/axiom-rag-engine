"""
Axiom Engine — FastAPI application factory.

``create_app(settings=None, *, search_backend=None)`` assembles an isolated app:
middleware (CORS, rate limiting, body-size cap, Prometheus), the exception
handler, and the routers. Its lifespan builds the app's runtime services —
engine, response cache, audit store, corpus store, search backend — onto
``app.state.services`` (see ``bootstrap.build_services``).

``app = create_app()`` is the ASGI entry point (``axiom_rag_engine.main:app``).

Where things live:
  - axiom_rag_engine.bootstrap          — settings → AppServices (startup wiring)
  - axiom_rag_engine.services           — the per-app AppServices container
  - axiom_rag_engine.api.routes.*       — endpoints, one module per resource
  - axiom_rag_engine.api.auth           — API key authentication
  - axiom_rag_engine.api.rate_limit     — slowapi limiter + bucketing
  - axiom_rag_engine.api.middleware     — request-body size limit
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request  # noqa: F401 — HTTPException re-exported
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from axiom_rag_engine.api.auth import verify_api_key  # noqa: F401 — re-exported
from axiom_rag_engine.api.middleware import make_body_size_middleware
from axiom_rag_engine.api.rate_limit import (  # noqa: F401 — re-exported
    build_limiter,
    get_real_ip,
    rate_limit_key,
)
from axiom_rag_engine.api.routes import audits, documents, ops, synthesize
from axiom_rag_engine.bootstrap import VERSION, build_services
from axiom_rag_engine.config.logging import configure_logging, request_id_ctx
from axiom_rag_engine.config.observability import setup_prometheus, setup_tracing
from axiom_rag_engine.config.settings import Settings, get_settings
from axiom_rag_engine.marshalling import (  # noqa: F401 — re-exported
    make_error_response,
    marshal_response,
)
from axiom_rag_engine.scoring import (  # noqa: F401 — re-exported
    compute_confidence_summary,
    determine_status,
)

logger = logging.getLogger("axiom_rag_engine")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build the app's runtime services at startup; release them at shutdown.

    Settings passed to ``create_app`` win; otherwise the process settings are
    read *now* (not at import), so environment changes made before startup are
    honoured.
    """
    configure_logging()
    settings: Settings = getattr(app.state, "settings_override", None) or get_settings()
    services = build_services(
        settings, search_backend=getattr(app.state, "search_backend_override", None)
    )
    app.state.services = services
    setup_tracing(app, "axiom-rag-engine", VERSION)
    try:
        yield
    finally:
        logger.info("Axiom Engine shutting down.")
        await services.cache.aclose()


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all for any exception that escapes an endpoint: a structured
    AxiomResponse with status="error" (details go to the server log only)."""
    request_id = request_id_ctx.get() or "unknown"
    logger.exception("Unhandled exception for request %s", request_id)
    error_response = make_error_response(request_id, exc)
    return JSONResponse(status_code=500, content=error_response.model_dump())


def _cors_origins(settings: Settings) -> list[str]:
    """Allowed origins; a wildcard is refused outright (browser clients must
    name the origins they trust, and a wildcard disables credentials anyway)."""
    origins = list(settings.cors_origins)
    if "*" in origins:
        logger.warning(
            "AXIOM_CORS_ORIGINS contained '*'; dropping and refusing to enable wildcard CORS."
        )
        origins = [o for o in origins if o != "*"]
    if not origins:
        logger.info(
            "AXIOM_CORS_ORIGINS is unset; browser clients will be blocked. "
            "Set AXIOM_CORS_ORIGINS=https://example.com to enable."
        )
    return origins


def create_app(
    settings: Settings | None = None,
    *,
    search_backend: Any = None,
) -> FastAPI:
    """Build an isolated Axiom Engine app.

    Args:
        settings: Configuration for this app. When omitted, the process settings
            (``get_settings()``) configure the middleware now and are re-read by
            the lifespan at startup.
        search_backend: Overrides the backend derived from settings (tests,
            embedding the engine in another service). Anything with
            ``search(query) -> list[dict]``.

    Middleware must be added before startup (Starlette forbids it afterwards),
    so body limits, rate limits, CORS, and docs are fixed when the app is built.
    Prometheus instrumentation is process-wide: only the first app built in a
    process exposes ``/metrics``.
    """
    config = settings or get_settings()

    app = FastAPI(
        title="Axiom Engine",
        version=VERSION,
        description="Configuration-driven Agentic RAG with 6-tier citation verification.",
        lifespan=lifespan,
        docs_url="/docs" if config.docs_enabled else None,
        redoc_url="/redoc" if config.docs_enabled else None,
    )
    app.state.settings_override = settings
    app.state.search_backend_override = search_backend

    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins(config),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["X-API-Key", "Content-Type"],
    )

    limiter = build_limiter(config)
    app.state.limiter = limiter
    app.add_middleware(SlowAPIMiddleware)
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]
    for endpoint in ops.RATE_LIMIT_EXEMPT:
        limiter.exempt(endpoint)

    app.middleware("http")(
        make_body_size_middleware(config.max_body_bytes, config.max_document_bytes)
    )
    setup_prometheus(app)
    app.add_exception_handler(Exception, unhandled_exception_handler)

    app.include_router(ops.router)
    app.include_router(synthesize.build_router(limiter, config.stream_rate_limit))
    app.include_router(audits.router)
    app.include_router(documents.router)
    return app


app = create_app()
