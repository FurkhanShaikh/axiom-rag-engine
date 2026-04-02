"""
Axiom Engine v2.3 — FastAPI Gateway (Module 1)

Responsibilities:
  - Exposes POST /v1/synthesize as the single entry point.
  - Exposes GET /health for liveness/readiness probes.
  - Validates input via the AxiomRequest Pydantic model.
  - Converts AxiomRequest into a GraphState, invokes the compiled
    LangGraph DAG, and marshals the result into AxiomResponse.
  - Computes the ConfidenceSummary (tier breakdown + overall score).
  - Catches all unhandled exceptions and returns a structured error
    response matching the AxiomResponse schema (architecture §7).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Any, Literal

from dotenv import load_dotenv

load_dotenv()  # loads .env from the project root; no-op if file is absent

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from axiom_engine.config.logging import configure_logging
from axiom_engine.graph import build_axiom_graph
from axiom_engine.nodes.retriever import set_search_backend
from axiom_engine.models import (
    AuditEvent,
    AxiomRequest,
    AxiomResponse,
    ConfidenceSummary,
    DebugInfo,
    FinalSentence,
    TierBreakdown,
)
from axiom_engine.state import make_initial_state

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("axiom_engine")


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

# Configurable via env var; default is 20 requests/minute per IP.
# Set AXIOM_RATE_LIMIT=100/minute in production if needed.
_RATE_LIMIT = os.environ.get("AXIOM_RATE_LIMIT", "20/minute")

limiter = Limiter(key_func=get_remote_address, default_limits=[_RATE_LIMIT])


# ---------------------------------------------------------------------------
# Runtime metrics
# ---------------------------------------------------------------------------

_metrics: dict[str, Any] = {
    "requests_total": 0,
    "requests_success": 0,
    "requests_partial": 0,
    "requests_unanswerable": 0,
    "requests_error": 0,
    "requests_cache_hits": 0,
    "tier_counts": defaultdict(int),
    "started_at": 0.0,  # set in lifespan
}


# ---------------------------------------------------------------------------
# Response cache (in-memory, TTL-based)
# ---------------------------------------------------------------------------

_CACHE_TTL_SECONDS: int = 300  # 5 minutes
_response_cache: dict[str, tuple[AxiomResponse, float]] = {}


def _cache_key(payload: AxiomRequest) -> str:
    """SHA-256 of the request fields that affect pipeline output."""
    raw = json.dumps(
        {
            "query": payload.user_query,
            "models": payload.models.model_dump(),
            "pipeline": payload.pipeline_config.model_dump(),
            "app": payload.app_config.model_dump(),
        },
        sort_keys=True,
    )
    return hashlib.sha256(raw.encode()).hexdigest()


def _get_cached(key: str) -> AxiomResponse | None:
    if key in _response_cache:
        response, ts = _response_cache[key]
        if time.monotonic() - ts < _CACHE_TTL_SECONDS:
            return response
        del _response_cache[key]
    return None


def _set_cached(key: str, response: AxiomResponse) -> None:
    _response_cache[key] = (response, time.monotonic())


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------

# Tier weights for the overall confidence score (architecture §4):
#   Tier 1 (Authoritative)    → 1.0
#   Tier 2 (Consensus)        → 0.85
#   Tier 3 (Model Assisted)   → 0.60
#   Tier 4 (Misrepresented)   → 0.20  (should rarely survive to final output)
#   Tier 5 (Hallucinated)     → 0.00  (should never survive to final output)
#   Tier 6 (Conflicted)       → 0.40
_TIER_WEIGHTS: dict[int, float] = {
    1: 1.0,
    2: 0.85,
    3: 0.60,
    4: 0.20,
    5: 0.00,
    6: 0.40,
}


def compute_confidence_summary(
    final_sentences: list[dict[str, Any]],
) -> ConfidenceSummary:
    """
    Compute tier breakdown and weighted overall confidence score from
    the verified final_sentences produced by the graph.
    """
    breakdown = TierBreakdown()
    weighted_sum = 0.0
    total_claims = 0

    for sentence in final_sentences:
        vr = sentence.get("verification", {})
        tier: int = vr.get("tier", 3)

        attr = f"tier_{tier}_claims"
        setattr(breakdown, attr, getattr(breakdown, attr, 0) + 1)

        weighted_sum += _TIER_WEIGHTS.get(tier, 0.0)
        total_claims += 1

    overall = round(weighted_sum / total_claims, 4) if total_claims > 0 else 0.0

    return ConfidenceSummary(
        overall_score=overall,
        tier_breakdown=breakdown,
    )


def determine_status(
    is_answerable: bool,
    final_sentences: list[dict[str, Any]],
) -> Literal["success", "partial", "unanswerable", "error"]:
    """
    Determine the response status string.

    Rules:
      - "unanswerable" if escape hatch fired.
      - "success" if all sentences are Tier 1–3.
      - "partial" if any sentence is Tier 4, 5, or 6.
      - "error" should only come from exception handling (not here).
    """
    if not is_answerable:
        return "unanswerable"

    for s in final_sentences:
        tier = s.get("verification", {}).get("tier", 3)
        if tier in (4, 5, 6):
            return "partial"

    return "success"


# ---------------------------------------------------------------------------
# Graph result → AxiomResponse marshalling
# ---------------------------------------------------------------------------

def marshal_response(
    request_id: str,
    graph_result: dict[str, Any],
    include_debug: bool = False,
) -> AxiomResponse:
    """
    Convert the raw GraphState dict returned by the compiled graph into
    a validated AxiomResponse.
    """
    is_answerable: bool = graph_result.get("is_answerable", False)
    raw_sentences: list[dict] = graph_result.get("final_sentences", [])

    # Validate each sentence through the Pydantic model to ensure
    # the response contract is fully honoured.
    final_sentences: list[FinalSentence] = [
        FinalSentence.model_validate(s) for s in raw_sentences
    ]

    status = determine_status(is_answerable, raw_sentences)
    confidence = compute_confidence_summary(raw_sentences)

    debug: DebugInfo | None = None
    if include_debug:
        raw_audit = graph_result.get("audit_trail", [])
        debug = DebugInfo(
            audit_trail=[AuditEvent.model_validate(e) for e in raw_audit],
            pipeline_stats={
                "chunks_retrieved": len(graph_result.get("indexed_chunks", [])),
                "chunks_ranked": len(graph_result.get("ranked_chunks", [])),
                "loop_count": graph_result.get("loop_count", 0),
                "retrieval_retry_count": graph_result.get("retrieval_retry_count", 0),
            },
        )

    return AxiomResponse(
        request_id=request_id,
        status=status,
        is_answerable=is_answerable,
        confidence_summary=confidence,
        final_response=final_sentences,
        debug=debug,
    )


def make_error_response(
    request_id: str,
    error: Exception,
) -> AxiomResponse:
    """
    Build a structured error response matching the AxiomResponse schema.
    Category 1 errors (architecture §7): unrecoverable system failures.
    """
    error_message = f"{type(error).__name__}: {error}"
    logger.error("Pipeline error for request %s: %s", request_id, error_message)

    return AxiomResponse(
        request_id=request_id,
        status="error",
        is_answerable=False,
        confidence_summary=ConfidenceSummary(
            overall_score=0.0,
            tier_breakdown=TierBreakdown(),
        ),
        final_response=[],
        error_message=error_message,
    )


# ---------------------------------------------------------------------------
# App factory & lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Compile the graph once at startup, attach to app state."""
    configure_logging()

    # Wire search backend — Tavily if key present, else MockSearchBackend.
    tavily_key = os.environ.get("TAVILY_API_KEY")
    if tavily_key:
        from axiom_engine.search.tavily import TavilySearchBackend
        set_search_backend(TavilySearchBackend(api_key=tavily_key))
        logger.info("Search backend: Tavily (live web search enabled).")
    else:
        logger.warning(
            "TAVILY_API_KEY not set — using MockSearchBackend. "
            "Set TAVILY_API_KEY in .env for live web search."
        )

    app.state.engine = build_axiom_graph()
    _metrics["started_at"] = time.monotonic()
    logger.info("Axiom Engine graph compiled and ready.")
    yield


app = FastAPI(
    title="Axiom Engine",
    version="2.3.0",
    description="Configuration-driven Agentic RAG with 6-tier verification.",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]


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
    # Try to extract request_id from the body; fall back to "unknown".
    request_id = "unknown"
    try:
        body = await request.json()
        request_id = body.get("request_id", "unknown")
    except Exception:
        pass

    logger.exception("Unhandled exception for request %s", request_id)
    error_response = make_error_response(request_id, exc)
    return JSONResponse(
        status_code=500,
        content=error_response.model_dump(),
    )


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health", summary="Liveness / readiness probe.")
@limiter.exempt
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metrics", summary="Basic runtime metrics for ops dashboards.")
@limiter.exempt
async def metrics() -> dict[str, Any]:
    """
    Returns request counts, cache hit rate, tier distribution, and uptime.
    Designed for Prometheus scraping or ops dashboards.
    """
    total = _metrics["requests_total"]
    hit_rate = (
        round(_metrics["requests_cache_hits"] / total, 4) if total > 0 else 0.0
    )
    uptime = time.monotonic() - _metrics["started_at"]
    return {
        "uptime_seconds": round(uptime, 1),
        "requests_total": total,
        "requests_success": _metrics["requests_success"],
        "requests_partial": _metrics["requests_partial"],
        "requests_unanswerable": _metrics["requests_unanswerable"],
        "requests_error": _metrics["requests_error"],
        "cache_hits": _metrics["requests_cache_hits"],
        "cache_hit_rate": hit_rate,
        "tier_distribution": dict(_metrics["tier_counts"]),
    }


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@app.post(
    "/v1/synthesize",
    response_model=AxiomResponse,
    summary="Run the Axiom Engine verification pipeline.",
)
async def synthesize(payload: AxiomRequest) -> AxiomResponse:
    """
    Accept an AxiomRequest, execute the LangGraph DAG, and return
    a fully validated AxiomResponse with tier breakdown and confidence score.
    """
    _metrics["requests_total"] += 1

    # Cache lookup — skip for error/unanswerable; only cache clean results.
    key = _cache_key(payload)
    cached = _get_cached(key)
    if cached is not None:
        _metrics["requests_cache_hits"] += 1
        logger.info("Cache hit for request %s", payload.request_id)
        return cached

    initial_state = make_initial_state(
        request_id=payload.request_id,
        user_query=payload.user_query,
        app_config=payload.app_config.model_dump(),
        models_config=payload.models.model_dump(),
        pipeline_config=payload.pipeline_config.model_dump(),
    )

    try:
        engine = app.state.engine
        graph_result = await asyncio.to_thread(engine.invoke, initial_state)
    except Exception as exc:
        _metrics["requests_error"] += 1
        return make_error_response(payload.request_id, exc)

    response = marshal_response(payload.request_id, graph_result, payload.include_debug)

    # Update metrics.
    _metrics[f"requests_{response.status}"] += 1
    for sentence in graph_result.get("final_sentences", []):
        tier = sentence.get("verification", {}).get("tier", 3)
        _metrics["tier_counts"][f"tier_{tier}"] += 1

    # Cache successful and partial responses (not errors or unanswerable).
    if response.status in ("success", "partial"):
        _set_cached(key, response)

    return response
