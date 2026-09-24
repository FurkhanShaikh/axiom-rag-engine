"""
Axiom Engine — Observability setup (Prometheus metrics + OpenTelemetry tracing).

Call setup_prometheus() and instrument_app() while building the app, and
setup_tracing() once at startup in the FastAPI lifespan.
"""

from __future__ import annotations

import functools
import hmac
import logging
import os
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request
from fastapi.responses import Response
from opentelemetry import trace
from opentelemetry.trace import Tracer
from prometheus_client import CONTENT_TYPE_LATEST, REGISTRY, Counter, Histogram, generate_latest

logger = logging.getLogger("axiom_rag_engine.observability")

# ---------------------------------------------------------------------------
# Prometheus — custom domain metrics
# ---------------------------------------------------------------------------

PIPELINE_DURATION = Histogram(
    "axiom_pipeline_duration_seconds",
    "End-to-end pipeline duration per request",
    buckets=[0.5, 1, 2, 5, 10, 30, 60, 120],
)

LLM_CALL_DURATION = Histogram(
    "axiom_llm_call_duration_seconds",
    "Wall-clock duration of a single LLM completion call",
    ["node", "model"],
    buckets=[0.5, 1, 2, 5, 10, 30, 60],
)

CACHE_HITS = Counter("axiom_cache_hits_total", "Response cache hits")
CACHE_MISSES = Counter("axiom_cache_misses_total", "Response cache misses")
REQUESTS_BY_STATUS = Counter(
    "axiom_requests_by_status_total",
    "Request outcomes by status",
    ["status"],
)
TIER_ASSIGNMENTS = Counter(
    "axiom_tier_assignments_total",
    "Verification tiers assigned to cited sentences (claims), matching the response's "
    "tier_breakdown; label separates e.g. tier 3 model_assisted from unverified",
    ["tier", "label"],
)
SEMANTIC_DEGRADATIONS = Counter(
    "axiom_semantic_degradations_total",
    "Number of citations that fell back to deterministic Tier 3 due to LLM failure",
)
LOOP_EXHAUSTED_TIER5 = Counter(
    "axiom_loop_exhausted_tier5_total",
    "Tier 5 sentences that survived all rewrite and retrieval retries and reached the final response",
)
LLM_RETRIES = Counter(
    "axiom_llm_retries_total",
    "LLM calls retried after a transient provider failure (rate limit, timeout, 5xx)",
    ["node", "model"],
)
NODE_DURATION = Histogram(
    "axiom_node_duration_seconds",
    "Wall-clock duration of each graph node execution",
    ["node"],
    buckets=[0.1, 0.5, 1, 2, 5, 10, 30, 60],
)

# ``kind`` is one of {"prompt", "completion"}. Model label is bounded by
# ``safe_model_label`` so customers can't spike cardinality via per-request
# model overrides.
LLM_TOKENS_TOTAL = Counter(
    "axiom_llm_tokens_total",
    "Cumulative LLM tokens consumed, labelled by model and kind.",
    ["model", "kind"],
)
LLM_COST_USD_TOTAL = Counter(
    "axiom_llm_cost_usd_total",
    "Cumulative LLM cost in USD (best-effort via litellm.completion_cost).",
    ["model"],
)

_prometheus_initialized = False


def setup_prometheus(
    app: FastAPI, metrics_token: str | None = None
) -> Callable[[Request], Awaitable[Response]]:
    """Serve ``/metrics`` on ``app`` and return its endpoint.

    HTTP request metrics are instrumented once per process (their collectors
    are process-wide). The endpoint is added to every app; with
    ``metrics_token`` it requires ``Authorization: Bearer <token>``, since the
    metrics expose model usage and spend.
    """
    global _prometheus_initialized
    if not _prometheus_initialized:
        from prometheus_fastapi_instrumentator import Instrumentator

        Instrumentator().instrument(app)
        _prometheus_initialized = True

    expected = f"Bearer {metrics_token}".encode() if metrics_token else None

    async def metrics(request: Request) -> Response:
        if expected is not None:
            given = request.headers.get("authorization", "").encode()
            if not hmac.compare_digest(given, expected):
                return Response(status_code=401, headers={"WWW-Authenticate": "Bearer"})
        return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)

    app.add_api_route("/metrics", metrics, methods=["GET"], include_in_schema=False)
    logger.info(
        "Prometheus metrics at /metrics (%s).",
        "bearer token required" if expected else "unauthenticated",
    )
    return metrics


# ---------------------------------------------------------------------------
# OpenTelemetry — distributed tracing
# ---------------------------------------------------------------------------

_tracer: Tracer = trace.get_tracer("axiom-rag-engine")
_tracer_provider_set = False


def tracing_enabled() -> bool:
    """True when OTEL_EXPORTER_OTLP_ENDPOINT is set (tracing is otherwise a no-op)."""
    return bool(os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"))


def instrument_app(app: FastAPI) -> None:
    """Add OpenTelemetry HTTP server spans to ``app`` when tracing is enabled.

    Must run while the app is being built: Starlette assembles its middleware
    stack on the first ASGI call (the lifespan startup), so instrumenting from
    the lifespan silently produced no request spans. The middleware resolves the
    global tracer provider lazily, so :func:`setup_tracing` may still install the
    provider later, at startup.
    """
    if not tracing_enabled():
        return
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

    FastAPIInstrumentor.instrument_app(app)


def setup_tracing(service_name: str, version: str) -> None:
    """
    Install the OTLP tracer provider if OTEL_EXPORTER_OTLP_ENDPOINT is set.

    Process-wide and idempotent (OpenTelemetry allows one global provider). When
    the endpoint is not configured, the tracer remains a no-op (zero overhead).
    """
    global _tracer, _tracer_provider_set

    if not tracing_enabled():
        logger.info("OTEL_EXPORTER_OTLP_ENDPOINT not set — tracing disabled (no-op).")
        return
    if _tracer_provider_set:
        return

    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    resource = Resource.create({"service.name": service_name, "service.version": version})
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)
    _tracer_provider_set = True

    _tracer = trace.get_tracer(service_name, version)
    logger.info("OpenTelemetry tracing enabled → %s", os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"))


def tag_current_span(request_id: str) -> None:
    """Attach the Axiom request id to the active span (the HTTP server span)."""
    trace.get_current_span().set_attribute("axiom.request_id", request_id)


def get_tracer() -> Tracer:
    """Return the configured tracer (no-op when tracing is disabled)."""
    return _tracer


# ---------------------------------------------------------------------------
# Prometheus label safety
# ---------------------------------------------------------------------------
# The ``model`` label on LLM_CALL_DURATION is user-influenced (callers can
# override the model per-request).  Unbounded label cardinality is a Prometheus
# footgun that can crash the metrics store.  Anything not in this set is
# collapsed to "other".

_LLM_LABEL_OTHER = "other"


@functools.cache
def _allowed_llm_label_models() -> frozenset[str]:
    """Snapshot allowlist from Settings. Cached so startup cost is paid once."""
    from axiom_rag_engine.config.settings import get_settings

    return frozenset(get_settings().allowed_metric_models)


def safe_model_label(model: str) -> str:
    """
    Return a Prometheus-safe model label.

    Exact matches against the allowlist pass through unchanged.  Ollama models
    (``ollama/<name>``) are collapsed to the prefix ``"ollama/…"`` to keep
    cardinality bounded while remaining identifiable.  Everything else becomes
    ``"other"``.
    """
    if model in _allowed_llm_label_models():
        return model
    if model.startswith("ollama/"):
        return "ollama/…"
    return _LLM_LABEL_OTHER
