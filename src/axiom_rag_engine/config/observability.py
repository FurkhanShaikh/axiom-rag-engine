"""
Axiom Engine — Observability setup (Prometheus metrics + OpenTelemetry tracing).

Call setup_prometheus() and setup_tracing() once at startup in the FastAPI lifespan.
"""

from __future__ import annotations

import functools
import logging
import os

from fastapi import FastAPI
from opentelemetry import trace
from opentelemetry.trace import Tracer
from prometheus_client import Counter, Histogram

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
    "Verification tier assignment count",
    ["tier"],
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


def setup_prometheus(app: FastAPI) -> None:
    """Instrument the FastAPI app with Prometheus metrics (idempotent)."""
    global _prometheus_initialized
    if _prometheus_initialized:
        return

    from prometheus_fastapi_instrumentator import Instrumentator

    Instrumentator().instrument(app).expose(app, include_in_schema=False)
    _prometheus_initialized = True
    logger.info("Prometheus metrics enabled at /metrics.")


# ---------------------------------------------------------------------------
# OpenTelemetry — distributed tracing
# ---------------------------------------------------------------------------

_tracer: Tracer = trace.get_tracer("axiom-rag-engine")


def setup_tracing(app: FastAPI, service_name: str, version: str) -> None:
    """
    Configure OpenTelemetry tracing if OTEL_EXPORTER_OTLP_ENDPOINT is set.

    When the endpoint is not configured, the tracer remains a no-op (zero overhead).
    """
    global _tracer

    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not endpoint:
        logger.info("OTEL_EXPORTER_OTLP_ENDPOINT not set — tracing disabled (no-op).")
        return

    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    resource = Resource.create({"service.name": service_name, "service.version": version})
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)

    FastAPIInstrumentor.instrument_app(app)

    _tracer = trace.get_tracer(service_name, version)
    logger.info("OpenTelemetry tracing enabled → %s", endpoint)


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
