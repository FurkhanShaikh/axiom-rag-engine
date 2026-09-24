"""
Axiom Engine — Structured logging configuration.

Call configure_logging() once at application startup (e.g. in the FastAPI
lifespan) to set up consistent, machine-readable log output.

Supports two output formats (controlled via LOG_FORMAT env var):
  - "text"  : Human-readable, coloured output for local development (default).
  - "json"  : Machine-readable JSON lines for production log aggregation.
"""

from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar
from datetime import UTC, datetime

from opentelemetry import trace

from axiom_rag_engine.config.settings import get_settings

# ---------------------------------------------------------------------------
# Context variable for request-scoped correlation
# ---------------------------------------------------------------------------

request_id_ctx: ContextVar[str | None] = ContextVar("request_id", default=None)


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------


class _TextFormatter(logging.Formatter):
    """Human-readable formatter for local development."""

    FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

    def __init__(self) -> None:
        super().__init__(fmt=self.FORMAT, datefmt="%Y-%m-%dT%H:%M:%S")

    def format(self, record: logging.LogRecord) -> str:
        rid = request_id_ctx.get()
        if rid:
            # Prefix a copy: the record is shared by every handler, so mutating
            # it would stack the prefix once per handler (or per format call).
            record = logging.makeLogRecord(record.__dict__)
            record.msg = f"[{rid}] {record.msg}"
        return super().format(record)


class _JSONFormatter(logging.Formatter):
    """Structured JSON formatter for production log aggregation."""

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, object] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        rid = request_id_ctx.get()
        if rid:
            entry["request_id"] = rid
        # Inject OpenTelemetry trace context for log-to-trace correlation.
        span = trace.get_current_span()
        ctx = span.get_span_context()
        if ctx and ctx.trace_id:
            entry["trace_id"] = format(ctx.trace_id, "032x")
            entry["span_id"] = format(ctx.span_id, "016x")
        # Surface the structured audit payload when present — set via
        # ``logger.info(..., extra={"axiom_audit": event})`` from main.py when
        # AXIOM_LOG_AUDIT_EVENTS=true.
        audit_extra = getattr(record, "axiom_audit", None)
        if audit_extra is not None:
            entry["axiom_audit"] = audit_extra
        if record.exc_info and record.exc_info[1]:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def configure_logging(level: int = logging.INFO) -> None:
    """
    Configure the root 'axiom_rag_engine' logger.

    Set LOG_FORMAT=json (env var) for structured JSON output, otherwise
    defaults to human-readable text.
    """
    root_logger = logging.getLogger("axiom_rag_engine")
    if root_logger.handlers:
        return  # Already configured (avoid duplicate handlers on reload)

    root_logger.setLevel(level)

    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(level)

    fmt = get_settings().log_format.lower()
    if fmt == "json":
        handler.setFormatter(_JSONFormatter())
    else:
        handler.setFormatter(_TextFormatter())

    root_logger.addHandler(handler)
