# ── Build stage ───────────────────────────────────────────────────────────────
# Pin to a specific patch release for reproducible builds.
# Update both FROM tags together when upgrading Python or uv.
# Use the digest form (python:3.11.12-slim@sha256:<digest>) in production CI
# for fully immutable image references.
FROM python:3.11.12-slim AS builder

# Install uv — pinned version for reproducible builds.
COPY --from=ghcr.io/astral-sh/uv:0.5.0 /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency manifests first so Docker can cache this layer.
COPY pyproject.toml uv.lock ./

# Install runtime dependencies into an isolated venv (no dev extras).
# --frozen: honour uv.lock exactly; --no-install-project: deps only, not src yet.
RUN uv sync --frozen --no-dev --no-install-project

# Copy source and install the project itself.
COPY src/ ./src/
RUN uv sync --frozen --no-dev


# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.11.12-slim

# Create a non-root user for security.
RUN useradd --create-home --shell /bin/bash axiom \
    && apt-get update -qq && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the populated venv and source from the build stage.
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
COPY --from=builder /app/pyproject.toml /app/pyproject.toml

# Activate the venv by prepending it to PATH.
ENV PATH="/app/.venv/bin:$PATH"

USER axiom

EXPOSE 8000

# Readiness health check — allows orchestrators to postpone traffic until the
# graph engine is compiled and ready.
HEALTHCHECK --interval=10s --timeout=3s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health/ready || exit 1

# Uvicorn: single worker per container; scale horizontally via compose/k8s.
CMD ["uvicorn", "axiom_engine.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
