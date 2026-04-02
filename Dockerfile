# ── Build stage ───────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

# Install uv — single binary, no pip needed.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

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
FROM python:3.11-slim

# Create a non-root user for security.
RUN useradd --create-home --shell /bin/bash axiom

WORKDIR /app

# Copy the populated venv and source from the build stage.
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
COPY --from=builder /app/pyproject.toml /app/pyproject.toml

# Activate the venv by prepending it to PATH.
ENV PATH="/app/.venv/bin:$PATH"

USER axiom

EXPOSE 8000

# Uvicorn: single worker per container; scale horizontally via compose/k8s.
CMD ["uvicorn", "axiom_engine.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
