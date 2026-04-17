# Axiom Engine

**Citation-verified RAG with 6-tier confidence scoring.**

Axiom Engine is a retrieval-augmented generation (RAG) service that
verifies every cited claim before presenting answers. Each claim is assigned
a confidence tier (1-6) based on deterministic + semantic verification.

## Install

```bash
pip install axiom-rag-engine
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add axiom-rag-engine
```

## Quick start

### From PyPI

```bash
# Set required env vars (or create a .env file)
export AXIOM_ENV=development
export TAVILY_API_KEY=your_key   # or use AXIOM_ALLOW_MOCK_SEARCH=true

# Start the server
axiom-rag-engine serve

# In another terminal — send a test query
axiom-rag-engine probe "What are solid-state batteries?"

# Check resolved configuration (secrets redacted)
axiom-rag-engine check-config
```

### From source

```bash
git clone https://github.com/FurkhanShaikh/axiom-rag-engine.git
cd axiom-rag-engine
python tasks.py install          # scaffold .env + install deps via uv
# Edit .env — fill in TAVILY_API_KEY for live web search
python tasks.py run              # start FastAPI server at http://localhost:8000
python tasks.py probe "your question"
```

## Configuration

All settings are controlled via environment variables (or a `.env` file).
No code changes required. Run `axiom-rag-engine check-config` to see the full
resolved configuration.

| Variable | Default | Description |
|---|---|---|
| `AXIOM_ENV` | `production` | Runtime environment. Set to `development` to disable auth. |
| `AXIOM_API_KEYS` | _(empty)_ | Comma-separated API keys. Required when env != development. |
| `TAVILY_API_KEY` | _(empty)_ | Tavily search API key for live web retrieval. |
| `AXIOM_DEFAULT_SYNTHESIZER_MODEL` | `claude-sonnet-4-5` | LiteLLM model ID for synthesis. |
| `AXIOM_DEFAULT_VERIFIER_MODEL` | `gpt-4o-mini` | LiteLLM model ID for semantic verification. |
| `AXIOM_RATE_LIMIT` | `20/minute` | Rate limit per API key or IP. |
| `AXIOM_CACHE_TTL_SECONDS` | `300` | Response cache TTL. |
| `AXIOM_REDIS_URL` | _(empty)_ | Optional Redis URL for distributed cache. |
| `AXIOM_CORS_ORIGINS` | _(empty)_ | Comma-separated allowed CORS origins. |
| `AXIOM_DOCS_ENABLED` | `true` | Set `false` to disable /docs and /redoc. |
| `AXIOM_SEMANTIC_VERIFICATION_ENABLED` | `true` | Enable/disable Stage 2 semantic verification. |
| `AXIOM_AUDIT_RETENTION` | `0` | Retain the last N audit trails in memory for `/v1/audits/{id}`. |
| `AXIOM_LOG_AUDIT_EVENTS` | `false` | Emit each audit event as a structured log line. |
| `LOG_FORMAT` | `text` | `json` for structured log output. |

See [.env.example](.env.example) for the full list with comments.

## Architecture

```
retriever -> scorer -> ranker -> synthesizer -> verifier -+
   ^                    ^                                 |
   |                    +-- (rewrite loop) <--------------+  (Tier 4/5 & loop < max)
   +-- (re-retrieve) <-----------------------------------+  (loop exhausted & retries left)
```

| Module | Responsibility |
|---|---|
| **Retriever** | Web search via Tavily, dedup, HTML strip, paragraph chunking |
| **Scorer** | Domain authority + content quality scoring (deterministic) |
| **Ranker** | BM25-based relevance ranking with quality blend |
| **Synthesizer** | LLM-powered answer generation with strict citation format |
| **Verifier** | Two-stage verification: mechanical (exact match) + semantic (LLM) |

## Verification tiers

| Tier | Label | Meaning |
|---|---|---|
| 1 | Authoritative | Verified against official/primary source |
| 2 | Multi-Source | Verified against multiple independent domains |
| 3 | Model Assisted | Mechanically verified; semantic relied on model knowledge |
| 4 | Misrepresented | Quote exists but claim distorts context |
| 5 | Hallucinated | Quote not found in source chunk |
| 6 | Conflicted | Reserved for future contradiction detection |

## CLI reference

```bash
axiom-rag-engine serve [--host 0.0.0.0] [--port 8000] [--reload]
axiom-rag-engine probe "question" [--url URL] [--model MODEL] [--debug]
axiom-rag-engine check-config [--format text|json]
axiom-rag-engine audit <request_id> [--url URL] [--api-key KEY] [--json]
```

## Operations

### Runtime status

`GET /v1/status` returns a JSON snapshot of version, uptime, active policy, and
configured backends. No secrets are exposed — API keys and Redis URLs are
reported as booleans.

```bash
curl http://localhost:8000/v1/status | jq .
```

Combine with `axiom-rag-engine check-config` to see every `AXIOM_*` value and
**where it came from** (env var, `.env`, or built-in default).

### Audit trails

Every request emits a full audit trail from each graph node. Two ways to view
them:

1. **Retained in-process** — set `AXIOM_AUDIT_RETENTION=200` to keep the last
   N trails in memory. Fetch any one by ID:

   ```bash
   # HTTP
   curl -H "X-API-Key: $KEY" http://localhost:8000/v1/audits/<request_id>

   # CLI (human-readable event log)
   axiom-rag-engine audit <request_id>
   ```

2. **Streamed to logs** — set `AXIOM_LOG_AUDIT_EVENTS=true` together with
   `LOG_FORMAT=json` to emit one structured line per audit event, ready to
   forward to a log aggregator.

The retention store is process-local and bounded — for durable history, use
the log stream into your existing aggregator.

### Metrics dashboard

`GET /metrics` exposes Prometheus metrics including `axiom_pipeline_duration_seconds`,
per-node and per-model LLM latency histograms, **per-model token + USD cost
counters** (`axiom_llm_tokens_total`, `axiom_llm_cost_usd_total`), tier
assignment rates, cache hit ratio, and verification-degradation counters.

A ready-to-import Grafana dashboard lives at
[`deploy/grafana/axiom-engine.json`](deploy/grafana/axiom-engine.json). The
quickest way to see it wired up end-to-end is the docker-compose stack below.

### Token + cost accounting

Every response includes a `usage` block with call count, prompt/completion
tokens, best-effort USD cost, and a per-model breakdown:

```json
"usage": {
  "calls": 2,
  "prompt_tokens": 2661,
  "completion_tokens": 169,
  "total_tokens": 2830,
  "cost_usd": 0.00042,
  "by_model": {
    "claude-sonnet-4-5": {"calls": 1, "prompt_tokens": 2500, "completion_tokens": 150, "cost_usd": 0.00040}
  }
}
```

Cost is computed via `litellm.completion_cost`. Local models (Ollama) and
untracked providers report `0.0`. Cache hits return `usage: null` because
the current request consumed no tokens.

### Quick health walk

```bash
# Liveness (process alive)
curl -fsS http://localhost:8000/health/live

# Readiness (engine compiled, keys + backend configured)
curl -fsS http://localhost:8000/health/ready

# Full operator snapshot
curl -fsS http://localhost:8000/v1/status | jq .
```

## Development

```bash
python tasks.py test             # unit tests (>=70% coverage required)
python tasks.py lint             # ruff + mypy
python tasks.py format           # auto-format
python tasks.py clean            # remove caches + venv
```

## API

- `POST /v1/synthesize` — Run the verification pipeline
- `GET /health` — Liveness probe
- `GET /health/ready` — Readiness probe
- `GET /metrics` — Prometheus metrics

See the interactive docs at `http://localhost:8000/docs` when the server is running.

## Docker

The bundled `docker-compose.yml` brings up the full stack — Axiom, Ollama,
Redis (cache backing store), Prometheus (scraping `/metrics`), and Grafana
with the dashboard pre-provisioned:

```bash
docker compose up --build
```

Endpoints once the stack is healthy:

- Axiom API  — http://localhost:8000
- Prometheus — http://localhost:9090
- Grafana    — http://localhost:3000  (login `admin` / `admin`)
  → *Dashboards → Axiom → Axiom Engine*

To run without the observability sidecars, comment out the `redis`,
`prometheus`, and `grafana` services.

## License

[MIT](LICENSE)
