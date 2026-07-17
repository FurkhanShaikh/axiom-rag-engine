# Axiom Engine

**Citation-verified RAG with 5-tier confidence scoring.**

Axiom Engine is a retrieval-augmented generation (RAG) service that
verifies every cited claim before presenting answers. Every claim carries a
verbatim source quote that is checked against the retrieved page
deterministically, then judged for faithfulness by a model, and assigned a
confidence tier.

A sixth tier (Conflicted) is defined in the response schema but is **not
implemented** — the verifier never assigns it today. See
[Verification tiers](#verification-tiers).

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
| `AXIOM_DEFAULT_SYNTHESIZER_MODEL` | `claude-opus-4-8` | LiteLLM model ID for synthesis. |
| `AXIOM_DEFAULT_VERIFIER_MODEL` | `gpt-4o-mini` | LiteLLM model ID for semantic verification. |
| `AXIOM_RATE_LIMIT` | `20/minute` | Rate limit per API key or IP. |
| `AXIOM_CACHE_TTL_SECONDS` | `300` | Response cache TTL. |
| `AXIOM_REDIS_URL` | _(empty)_ | Optional Redis URL for distributed cache. |
| `AXIOM_CORS_ORIGINS` | _(empty)_ | Comma-separated allowed CORS origins. |
| `AXIOM_DOCS_ENABLED` | `true` | Set `false` to disable /docs and /redoc. |
| `AXIOM_SEMANTIC_VERIFICATION_ENABLED` | `true` | Enable/disable Stage 2 semantic verification. |
| `AXIOM_FETCH_FULL_PAGES` | `true` | Verify citations against full page text rather than search snippets. See [Verification sources](#verification-sources). |
| `AXIOM_MAX_RAW_CONTENT_CHARS` | `200000` | Per-document cap on full page text. Oversized pages are truncated, not dropped. |
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

Each tier states exactly what the engine checked — no more.

| Tier | Label | What it proves | What it does *not* prove |
|---|---|---|---|
| 1 | Authoritative | Quote is verbatim in the source, faithfully represents it, and at least one cited domain is on the configured primary-source list | That the primary-source list is complete — a genuinely authoritative domain that isn't on it lands at Tier 3 |
| 2 | Multi-Domain | Quote is verbatim, faithfully represents the source, and the sentence cites ≥2 distinct domains | **That those domains agree.** This is coverage, not corroboration — the sources are not compared to each other |
| 3 | Model Assisted | Quote is verbatim and faithfully represents the source | Any authority or cross-source claim |
| 4 | Misrepresented | Quote is verbatim, but the claim distorts what the source says | — |
| 5 | Hallucinated | Quote was **not found** in the cited chunk | — |
| 6 | Conflicted | *Not implemented.* Reserved for cross-source contradiction detection; the verifier never assigns this tier today | — |

Tier 1 and Tier 2 are deterministic judgements about **sources**, computed
from domain metadata — never inferred by a model. Tiers 3–5 describe the
**claim-to-source** relationship: Tier 5 is a deterministic substring check
(no LLM involved), Tier 4 is the model's faithfulness verdict.

Tier 2 is named "Multi-Domain" rather than "Multi-Source" deliberately: it
only establishes that a sentence draws on more than one domain. Detecting
whether those sources actually *agree* requires an entailment check across
citations, which is not yet implemented.

## Verification sources

A citation is only as verified as the text it was checked against.

Search APIs return a short, query-biased **snippet** per result alongside the
full page. A snippet is a summary — a quote the model copied verbatim from the
real page can be missing from it. Verifying against snippets therefore produces
Tier 5 (Hallucinated) verdicts for claims the source actually supports, which is
the worst possible failure for this product: it discredits correct answers.

Axiom requests full page text by default (`AXIOM_FETCH_FULL_PAGES=true`) so
"verified against the source" means the source. Pages that yield no extractable
text — paywalls, JS-only rendering, robots-blocked — fall back to the snippet
rather than being dropped, and every such document is recorded in the audit
trail as `retriever_snippet_only_source`, with a `snippet_only_docs` count on
`retriever_complete`. Each indexed chunk also carries `content_mode`
(`raw` | `snippet`), so a Tier 5 traced back to a snippet-only source can be
told apart from a genuine hallucination.

Fetching pages costs latency and payload size. Set `AXIOM_FETCH_FULL_PAGES=false`
to opt out — the startup log will warn that citations are being verified against
snippets.

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
    "claude-opus-4-8": {"calls": 1, "prompt_tokens": 2500, "completion_tokens": 150, "cost_usd": 0.00040}
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
python tasks.py security         # dependency vulnerability scan
python tasks.py ci               # CI-style pre-push check
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
