# Changelog

All notable changes to Axiom Engine are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `GET /v1/status` — operator snapshot (version, uptime, policy, backends, limits). No secrets exposed.
- `GET /v1/audits/{request_id}` — retrieve the full audit trail for a recent request. Controlled by `AXIOM_AUDIT_RETENTION` (in-memory ring buffer, 0 = disabled).
- `axiom-rag-engine audit <request_id>` CLI subcommand — human-readable event log, with `--json` for raw output.
- `AXIOM_LOG_AUDIT_EVENTS` — when true, every audit event is emitted as a structured log line (pairs with `LOG_FORMAT=json`).
- `source_weight` / `chunk_weight` on `AppConfig` — formal request-body fields for the ranker weight blend.
- Pre-built Grafana dashboard at `deploy/grafana/axiom-engine.json` for the exposed Prometheus metrics.
- **Per-request token + cost accounting**: every response carries a `usage` block (calls, prompt/completion/total tokens, best-effort USD cost via `litellm.completion_cost`, per-model breakdown). Cache hits report `usage: null`.
- Prometheus counters `axiom_llm_tokens_total{model, kind}` and `axiom_llm_cost_usd_total{model}`. Model labels are bounded by the existing `safe_model_label` allowlist.
- Synthetic terminal `usage_summary` audit event so the CLI + `/v1/audits/{id}` show per-request cost without joining separate streams.
- **docker-compose stack** now bundles Redis (cache), Prometheus (scraper), and Grafana (auto-provisioned with the Axiom dashboard + Prometheus datasource) alongside Axiom and Ollama.

### Changed
- `check-config` now prints values grouped by section along with the effective source (`env` / `.env` / `default`) and the canonical env var name.
- All `AXIOM_*` environment reads now flow through `Settings`. Stragglers in `utils/llm.py`, `nodes/synthesizer.py`, `config/observability.py`, and `models.py` were migrated; `max_llm_calls_per_request`, `max_tokens_per_request`, `max_concurrent_llm`, `min_usable_ranking_score`, `allowed_metric_models`, and `ollama_api_base` are now first-class `Settings` fields.
- `.env.example` regenerated to reflect every new field with comments.

### Fixed
- Duplicate entry in the package-version discovery list in `main.py`.

## [0.1.0b1] - 2026-04-15

First public beta release.

### Added
- **RAG pipeline** — LangGraph DAG with retriever, scorer, ranker, synthesizer, and two-stage verifier (mechanical + semantic).
- **6-tier confidence scoring** — every cited claim is assigned a verification tier (1-Authoritative through 6-Conflicted).
- **Central configuration** (`config/settings.py`) — all `AXIOM_*` env vars in one typed `Settings` class backed by `pydantic-settings`. No code changes needed to configure.
- **CLI entry point** (`axiom-rag-engine`) with `serve`, `probe`, and `check-config` subcommands.
- **FastAPI HTTP API** — `POST /v1/synthesize`, health probes, Prometheus metrics.
- **Search backends** — Tavily live web search with automatic fallback to mock backend.
- **LLM flexibility** — any LiteLLM-supported model, including local Ollama.
- **Response cache** — in-memory TTLCache with optional Redis backing layer.
- **Security hardening** — fail-closed auth, CORS lockdown, SSRF defense, rate limiting, body-size cap.
- **Observability** — Prometheus metrics, OpenTelemetry tracing, structured JSON logging.
- **CI pipeline** — GitHub Actions for lint, typecheck, test (3.11/3.12/3.13), security audit, Docker build.
- **Publish workflow** — tag-triggered release to TestPyPI (rc tags) and PyPI via Trusted Publishing.
- `tasks.py` developer task runner (install, run, test, lint, format, probe, clean).

[0.1.0b1]: https://github.com/FurkhanShaikh/axiom-rag-engine/releases/tag/v0.1.0b1
