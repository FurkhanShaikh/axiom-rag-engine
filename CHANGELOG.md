# Changelog

All notable changes to Axiom Engine are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added — OpenRouter provider
- **`OPENROUTER_API_KEY` is a first-class provider key.** With no Anthropic or OpenAI key, startup auto-selects `AXIOM_OPENROUTER_SYNTHESIZER_MODEL` (default `openrouter/openai/gpt-4o`) and `AXIOM_OPENROUTER_VERIFIER_MODEL` (default `openrouter/openai/gpt-4o-mini`, the same model as the OpenAI-key verifier). The key counts as an available provider for production's fail-closed check, is pushed to LiteLLM from `.env`, and is redacted by `check-config`.

### Changed — semantic verifier baseline (EVAL-1)
- **`--record` writes the enforced verifier baseline in one step.** `semantic_verifier_eval.py --model <verifier> --record` refuses samples under 200 (now the default `--limit`) and sets floors at the run's 95% Wilson lower bounds, so the first keyed run activates the gate. A baseline recorded on a different model is reported, not enforced (`openrouter/openai/gpt-4o-mini` counts as `gpt-4o-mini`).
- **The nightly job uses whichever key is configured** (`OPENAI_API_KEY`, `OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY`, choosing the matching verifier), and fails when no key is set once the baseline is enforced.

### Fixed — explicit model configuration
- **Setting a model to its default value is respected.** Startup decided whether the operator chose a model by comparing it with the built-in default, so `AXIOM_DEFAULT_VERIFIER_MODEL=gpt-4o-mini` with only an Anthropic key was silently switched to Haiku. Explicit configuration is now read from the settings sources (`model_fields_set`).

### Changed — synthesis
- **Tier 2 is reachable.** Chunk headers now show each chunk's source domain (sanitized to hostname characters), and the synthesizer is asked to cite every chunk from a different source that states a fact (up to 3, each with its own verbatim quote). On 100 ASQA questions (local qwen3.5:9b) sentences citing ≥ 2 domains went from 0.6% to 23.5% and Tier 2 from 0% to 20%, with Tier 2 sentences judged as well supported as Tier 3 (0.98 vs 0.95). See BENCHMARKS.md → Tier calibration.

### Changed — retrieval
- **First-pass retrieval sends the original query only.** The "What is <q>" / "Explain <q>" reformulations were measured with the new query-expansion eval (15 questions, live Tavily, LLM-judged): 2.6× the searches, no relevance gain (grade@5 2.51 vs 2.57, nDCG@5 0.83 vs 0.86). Dropping them cuts web searches ~60%. Retry-pass reformulations are unchanged. See BENCHMARKS.md.

### Changed — application structure
- **`main.py` split into an app factory.** `create_app(settings=None, *, search_backend=None)` builds an isolated app; `app = create_app()` remains the ASGI entry point (`axiom_rag_engine.main:app`). Endpoints live in `api/routes/{synthesize,documents,audits,ops}.py`, startup wiring in `bootstrap.py`, and per-app state in `services.AppServices` on `app.state.services`.
- **Less process-wide app state.** The response cache, rate limiter, audit store, corpus store, body-size limits, and search backend are per app, and the pipeline (nodes, budget caps, LLM timeouts and retries, embeddings) reads the serving app's settings via `current_settings()`, bound per request. The LLM concurrency limit, Prometheus registry, and tracer provider stay process-wide by design. The search backend reaches the retriever through the LangGraph run config (`configurable.search_backend`); `set_search_backend` still sets the default used by direct node calls, tests, and evals. Auth and rate-limit bucketing read the app's settings.
- **Async response cache.** `CacheBackend` is now async; the Redis backend uses `redis.asyncio`, so a slow Redis can no longer stall the event loop (failures still degrade to a cache miss). Cache connections close at shutdown.
- **Structured outputs.** Synthesizer, semantic-verifier, corroboration, and contradiction calls send a JSON Schema (`schemas.py`) where the provider supports it (LiteLLM `supports_response_schema`; Ollama via its native `format`), falling back to JSON mode otherwise. The lenient parser stays as a safety net. Measured on local Ollama (llama3.2:1b, qwen3.5:9b over the golden questions) parse success was already 100% in both modes, so the gain is shape guarantees on cloud providers rather than a measured local improvement.
- **`is_cited` is derived from `citations`** when parsing synthesizer output, instead of failing the parse (and spending a retry) when a model gets the redundant flag wrong — observed live on llama3.2:1b. Every citation is still verified; a sentence without citations is labelled unverified either way.

### Changed — budget exhaustion status (429 → 422)
- **Running out of the LLM budget before any verified pass returns 422, not 429.** Clients and proxies treat 429 as "retry later", but the same request would exhaust the same budget again. The error message now names the limit (`AXIOM_MAX_LLM_CALLS_PER_REQUEST` / `AXIOM_MAX_TOKENS_PER_REQUEST`). Exhaustion after a verified pass still returns that pass with 200 (`partial`); the stream endpoint's `budget_exceeded` error frame is unchanged.

### Changed — auth mode (breaking for setups that relied on a dev environment to switch off configured keys)
- **Configured API keys are always enforced.** Auth used to depend on `AXIOM_ENV` alone, so a deployment that set `AXIOM_API_KEYS` but carried `AXIOM_ENV=dev` or `test` (a copied `.env`, a typo) served every endpoint without authentication. Now any configured key (`AXIOM_API_KEYS` or `AXIOM_ADMIN_API_KEYS`) turns auth on; a non-production environment disables auth only when no keys are set, so the development quick start is unchanged. Production's fail-closed startup checks (LLM provider, live search) now follow the environment only (`Settings.is_production()`), so configuring keys locally does not bring them.
- **`/docs` and `/redoc` are off by default when auth is required.** `AXIOM_DOCS_ENABLED` now defaults to the auth mode (on in development, off in production); set it explicitly to override.

### Changed — uncited claims (status can change from `success` to `partial`)
- **Uncited sentences with checkable content make the response `partial`.** The synthesizer may leave only transitional sentences uncited, but nothing enforced it, so an answer with verified sentences plus an uncited "Tesla sold 1.8 million cars in 2023" came back `status: "success"`. An uncited sentence carrying numbers or names (capitalised words after the first) now makes it `partial`, and its `failure_reason` says so. `confidence_summary` gains `uncited_sentences` and `uncited_checkable_sentences`; uncited sentences still stay out of the score and the tier breakdown.

### Changed — corpus write access (breaking for production deployments that ingest)
- **Ingest and delete require an admin key when auth is required.** The corpus is shared by every tenant, yet any valid API key could ingest or delete any document, letting one tenant poison or erase what every other tenant's answers are built from. `POST /v1/documents`, `POST /v1/documents/upload`, and `DELETE /v1/documents/{id}` now need a key listed in the new `AXIOM_ADMIN_API_KEYS` (403 otherwise; refused outright when none is configured). Reads still accept any valid key, and admin keys are valid API keys. With auth disabled, writes stay open.

### Changed — model policy (breaking for production callers that set `models`)
- **The verifier is server policy when auth is required.** The verifier grants the confidence tiers, so a caller could previously pick a lenient judge for its own answers. `models.verifier` is now ignored (and logged) unless auth is disabled.
- **Caller-chosen synthesizers are allowlisted when auth is required.** `models.synthesizer` must be the server default or listed in `AXIOM_ALLOWED_SYNTHESIZER_MODELS`; anything else returns 422 before any model is called. Previously any LiteLLM model reachable with the operator's keys could be requested.
- The response cache key uses the effective models, so an ignored override shares the cache entry of the request it resolves to.
- Development/test environments (auth disabled) honour caller model choices as before.

### Changed — performance
- **Semantic verdicts are reused across passes.** A rewrite no longer re-judges sentences it kept unchanged: completed verdicts are cached per request by (claim, chunk, quote, verifier model), saving verifier calls and budget. Audited as `semantic_verdicts_reused`.

### Changed — verification honesty (breaking for clients that assumed every Tier 3 was checked)
- **New `tier_label: "unverified"` (tier 3).** A cited sentence whose semantic check could not run — provider error, unparseable verifier output, exhausted LLM budget — was silently reported as Tier 3 "model_assisted" with `status: "success"`, identical to a verified answer. It is now labelled `unverified`, scores 0.30 (vs 0.60), and makes the response `status: "partial"`. Enforced by `VerificationResult`'s validator: `model_assisted` now requires a mechanical pass.
- **Uncited sentences are `unverified`, not Tier 3 "model_assisted".** They remain allowed (transitional text) but are excluded from the confidence score, the tier breakdown, and the success decision; an answer with no cited sentence is `partial`.
- **Mechanical verifier no longer deletes punctuation.** Punctuation and symbols become word boundaries, a minus sign before a number is kept, matches must align to word boundaries in spaced scripts, and the 12-character minimum applies only to unspaced scripts (CJK, Thai, …). Previously `"rose 1.5 percent"` matched a source saying `"rose 15 percent"`, `"5 degrees"` matched `"-5 degrees"`, and a single long word (`"cardiovascular"`) counted as a quote.
- **`max_rewrite_loops` now counts rewrites.** It used to count synthesis passes, so `max_rewrite_loops=1` meant no rewrite at all. The default moved from 3 to 2 so default LLM cost is unchanged (still 3 synthesis passes per retrieval round); `0` now disables rewrites. Callers who set it explicitly get one more rewrite pass than before.
- **Pre-LLM answerability gate requires lexical overlap.** Chunk quality alone cleared the old `ranking_score` floor, so the gate only fired on empty retrieval. It now also requires at least one chunk to share a query term (waived when hybrid retrieval ran).
- **BM25 tokenizer is Unicode-aware.** Arabic, accented Latin, and other scripts are tokenized; CJK/Thai runs become character bigrams. Previously every non-Latin query scored zero relevance. SciFact BM25 metrics are unchanged.
- **Tier 1 is judged per page.** Forum, mailing-list, Q&A, and public-comment pages hosted on primary domains (`users.rust-lang.org`, `lists.w3.org`, `learn.microsoft.com/…/answers/`, `regulations.gov/comment/…`) no longer inherit Tier 1.
- **`GET /v1/status` requires an API key.** Health probes remain open.

### Fixed
- **The faithfulness verifier saw source metadata.** Its prompt carried every chunk field — domain, URL, title, and the scorer's authority and quality scores — while telling the model not to infer authority. It now sees only the claim, quote, and chunk text; authority stays with the deterministic tiers.
- **Readiness ignored dependencies.** `/health/ready` returned 200 whenever configuration looked right. It now checks reachability (cached for 5 s): a configured corpus database that cannot be read returns 503; an unreachable Redis cache reports `"status": "degraded"` with 200, since the cache is optional and its failures already degrade to misses. The response now includes per-dependency `checks`.
- **PDF extraction had no limits.** Uploaded PDFs were parsed in a worker thread with no page cap or time limit; a thread cannot be cancelled, so a crafted file could pin a worker indefinitely, and parser errors outside a few caught types became 500s. Extraction now runs in a spawned child process that is killed after `AXIOM_CORPUS_PDF_TIMEOUT_SECONDS` (default 60), PDFs over `AXIOM_CORPUS_MAX_PDF_PAGES` (default 500) are refused before parsing, and every failure is a 422.
- **HTTP request tracing never activated.** FastAPI instrumentation ran from the lifespan hook, after Starlette had built its middleware stack, so no request spans were emitted and pipeline spans were scattered across traces. It now runs in `create_app`; request spans carry `axiom.request_id` and pipeline spans nest under them.
- **Audit events leaked provider error text.** Search, synthesizer, dense-ranking, and semantic/corroboration/contradiction error events stored raw exception messages, which callers can read via `include_debug` and `/v1/audits`. They now record `error_type` only; full messages go to the server log.
- **Deleted documents were still cited from the response cache** until the entry expired. The corpus now keeps a version bumped by every ingest and delete, and the cache key includes it.
- **The corpus store could hit "database is locked"** when a search ran during an ingest. It now uses WAL mode and a 10 s busy timeout, and versions its schema (`PRAGMA user_version`) with in-place migrations.
- **The JSON endpoint kept running after its client disconnected**, spending LLM budget on a response nobody would read. The pipeline is now cancelled (counted as `status="cancelled"`, HTTP 499).
- **No bound on a request's duration.** The per-call LLM timeout was a fixed 600 s and nothing bounded a whole request. `AXIOM_LLM_TIMEOUT_SECONDS` (default 120) and `AXIOM_REQUEST_DEADLINE_SECONDS` (default 300) now bound them; a deadline after a verified pass returns that pass, before one it returns 504.
- **A failing later pass discarded a verified answer.** Once one pass was verified, a rewrite or re-retrieval that failed — exhausted LLM budget, a provider error, every search erroring — failed the whole request (HTTP 429/500), and a rewrite on which the synthesizer declared the query unanswerable turned the response `unanswerable`. The run now halts and returns the best verified pass (usually `status: "partial"`), audited as `pipeline_halted_best_pass_returned` and reported as `pipeline_stats.halt_reason`. Failures before the first verified pass still return 429/500.
- **Transient provider failures were not retried.** One rate-limit or overloaded response failed a synthesis pass or left a citation unverified. `call_llm` now retries rate limits, timeouts, dropped connections, and 5xx (`AXIOM_LLM_MAX_RETRIES`, default 2) with jittered backoff that honours `Retry-After` (capped by `AXIOM_LLM_RETRY_MAX_WAIT_SECONDS`). A retried call consumes one unit of per-request budget; retries are counted in `axiom_llm_retries_total`.
- CORS now allows `DELETE`, so browser clients can call `DELETE /v1/documents/{id}`.
- **Audit trails leaked across tenants.** Any API key could list, read, and overwrite (via a reused `request_id`) another key's audit trails. Trails are now scoped to the producing key.
- **Budget exhaustion returned HTTP 500 instead of 429.** The synthesizer wrapped `LLMBudgetExceededError` in a `RuntimeError`.
- **Rewrite passes were blind.** The correction list referenced sentence/citation IDs from a draft the model never saw; the previous draft is now included. When every retry is exhausted, the best pass seen is returned instead of the last one (audited as `best_pass_selected`).
- **Streaming loop events.** The `re_retrieve` loop event could never fire, and `rewrite` fired even when a failed pass ended the run. Events are now emitted when the rewrite / re-retrieval actually starts. The SSE docs no longer claim unverified text never reaches the client — failed sentences are streamed, labelled, exactly as in the JSON response.
- **Chunk cap was first-come.** One long page could consume the whole per-request cap (and overshoot it). The cap is now shared round-robin across documents.
- **Retrieval retries discarded the best sources.** A retry now keeps the previous round's top-ranked chunks alongside fresh results.
- **Startup refused explicitly configured models** (e.g. `gemini/…`, `bedrock/…`) unless an Anthropic/OpenAI key or Ollama was present. Setting both `AXIOM_DEFAULT_*_MODEL` now suffices.
- **Corpus provenance.** The ingest `source` label now reaches citations (`CitationSource.source_label`).
- Document-ingest 502 responses no longer echo backend error text; upload form fields are length-limited like the JSON endpoint.
- PDF/HTML extraction, chunking, and corpus SQLite calls no longer block the event loop; corpus query embedding uses LiteLLM's sync client instead of `asyncio.run` in worker threads (which left aiohttp sessions bound to dead loops).
- `docker-compose` Ollama healthcheck uses `ollama list` instead of `curl`.

### Internal
- **Retrieval gate floors pinned to observed values.** The deterministic BM25 gate's floors sat ~2 points below what it measures, so recall@10 could fall from 0.916 to 0.895 without failing. They now equal the observed values with a 0.001 tolerance. `--ratchet` (retrieval and pipeline-retrieval evals) raises floors after a passing run that beat them and never lowers them; the gate report names the metrics that improved.
- **Paired bootstrap for method comparisons.** `retrieval_eval.py --compare A B` reports the mean per-query difference between two results files with a 95% paired-bootstrap CI (`gate.paired_bootstrap`); results files now record per-query reciprocal rank.
- One LLM call path (`utils.llm.call_llm`) and one JSON parser (`parse_json_object`) replace five copies of budget/semaphore/usage/salvage logic.
- Contradiction and corroboration gates run concurrently across sentences.
- Removed dead code (`_build_uncited_sentence_request`, `run_with_otel_context`).
- Contract tests pin the README tier table, mechanical-verifier integrity cases, verifier fault injection, and tenant isolation.

### Added
- **Tier calibration harness (`evals/calibration_eval.py`, `python tasks.py evals calibration`)**, with `--batch` (short resumable chunks), `--tag` + `--phase compare` (A/B over the same cached sources), per-citation retrieval scores in run records, and resume that retries infrastructure failures. Runs the pipeline over ASQA questions (live Tavily, cached), grades every sentence with a separate judge model against the full text of its cited passages, and reports the judged-supported rate per tier (95% Wilson CI) next to each tier's confidence weight, plus answer-level STR-EM against ASQA gold answers vs `overall_score`. Resumable run / judge / report phases. `python tasks.py evals download-asqa` fetches ASQA dev (via the `din0s/asqa` mirror; the original URL is gone). First results (100 questions, local models): every verified tier judged ~95–100% supported by a lenient 4B judge, and the confidence score does not predict answer correctness (Spearman ≈ 0 vs gold answers) — tier weights left unchanged pending a stronger judge.
- **Query-expansion eval (`evals/query_expansion_eval.py`, `python tasks.py evals query-expansion`).** Compares original-only, the production reformulations, and an LLM keyword rewrite over live Tavily results (cached), graded by an LLM judge. See BENCHMARKS.md.
- **Second-stage reranking (opt-in).** Set `AXIOM_RERANKER_MODEL` (a LiteLLM chat model) to add an LLM reranker that regrades the top `AXIOM_RERANK_TOP_K` candidates (query + passage judged together) and reorders by relevance — after BM25/hybrid and before the trim, so a strong chunk the base ranker buried below the cutoff can still reach the answer. A refinement, not a reshuffle: equal grades keep the base order, `ranking_score` is untouched, each reranked chunk gets a `rerank_grade`, and `ranker_complete` reports `ranking_mode` as `…+rerank`. Off by default; **fails open** (a grading error sinks that candidate; a total failure keeps the base order). Adds up to `AXIOM_RERANK_TOP_K` LLM calls per request — pick a fast model. Measured lift (nDCG@10 +~0.10, recall@1 +~0.13 over BM25) and the harness (`retrieval_eval.py --method rerank`) are in `BENCHMARKS.md`. `GET /v1/status` reports the reranker.
- **Tier 6 (Conflicted) — cross-source contradiction detection (opt-in).** `AXIOM_CONTRADICTION_DETECTION_ENABLED=true` makes a multi-domain sentence whose cited sources *actively contradict each other* (opposite conclusions, incompatible figures) resolve to Tier 6 instead of a confident Tier 1/2 — surfacing disagreement rather than hiding it. An extra verifier check over the distinct-domain quotes runs first and overrides Tier 1/2 (a conflict short-circuits the corroboration gate). Fails safe: a check error keeps the original tier rather than asserting a conflict it could not verify. Default false keeps Tier 6 unassigned. Audited as `contradiction_result` / `contradiction_error`. This completes the 6-tier taxonomy — Tier 6 was previously reserved in the schema but never assigned.
- **Document ingestion (bring-your-own corpus).** Ingest your own documents and answer over them through the same citation-verified pipeline used for web search. Set `AXIOM_CORPUS_DB_PATH` to enable a single-node SQLite corpus store, then manage it via a new API: `POST /v1/documents` (ingest raw text), `POST /v1/documents/upload` (multipart file upload — text / markdown / HTML / **PDF**), `GET /v1/documents` (list), `GET /v1/documents/{id}`, and `DELETE /v1/documents/{id}`. `AXIOM_RETRIEVAL_SOURCE` selects where the retriever draws from — `web` (default), `corpus`, or `both` (merged and deduplicated). Ingestion reuses the production chunker and embedder, so a corpus chunk is scored, ranked, and verified exactly like a web chunk. Corpus retrieval is dense (chunks are embedded at ingest and matched by cosine at query time), so it requires `AXIOM_EMBEDDING_MODEL`. Dependency-light by design: stdlib SQLite, no vector-DB server and no numpy. `GET /v1/status` reports corpus size and the embedding models in use. (`python-multipart` added for file uploads; `pypdf` for PDF text extraction.) A new eval, `evals/corpus_eval.py` (`python tasks.py evals corpus`), measures document-level recall through the shipped ingest→store→search path.
- **Authority scoring v2 — government-domain rule + expanded primary packs.** Official government / treaty-org domains now reach Tier 1 by TLD rule, with no per-agency allowlist: globally restricted `.gov` / `.mil` / `.int`, plus a curated allowlist of national government second-levels (`gov.uk`, `gob.mx`, `gouv.fr`, `canada.ca`, …). The rule is deliberately tight — it does **not** blanket-match `gov.<cc>`, so lookalikes on open-registration ccTLDs (e.g. `records.gov.io`, where `.io` runs no restricted government zone) stay out of Tier 1. The curated primary-source list was also broadened into per-vertical packs (intergovernmental bodies, standards organizations, and official language / platform / vendor docs such as `docs.python.org`, `developer.mozilla.org`, `learn.microsoft.com`). Reference sources (Wikipedia, arXiv) remain Tier-2-max, never primary.
- **Cross-source corroboration for Tier 2 (opt-in).** `AXIOM_CORROBORATION_ENABLED=true` makes a multi-domain sentence reach Tier 2 only when ≥2 distinct sources independently corroborate its central claim (an extra verifier call over the cited quotes). Sentences whose sources merely cover different aspects drop to Tier 3. Default false keeps Tier 2 as multi-domain coverage. Fails safe to Tier 3 on any check error — never claims corroboration it could not verify. Audited as `corroboration_result` / `corroboration_error`.
- **Hybrid retrieval (opt-in).** Set `AXIOM_EMBEDDING_MODEL` (a LiteLLM embedding model, e.g. `ollama/nomic-embed-text` or `text-embedding-3-small`) to fuse BM25 with dense cosine similarity via reciprocal-rank fusion in the ranker. Empty (default) = BM25-only, so existing deployments are unaffected. Falls back to BM25 automatically if the embedder errors. `AXIOM_RRF_K` tunes the fusion constant. `GET /v1/status` reports the `retrieval` mode. Dense retrieval helps most on vocabulary-mismatch queries — see `BENCHMARKS.md`.
- **Retrieval-quality eval (`evals/retrieval_eval.py`)** measuring recall@k / nDCG@10 / MRR over SciFact and BEIR datasets, with BM25 / dense / hybrid methods and a deterministic BM25 regression gate. `BENCHMARKS.md` records the measured BM25-vs-hybrid crossover.

## [0.1.0b2] - 2026-04-16

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

[0.1.0b2]: https://github.com/FurkhanShaikh/axiom-rag-engine/releases/tag/v0.1.0b2
[0.1.0b1]: https://github.com/FurkhanShaikh/axiom-rag-engine/releases/tag/v0.1.0b1
