"""
Axiom Engine — Centralized runtime configuration.

Every `AXIOM_*` environment variable the service reads is declared here,
with a type, a default, and a short description. A single `Settings`
instance is the authoritative source — no code should call `os.getenv` for
an `AXIOM_*` variable directly.

Usage:

    from axiom_rag_engine.config.settings import get_settings

    settings = get_settings()
    if settings.allow_mock_search:
        ...

`get_settings()` is cached, so repeated calls are cheap. Tests that need
to override configuration should call `get_settings.cache_clear()` between
cases (see `tests/conftest.py`).
"""

from __future__ import annotations

from contextvars import ContextVar
from functools import lru_cache
from typing import Annotated, Any, Literal

from pydantic import BeforeValidator, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic_settings.sources import (
    DotEnvSettingsSource,
    EnvSettingsSource,
    PydanticBaseSettingsSource,
)

# ---------------------------------------------------------------------------
# Custom field type — comma-separated list
# ---------------------------------------------------------------------------
# pydantic-settings' default env parser expects JSON (e.g. '["a","b"]') for
# list[str] fields. Every existing AXIOM_* list variable is comma-separated,
# so we define a custom env source that falls back to the raw string when
# JSON decoding fails, letting pydantic's BeforeValidator split on commas.


def _split_csv(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return [str(value)]


CommaSepList = Annotated[list[str], BeforeValidator(_split_csv)]


class _CsvFriendlyEnvSource(EnvSettingsSource):
    """Env source that falls back to the raw string when JSON decoding fails.

    This lets CommaSepList fields accept both ``"a,b,c"`` and ``'["a","b","c"]'``.
    """

    def decode_complex_value(self, field_name: str, field: Any, value: Any) -> Any:
        try:
            return super().decode_complex_value(field_name, field, value)
        except ValueError:
            return value


class _CsvFriendlyDotEnvSource(DotEnvSettingsSource):
    """Same fallback for .env file values."""

    def decode_complex_value(self, field_name: str, field: Any, value: Any) -> Any:
        try:
            return super().decode_complex_value(field_name, field, value)
        except ValueError:
            return value


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


class Settings(BaseSettings):
    """Axiom Engine runtime configuration.

    All fields are populated from environment variables (and optionally a
    `.env` file in the working directory). Field names map to env vars by
    prefixing with `AXIOM_` and uppercasing — e.g. `rate_limit` is read
    from `AXIOM_RATE_LIMIT`.
    """

    model_config = SettingsConfigDict(
        env_prefix="AXIOM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ── Runtime ──────────────────────────────────────────────────────────
    env: Literal["production", "development", "dev", "local", "test"] = Field(
        default="production",
        description="Runtime environment. Non-production values disable auth requirements.",
    )
    docs_enabled: bool | None = Field(
        default=None,
        description=(
            "Serve /docs and /redoc. Unset: on when auth is disabled (development), "
            "off when auth is required, so production does not publish its API schema."
        ),
    )

    # ── Auth ─────────────────────────────────────────────────────────────
    api_keys: CommaSepList = Field(
        default_factory=list,
        description="Comma-separated list of valid API keys. Required when env != development.",
    )
    admin_api_keys: CommaSepList = Field(
        default_factory=list,
        description=(
            "Keys allowed to change the corpus (ingest and delete documents) when auth is "
            "required. Admin keys are also valid API keys. Empty = corpus writes are refused."
        ),
    )

    # ── LLM defaults ─────────────────────────────────────────────────────
    # When these are not set explicitly (env, .env or constructor), startup
    # auto-selects models from the providers it finds — see
    # resolve_llm_defaults in bootstrap.py, which checks model_fields_set.
    default_synthesizer_model: str = Field(
        default="claude-opus-4-8",
        description="Default synthesizer LiteLLM model ID.",
    )
    default_verifier_model: str = Field(
        default="gpt-4o-mini",
        description="Default verifier LiteLLM model ID.",
    )
    allowed_synthesizer_models: CommaSepList = Field(
        default_factory=list,
        description=(
            "Synthesizer models a caller may request via models.synthesizer when auth "
            "is required, besides the server default. Anything else is rejected with "
            "422. The verifier is always server-controlled when auth is required."
        ),
    )

    # ── Rate limiting / response cache ───────────────────────────────────
    rate_limit: str = Field(
        default="20/minute",
        description="SlowAPI rate-limit string applied per API key or IP.",
    )
    stream_rate_limit: str = Field(
        default="20/minute",
        description=(
            "SlowAPI rate-limit string applied per API key or IP for the "
            "streaming endpoint (/v1/synthesize/stream). Defaults to the same "
            "limit as ``rate_limit``."
        ),
    )
    cache_ttl_seconds: int = Field(
        default=300,
        description="TTL for the in-process response cache.",
    )
    cache_max_size: int = Field(
        default=256,
        description="Max entries in the in-process response cache.",
    )
    redis_url: str | None = Field(
        default=None,
        description="If set, use Redis for the response cache instead of in-memory TTLCache.",
        alias="AXIOM_REDIS_URL",
    )

    # ── Search / retrieval ───────────────────────────────────────────────
    allow_mock_search: bool = Field(
        default=False,
        description="If true, allow MockSearchBackend in non-development envs.",
    )
    fetch_full_pages: bool = Field(
        default=True,
        description=(
            "Request full page text from the search backend instead of verifying "
            "against its short result snippet. Snippets are a summary of the page, "
            "so a quote can be genuinely present on the source and still fail "
            "mechanical verification as Tier 5. Disable to cut retrieval latency "
            "and payload size at the cost of that false-negative rate."
        ),
    )
    max_raw_content_chars: int = Field(
        default=200_000,
        ge=1_000,
        description=(
            "Per-document cap on full page text. Bounds memory and chunking work "
            "when a single result is very large; the page is truncated, not dropped."
        ),
    )
    authoritative_domains: CommaSepList = Field(
        default_factory=list,
        description="Extra domains treated as authoritative by the scorer.",
    )
    low_quality_domains: CommaSepList = Field(
        default_factory=list,
        description="Domains to down-rank during scoring.",
    )
    exclude_default_domains: CommaSepList = Field(
        default_factory=list,
        description="Domains to strip from the built-in authoritative list.",
    )

    # ── Retrieval: hybrid (dense + BM25) ─────────────────────────────────
    embedding_model: str | None = Field(
        default=None,
        description=(
            "LiteLLM embedding model for hybrid retrieval (e.g. "
            "'ollama/nomic-embed-text' or 'text-embedding-3-small'). When set, the "
            "ranker fuses BM25 with dense cosine via reciprocal-rank fusion. "
            "Unset (default) = BM25-only ranking. Dense retrieval helps most on "
            "vocabulary-mismatch queries (colloquial query vs formal source); see "
            "BENCHMARKS.md."
        ),
    )
    rrf_k: int = Field(
        default=60,
        ge=1,
        description="Reciprocal-rank-fusion constant for hybrid retrieval. 60 is the RRF-paper default.",
    )
    reranker_model: str | None = Field(
        default=None,
        description=(
            "LiteLLM chat model for second-stage reranking (e.g. 'gpt-4o-mini'). "
            "When set, the ranker regrades the top AXIOM_RERANK_TOP_K candidates "
            "(query+passage judged together) and reorders by relevance before "
            "trimming — lifting precision@k. Unset (default) = no reranking. Adds "
            "up to AXIOM_RERANK_TOP_K LLM calls per request; pick a *fast* model "
            "(a local thinking model adds seconds per candidate). Fails open to "
            "the pre-rerank order on any error. See BENCHMARKS.md."
        ),
    )
    rerank_top_k: int = Field(
        default=20,
        ge=1,
        description=(
            "How many top candidates the reranker regrades. Must exceed "
            "max_ranked_chunks to change which chunks survive the trim; candidates "
            "below this depth keep their pre-rerank order."
        ),
    )

    # ── Corpus (bring-your-own documents) ────────────────────────────────
    corpus_db_path: str | None = Field(
        default=None,
        description=(
            "Path to the SQLite corpus database for ingested documents. Set it to "
            "enable the document-management API and corpus retrieval. Unset "
            "(default) = web-only, no corpus. Requires AXIOM_EMBEDDING_MODEL, since "
            "ingestion and corpus search are dense (chunks are embedded at ingest "
            "and matched by cosine at query time)."
        ),
    )
    retrieval_source: Literal["web", "corpus", "both"] = Field(
        default="web",
        description=(
            "Where the retriever draws sources from: 'web' (Tavily, default), "
            "'corpus' (only ingested documents), or 'both' (merge web + corpus, "
            "deduplicated). 'corpus'/'both' require AXIOM_CORPUS_DB_PATH."
        ),
    )
    corpus_max_results: int = Field(
        default=5,
        ge=1,
        description="Top-k chunks corpus retrieval returns per search query.",
    )
    corpus_max_chunks_per_document: int = Field(
        default=2000,
        ge=1,
        description="Cap on chunks stored per ingested document (guards a runaway upload).",
    )
    corpus_max_pdf_pages: int = Field(
        default=500,
        ge=1,
        description="Uploaded PDFs with more pages are refused (422) before any page is parsed.",
    )
    corpus_pdf_timeout_seconds: float = Field(
        default=60.0,
        gt=0,
        description=(
            "Wall-clock limit for extracting one PDF. Extraction runs in a child process "
            "that is killed on overrun, so a crafted file cannot pin a worker."
        ),
    )
    max_document_bytes: int = Field(
        default=10_485_760,  # 10 MiB
        ge=1,
        description=(
            "Body-size cap for the document-ingest endpoints, separate from "
            "AXIOM_MAX_BODY_BYTES (which stays small for the synthesize API). "
            "Documents are legitimately large, so this is higher by default."
        ),
    )

    # ── Verification ─────────────────────────────────────────────────────
    semantic_verification_enabled: bool = Field(
        default=True,
        description="Server policy for semantic verification (Stage 2).",
    )
    corroboration_enabled: bool = Field(
        default=False,
        description=(
            "When true, a sentence reaches Tier 2 only if >=2 distinct-domain sources "
            "independently corroborate its central claim (an extra LLM check over the "
            "cited quotes). Multi-domain sentences that merely cover different aspects "
            "drop to Tier 3. Default false keeps Tier 2 as multi-domain coverage. "
            "Adds one verifier call per Tier-2-candidate sentence."
        ),
    )
    contradiction_detection_enabled: bool = Field(
        default=False,
        description=(
            "When true, a multi-domain sentence whose cited sources actively "
            "contradict each other is assigned Tier 6 (Conflicted) instead of "
            "Tier 1/2 — surfacing source disagreement rather than hiding it behind "
            "a confident tier (an extra LLM check over the distinct-domain quotes). "
            "Default false: contradiction is not checked and Tier 6 is never "
            "assigned. Fails safe — a check error keeps the original tier rather "
            "than asserting a conflict it could not verify. Adds one verifier call "
            "per multi-domain sentence."
        ),
    )
    min_usable_ranking_score: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Pre-LLM threshold — if the best ranking_score is below this, the synthesizer is skipped and is_answerable=false is returned.",
    )

    # ── LLM budget & concurrency ─────────────────────────────────────────
    max_llm_calls_per_request: int = Field(
        default=64,
        ge=1,
        description="Hard cap on LLM completions per request.",
    )
    max_tokens_per_request: int = Field(
        default=0,
        ge=0,
        description="Hard cap on total LLM tokens per request. 0 = unlimited.",
    )
    max_concurrent_llm: int = Field(
        default=5,
        ge=1,
        description="Maximum concurrent in-flight LLM calls across all requests.",
    )
    llm_timeout_seconds: float = Field(
        default=120.0,
        gt=0,
        le=3600,
        description=(
            "Timeout for one LLM call. Raise it for slow local models (Ollama on "
            "CPU can need several minutes for a long prompt)."
        ),
    )
    request_deadline_seconds: float = Field(
        default=300.0,
        ge=0,
        description=(
            "Wall-clock limit for one pipeline run. When it expires after a verified "
            "pass, that pass is returned (status partial); before any verified pass the "
            "request fails with 504. 0 disables the deadline."
        ),
    )
    llm_max_retries: int = Field(
        default=2,
        ge=0,
        le=5,
        description=(
            "Retries for a transient provider failure (rate limit, timeout, connection "
            "drop, 5xx) on one LLM call. 0 disables retries. A retry does not consume "
            "extra per-request call budget."
        ),
    )
    llm_retry_max_wait_seconds: float = Field(
        default=8.0,
        ge=0.0,
        le=60.0,
        description="Upper bound on one retry backoff (also caps a provider's Retry-After).",
    )
    allowed_metric_models: CommaSepList = Field(
        default_factory=lambda: [
            # Claude 5 family
            "claude-fable-5",
            "claude-sonnet-5",
            # Claude 4.x
            "claude-opus-4-8",
            "claude-opus-4-7",
            "claude-opus-4-6",
            "claude-sonnet-4-6",
            "claude-haiku-4-5",
            "claude-haiku-4-5-20251001",
            # Claude 4.5 legacy
            "claude-sonnet-4-5",
            "claude-opus-4-5",
            # OpenAI
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            # OpenRouter auto-selected defaults
            "openrouter/openai/gpt-4o",
            "openrouter/openai/gpt-4o-mini",
            # Local (prefix-matched)
            "ollama",
        ],
        description="Models allowed as Prometheus labels. Others are collapsed to 'other' to bound cardinality.",
    )
    ollama_api_base: str = Field(
        default="http://localhost:11434",
        description="Ollama server URL used when a request specifies ollama/<model>.",
        alias="OLLAMA_API_BASE",
    )
    tavily_api_key: str | None = Field(
        default=None,
        description="Tavily search API key. When set, enables live web retrieval.",
        alias="TAVILY_API_KEY",
    )
    anthropic_api_key: str | None = Field(
        default=None,
        description="Anthropic API key. Presence enables claude-* model selection.",
        alias="ANTHROPIC_API_KEY",
    )
    openai_api_key: str | None = Field(
        default=None,
        description="OpenAI API key. Presence enables gpt-* model selection.",
        alias="OPENAI_API_KEY",
    )
    openrouter_api_key: str | None = Field(
        default=None,
        description=(
            "OpenRouter API key. Presence enables openrouter/* models, and the "
            "openrouter_* defaults below when no Anthropic or OpenAI key is set."
        ),
        alias="OPENROUTER_API_KEY",
    )
    openrouter_synthesizer_model: str = Field(
        default="openrouter/openai/gpt-4o",
        description="Synthesizer auto-selected when OpenRouter is the only cloud provider.",
    )
    openrouter_verifier_model: str = Field(
        default="openrouter/openai/gpt-4o-mini",
        description=(
            "Verifier auto-selected when OpenRouter is the only cloud provider. The "
            "default is the same model as the OpenAI-key verifier, routed via OpenRouter."
        ),
    )

    # ── Audit ────────────────────────────────────────────────────────────
    audit_retention: int = Field(
        default=0,
        ge=0,
        description="Number of recent audit trails to keep in memory for GET /v1/audits/{request_id}. 0 = disabled.",
    )
    log_audit_events: bool = Field(
        default=False,
        description="When true, every audit event is emitted as a structured log line (best with LOG_FORMAT=json).",
    )

    # ── Security / limits ────────────────────────────────────────────────
    cors_origins: CommaSepList = Field(
        default_factory=list,
        description="Allowed CORS origins. Wildcard is rejected.",
    )
    trusted_proxy_ips: CommaSepList = Field(
        default_factory=list,
        description="IPs whose X-Forwarded-For headers may be trusted. Use '*' only behind a private ingress.",
    )
    max_body_bytes: int = Field(
        default=128 * 1024,
        description="Hard cap on request body size.",
    )

    # ── Observability ────────────────────────────────────────────────────
    log_format: Literal["text", "json"] = Field(
        default="text",
        description="Log output format. 'json' is recommended for production aggregation.",
        alias="LOG_FORMAT",
    )

    # ─────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            _CsvFriendlyEnvSource(settings_cls),
            _CsvFriendlyDotEnvSource(settings_cls),
            file_secret_settings,
        )

    _NON_PROD_ENVS = frozenset({"development", "dev", "local", "test"})

    def auth_required(self) -> bool:
        """Whether requests must carry a valid API key.

        Configured keys are always enforced: a deployment that set
        AXIOM_API_KEYS but mistyped or copied AXIOM_ENV (``dev``, ``test``)
        used to run with every endpoint open. Without keys, auth is off only
        for an explicit non-production environment.
        """
        if any(self.api_keys) or any(self.admin_api_keys):
            return True
        return self.env.lower() not in self._NON_PROD_ENVS

    def is_production(self) -> bool:
        """Whether the environment is production, which makes startup fail closed
        (no LLM provider, or mock search without AXIOM_ALLOW_MOCK_SEARCH, refuses
        to boot). Separate from ``auth_required``: a developer who configures
        keys locally gets authentication, not production strictness."""
        return self.env.lower() not in self._NON_PROD_ENVS

    def docs_on(self) -> bool:
        """Whether to serve /docs and /redoc (see ``docs_enabled``)."""
        return self.docs_enabled if self.docs_enabled is not None else not self.auth_required()

    def redacted_dict(self) -> dict[str, Any]:
        """Return settings as a dict with secrets masked. Used by `check-config`.

        Secrets are recognised by field name (``_SECRET_FIELD_SUFFIXES``), so a
        secret added later is masked by default rather than printed.
        """
        data = self.model_dump()
        for name, value in data.items():
            if not value or not name.endswith(_SECRET_FIELD_SUFFIXES):
                continue
            if isinstance(value, list):
                data[name] = [f"***{len(str(v))}" for v in value]
            else:
                data[name] = f"***{len(str(value))}"
        if data.get("redis_url"):
            data["redis_url"] = _redact_url(data["redis_url"])
        return data


# Settings fields holding credentials; their values never leave redacted_dict.
_SECRET_FIELD_SUFFIXES = ("api_key", "api_keys", "_token", "_secret", "_password")


def _redact_url(url: str) -> str:
    """Mask the password in a URL like redis://user:pw@host:6379/0."""
    try:
        from urllib.parse import urlparse, urlunparse

        parsed = urlparse(url)
        if parsed.password:
            netloc = parsed.netloc.replace(f":{parsed.password}@", ":***@")
            return urlunparse(parsed._replace(netloc=netloc))
    except Exception:
        return "***"
    return url


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide Settings instance (cached)."""
    return Settings()


# The settings of the app serving the current request. Set by the API layer
# (``use_settings``) before the pipeline runs; asyncio tasks the request spawns
# inherit it, so every node and LLM call sees its own app's configuration.
_request_settings: ContextVar[Settings | None] = ContextVar("axiom_request_settings", default=None)


def use_settings(settings: Settings) -> None:
    """Make ``settings`` the configuration for the rest of the current request."""
    _request_settings.set(settings)


def current_settings() -> Settings:
    """Settings for the code running now: the serving app's when inside a
    request (see ``use_settings``), otherwise the process settings. Pipeline
    code reads this, never ``get_settings()``, so apps built by ``create_app``
    with explicit settings are honoured end to end."""
    return _request_settings.get() or get_settings()
