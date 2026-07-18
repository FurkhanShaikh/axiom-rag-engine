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
    docs_enabled: bool = Field(
        default=True,
        description="If false, /docs and /redoc are disabled.",
    )

    # ── Auth ─────────────────────────────────────────────────────────────
    api_keys: CommaSepList = Field(
        default_factory=list,
        description="Comma-separated list of valid API keys. Required when env != development.",
    )

    # ── LLM defaults ─────────────────────────────────────────────────────
    # These doubles as the "operator did not choose a model" sentinel — see
    # _resolve_llm_defaults in main.py, which reads them via model_fields.
    default_synthesizer_model: str = Field(
        default="claude-opus-4-8",
        description="Default synthesizer LiteLLM model ID.",
    )
    default_verifier_model: str = Field(
        default="gpt-4o-mini",
        description="Default verifier LiteLLM model ID.",
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
        """Return True unless the runtime env is an explicit non-prod alias."""
        return self.env.lower() not in self._NON_PROD_ENVS

    def redacted_dict(self) -> dict[str, Any]:
        """Return settings as a dict with secrets masked. Used by `check-config`."""
        data = self.model_dump()
        if data.get("api_keys"):
            data["api_keys"] = [f"***{len(k)}" for k in data["api_keys"]]
        if data.get("redis_url"):
            data["redis_url"] = _redact_url(data["redis_url"])
        return data


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
