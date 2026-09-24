"""
Axiom Engine — startup wiring.

Turns ``Settings`` into the ``AppServices`` an app runs on: LLM default
resolution, the response cache, the audit store, the corpus store, the search
backend, and the compiled graph. Called from the app lifespan (``main.py``);
nothing here touches module-level state of other modules.
"""

from __future__ import annotations

import contextlib
import importlib.metadata
import json
import logging
import os
import time
from typing import Any

from axiom_rag_engine.api.auth import _api_keys, _auth_required
from axiom_rag_engine.audit_store import AuditStore
from axiom_rag_engine.cache import CacheBackend, MemoryCacheBackend, RedisCacheBackend
from axiom_rag_engine.config.settings import Settings
from axiom_rag_engine.corpus.store import CorpusStore
from axiom_rag_engine.graph import build_axiom_graph
from axiom_rag_engine.services import AppServices
from axiom_rag_engine.spend import MemorySpendLedger, RedisSpendLedger, SpendLedger

logger = logging.getLogger("axiom_rag_engine")

# pyproject.toml is the single source of truth; this fallback only applies when
# running from a source tree with no distribution metadata installed.
VERSION = "0.0.0+unknown"
with contextlib.suppress(importlib.metadata.PackageNotFoundError):
    VERSION = importlib.metadata.version("axiom-rag-engine")

# ---------------------------------------------------------------------------
# LLM provider detection
# ---------------------------------------------------------------------------

# Ollama model preference order (first match wins)
_OLLAMA_PREFERENCE = [
    "qwen3:8b",
    "qwen3:4b",
    "qwen3:1.7b",
    "llama3.2:3b",
    "llama3:8b",
    "mistral:7b",
    "gemma2:9b",
]


def _list_ollama_models(base_url: str) -> list[str]:
    """Return available Ollama model names, or [] if Ollama is unreachable."""
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(  # noqa: S310 — base_url is operator-controlled, http(s) only
            f"{base_url}/api/tags", timeout=2
        ) as resp:
            data = json.loads(resp.read())
            return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


def _best_ollama_model(models: list[str]) -> str | None:
    for preferred in _OLLAMA_PREFERENCE:
        family = preferred.split(":")[0]
        match = next((m for m in models if m.startswith(family)), None)
        if match:
            return match
    return models[0] if models else None


def resolve_llm_defaults(settings: Settings) -> tuple[str, str]:
    """
    Detect available LLM providers and return (synthesizer_model, verifier_model).

    Resolution order:
      1. If the operator explicitly set AXIOM_DEFAULT_*_MODEL (in the
         environment, .env or the Settings constructor — even to the built-in
         default value), trust their choice unconditionally.
      2. Otherwise, auto-select by probing for API keys and Ollama availability:
         synthesizer: Anthropic > OpenAI    > OpenRouter > Ollama
         verifier:    OpenAI   > Anthropic > OpenRouter > Ollama
      3. If no provider is reachable, raise RuntimeError at startup rather than
         letting requests fail mid-pipeline with an opaque error.
    """
    has_anthropic = bool(settings.anthropic_api_key)
    has_openai = bool(settings.openai_api_key)
    has_openrouter = bool(settings.openrouter_api_key)

    ollama_models = _list_ollama_models(settings.ollama_api_base)
    best_ollama = _best_ollama_model(ollama_models)
    has_ollama = best_ollama is not None

    configured_synth = settings.default_synthesizer_model
    configured_verif = settings.default_verifier_model
    # Explicitly set, not "differs from the default": an operator who pins the
    # default model must not be auto-switched to another provider.
    operator_set_synth = "default_synthesizer_model" in settings.model_fields_set
    operator_set_verif = "default_verifier_model" in settings.model_fields_set

    def _select(role: str, operator_set: bool, configured: str) -> str:
        if operator_set:
            return configured

        if role == "synthesizer":
            # Synthesis is the quality-critical step — every cited claim
            # originates here — so prefer the most capable available model.
            if has_anthropic:
                return "claude-opus-4-8"
            if has_openai:
                return "gpt-4o"
            if has_openrouter:
                return settings.openrouter_synthesizer_model
            if has_ollama:
                return f"ollama/{best_ollama}"
        else:  # verifier
            # Verification is a per-citation entailment check: high volume,
            # narrow judgment. A small fast model is the right trade here.
            if has_openai:
                return "gpt-4o-mini"
            if has_anthropic:
                return "claude-haiku-4-5"
            if has_openrouter:
                return settings.openrouter_verifier_model
            if has_ollama:
                return f"ollama/{best_ollama}"

        return configured

    if operator_set_synth and operator_set_verif:
        # The operator chose both models explicitly (e.g. gemini/..., bedrock/...,
        # openrouter/...). We cannot detect every LiteLLM provider, so trust the
        # choice; a wrong key surfaces on the first request.
        logger.info(
            "Models operator-configured: synthesizer=%s verifier=%s",
            configured_synth,
            configured_verif,
        )
        return configured_synth, configured_verif

    if not (has_anthropic or has_openai or has_openrouter or has_ollama):
        # Only fail-closed in production. In dev/test envs we fall back to the
        # configured defaults so the app can boot without provider credentials —
        # individual requests will surface the missing-key error at call time.
        if settings.is_production():
            raise RuntimeError(
                "No LLM provider is available. Configure one of:\n"
                "  • ANTHROPIC_API_KEY  (recommended for production)\n"
                "  • OPENAI_API_KEY\n"
                "  • OPENROUTER_API_KEY\n"
                f"  • Ollama running at {settings.ollama_api_base} with at least one model pulled\n"
                "Or set BOTH AXIOM_DEFAULT_SYNTHESIZER_MODEL and AXIOM_DEFAULT_VERIFIER_MODEL "
                "to models your environment can reach."
            )
        logger.warning(
            "No LLM provider detected; using configured defaults (synth=%s, verif=%s). "
            "Requests will fail until ANTHROPIC_API_KEY, OPENAI_API_KEY, OPENROUTER_API_KEY, "
            "or Ollama is available.",
            configured_synth,
            configured_verif,
        )
        return configured_synth, configured_verif

    synth = _select("synthesizer", operator_set_synth, configured_synth)
    verif = _select("verifier", operator_set_verif, configured_verif)

    if synth != configured_synth:
        logger.warning(
            "Synthesizer default auto-selected: '%s' (configured '%s' requires a missing API key). "
            "Set AXIOM_DEFAULT_SYNTHESIZER_MODEL to silence this.",
            synth,
            configured_synth,
        )
    else:
        logger.info(
            "Synthesizer model: %s%s", synth, " (operator-configured)" if operator_set_synth else ""
        )

    if verif != configured_verif:
        logger.warning(
            "Verifier default auto-selected: '%s' (configured '%s' requires a missing API key). "
            "Set AXIOM_DEFAULT_VERIFIER_MODEL to silence this.",
            verif,
            configured_verif,
        )
    else:
        logger.info(
            "Verifier model: %s%s", verif, " (operator-configured)" if operator_set_verif else ""
        )

    if has_ollama:
        logger.info(
            "Ollama reachable at %s — available models: %s",
            settings.ollama_api_base,
            ", ".join(ollama_models),
        )
    else:
        logger.debug("Ollama not reachable at %s.", settings.ollama_api_base)

    return synth, verif


# ---------------------------------------------------------------------------
# Stores, cache, search backend
# ---------------------------------------------------------------------------


def build_cache(settings: Settings) -> CacheBackend:
    """Redis when AXIOM_REDIS_URL is set and usable, else an in-memory TTL cache."""
    if settings.redis_url:
        try:
            cache = RedisCacheBackend(
                redis_url=settings.redis_url, ttl_seconds=settings.cache_ttl_seconds
            )
            logger.info("Response cache: Redis backing layer initialized.")
            return cache
        except ImportError:
            logger.warning(
                "AXIOM_REDIS_URL is set but 'redis' package is not installed. "
                "Falling back to MemoryCacheBackend. (Install the 'redis' extra.)"
            )
        except Exception as exc:
            logger.error(
                "Failed to initialize Redis cache: %s. Falling back to MemoryCacheBackend.", exc
            )
    return MemoryCacheBackend(
        maxsize=settings.cache_max_size, ttl_seconds=settings.cache_ttl_seconds
    )


def build_spend_ledger(cache: CacheBackend) -> SpendLedger:
    """Share per-key spend through the cache's Redis when there is one."""
    if isinstance(cache, RedisCacheBackend):
        return RedisSpendLedger(cache.client)
    return MemorySpendLedger()


def build_corpus_store(settings: Settings) -> CorpusStore | None:
    """Open the corpus store when AXIOM_CORPUS_DB_PATH is set, else return None.

    Presence of the store is what enables the document-management API — separate
    from whether corpus results are wired into retrieval (AXIOM_RETRIEVAL_SOURCE).
    """
    if not settings.corpus_db_path:
        return None
    store = CorpusStore(settings.corpus_db_path)
    logger.info(
        "Corpus store open at %s (%d documents, %d chunks).",
        settings.corpus_db_path,
        store.count_documents(),
        store.count_chunks(),
    )
    return store


def build_search_backend(settings: Settings, corpus_store: CorpusStore | None) -> tuple[Any, str]:
    """Build the retriever's search backend per AXIOM_RETRIEVAL_SOURCE.

    Returns ``(backend, mode)``. ``backend`` is None when no live backend is
    configured (development without a Tavily key) — the retriever then uses its
    module default (MockSearchBackend).

    'web' → Tavily (or mock in dev); 'corpus' → ingested documents only;
    'both' → web + corpus merged (the retriever deduplicates the union). Corpus
    modes require a corpus store and an embedding model. When 'both' is requested
    but Tavily is unavailable, it degrades to corpus-only rather than failing.
    """
    source = settings.retrieval_source
    want_web = source in ("web", "both")
    want_corpus = source in ("corpus", "both")

    backends: list[Any] = []
    modes: list[str] = []

    # --- Web (Tavily) ---
    if want_web:
        tavily_key = settings.tavily_api_key
        if tavily_key:
            from axiom_rag_engine.search.tavily import TavilySearchBackend

            backends.append(
                TavilySearchBackend(
                    api_key=tavily_key,
                    fetch_full_pages=settings.fetch_full_pages,
                    max_raw_content_chars=settings.max_raw_content_chars,
                    timeout_seconds=settings.search_timeout_seconds,
                )
            )
            modes.append("tavily")
            if settings.fetch_full_pages:
                logger.info("Web search: Tavily (verifying against full page text).")
            else:
                logger.warning(
                    "Web search: Tavily with AXIOM_FETCH_FULL_PAGES=false — citations are "
                    "verified against search snippets, not the source page. Quotes that exist "
                    "on the page but not in the snippet will be marked Tier 5 (hallucinated)."
                )
        elif want_corpus:
            logger.warning(
                "AXIOM_RETRIEVAL_SOURCE=both but TAVILY_API_KEY is not set — "
                "serving corpus results only."
            )
        elif settings.is_production() and not settings.allow_mock_search:
            raise RuntimeError(
                "TAVILY_API_KEY must be configured in non-development environments unless "
                "AXIOM_ALLOW_MOCK_SEARCH=true."
            )
        else:
            logger.warning(
                "TAVILY_API_KEY not set — using MockSearchBackend. "
                "Set TAVILY_API_KEY in .env for live web search."
            )
            modes.append("mock")

    # --- Corpus (ingested documents) ---
    if want_corpus:
        if corpus_store is None:
            raise RuntimeError(f"AXIOM_RETRIEVAL_SOURCE={source!r} requires AXIOM_CORPUS_DB_PATH.")
        if not settings.embedding_model:
            raise RuntimeError(
                "Corpus retrieval requires AXIOM_EMBEDDING_MODEL — chunks are embedded at "
                "ingest and matched by cosine at query time."
            )
        from axiom_rag_engine.search.corpus_backend import CorpusSearchBackend

        backends.append(
            CorpusSearchBackend(
                corpus_store,
                settings.embedding_model,
                max_results=settings.corpus_max_results,
            )
        )
        modes.append("corpus")
        logger.info(
            "Corpus retrieval enabled (embedding model %s, top-%d per query).",
            settings.embedding_model,
            settings.corpus_max_results,
        )

    mode = "+".join(modes) if modes else "mock"
    if not backends:
        return None, mode
    if len(backends) == 1:
        return backends[0], mode
    from axiom_rag_engine.search.corpus_backend import CompositeSearchBackend

    return CompositeSearchBackend(backends), mode


def build_services(settings: Settings, *, search_backend: Any = None) -> AppServices:
    """Validate ``settings`` and build everything an app runs on.

    ``search_backend`` overrides the backend derived from settings (tests,
    embedding the engine in another service).
    """
    if _auth_required(settings) and not _api_keys(settings):
        raise RuntimeError("AXIOM_API_KEYS must be configured when AXIOM_ENV is not development.")

    # Push vendor API keys from Settings into os.environ so LiteLLM can find them.
    # pydantic-settings reads .env into the Settings model but does not populate
    # os.environ; LiteLLM reads keys from the process environment directly.
    if settings.anthropic_api_key:
        os.environ.setdefault("ANTHROPIC_API_KEY", settings.anthropic_api_key)
    if settings.openai_api_key:
        os.environ.setdefault("OPENAI_API_KEY", settings.openai_api_key)
    if settings.openrouter_api_key:
        os.environ.setdefault("OPENROUTER_API_KEY", settings.openrouter_api_key)

    audit_store = AuditStore(maxsize=settings.audit_retention)
    if settings.audit_retention:
        logger.info(
            "Audit retention enabled: last %d requests retrievable at /v1/audits/{request_id}.",
            settings.audit_retention,
        )

    synth_model, verif_model = resolve_llm_defaults(settings)

    # The corpus store exists whenever AXIOM_CORPUS_DB_PATH is set — that enables
    # the document API independently of whether corpus results are wired into
    # retrieval (you can ingest under 'web', then switch to 'both').
    corpus_store = build_corpus_store(settings)
    if search_backend is not None:
        backend, mode = search_backend, type(search_backend).__name__
    else:
        backend, mode = build_search_backend(settings, corpus_store)

    engine = build_axiom_graph()
    logger.info("Axiom Engine graph compiled and ready.")

    cache = build_cache(settings)
    return AppServices(
        settings=settings,
        engine=engine,
        cache=cache,
        audit_store=audit_store,
        corpus_store=corpus_store,
        search_backend=backend,
        search_backend_mode=mode,
        default_synthesizer_model=synth_model,
        default_verifier_model=verif_model,
        started_at=time.time(),
        version=VERSION,
        spend_ledger=build_spend_ledger(cache),
    )
