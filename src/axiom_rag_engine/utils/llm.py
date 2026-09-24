"""
Axiom Engine — LLM utility helpers.

Centralises provider-specific quirks (Ollama api_base injection,
response_format gating) and enforces a consistent timeout across all
litellm.acompletion() call sites.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import functools
import json
import logging
import random
import re
import time
from collections.abc import Awaitable, Callable
from typing import Any

from axiom_rag_engine.config.settings import current_settings, get_settings

logger = logging.getLogger("axiom_rag_engine.llm")


# ---------------------------------------------------------------------------
# Per-request LLM call budget
# ---------------------------------------------------------------------------
# A hard cap on both the number of paid LLM completions AND the total tokens
# a single request may consume. Both limits are stored in a mutable dict inside
# a ContextVar so asyncio.gather children share the same counter object.
#
# Values are sourced from Settings (AXIOM_MAX_LLM_CALLS_PER_REQUEST,
# AXIOM_MAX_TOKENS_PER_REQUEST) at reset time.

# ContextVar holds a mutable dict so all tasks created from the same request
# coroutine share one counter object. Keys:
#   remaining        — int, call budget left before LLMBudgetExceededError
#   tokens_used      — int, running total of total_tokens (for the token cap)
#   token_cap        — int, 0 = unlimited
#   calls            — int, count of completed LLM calls (usage observed)
#   prompt_tokens    — int, cumulative prompt tokens
#   completion_tokens— int, cumulative completion tokens
#   cost_usd         — float, cumulative USD cost (best-effort via litellm)
#   by_model         — dict[str, dict] — per-model breakdown with the same fields
_llm_budget_ctx: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "axiom_llm_budget", default=None
)


class LLMBudgetExceededError(RuntimeError):
    """Raised when a single request exhausts its LLM call or token budget."""


def reset_llm_budget(max_calls: int | None = None, max_tokens: int | None = None) -> int:
    """Initialize the per-request LLM budgets. Returns the call cap that was set."""
    settings = current_settings()
    cap = max_calls if max_calls is not None else settings.max_llm_calls_per_request
    token_cap = max_tokens if max_tokens is not None else settings.max_tokens_per_request
    _llm_budget_ctx.set(
        {
            "remaining": cap,
            "tokens_used": 0,
            "token_cap": token_cap,
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cost_usd": 0.0,
            "by_model": {},
        }
    )
    return cap


def consume_llm_budget(node: str) -> None:
    """
    Decrement the per-request call budget before issuing an LLM call.

    No-op when no budget has been initialized (unit tests / direct-call paths).
    Raises LLMBudgetExceededError if the call budget is exhausted.
    """
    budget = _llm_budget_ctx.get()
    if budget is None:
        return
    if budget["remaining"] <= 0:
        from axiom_rag_engine.config.observability import LLM_BUDGET_EXHAUSTED

        LLM_BUDGET_EXHAUSTED.labels(cap="calls").inc()
        raise LLMBudgetExceededError(
            f"LLM call budget exhausted before {node} could issue its call."
        )
    budget["remaining"] -= 1


def record_llm_usage(usage: Any, node: str, model: str | None = None) -> None:
    """
    Record token counts + cost from a completed LLM or embedding response.

    Always emits the Prometheus counters (``axiom_llm_tokens_total``,
    ``axiom_llm_cost_usd_total``) when ``model`` is provided — including for
    work outside a request budget, such as document ingestion. When a
    per-request budget is active it also accumulates the usage there and
    enforces the token cap.

    Missing provider usage is tolerated — only counters with non-zero data are
    updated. Cost is best-effort via ``litellm.completion_cost``; Ollama and
    other local backends will report 0.

    Raises LLMBudgetExceededError if the cumulative token count exceeds the cap.
    """
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0) if usage is not None else 0
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0) if usage is not None else 0
    total_tokens = int(getattr(usage, "total_tokens", 0) or 0) if usage is not None else 0
    if total_tokens == 0:
        total_tokens = prompt_tokens + completion_tokens

    # Providers without a price entry (Ollama, custom endpoints) or version-skew
    # raise inside completion_cost; a missing cost is not a request failure.
    cost_usd = 0.0
    if usage is not None and model is not None:
        with contextlib.suppress(Exception):
            import litellm

            cost_usd = float(
                litellm.completion_cost(
                    model=model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )
                or 0.0
            )

    if model is not None:
        # Label cardinality is bounded by safe_model_label. Never let metrics
        # emission break a request.
        with contextlib.suppress(Exception):
            from axiom_rag_engine.config.observability import (
                LLM_COST_USD_TOTAL,
                LLM_TOKENS_TOTAL,
                safe_model_label,
            )

            label = safe_model_label(model)
            if prompt_tokens:
                LLM_TOKENS_TOTAL.labels(model=label, kind="prompt").inc(prompt_tokens)
            if completion_tokens:
                LLM_TOKENS_TOTAL.labels(model=label, kind="completion").inc(completion_tokens)
            if cost_usd:
                LLM_COST_USD_TOTAL.labels(model=label).inc(cost_usd)

    budget = _llm_budget_ctx.get()
    if budget is None:
        return

    budget["calls"] = budget.get("calls", 0) + 1
    budget["prompt_tokens"] = budget.get("prompt_tokens", 0) + prompt_tokens
    budget["completion_tokens"] = budget.get("completion_tokens", 0) + completion_tokens
    budget["tokens_used"] = budget.get("tokens_used", 0) + total_tokens
    budget["cost_usd"] = budget.get("cost_usd", 0.0) + cost_usd

    if model is not None:
        by_model: dict[str, dict[str, Any]] = budget.setdefault("by_model", {})
        row = by_model.setdefault(
            model,
            {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0},
        )
        row["calls"] += 1
        row["prompt_tokens"] += prompt_tokens
        row["completion_tokens"] += completion_tokens
        row["cost_usd"] += cost_usd

    token_cap: int = int(budget.get("token_cap", 0) or 0)
    if token_cap > 0 and budget["tokens_used"] > token_cap:
        from axiom_rag_engine.config.observability import LLM_BUDGET_EXHAUSTED

        LLM_BUDGET_EXHAUSTED.labels(cap="tokens").inc()
        raise LLMBudgetExceededError(
            f"Token budget exceeded after {node} call: "
            f"{budget['tokens_used']} tokens used (cap {token_cap})."
        )


def get_llm_usage_snapshot() -> dict[str, Any]:
    """Return an immutable snapshot of accumulated LLM usage for this request.

    Empty snapshot (all zeros, empty by_model) when no budget has been
    initialized. Safe to call at any point — typically invoked after the
    graph has finished to attach usage to the response payload.
    """
    budget = _llm_budget_ctx.get()
    if budget is None:
        return {
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
            "by_model": {},
        }
    by_model = {m: dict(row) for m, row in (budget.get("by_model") or {}).items()}
    return {
        "calls": int(budget.get("calls", 0) or 0),
        "prompt_tokens": int(budget.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(budget.get("completion_tokens", 0) or 0),
        "total_tokens": int(budget.get("tokens_used", 0) or 0),
        "cost_usd": float(budget.get("cost_usd", 0.0) or 0.0),
        "by_model": by_model,
    }


# ---------------------------------------------------------------------------
# LLM concurrency limiters
# ---------------------------------------------------------------------------
# One asyncio.Semaphore per pool bounds in-flight calls to protect provider
# rate limits. Pools are separate so a few slow synthesis calls cannot hold
# every slot while cheap verification calls — often on another provider, and
# 10-30 per request — wait behind them.

# node -> pool; nodes not listed share the "auxiliary" pool.
_POOL_BY_NODE = {
    "synthesizer": "synthesis",
    "semantic": "verification",
    "corroboration": "verification",
    "contradiction": "verification",
}

_llm_semaphores: dict[str, asyncio.Semaphore] = {}


def llm_pool(node: str) -> str:
    """The concurrency pool a node's calls are limited in."""
    return _POOL_BY_NODE.get(node, "auxiliary")


def get_llm_semaphore(pool: str = "synthesis") -> asyncio.Semaphore:
    """Return the semaphore limiting in-flight LLM calls in ``pool``.

    Lazily instantiated so tests / callers can change
    ``AXIOM_MAX_CONCURRENT_LLM`` / ``AXIOM_MAX_CONCURRENT_VERIFIER_LLM`` via env
    before the first call. Deliberately process-wide (``get_settings``, not the
    request's): it bounds calls across every app in the process.
    """
    semaphore = _llm_semaphores.get(pool)
    if semaphore is None:
        settings = get_settings()
        limit = settings.max_concurrent_llm
        if pool == "verification" and settings.max_concurrent_verifier_llm:
            limit = settings.max_concurrent_verifier_llm
        semaphore = _llm_semaphores[pool] = asyncio.Semaphore(limit)
    return semaphore


# (schema name, JSON Schema) for structured output.
JsonSchemaSpec = tuple[str, dict[str, Any]]


@functools.lru_cache(maxsize=256)
def _supports_response_schema(model: str) -> bool:
    """Whether LiteLLM can send a JSON Schema to ``model``'s provider."""
    try:
        import litellm

        return bool(litellm.supports_response_schema(model=model))
    except Exception:
        return False


def build_completion_kwargs(
    model: str,
    messages: list[dict[str, Any]],
    temperature: float = 0.0,
    timeout: float | None = None,
    json_mode: bool = True,
    json_schema: JsonSchemaSpec | None = None,
) -> dict[str, Any]:
    """
    Build a kwargs dict for litellm.acompletion(), handling provider quirks.

    - Ollama models: injects api_base from OLLAMA_API_BASE (defaults to
      http://localhost:11434). response_format is NOT set because Ollama returns
      empty content with it; instead Ollama's native ``format`` carries the JSON
      Schema (``json_schema``) or plain ``"json"`` (``json_mode``).
    - Other providers: with ``json_schema``, a ``json_schema`` response_format when
      LiteLLM reports the model supports it; otherwise (or with only
      ``json_mode``) ``{"type": "json_object"}``.
    - Always sets a timeout (``AXIOM_LLM_TIMEOUT_SECONDS`` unless given) to
      prevent indefinite hangs.
    """
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "timeout": timeout if timeout is not None else current_settings().llm_timeout_seconds,
    }

    if model.startswith("ollama/"):
        kwargs["api_base"] = current_settings().ollama_api_base
        extra: dict[str, Any] = {}
        # Qwen3 models expose a `think` parameter to suppress chain-of-thought.
        # Other models reject it, so only set it for qwen3/* variants.
        if "qwen3" in model.lower():
            extra["think"] = False
        if json_schema is not None:
            extra["format"] = json_schema[1]
        elif json_mode:
            extra["format"] = "json"
        if extra:
            kwargs["extra_body"] = extra
    elif json_schema is not None and _supports_response_schema(model):
        name, schema = json_schema
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": name, "schema": schema},
        }
    elif json_mode or json_schema is not None:
        kwargs["response_format"] = {"type": "json_object"}

    return kwargs


# ---------------------------------------------------------------------------
# Transient-failure retries
# ---------------------------------------------------------------------------

# Status codes worth retrying: request timeout, rate limit, server errors, and
# Anthropic's 529 "overloaded".
_TRANSIENT_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504, 529})
_TRANSIENT_ERROR_NAMES = (
    "RateLimitError",
    "APIConnectionError",  # includes litellm.Timeout
    "InternalServerError",
    "ServiceUnavailableError",
    "BadGatewayError",
)
_RETRY_BASE_SECONDS = 0.5


def is_transient_llm_error(exc: BaseException) -> bool:
    """True for provider failures that a retry can fix: rate limits, timeouts,
    dropped connections, and 5xx. Auth, bad-request, and content errors are not."""
    import litellm

    transient_types = tuple(
        t
        for t in (getattr(litellm, name, None) for name in _TRANSIENT_ERROR_NAMES)
        if isinstance(t, type)
    )
    if transient_types and isinstance(exc, transient_types):
        return True
    return getattr(exc, "status_code", None) in _TRANSIENT_STATUS_CODES


def _retry_delay(exc: BaseException, attempt: int, max_wait: float) -> float:
    """Seconds to wait before retry ``attempt`` (0-based).

    Honours a provider ``Retry-After`` header when present; otherwise uses
    exponential backoff with full jitter. Always capped at ``max_wait``.
    """
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is not None:
        with contextlib.suppress(TypeError, ValueError):
            return min(max_wait, max(0.0, float(headers.get("retry-after"))))
    return random.uniform(0.0, min(max_wait, _RETRY_BASE_SECONDS * 2**attempt))  # noqa: S311


def _log_retry(
    node: str, model: str, exc: BaseException, attempt: int, of: int, delay: float
) -> None:
    from axiom_rag_engine.config.observability import LLM_RETRIES, safe_model_label

    LLM_RETRIES.labels(node=node, model=safe_model_label(model)).inc()
    logger.warning(
        "Transient %s error from %s (%s); retry %d/%d in %.1fs.",
        node,
        model,
        type(exc).__name__,
        attempt,
        of,
        delay,
    )


async def _with_retry(node: str, model: str, call: Callable[[], Awaitable[Any]]) -> Any:
    """Await ``call()`` with retries for transient provider failures.

    The concurrency semaphore is held per attempt, never across a backoff sleep,
    so a throttled call does not block other requests while it waits.
    """
    settings = current_settings()
    attempt = 0
    while True:
        try:
            async with get_llm_semaphore(llm_pool(node)):
                return await call()
        except Exception as exc:
            if attempt >= settings.llm_max_retries or not is_transient_llm_error(exc):
                raise
            delay = _retry_delay(exc, attempt, settings.llm_retry_max_wait_seconds)
            attempt += 1
            _log_retry(node, model, exc, attempt, settings.llm_max_retries, delay)
            await asyncio.sleep(delay)


def _with_retry_sync(node: str, model: str, call: Callable[[], Any]) -> Any:
    """Blocking twin of :func:`_with_retry` for code already off the event loop
    (worker threads). The asyncio semaphore cannot be used from a thread."""
    settings = current_settings()
    attempt = 0
    while True:
        try:
            return call()
        except Exception as exc:
            if attempt >= settings.llm_max_retries or not is_transient_llm_error(exc):
                raise
            delay = _retry_delay(exc, attempt, settings.llm_retry_max_wait_seconds)
            attempt += 1
            _log_retry(node, model, exc, attempt, settings.llm_max_retries, delay)
            time.sleep(delay)


# ---------------------------------------------------------------------------
# The single LLM call path
# ---------------------------------------------------------------------------


async def call_llm(
    node: str,
    model: str,
    messages: list[dict[str, Any]],
    *,
    temperature: float = 0.0,
    json_mode: bool = True,
    json_schema: JsonSchemaSpec | None = None,
    max_tokens: int | None = None,
) -> str:
    """Issue one chat completion under the shared call policy; return its text.

    Every LLM call in the pipeline goes through here so the policy is uniform:
      - the per-request call budget is consumed *before* the provider is hit
        (``LLMBudgetExceededError`` propagates unwrapped — callers decide whether
        to degrade or abort, and the endpoint maps it to HTTP 422);
      - the node's concurrency pool bounds in-flight calls (``llm_pool``);
      - transient provider failures (rate limit, timeout, 5xx) are retried up
        to ``AXIOM_LLM_MAX_RETRIES`` times with backoff, within one budget unit;
      - duration, tokens, and cost are recorded under ``node``;
      - provider quirks come from :func:`build_completion_kwargs`; pass
        ``json_schema`` to request schema-conforming output where supported.

    Non-transient provider errors, and transient ones that outlast the retries,
    propagate unchanged. ``None`` content becomes ``""``.
    """
    import litellm

    from axiom_rag_engine.config.observability import (
        LLM_CALL_DURATION,
        get_tracer,
        safe_model_label,
    )

    kwargs = build_completion_kwargs(
        model=model,
        messages=messages,
        temperature=temperature,
        json_mode=json_mode,
        json_schema=json_schema,
    )
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    with get_tracer().start_as_current_span(f"{node}.llm_call", attributes={"model": model}):
        consume_llm_budget(node)
        start = time.monotonic()
        response = await _with_retry(node, model, lambda: litellm.acompletion(**kwargs))
        LLM_CALL_DURATION.labels(node=node, model=safe_model_label(model)).observe(
            time.monotonic() - start
        )
        record_llm_usage(getattr(response, "usage", None), node, model)
    return str(response.choices[0].message.content or "")


# ---------------------------------------------------------------------------
# Embedding calls — same budget, concurrency, retry, and usage policy
# ---------------------------------------------------------------------------


async def call_embedding(node: str, model: str, kwargs: dict[str, Any]) -> Any:
    """``litellm.aembedding`` under the shared call policy (see :func:`call_llm`).

    One batched embedding request consumes one unit of per-request budget; its
    tokens and cost are recorded under ``node``.
    """
    import litellm

    from axiom_rag_engine.config.observability import (
        EMBEDDING_INPUTS,
        LLM_CALL_DURATION,
        safe_model_label,
    )

    consume_llm_budget(node)
    EMBEDDING_INPUTS.labels(model=safe_model_label(model)).inc(len(kwargs.get("input") or []))
    start = time.monotonic()
    response = await _with_retry(node, model, lambda: litellm.aembedding(**kwargs))
    LLM_CALL_DURATION.labels(node=node, model=safe_model_label(model)).observe(
        time.monotonic() - start
    )
    record_llm_usage(getattr(response, "usage", None), node, model)
    return response


def call_embedding_sync(node: str, model: str, kwargs: dict[str, Any]) -> Any:
    """Blocking :func:`call_embedding` for worker threads (``asyncio.to_thread``
    copies the request context, so the request budget still applies)."""
    import litellm

    from axiom_rag_engine.config.observability import (
        EMBEDDING_INPUTS,
        LLM_CALL_DURATION,
        safe_model_label,
    )

    consume_llm_budget(node)
    EMBEDDING_INPUTS.labels(model=safe_model_label(model)).inc(len(kwargs.get("input") or []))
    start = time.monotonic()
    response = _with_retry_sync(node, model, lambda: litellm.embedding(**kwargs))
    LLM_CALL_DURATION.labels(node=node, model=safe_model_label(model)).observe(
        time.monotonic() - start
    )
    record_llm_usage(getattr(response, "usage", None), node, model)
    return response


# Caps the O(n) salvage scan so a runaway or adversarial response cannot burn
# CPU before we give up.
_MAX_JSON_SEARCH_CHARS = 200_000
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_LEADING_FENCE_RE = re.compile(r"^```(?:json)?\s*", re.IGNORECASE)
_TRAILING_FENCE_RE = re.compile(r"\s*```$")


def _extract_first_json_object(text: str) -> str | None:
    """Quote-aware balanced-brace scan for the first ``{...}`` block."""
    if len(text) > _MAX_JSON_SEARCH_CHARS:
        return None
    depth = 0
    start = -1
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if esc:
            esc = False
            continue
        if ch == "\\" and in_str:
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start != -1:
                return text[start : i + 1]
    return None


def parse_json_object(raw: str) -> dict[str, Any]:
    """Parse an LLM reply that should be a single JSON object.

    Tolerates the common failure shapes: ``<think>`` blocks (Qwen-family),
    markdown fences, and prose around the object (balanced-brace salvage).
    Raises ``ValueError`` when no JSON object can be recovered.
    """
    clean = _THINK_BLOCK_RE.sub("", raw.strip())
    clean = _LEADING_FENCE_RE.sub("", clean.strip())
    clean = _TRAILING_FENCE_RE.sub("", clean.strip())
    if len(clean) > _MAX_JSON_SEARCH_CHARS:
        raise ValueError("LLM response is not valid JSON: response too large to parse.")

    try:
        data = json.loads(clean)
    except json.JSONDecodeError as first_err:
        salvaged = _extract_first_json_object(clean)
        if salvaged is None:
            raise ValueError(f"LLM response is not valid JSON: {first_err}") from first_err
        try:
            data = json.loads(salvaged)
        except json.JSONDecodeError as exc:
            raise ValueError(f"LLM response is not valid JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"LLM response must be a JSON object, got {type(data).__name__}.")
    return data
