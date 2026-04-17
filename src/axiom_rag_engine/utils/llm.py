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
from typing import Any

from axiom_rag_engine.config.settings import get_settings

# Default timeout for all LLM calls.  Local models (Ollama) on CPU-only hardware
# can be slow on large prompts — 600 s is the ceiling; cloud models finish in <10 s.
_DEFAULT_TIMEOUT: int = 600

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
    settings = get_settings()
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
        raise LLMBudgetExceededError(
            f"LLM call budget exhausted before {node} could issue its call."
        )
    budget["remaining"] -= 1


def record_llm_usage(usage: Any, node: str, model: str | None = None) -> None:
    """
    Accumulate token counts + cost from a completed LLM response and enforce
    the token cap. Also emits Prometheus counters (``axiom_llm_tokens_total``,
    ``axiom_llm_cost_usd_total``) when ``model`` is provided.

    Call this immediately after a successful ``litellm.acompletion()`` call:
        record_llm_usage(response.usage, "synthesizer", model)

    No-op when no budget has been initialized (unit tests / direct-call paths).
    Missing provider usage is tolerated — only counters with non-zero data are
    updated. Cost is best-effort via ``litellm.completion_cost``; Ollama and
    other local backends will report 0.

    Raises LLMBudgetExceededError if the cumulative token count exceeds the cap.
    """
    budget = _llm_budget_ctx.get()
    if budget is None:
        return

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

        # Emit Prometheus counters; label cardinality is bounded by safe_model_label.
        # Never let metrics emission break a request.
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

    token_cap: int = int(budget.get("token_cap", 0) or 0)
    if token_cap > 0 and budget["tokens_used"] > token_cap:
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
# Shared LLM concurrency limiter
# ---------------------------------------------------------------------------
# asyncio.Semaphore — limits total concurrent in-flight acompletion() calls
# across all nodes so we don't breach provider rate limits.
#
# Python 3.10+ asyncio primitives no longer bind to a running event loop at
# construction time, so it is safe to create the semaphore at module level.
# This project requires Python >= 3.11, so the old lazy-init pattern that
# worked around the "no running event loop" error is no longer needed.

_llm_semaphore: asyncio.Semaphore | None = None


def get_llm_semaphore() -> asyncio.Semaphore:
    """Return the shared asyncio.Semaphore for LLM concurrency limiting.

    Lazily instantiated so tests / callers can change
    ``AXIOM_MAX_CONCURRENT_LLM`` via env before the first call.
    """
    global _llm_semaphore
    if _llm_semaphore is None:
        _llm_semaphore = asyncio.Semaphore(get_settings().max_concurrent_llm)
    return _llm_semaphore


def build_completion_kwargs(
    model: str,
    messages: list[dict[str, Any]],
    temperature: float = 0.0,
    timeout: int = _DEFAULT_TIMEOUT,
    json_mode: bool = True,
) -> dict[str, Any]:
    """
    Build a kwargs dict for litellm.acompletion(), handling provider quirks.

    - Ollama models: injects api_base from OLLAMA_API_BASE env var (defaults
      to http://localhost:11434). response_format is NOT set because Ollama
      returns empty content when it is.
    - All other providers: sets response_format={"type": "json_object"} when
      json_mode=True (OpenAI-compatible structured output).
    - Always sets timeout to prevent indefinite hangs.
    """
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "timeout": timeout,
    }

    if model.startswith("ollama/"):
        kwargs["api_base"] = get_settings().ollama_api_base
        # Disable chain-of-thought thinking for Qwen3-family models.
        kwargs["extra_body"] = {"think": False}
    elif json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    return kwargs
