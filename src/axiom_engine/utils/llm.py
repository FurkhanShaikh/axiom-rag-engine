"""
Axiom Engine — LLM utility helpers.

Centralises provider-specific quirks (Ollama api_base injection,
response_format gating) and enforces a consistent timeout across all
litellm.acompletion() call sites.
"""

from __future__ import annotations

import asyncio
import contextvars
import os
from typing import Any

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
# Call budget defaults to 64: ~3 rewrite loops x (1 synthesizer + up to
# ~15 semantic citations) leaves a safe margin.
# Token budget defaults to 0 (unlimited) — set AXIOM_MAX_TOKENS_PER_REQUEST
# to a dollar-equivalent ceiling for your deployment.
_DEFAULT_MAX_LLM_CALLS = int(os.environ.get("AXIOM_MAX_LLM_CALLS_PER_REQUEST", "64"))
_DEFAULT_MAX_TOKENS = int(os.environ.get("AXIOM_MAX_TOKENS_PER_REQUEST", "0"))

# ContextVar holds {"remaining": N, "tokens_used": 0, "token_cap": M} so all
# tasks created from the same request coroutine share one counter object.
_llm_budget_ctx: contextvars.ContextVar[dict[str, int] | None] = contextvars.ContextVar(
    "axiom_llm_budget", default=None
)


class LLMBudgetExceededError(RuntimeError):
    """Raised when a single request exhausts its LLM call or token budget."""


def reset_llm_budget(max_calls: int | None = None, max_tokens: int | None = None) -> int:
    """Initialize the per-request LLM budgets. Returns the call cap that was set."""
    cap = max_calls if max_calls is not None else _DEFAULT_MAX_LLM_CALLS
    token_cap = max_tokens if max_tokens is not None else _DEFAULT_MAX_TOKENS
    _llm_budget_ctx.set({"remaining": cap, "tokens_used": 0, "token_cap": token_cap})
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


def record_llm_usage(usage: Any, node: str) -> None:
    """
    Accumulate token counts from a completed LLM response and enforce the token cap.

    Call this immediately after a successful ``litellm.acompletion()`` call:
        record_llm_usage(response.usage, "synthesizer")

    No-op when:
      - No budget has been initialized (unit tests / direct-call paths).
      - ``usage`` is None (provider did not return usage metadata).
      - Token cap is 0 (unlimited).

    Raises LLMBudgetExceededError if the cumulative token count exceeds the cap.
    """
    budget = _llm_budget_ctx.get()
    if budget is None or usage is None:
        return
    total_tokens: int = int(getattr(usage, "total_tokens", 0) or 0)
    budget["tokens_used"] = budget.get("tokens_used", 0) + total_tokens
    token_cap: int = budget.get("token_cap", 0)
    if token_cap > 0 and budget["tokens_used"] > token_cap:
        raise LLMBudgetExceededError(
            f"Token budget exceeded after {node} call: "
            f"{budget['tokens_used']} tokens used (cap {token_cap})."
        )


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

_MAX_CONCURRENT_LLM = int(os.environ.get("AXIOM_MAX_CONCURRENT_LLM", "5"))
_llm_semaphore: asyncio.Semaphore = asyncio.Semaphore(_MAX_CONCURRENT_LLM)


def get_llm_semaphore() -> asyncio.Semaphore:
    """Return the shared asyncio.Semaphore for LLM concurrency limiting."""
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
        kwargs["api_base"] = os.environ.get("OLLAMA_API_BASE", "http://localhost:11434")
        # Disable chain-of-thought thinking for Qwen3-family models.
        kwargs["extra_body"] = {"think": False}
    elif json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    return kwargs
