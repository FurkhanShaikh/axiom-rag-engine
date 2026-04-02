"""
Axiom Engine — LLM utility helpers.

Centralises provider-specific quirks (Ollama api_base injection,
response_format gating) and enforces a consistent timeout across all
litellm.completion() call sites.
"""

from __future__ import annotations

import os
from typing import Any

# Default timeout for all LLM calls.  Local models (Ollama) on CPU-only hardware
# can be slow on large prompts — 600 s is the ceiling; cloud models finish in <10 s.
_DEFAULT_TIMEOUT: int = 600


def build_completion_kwargs(
    model: str,
    messages: list[dict[str, Any]],
    temperature: float = 0.0,
    timeout: int = _DEFAULT_TIMEOUT,
    json_mode: bool = True,
) -> dict[str, Any]:
    """
    Build a kwargs dict for litellm.completion(), handling provider quirks.

    - Ollama models: injects api_base from OLLAMA_API_BASE env var (defaults
      to http://localhost:11434). response_format is NOT set because Ollama
      returns empty content when it is.
    - All other providers: sets response_format={"type": "json_object"} when
      json_mode=True (OpenAI-compatible structured output).
    - Always sets timeout to prevent indefinite hangs.

    Args:
        model:       LiteLLM model identifier (e.g. "gpt-4o-mini", "ollama/qwen3:9b").
        messages:    Chat messages list.
        temperature: Sampling temperature (default 0.0 for deterministic output).
        timeout:     Request timeout in seconds (default 120).
        json_mode:   Whether to request JSON structured output (ignored for Ollama).

    Returns:
        Dict ready to be unpacked into litellm.completion(**kwargs).
    """
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "timeout": timeout,
    }

    if model.startswith("ollama/"):
        kwargs["api_base"] = os.environ.get(
            "OLLAMA_API_BASE", "http://localhost:11434"
        )
        # Disable chain-of-thought thinking for Qwen3-family models.
        # `think` is a top-level Ollama API parameter; other models ignore it.
        kwargs["extra_body"] = {"think": False}
    elif json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    return kwargs
