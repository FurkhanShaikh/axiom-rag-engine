"""
Axiom Engine — failure types and the client-facing ``error_type`` vocabulary.

A request that fails before any answer is verified ends with ``status="error"``
and an ``error_type`` (the same vocabulary in JSON responses and SSE error
frames), so a provider outage, an exhausted budget, and an engine bug are
distinguishable without server logs. A failure *after* a verified pass is not an
error at all: the best pass is returned with ``halt_reason`` set (graph.py).
"""

from __future__ import annotations

from typing import Literal

from axiom_rag_engine.utils.llm import LLMBudgetExceededError

ErrorType = Literal[
    "budget_exceeded",
    "deadline_exceeded",
    "llm_unavailable",
    "llm_output_unusable",
    "internal",
]

# Why a run stopped early but still returned its best verified pass.
HaltReason = Literal["node_error", "synthesizer_gave_up", "deadline"]


class SynthesizerUnavailableError(RuntimeError):
    """The synthesizer's LLM provider failed after call_llm's retries."""


class SynthesizerOutputError(RuntimeError):
    """The synthesizer's output stayed unusable after every parse retry."""


def error_type_for(exc: BaseException) -> ErrorType:
    """Classify a pipeline failure for the client."""
    from axiom_rag_engine.graph import PipelineDeadlineError  # graph imports the nodes

    if isinstance(exc, PipelineDeadlineError):
        return "deadline_exceeded"
    if isinstance(exc, LLMBudgetExceededError):
        return "budget_exceeded"
    if isinstance(exc, SynthesizerUnavailableError):
        return "llm_unavailable"
    if isinstance(exc, SynthesizerOutputError):
        return "llm_output_unusable"
    return "internal"


def is_upstream_failure(exc: BaseException) -> bool:
    """True when the failure was the LLM provider's (HTTP 502), not the engine's."""
    return isinstance(exc, SynthesizerUnavailableError | SynthesizerOutputError)
