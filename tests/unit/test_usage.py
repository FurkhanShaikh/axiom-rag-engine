"""Tests for the per-request LLM usage accounting + UsageSummary response field."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from axiom_rag_engine.marshalling import (
    _usage_summary_from_snapshot,
    make_error_response,
    marshal_response,
)
from axiom_rag_engine.models import UsageSummary
from axiom_rag_engine.utils.llm import (
    get_llm_usage_snapshot,
    record_llm_usage,
    reset_llm_budget,
)


def _usage(prompt: int, completion: int, total: int | None = None) -> SimpleNamespace:
    """Build a minimal stand-in for a litellm response.usage object."""
    return SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion if total is None else total,
    )


# ---------------------------------------------------------------------------
# Accumulator semantics
# ---------------------------------------------------------------------------


def test_snapshot_is_empty_before_budget_reset() -> None:
    snap = get_llm_usage_snapshot()
    assert snap == {
        "calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cost_usd": 0.0,
        "by_model": {},
    }


def test_record_accumulates_tokens_and_per_model_breakdown() -> None:
    reset_llm_budget(max_calls=10, max_tokens=0)

    record_llm_usage(_usage(100, 50), "synthesizer", model="gpt-4o-mini")
    record_llm_usage(_usage(60, 20), "semantic", model="gpt-4o-mini")
    record_llm_usage(_usage(30, 15), "synthesizer", model="ollama/qwen3.5:9b")

    snap = get_llm_usage_snapshot()
    assert snap["calls"] == 3
    assert snap["prompt_tokens"] == 190
    assert snap["completion_tokens"] == 85
    assert snap["total_tokens"] == 275

    by_model = snap["by_model"]
    assert set(by_model.keys()) == {"gpt-4o-mini", "ollama/qwen3.5:9b"}
    assert by_model["gpt-4o-mini"]["calls"] == 2
    assert by_model["gpt-4o-mini"]["prompt_tokens"] == 160
    assert by_model["gpt-4o-mini"]["completion_tokens"] == 70
    assert by_model["ollama/qwen3.5:9b"]["calls"] == 1


def test_missing_total_tokens_is_derived_from_prompt_plus_completion() -> None:
    reset_llm_budget()
    # total_tokens=0 forces the fallback path.
    record_llm_usage(_usage(100, 50, total=0), "synthesizer", model="gpt-4o-mini")
    assert get_llm_usage_snapshot()["total_tokens"] == 150


def test_none_usage_is_tolerated_as_noop() -> None:
    reset_llm_budget()
    record_llm_usage(None, "synthesizer", model="gpt-4o-mini")
    snap = get_llm_usage_snapshot()
    # Still counts the call but contributes no tokens.
    assert snap["calls"] == 1
    assert snap["prompt_tokens"] == 0
    assert snap["completion_tokens"] == 0


def test_record_without_model_skips_by_model_breakdown() -> None:
    reset_llm_budget()
    record_llm_usage(_usage(10, 5), "synthesizer", model=None)
    snap = get_llm_usage_snapshot()
    assert snap["calls"] == 1
    assert snap["prompt_tokens"] == 10
    assert snap["by_model"] == {}


# ---------------------------------------------------------------------------
# UsageSummary marshalling
# ---------------------------------------------------------------------------


def test_usage_summary_is_none_when_no_calls_observed() -> None:
    assert _usage_summary_from_snapshot(None) is None
    assert (
        _usage_summary_from_snapshot(
            {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "by_model": {}}
        )
        is None
    )


def test_usage_summary_is_validated_from_snapshot() -> None:
    snap = {
        "calls": 2,
        "prompt_tokens": 100,
        "completion_tokens": 40,
        "total_tokens": 140,
        "cost_usd": 0.00042,
        "by_model": {
            "gpt-4o-mini": {
                "calls": 2,
                "prompt_tokens": 100,
                "completion_tokens": 40,
                "cost_usd": 0.00042,
            },
        },
    }
    summary = _usage_summary_from_snapshot(snap)
    assert isinstance(summary, UsageSummary)
    assert summary.total_tokens == 140
    assert summary.by_model["gpt-4o-mini"].calls == 2
    assert summary.cost_usd == pytest.approx(0.00042)


def test_marshal_response_attaches_usage_summary() -> None:
    graph_result = {"is_answerable": True, "final_sentences": []}
    snap = {
        "calls": 1,
        "prompt_tokens": 50,
        "completion_tokens": 20,
        "total_tokens": 70,
        "cost_usd": 0.0,
        "by_model": {
            "ollama/qwen3.5:9b": {
                "calls": 1,
                "prompt_tokens": 50,
                "completion_tokens": 20,
                "cost_usd": 0.0,
            },
        },
    }
    resp = marshal_response("req_x", graph_result, include_debug=False, usage_snapshot=snap)
    assert resp.usage is not None
    assert resp.usage.calls == 1
    assert "ollama/qwen3.5:9b" in resp.usage.by_model


def test_error_response_also_reports_usage() -> None:
    snap = {
        "calls": 3,
        "prompt_tokens": 500,
        "completion_tokens": 0,
        "total_tokens": 500,
        "cost_usd": 0.0,
        "by_model": {},
    }
    resp = make_error_response("req_y", RuntimeError("boom"), usage_snapshot=snap)
    assert resp.status == "error"
    assert resp.usage is not None
    assert resp.usage.calls == 3
