"""
The semantic eval measures the production retry behaviour.

The eval used to retry rate limits itself (4 times, up to 40 s) while
production retried nothing, so its 0% error rate hid a real production failure
mode. It now calls the production verifier directly, which retries through
``call_llm``, and reports transient give-ups and provider retries separately.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import litellm
import pytest

from axiom_rag_engine.config.settings import get_settings

_EVALS = Path(__file__).resolve().parents[2] / "evals"
sys.path.insert(0, str(_EVALS))
_spec = importlib.util.spec_from_file_location(
    "axiom_semantic_eval", _EVALS / "semantic_verifier_eval.py"
)
assert _spec and _spec.loader
seval = importlib.util.module_from_spec(_spec)
sys.modules["axiom_semantic_eval"] = seval
_spec.loader.exec_module(seval)

_EXAMPLE = seval.Example(
    example_id="ex1",
    claim="Solid-state batteries use ceramics.",
    quote="Solid-state batteries replace liquid electrolytes with solid ceramics.",
    chunk="Solid-state batteries replace liquid electrolytes with solid ceramics.",
    label="SUPPORT",
)


def _reply(content: str) -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content=content))]
    return response


def _rate_limited() -> litellm.RateLimitError:
    return litellm.RateLimitError("slow down", llm_provider="openai", model="m")


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AXIOM_LLM_RETRY_MAX_WAIT_SECONDS", "0")
    get_settings.cache_clear()


async def test_rate_limit_is_retried_by_the_production_path() -> None:
    llm = AsyncMock(side_effect=[_rate_limited(), _reply('{"semantic_check": "passed"}')])
    with patch("litellm.acompletion", llm):
        record = await seval._judge(_EXAMPLE, "mock/v")
    assert record.got == "passed"
    assert record.correct
    assert llm.await_count == 2


async def test_rate_limit_outlasting_the_policy_is_a_transient_error() -> None:
    llm = AsyncMock(side_effect=[_rate_limited() for _ in range(10)])
    with patch("litellm.acompletion", llm):
        record = await seval._judge(_EXAMPLE, "mock/v")
    assert record.got == "error"
    assert record.transient
    assert llm.await_count == get_settings().llm_max_retries + 1

    summary = seval.summarize([record])
    assert summary["errors"] == 1
    assert summary["transient_errors"] == 1


def test_eval_has_no_retry_of_its_own() -> None:
    assert not hasattr(seval, "_verify_with_retry")


def test_summary_reports_wilson_intervals() -> None:
    def rec(label: str, got: str) -> object:
        expected = "passed" if label == "SUPPORT" else "failed"
        return seval.Record("e", label, expected, got, got == expected, None, 0.0)

    # 12 of 14 unfaithful claims caught, 6 of 16 faithful claims flagged.
    records = (
        [rec("CONTRADICT", "failed")] * 12
        + [rec("CONTRADICT", "passed")] * 2
        + [rec("SUPPORT", "failed")] * 6
        + [rec("SUPPORT", "passed")] * 10
    )
    ci = seval.summarize(records)["ci95"]
    low, high = ci["unfaithful_recall"]
    assert low < 12 / 14 < high
    assert 0.55 < low < 0.65  # n = 14 leaves a wide interval
    low, high = ci["unfaithful_precision"]
    assert low < 12 / 18 < high


def _summary(n_per_cell: int, errors: int = 0) -> dict:
    def rec(label: str, got: str) -> object:
        expected = "passed" if label == "SUPPORT" else "failed"
        return seval.Record("e", label, expected, got, got == expected, None, 0.0)

    records = (
        [rec("CONTRADICT", "failed")] * (4 * n_per_cell)
        + [rec("CONTRADICT", "passed")] * n_per_cell
        + [rec("SUPPORT", "failed")] * n_per_cell
        + [rec("SUPPORT", "passed")] * (4 * n_per_cell)
        + [rec("SUPPORT", "error")] * errors
    )
    return seval.summarize(records)


def test_record_writes_an_enforced_baseline_at_the_lower_bounds(tmp_path: Path) -> None:
    path = tmp_path / "semantic-verifier.json"
    path.write_text(seval.BASELINE_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    summary = _summary(n_per_cell=20, errors=2)  # 202 examples

    data = seval.record_baseline(path, summary, "openrouter/openai/gpt-4o-mini", "2026-09-24")

    assert data == seval.json.loads(path.read_text(encoding="utf-8"))
    assert data["enforcement"] == "enforce"
    assert data["recorded_at"] == "2026-09-24"
    assert data["model"] == "openrouter/openai/gpt-4o-mini"
    metrics = data["metrics"]
    assert metrics["unfaithful_recall"]["floor"] == summary["ci95"]["unfaithful_recall"][0]
    assert metrics["accuracy"]["floor"] == summary["ci95"]["accuracy"][0]
    assert metrics["unfaithful_recall"]["floor"] < summary["unfaithful_recall"]
    assert metrics["error_rate"]["ceiling"] >= 0.05
    # The run it was recorded from passes its own gate.
    report = seval.gate.evaluate_gate(seval._gate_metrics(summary), data)
    assert report.metrics_ok


def test_record_refuses_small_samples(tmp_path: Path) -> None:
    path = tmp_path / "semantic-verifier.json"
    path.write_text(seval.BASELINE_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(ValueError, match="at least 200"):
        seval.record_baseline(path, _summary(n_per_cell=3), "gpt-4o-mini", "2026-09-24")


def test_routed_model_is_the_same_model() -> None:
    assert seval.model_family("openrouter/openai/gpt-4o-mini") == "gpt-4o-mini"
    enforced = {"model": "gpt-4o-mini", "enforcement": "enforce", "metrics": {}}
    assert seval.baseline_for("openrouter/openai/gpt-4o-mini", enforced) is enforced


def test_another_models_baseline_is_not_enforced() -> None:
    enforced = {"model": "gpt-4o-mini", "enforcement": "enforce", "metrics": {}}
    assert seval.baseline_for("claude-haiku-4-5", enforced)["enforcement"] == "report_only"


def test_default_sample_is_large_enough_to_record() -> None:
    assert seval.MIN_RECORD_EXAMPLES == 200
