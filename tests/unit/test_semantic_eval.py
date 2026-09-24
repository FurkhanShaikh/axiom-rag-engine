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
