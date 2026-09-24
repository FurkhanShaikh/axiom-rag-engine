"""
Transient provider failures are retried inside ``call_llm``.

Without retries a single rate-limit or overloaded response failed a whole
synthesis pass, or marked a verified citation "unverified". Retries cover rate
limits, timeouts, dropped connections and 5xx only; auth and bad-request errors
fail at once. A retried call consumes one unit of per-request budget, not one
per attempt.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import litellm
import pytest

from axiom_rag_engine.config.settings import get_settings
from axiom_rag_engine.utils.llm import (
    _retry_delay,
    call_llm,
    get_llm_usage_snapshot,
    is_transient_llm_error,
    reset_llm_budget,
)


def _ok(content: str = "ok") -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content=content))]
    response.usage = MagicMock(prompt_tokens=5, completion_tokens=2, total_tokens=7)
    return response


def _rate_limited() -> litellm.RateLimitError:
    return litellm.RateLimitError("slow down", llm_provider="openai", model="gpt-4o-mini")


def _overloaded() -> litellm.InternalServerError:
    return litellm.InternalServerError("overloaded", llm_provider="anthropic", model="claude")


def _auth_error() -> litellm.AuthenticationError:
    return litellm.AuthenticationError("bad key", llm_provider="openai", model="gpt-4o-mini")


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """Zero backoff so retries run instantly."""
    monkeypatch.setenv("AXIOM_LLM_RETRY_MAX_WAIT_SECONDS", "0")
    get_settings.cache_clear()


def _messages() -> list[dict[str, Any]]:
    return [{"role": "user", "content": "hi"}]


class TestRetries:
    async def test_rate_limit_then_success(self) -> None:
        reset_llm_budget(max_calls=5)
        mock = AsyncMock(side_effect=[_rate_limited(), _overloaded(), _ok("done")])
        with patch("litellm.acompletion", mock):
            assert await call_llm("semantic", "gpt-4o-mini", _messages()) == "done"
        assert mock.await_count == 3

    async def test_retries_consume_one_budget_unit(self) -> None:
        reset_llm_budget(max_calls=1)
        mock = AsyncMock(side_effect=[_rate_limited(), _ok()])
        with patch("litellm.acompletion", mock):
            await call_llm("semantic", "gpt-4o-mini", _messages())
        assert mock.await_count == 2
        assert get_llm_usage_snapshot()["calls"] == 1

    async def test_gives_up_after_max_retries(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AXIOM_LLM_MAX_RETRIES", "2")
        get_settings.cache_clear()
        reset_llm_budget(max_calls=5)
        mock = AsyncMock(side_effect=[_rate_limited() for _ in range(5)])
        with patch("litellm.acompletion", mock), pytest.raises(litellm.RateLimitError):
            await call_llm("semantic", "gpt-4o-mini", _messages())
        assert mock.await_count == 3  # one call + two retries

    async def test_non_transient_error_is_not_retried(self) -> None:
        reset_llm_budget(max_calls=5)
        mock = AsyncMock(side_effect=[_auth_error(), _ok()])
        with patch("litellm.acompletion", mock), pytest.raises(litellm.AuthenticationError):
            await call_llm("semantic", "gpt-4o-mini", _messages())
        assert mock.await_count == 1

    async def test_retries_can_be_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AXIOM_LLM_MAX_RETRIES", "0")
        get_settings.cache_clear()
        reset_llm_budget(max_calls=5)
        mock = AsyncMock(side_effect=[_rate_limited(), _ok()])
        with patch("litellm.acompletion", mock), pytest.raises(litellm.RateLimitError):
            await call_llm("semantic", "gpt-4o-mini", _messages())
        assert mock.await_count == 1


class TestClassification:
    @pytest.mark.parametrize(
        "exc",
        [
            _rate_limited(),
            _overloaded(),
            litellm.Timeout("timed out", model="m", llm_provider="p"),
            litellm.APIConnectionError("reset", llm_provider="p", model="m"),
            litellm.ServiceUnavailableError("down", llm_provider="p", model="m"),
        ],
    )
    def test_transient(self, exc: Exception) -> None:
        assert is_transient_llm_error(exc)

    @pytest.mark.parametrize(
        "exc",
        [
            _auth_error(),
            litellm.BadRequestError("bad", model="m", llm_provider="p"),
            ValueError("parse failure"),
            RuntimeError("anything else"),
        ],
    )
    def test_not_transient(self, exc: Exception) -> None:
        assert not is_transient_llm_error(exc)

    def test_status_code_529_is_transient(self) -> None:
        exc = RuntimeError("overloaded")
        exc.status_code = 529  # type: ignore[attr-defined]
        assert is_transient_llm_error(exc)


class TestRetryDelay:
    def test_honours_retry_after_header_capped(self) -> None:
        response = httpx.Response(
            429, headers={"retry-after": "3"}, request=httpx.Request("POST", "https://x")
        )
        exc = litellm.RateLimitError("slow", llm_provider="p", model="m", response=response)
        assert _retry_delay(exc, attempt=0, max_wait=8.0) == 3.0
        assert _retry_delay(exc, attempt=0, max_wait=1.0) == 1.0

    def test_backoff_is_bounded(self) -> None:
        for attempt in range(6):
            assert 0.0 <= _retry_delay(RuntimeError("x"), attempt, max_wait=2.0) <= 2.0
