"""
Shared LLM call path — one budget/semaphore/usage/error policy for every node.

Before this helper, five call sites each re-implemented budget accounting and
JSON salvage, and handled a budget overrun differently (the synthesizer wrapped
it in a RuntimeError, turning a documented HTTP 429 into a 500).
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from axiom_rag_engine.utils.llm import (
    LLMBudgetExceededError,
    call_llm,
    get_llm_usage_snapshot,
    parse_json_object,
    reset_llm_budget,
)


def _response(content: str, prompt_tokens: int = 7, completion_tokens: int = 3) -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content=content))]
    response.usage = MagicMock(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    return response


class TestCallLlm:
    async def test_returns_message_content_and_records_usage(self) -> None:
        reset_llm_budget(max_calls=5)
        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=_response("hi")):
            content = await call_llm("semantic", "gpt-4o-mini", [{"role": "user", "content": "x"}])
        assert content == "hi"
        usage = get_llm_usage_snapshot()
        assert usage["calls"] == 1
        assert usage["prompt_tokens"] == 7

    async def test_raises_budget_error_before_calling_the_provider(self) -> None:
        reset_llm_budget(max_calls=0)
        with (
            patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm,
            pytest.raises(LLMBudgetExceededError),
        ):
            await call_llm("semantic", "gpt-4o-mini", [{"role": "user", "content": "x"}])
        mock_llm.assert_not_called()

    async def test_passes_json_mode_and_max_tokens(self) -> None:
        reset_llm_budget(max_calls=5)
        with patch(
            "litellm.acompletion", new_callable=AsyncMock, return_value=_response("2")
        ) as mock_llm:
            await call_llm("reranker", "gpt-4o-mini", [], json_mode=False, max_tokens=16)
        kwargs = mock_llm.call_args.kwargs
        assert "response_format" not in kwargs
        assert kwargs["max_tokens"] == 16

    async def test_none_content_becomes_empty_string(self) -> None:
        reset_llm_budget(max_calls=5)
        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=_response(None)):  # type: ignore[arg-type]
            assert await call_llm("semantic", "m", []) == ""


class TestParseJsonObject:
    def test_plain_object(self) -> None:
        assert parse_json_object('{"a": 1}') == {"a": 1}

    def test_strips_think_block_and_fences(self) -> None:
        raw = '<think>hmm {"no": 1}</think>\n```json\n{"a": 2}\n```'
        assert parse_json_object(raw) == {"a": 2}

    def test_salvages_object_wrapped_in_prose(self) -> None:
        assert parse_json_object('Sure! Here it is: {"a": "b}"} hope that helps') == {"a": "b}"}

    def test_rejects_non_object_json(self) -> None:
        with pytest.raises(ValueError, match="JSON object"):
            parse_json_object("[1, 2, 3]")

    def test_rejects_garbage(self) -> None:
        with pytest.raises(ValueError, match="not valid JSON"):
            parse_json_object("the claim looks faithful")

    def test_refuses_to_scan_pathologically_large_input(self) -> None:
        with pytest.raises(ValueError):
            parse_json_object("x" * 300_000 + '{"a": 1}')


class TestBudgetExhaustionIsHttp429:
    def test_synthesizer_budget_overrun_returns_429(self, client, monkeypatch) -> None:
        from axiom_rag_engine.nodes.retriever import MockSearchBackend, set_search_backend

        monkeypatch.setenv("AXIOM_MAX_LLM_CALLS_PER_REQUEST", "1")
        from axiom_rag_engine.config.settings import get_settings

        get_settings.cache_clear()
        set_search_backend(
            MockSearchBackend(
                [
                    {
                        "url": "https://example.com/a",
                        "title": "A",
                        "content": (
                            "Solid-state batteries replace liquid electrolytes with solid "
                            "ceramics, which improves thermal stability."
                        ),
                    }
                ]
            )
        )

        async def fake_llm(**kwargs: Any) -> MagicMock:
            return _response("not json")  # forces a parse retry -> second call -> over budget

        with patch("litellm.acompletion", side_effect=fake_llm):
            resp = client.post(
                "/v1/synthesize",
                json={"request_id": "budget-429", "user_query": "solid-state batteries"},
            )
        assert resp.status_code == 429
        body = resp.json()
        assert body["status"] == "error"
        assert json.dumps(body)  # still a structured AxiomResponse
