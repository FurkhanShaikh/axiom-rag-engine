"""
Fail-soft after the first verified pass — a later pass failing must not cost the
caller the answer already in hand.

Once one pass has been verified, rewrites and re-retrieval are refinement. A
failing later node (exhausted LLM budget, provider error, the synthesizer giving
up, every search erroring) used to fail the whole request with HTTP 429/500 or
turn it "unanswerable", discarding a verified first pass. The run now halts and
returns the best verified pass. Before the first verified pass there is nothing
to fall back to, so errors still propagate.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import pytest

from axiom_rag_engine.graph import build_axiom_graph
from axiom_rag_engine.marshalling import marshal_response
from axiom_rag_engine.state import make_initial_state
from axiom_rag_engine.utils.llm import (
    LLMBudgetExceededError,
    consume_llm_budget,
    reset_llm_budget,
)

_TEXT = (
    "Solid-state batteries replace liquid electrolytes with solid ceramics. "
    "This substitution significantly improves thermal stability and energy density of cells."
)
_GOOD_QUOTE = "Solid-state batteries replace liquid electrolytes with solid ceramics"
_BAD_QUOTE = "powered by tiny nuclear reactors inside every cell"

# First pass: one verifiable sentence, one hallucinated quote -> triggers a rewrite.
_FIRST_DRAFT = json.dumps(
    {
        "is_answerable": True,
        "sentences": [
            {
                "sentence_id": "s_01",
                "text": "Solid-state batteries use solid ceramics.",
                "is_cited": True,
                "citations": [
                    {
                        "citation_id": "cite_1",
                        "chunk_id": "doc_1_chunk_A",
                        "exact_source_quote": _GOOD_QUOTE,
                    }
                ],
            },
            {
                "sentence_id": "s_02",
                "text": "They run on nuclear power.",
                "is_cited": True,
                "citations": [
                    {
                        "citation_id": "cite_2",
                        "chunk_id": "doc_1_chunk_A",
                        "exact_source_quote": _BAD_QUOTE,
                    }
                ],
            },
        ],
    }
)
_GIVE_UP = json.dumps({"is_answerable": False, "sentences": []})
_SEMANTIC_PASS = json.dumps({"semantic_check": "passed", "failure_reason": None})


class _Backend:
    """Search backend that can be told to fail after its first call."""

    def __init__(self, fail_after_first: bool = False) -> None:
        self.calls = 0
        self.fail_after_first = fail_after_first

    def search(self, query: str) -> list[dict[str, Any]]:
        self.calls += 1
        if self.fail_after_first and self.calls > 1:
            raise ConnectionError("search provider down")
        return [{"url": "https://example.com/a", "title": "A", "content": _TEXT}]


def _llm(synth_replies: list[Any]) -> Any:
    """Fake call_llm: synthesizer replies in order (an Exception is raised),
    semantic checks always pass. Consumes budget like the real call path."""
    replies = list(synth_replies)

    async def _fake(node: str, model: str, messages: Any, **_: Any) -> str:
        consume_llm_budget(node)
        if node == "synthesizer":
            reply = replies.pop(0)
            if isinstance(reply, Exception):
                raise reply
            return str(reply)
        return _SEMANTIC_PASS

    return _fake


async def _run(
    synth_replies: list[Any],
    *,
    budget: int = 50,
    backend: Any = None,
    stages: dict[str, Any] | None = None,
) -> dict[str, Any]:
    state = make_initial_state(
        request_id="req_fail_soft",
        user_query="What are solid-state batteries?",
        app_config={},
        models_config={"synthesizer": "mock/s", "verifier": "mock/v"},
        pipeline_config={"stages": {"semantic_verification_enabled": True, **(stages or {})}},
    )
    reset_llm_budget(max_calls=budget)
    fake = _llm(synth_replies)
    with (
        patch("axiom_rag_engine.nodes.synthesizer.call_llm", fake),
        patch("axiom_rag_engine.nodes.semantic.call_llm", fake),
    ):
        return await build_axiom_graph().ainvoke(
            state, config={"configurable": {"search_backend": backend or _Backend()}}
        )


def _tiers(result: dict[str, Any]) -> list[tuple[str, int]]:
    return [(s["sentence_id"], s["verification"]["tier"]) for s in result["final_sentences"]]


def _halt_events(result: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        e for e in result["audit_trail"] if e["event_type"] == "pipeline_halted_best_pass_returned"
    ]


class TestLaterPassFailureReturnsBestPass:
    async def test_budget_exhausted_on_rewrite(self) -> None:
        # Budget 2 = first synthesis + one semantic check; the rewrite has none left.
        result = await _run([_FIRST_DRAFT, _FIRST_DRAFT], budget=2)

        assert result["halt_reason"] == "node_error"
        assert _tiers(result) == [("s_01", 3), ("s_02", 5)]
        assert _halt_events(result)[0]["payload"]["error_type"] == "LLMBudgetExceededError"
        response = marshal_response("req_fail_soft", result)
        assert response.status == "partial"
        assert response.is_answerable is True

    async def test_provider_error_on_rewrite(self) -> None:
        result = await _run([_FIRST_DRAFT, RuntimeError("provider 529 overloaded")])

        assert result["halt_reason"] == "node_error"
        assert _tiers(result) == [("s_01", 3), ("s_02", 5)]
        # Error type only — provider messages can carry account details.
        payload = _halt_events(result)[0]["payload"]
        assert payload["error_type"] == "RuntimeError"
        assert "529" not in json.dumps(payload)

    async def test_synthesizer_gives_up_on_rewrite(self) -> None:
        result = await _run([_FIRST_DRAFT, _GIVE_UP])

        assert result["halt_reason"] == "synthesizer_gave_up"
        assert result["is_answerable"] is True
        assert _tiers(result) == [("s_01", 3), ("s_02", 5)]
        assert marshal_response("req_fail_soft", result).status == "partial"

    async def test_all_searches_fail_on_re_retrieval(self) -> None:
        # No rewrites: the first pass's Tier 5 goes straight to re-retrieval,
        # where every search now errors.
        result = await _run(
            [_FIRST_DRAFT],
            backend=_Backend(fail_after_first=True),
            stages={"max_rewrite_loops": 0, "max_retrieval_retries": 1},
        )

        assert result["halt_reason"] == "node_error"
        assert _tiers(result) == [("s_01", 3), ("s_02", 5)]
        assert _halt_events(result)[0]["node"] == "re_retriever"


class TestFirstPassFailureStillPropagates:
    async def test_provider_error_before_any_verified_pass(self) -> None:
        with pytest.raises(RuntimeError, match="Synthesizer stage failed"):
            await _run([RuntimeError("provider down")])

    async def test_budget_exhausted_before_any_verified_pass(self) -> None:
        with pytest.raises(LLMBudgetExceededError):
            await _run([_FIRST_DRAFT], budget=0)


class TestHealthyRunIsUnchanged:
    async def test_no_halt_when_rewrite_succeeds(self) -> None:
        good = json.loads(_FIRST_DRAFT)
        good["sentences"] = good["sentences"][:1]
        result = await _run([_FIRST_DRAFT, json.dumps(good)])

        assert result["halt_reason"] is None
        assert _halt_events(result) == []
        assert _tiers(result) == [("s_01", 3)]
