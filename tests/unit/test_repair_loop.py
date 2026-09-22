"""
Repair loop — rewrites must be informed, correctly counted, and never make the
returned answer worse.

  * A rewrite pass shows the model its previous draft; the correction list
    ("Sentence s_02, citation cite_3 failed ...") is meaningless without it.
  * ``max_rewrite_loops=N`` allows exactly N rewrite passes per retrieval round
    (it used to allow N-1, so ``max_rewrite_loops=1`` meant no rewrite at all).
  * When every retry is exhausted, the best pass seen is returned — not
    whichever pass happened to run last.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from axiom_rag_engine import graph as graph_module
from axiom_rag_engine.models import PipelineStagesConfig
from axiom_rag_engine.nodes.synthesizer import synthesizer_node
from axiom_rag_engine.state import make_initial_state

_CHUNK_TEXT = (
    "Solid-state batteries replace liquid electrolytes with solid ceramics. "
    "This substitution significantly improves thermal stability and energy density."
)
_CHUNK = {
    "chunk_id": "doc_1_chunk_A",
    "text": _CHUNK_TEXT,
    "source_url": "https://example.com/a",
    "domain": "example.com",
    "ranking_score": 0.9,
}
_GOOD_QUOTE = "Solid-state batteries replace liquid electrolytes with solid ceramics"
_BAD_QUOTE = "Solid-state batteries are powered by tiny nuclear reactors inside"


def _sentence(i: int, quote: str) -> dict[str, Any]:
    return {
        "sentence_id": f"s_{i:02d}",
        "text": f"Claim number {i}.",
        "is_cited": True,
        "citations": [
            {"citation_id": f"cite_{i}", "chunk_id": "doc_1_chunk_A", "exact_source_quote": quote}
        ],
    }


def _state(stages: dict[str, Any] | None = None) -> dict[str, Any]:
    return dict(
        make_initial_state(
            request_id="req_loop",
            user_query="What are solid-state batteries?",
            app_config={},
            models_config={"synthesizer": "mock/s", "verifier": "mock/v"},
            pipeline_config={"stages": {"semantic_verification_enabled": False, **(stages or {})}},
        )
    )


# ---------------------------------------------------------------------------
# 1. The rewrite prompt carries the previous draft
# ---------------------------------------------------------------------------


class TestRewritePromptShowsPreviousDraft:
    async def test_rewrite_pass_includes_previous_draft(self) -> None:
        state = _state()
        state["ranked_chunks"] = [_CHUNK]
        state["draft_sentences"] = [_sentence(1, _BAD_QUOTE)]
        state["rewrite_requests"] = [
            "Sentence s_01, citation cite_1 (chunk doc_1_chunk_A): Tier 5 (hallucinated) failure"
        ]
        state["loop_count"] = 1

        reply = MagicMock()
        reply.choices = [
            MagicMock(
                message=MagicMock(
                    content=json.dumps(
                        {"is_answerable": True, "sentences": [_sentence(1, _GOOD_QUOTE)]}
                    )
                )
            )
        ]
        with patch(
            "axiom_rag_engine.nodes.synthesizer.litellm.acompletion",
            new_callable=AsyncMock,
            return_value=reply,
        ) as mock_llm:
            await synthesizer_node(state)

        user_prompt = mock_llm.call_args.kwargs["messages"][1]["content"]
        assert "PREVIOUS DRAFT" in user_prompt
        assert _BAD_QUOTE in user_prompt
        assert "Claim number 1." in user_prompt

    async def test_first_pass_has_no_previous_draft_section(self) -> None:
        state = _state()
        state["ranked_chunks"] = [_CHUNK]
        reply = MagicMock()
        reply.choices = [MagicMock(message=MagicMock(content=json.dumps({"is_answerable": False})))]
        with patch(
            "axiom_rag_engine.nodes.synthesizer.litellm.acompletion",
            new_callable=AsyncMock,
            return_value=reply,
        ) as mock_llm:
            await synthesizer_node(state)
        assert "PREVIOUS DRAFT" not in mock_llm.call_args.kwargs["messages"][1]["content"]


# ---------------------------------------------------------------------------
# Graph harness: fake retrieval + scripted synthesizer, REAL verifier
# ---------------------------------------------------------------------------


class _ScriptedSynth:
    """Returns the scripted drafts in order (repeating the last one)."""

    def __init__(self, drafts: list[list[dict[str, Any]]]) -> None:
        self.drafts = drafts
        self.calls = 0
        self.__name__ = "synthesizer_node"  # graph._timed_node copies the name

    async def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        draft = self.drafts[min(self.calls, len(self.drafts) - 1)]
        self.calls += 1
        return {"is_answerable": True, "draft_sentences": draft, "audit_trail": []}


async def _fake_retriever(state: dict[str, Any]) -> dict[str, Any]:
    return {"indexed_chunks": [_CHUNK], "audit_trail": []}


async def _passthrough(state: dict[str, Any]) -> dict[str, Any]:
    return {"audit_trail": []}


async def _run_graph(synth: _ScriptedSynth, stages: dict[str, Any]) -> dict[str, Any]:
    with (
        patch.object(graph_module, "retriever_node", _fake_retriever),
        patch.object(graph_module, "scorer_node", _passthrough),
        patch.object(graph_module, "ranker_node", _passthrough),
        patch.object(graph_module, "synthesizer_node", synth),
    ):
        engine = graph_module.build_axiom_graph()
        return await engine.ainvoke(_state(stages))


# ---------------------------------------------------------------------------
# 2. max_rewrite_loops counts rewrites, not passes
# ---------------------------------------------------------------------------


class TestRewriteLoopCount:
    async def test_max_rewrite_loops_allows_exactly_that_many_rewrites(self) -> None:
        synth = _ScriptedSynth([[_sentence(1, _BAD_QUOTE)]])  # never fixed
        await _run_graph(synth, {"max_rewrite_loops": 3, "max_retrieval_retries": 0})
        assert synth.calls == 1 + 3  # initial pass + 3 rewrites

    async def test_one_rewrite_loop_means_one_rewrite(self) -> None:
        synth = _ScriptedSynth([[_sentence(1, _BAD_QUOTE)]])
        await _run_graph(synth, {"max_rewrite_loops": 1, "max_retrieval_retries": 0})
        assert synth.calls == 2

    async def test_zero_rewrite_loops_disables_rewrites(self) -> None:
        synth = _ScriptedSynth([[_sentence(1, _BAD_QUOTE)]])
        await _run_graph(synth, {"max_rewrite_loops": 0, "max_retrieval_retries": 0})
        assert synth.calls == 1

    async def test_default_keeps_three_synthesis_passes_per_round(self) -> None:
        # The default moved from 3 (which really meant 2 rewrites) to 2 so the
        # default LLM cost is unchanged by the counting fix.
        defaults = PipelineStagesConfig().model_dump()
        synth = _ScriptedSynth([[_sentence(1, _BAD_QUOTE)]])
        await _run_graph(
            synth,
            {
                "max_rewrite_loops": defaults["max_rewrite_loops"],
                "max_retrieval_retries": 0,
            },
        )
        assert synth.calls == 3


# ---------------------------------------------------------------------------
# 3. Loop exhaustion returns the best pass, not the last one
# ---------------------------------------------------------------------------


class TestBestPassIsReturned:
    async def test_exhaustion_returns_the_pass_with_fewest_failures(self) -> None:
        best = [_sentence(1, _GOOD_QUOTE), _sentence(2, _BAD_QUOTE)]  # 1 failure
        worse = [_sentence(1, _BAD_QUOTE), _sentence(2, _BAD_QUOTE), _sentence(3, _BAD_QUOTE)]
        synth = _ScriptedSynth([best, worse])
        result = await _run_graph(synth, {"max_rewrite_loops": 2, "max_retrieval_retries": 0})

        tiers = [s["verification"]["tier"] for s in result["final_sentences"]]
        assert tiers == [3, 5]
        assert any(e["event_type"] == "best_pass_selected" for e in result["audit_trail"])

    async def test_best_pass_survives_a_retrieval_retry(self) -> None:
        best = [_sentence(1, _GOOD_QUOTE), _sentence(2, _BAD_QUOTE)]
        worse = [_sentence(1, _BAD_QUOTE), _sentence(2, _BAD_QUOTE)]
        synth = _ScriptedSynth([best, worse])
        result = await _run_graph(synth, {"max_rewrite_loops": 1, "max_retrieval_retries": 1})
        assert synth.calls == 4
        assert [s["verification"]["tier"] for s in result["final_sentences"]] == [3, 5]

    async def test_clean_final_pass_is_returned_as_is(self) -> None:
        synth = _ScriptedSynth([[_sentence(1, _BAD_QUOTE)], [_sentence(1, _GOOD_QUOTE)]])
        result = await _run_graph(synth, {"max_rewrite_loops": 2, "max_retrieval_retries": 0})
        assert synth.calls == 2
        assert [s["verification"]["tier"] for s in result["final_sentences"]] == [3]
        assert not any(e["event_type"] == "best_pass_selected" for e in result["audit_trail"])
