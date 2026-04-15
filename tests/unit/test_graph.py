"""
Phase 4 — TDD: LangGraph wiring, route_post_verification, and end-to-end loop tests.

Test categories:
  A. route_post_verification — unit tests for routing logic
  B. Graph structure — node/edge validation
  C. End-to-end loop tests — mocked LLM, verifying the graph catches
     hallucinations and routes backward for rewrite
"""

from __future__ import annotations

import json
from collections import deque
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from axiom_engine.graph import build_axiom_graph, route_post_verification
from axiom_engine.nodes.retriever import MockSearchBackend, set_search_backend
from axiom_engine.nodes.verification import verification_node
from axiom_engine.state import make_initial_state

# Synthesizer and Semantic models used in tests — must match _base_state.
_SYNTH_MODEL = "claude-3-5-sonnet-20241022"
_VERIFIER_MODEL = "gpt-4o-mini"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_litellm_response(content: str) -> MagicMock:
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


def _make_model_router(
    synth_responses: list[str],
    semantic_responses: list[str],
) -> AsyncMock:
    """
    Returns an async side_effect callable that routes litellm.acompletion calls
    based on the `model` kwarg. This avoids the module-singleton patch
    collision where nodes.synthesizer.litellm and nodes.semantic.litellm
    are the same object.
    """
    synth_q: deque[str] = deque(synth_responses)
    semantic_q: deque[str] = deque(semantic_responses)

    async def _router(*args: Any, **kwargs: Any) -> MagicMock:
        model = kwargs.get("model") or (args[0] if args else "")
        if model == _SYNTH_MODEL:
            if not synth_q:
                raise RuntimeError("Unexpected extra synthesizer call")
            return _mock_litellm_response(synth_q.popleft())
        elif model == _VERIFIER_MODEL:
            if not semantic_q:
                raise RuntimeError("Unexpected extra semantic call")
            return _mock_litellm_response(semantic_q.popleft())
        else:
            raise RuntimeError(f"Unexpected model in litellm.acompletion: {model!r}")

    return _router


def _base_state(**overrides: Any) -> dict[str, Any]:
    state = make_initial_state(
        request_id="req_test",
        user_query="What is a solid-state battery?",
        app_config={"expertise_level": "intermediate", "banned_domains": []},
        models_config={
            "synthesizer": _SYNTH_MODEL,
            "verifier": _VERIFIER_MODEL,
        },
        pipeline_config={
            "stages": {
                "semantic_verification_enabled": True,
                "max_ranked_chunks": 10,
                "max_rewrite_loops": 3,
            }
        },
    )
    state.update(overrides)
    return state


_SAMPLE_CHUNKS = [
    {
        "chunk_id": "doc_1_chunk_A",
        "text": (
            "Solid-state batteries replace liquid electrolytes with solid ceramics. "
            "This substitution significantly improves thermal stability and energy density."
        ),
        # Use a primary-source domain so Tier 1 is reachable in tier-assignment tests.
        "source_url": "https://nih.gov/article",
        "domain": "nih.gov",
    },
]

_SEARCH_RESULTS = [
    {
        "url": "https://nih.gov/article",
        "content": _SAMPLE_CHUNKS[0]["text"],
        "title": "Solid-State Battery Review",
    }
]


# ===========================================================================
# A. route_post_verification — unit tests
# ===========================================================================


class TestRoutePostVerification:
    def test_ends_when_unanswerable(self) -> None:
        state = _base_state(is_answerable=False)
        assert route_post_verification(state) == "__end__"

    def test_reretrieve_when_loop_exhausted_and_retries_left(self) -> None:
        state = _base_state(loop_count=3, pending_rewrite_count=1, retrieval_retry_count=0)
        assert route_post_verification(state) == "re_retriever"

    def test_ends_when_loop_exhausted_and_retries_spent(self) -> None:
        state = _base_state(loop_count=3, pending_rewrite_count=1, retrieval_retry_count=1)
        assert route_post_verification(state) == "__end__"

    def test_ends_when_loop_exceeds_max_and_retries_spent(self) -> None:
        state = _base_state(loop_count=5, pending_rewrite_count=1, retrieval_retry_count=1)
        assert route_post_verification(state) == "__end__"

    def test_loops_back_on_pending_rewrites(self) -> None:
        state = _base_state(loop_count=1, pending_rewrite_count=1)
        assert route_post_verification(state) == "synthesizer"

    def test_ends_when_all_passed(self) -> None:
        state = _base_state(loop_count=1, pending_rewrite_count=0)
        assert route_post_verification(state) == "__end__"

    def test_ends_with_default_is_answerable_true(self) -> None:
        """is_answerable defaults to True when missing from state."""
        state = _base_state(loop_count=1, pending_rewrite_count=0)
        del state["is_answerable"]
        assert route_post_verification(state) == "__end__"

    def test_respects_custom_max_rewrite_loops(self) -> None:
        state = _base_state(loop_count=2, pending_rewrite_count=1, retrieval_retry_count=1)
        state["pipeline_config"]["stages"]["max_rewrite_loops"] = 2
        assert route_post_verification(state) == "__end__"

    def test_loops_when_under_custom_max(self) -> None:
        state = _base_state(loop_count=1, pending_rewrite_count=1)
        state["pipeline_config"]["stages"]["max_rewrite_loops"] = 5
        assert route_post_verification(state) == "synthesizer"

    def test_unanswerable_takes_priority_over_pending_rewrites(self) -> None:
        """Even with pending rewrites, unanswerable should END immediately."""
        state = _base_state(
            is_answerable=False,
            loop_count=0,
            pending_rewrite_count=2,
        )
        assert route_post_verification(state) == "__end__"


# ===========================================================================
# B. Graph structure — node/edge validation
# ===========================================================================


class TestGraphStructure:
    def test_graph_compiles(self) -> None:
        graph = build_axiom_graph()
        assert graph is not None

    def test_graph_has_expected_nodes(self) -> None:
        graph = build_axiom_graph()
        node_names = set(graph.get_graph().nodes.keys())
        # LangGraph adds __start__ and __end__ nodes.
        assert "retriever" in node_names
        assert "scorer" in node_names
        assert "ranker" in node_names
        assert "synthesizer" in node_names
        assert "verifier" in node_names


# ===========================================================================
# C. End-to-end loop tests (mocked LLM)
# ===========================================================================


class TestEndToEndLoop:
    """
    These tests invoke the compiled graph with mocked LLM calls.
    They verify that:
      - A correct citation flows through in one pass.
      - A hallucinated citation triggers a rewrite loop.
      - The escape hatch terminates early.
      - Loop exhaustion terminates after max_rewrite_loops.

    All tests use a single @patch("litellm.acompletion") with a model-based
    router, because nodes.synthesizer.litellm and nodes.semantic.litellm
    are the same module-singleton — dual patching causes one to overwrite
    the other.
    """

    @pytest.fixture(autouse=True)
    def _reset_search_backend(self):
        """Prevent cross-test contamination from set_search_backend calls."""
        set_search_backend(MockSearchBackend())
        yield
        set_search_backend(MockSearchBackend())

    @patch("litellm.acompletion", new_callable=AsyncMock)
    async def test_happy_path_single_pass(self, mock_llm: AsyncMock) -> None:
        """Correct verbatim quote → mechanical pass → semantic pass → END."""
        set_search_backend(MockSearchBackend(_SEARCH_RESULTS))
        synth_json = json.dumps(
            {
                "is_answerable": True,
                "sentences": [
                    {
                        "sentence_id": "s_01",
                        "text": "Solid-state batteries replace liquid electrolytes with solid ceramics.",
                        "is_cited": True,
                        "citations": [
                            {
                                "citation_id": "cite_1",
                                "chunk_id": "doc_1_chunk_A",
                                "exact_source_quote": (
                                    "Solid-state batteries replace liquid electrolytes with solid ceramics."
                                ),
                            }
                        ],
                    }
                ],
            }
        )
        semantic_json = json.dumps(
            {
                "semantic_check": "passed",
                "failure_reason": None,
                "reasoning": "Claim faithfully represents the authoritative source.",
            }
        )
        mock_llm.side_effect = _make_model_router([synth_json], [semantic_json])

        graph = build_axiom_graph()
        result = await graph.ainvoke(_base_state())

        assert result["is_answerable"] is True
        assert len(result["final_sentences"]) == 1
        assert result["final_sentences"][0]["verification"]["tier"] == 1
        assert result["loop_count"] == 1

    @patch("litellm.acompletion", new_callable=AsyncMock)
    async def test_hallucination_triggers_rewrite_loop(self, mock_llm: AsyncMock) -> None:
        """
        Pass 1: Fabricated quote → Mechanical Tier 5 → loops back.
        Pass 2: Corrected verbatim quote → passes → END.
        """
        set_search_backend(MockSearchBackend(_SEARCH_RESULTS))
        hallucinated_json = json.dumps(
            {
                "is_answerable": True,
                "sentences": [
                    {
                        "sentence_id": "s_01",
                        "text": "Solid-state batteries use ceramic electrolytes.",
                        "is_cited": True,
                        "citations": [
                            {
                                "citation_id": "cite_1",
                                "chunk_id": "doc_1_chunk_A",
                                "exact_source_quote": "This quote is completely fabricated and does not exist.",
                            }
                        ],
                    }
                ],
            }
        )
        corrected_json = json.dumps(
            {
                "is_answerable": True,
                "sentences": [
                    {
                        "sentence_id": "s_01",
                        "text": "Solid-state batteries replace liquid electrolytes with solid ceramics.",
                        "is_cited": True,
                        "citations": [
                            {
                                "citation_id": "cite_1",
                                "chunk_id": "doc_1_chunk_A",
                                "exact_source_quote": (
                                    "Solid-state batteries replace liquid electrolytes with solid ceramics."
                                ),
                            }
                        ],
                    }
                ],
            }
        )
        semantic_json = json.dumps(
            {
                "semantic_check": "passed",
                "failure_reason": None,
                "reasoning": "Claim faithfully represents the source.",
            }
        )
        mock_llm.side_effect = _make_model_router(
            [hallucinated_json, corrected_json],
            [semantic_json],  # Only called on pass 2 (pass 1 fails mechanical)
        )

        graph = build_axiom_graph()
        result = await graph.ainvoke(_base_state())

        assert result["is_answerable"] is True
        assert len(result["final_sentences"]) == 1
        assert result["final_sentences"][0]["verification"]["tier"] == 1
        assert result["loop_count"] == 2

    @patch("litellm.acompletion", new_callable=AsyncMock)
    async def test_escape_hatch_terminates_immediately(self, mock_llm: AsyncMock) -> None:
        """is_answerable=false → END without entering verification."""
        set_search_backend(MockSearchBackend(_SEARCH_RESULTS))
        unanswerable_json = json.dumps(
            {
                "is_answerable": False,
                "sentences": [],
            }
        )
        mock_llm.side_effect = _make_model_router([unanswerable_json], [])

        graph = build_axiom_graph()
        result = await graph.ainvoke(_base_state())

        assert result["is_answerable"] is False
        assert result.get("final_sentences", []) == [] or result["final_sentences"] == []

    @patch("litellm.acompletion", new_callable=AsyncMock)
    async def test_loop_exhaustion_terminates_after_max_loops(self, mock_llm: AsyncMock) -> None:
        """
        Synthesizer keeps hallucinating. After 3 loops the graph terminates.
        """
        set_search_backend(MockSearchBackend(_SEARCH_RESULTS))
        hallucinated_json = json.dumps(
            {
                "is_answerable": True,
                "sentences": [
                    {
                        "sentence_id": "s_01",
                        "text": "Some claim.",
                        "is_cited": True,
                        "citations": [
                            {
                                "citation_id": "cite_1",
                                "chunk_id": "doc_1_chunk_A",
                                "exact_source_quote": "This fabricated quote never appears in the chunk.",
                            }
                        ],
                    }
                ],
            }
        )
        # 3 synthesis calls, all hallucinated. No semantic calls (mechanical fails).
        mock_llm.side_effect = _make_model_router(
            [hallucinated_json] * 3,
            [],
        )

        graph = build_axiom_graph()
        # Set retrieval_retry_count=1 so re-retrieve is exhausted; we're testing
        # rewrite loop exhaustion specifically, not re-retrieve.
        result = await graph.ainvoke(_base_state(retrieval_retry_count=1))

        assert result["loop_count"] == 3
        assert result["final_sentences"][0]["verification"]["tier"] == 5

    @patch("litellm.acompletion", new_callable=AsyncMock)
    async def test_tier_4_semantic_failure_triggers_rewrite(self, mock_llm: AsyncMock) -> None:
        """
        Mechanical passes but semantic finds misrepresentation (Tier 4).
        Graph loops back for a rewrite, second pass succeeds.
        """
        set_search_backend(MockSearchBackend(_SEARCH_RESULTS))
        synth_json = json.dumps(
            {
                "is_answerable": True,
                "sentences": [
                    {
                        "sentence_id": "s_01",
                        "text": "Solid-state batteries have perfect thermal stability.",
                        "is_cited": True,
                        "citations": [
                            {
                                "citation_id": "cite_1",
                                "chunk_id": "doc_1_chunk_A",
                                "exact_source_quote": (
                                    "This substitution significantly improves thermal stability and energy density."
                                ),
                            }
                        ],
                    }
                ],
            }
        )
        corrected_json = json.dumps(
            {
                "is_answerable": True,
                "sentences": [
                    {
                        "sentence_id": "s_01",
                        "text": "This substitution significantly improves thermal stability and energy density.",
                        "is_cited": True,
                        "citations": [
                            {
                                "citation_id": "cite_1",
                                "chunk_id": "doc_1_chunk_A",
                                "exact_source_quote": (
                                    "This substitution significantly improves thermal stability and energy density."
                                ),
                            }
                        ],
                    }
                ],
            }
        )
        tier4_json = json.dumps(
            {
                "semantic_check": "failed",
                "failure_reason": "Claim overstates 'improves' as 'perfect'.",
                "reasoning": "The source says 'improves' not 'perfect'.",
            }
        )
        tier1_json = json.dumps(
            {
                "semantic_check": "passed",
                "failure_reason": None,
                "reasoning": "Claim now faithfully represents the source.",
            }
        )
        mock_llm.side_effect = _make_model_router(
            [synth_json, corrected_json],
            [tier4_json, tier1_json],
        )

        graph = build_axiom_graph()
        result = await graph.ainvoke(_base_state())

        assert result["is_answerable"] is True
        assert result["final_sentences"][0]["verification"]["tier"] == 1

    @patch("litellm.acompletion", new_callable=AsyncMock)
    async def test_audit_trail_accumulates_across_loops(self, mock_llm: AsyncMock) -> None:
        """The audit_trail must accumulate events from every pass, never overwrite."""
        set_search_backend(MockSearchBackend(_SEARCH_RESULTS))
        hallucinated_json = json.dumps(
            {
                "is_answerable": True,
                "sentences": [
                    {
                        "sentence_id": "s_01",
                        "text": "Claim.",
                        "is_cited": True,
                        "citations": [
                            {
                                "citation_id": "cite_1",
                                "chunk_id": "doc_1_chunk_A",
                                "exact_source_quote": "Fabricated nonsense that does not exist in chunk.",
                            }
                        ],
                    }
                ],
            }
        )
        corrected_json = json.dumps(
            {
                "is_answerable": True,
                "sentences": [
                    {
                        "sentence_id": "s_01",
                        "text": "Solid-state batteries replace liquid electrolytes with solid ceramics.",
                        "is_cited": True,
                        "citations": [
                            {
                                "citation_id": "cite_1",
                                "chunk_id": "doc_1_chunk_A",
                                "exact_source_quote": (
                                    "Solid-state batteries replace liquid electrolytes with solid ceramics."
                                ),
                            }
                        ],
                    }
                ],
            }
        )
        semantic_json = json.dumps(
            {
                "semantic_check": "passed",
                "failure_reason": None,
                "reasoning": "ok",
            }
        )
        mock_llm.side_effect = _make_model_router(
            [hallucinated_json, corrected_json],
            [semantic_json],
        )

        graph = build_axiom_graph()
        result = await graph.ainvoke(_base_state())

        trail = result.get("audit_trail", [])
        assert len(trail) > 0
        nodes_in_trail = {e["node"] for e in trail}
        assert "synthesizer" in nodes_in_trail
        assert "verifier" in nodes_in_trail


# ===========================================================================
# D. Full pipeline integration — retriever → scorer → ranker → synth → verify
# ===========================================================================


class TestFullPipelineIntegration:
    """
    End-to-end tests that start from the retriever with mock search results,
    flow through scorer and ranker, then into synthesizer and verifier.
    Verifies the complete DAG wiring and audit trail accumulation.
    """

    @patch("litellm.acompletion", new_callable=AsyncMock)
    async def test_full_pipeline_from_retriever_to_end(self, mock_llm: AsyncMock) -> None:
        """Mock search → retriever → scorer → ranker → synth → verify → END."""
        from axiom_engine.nodes.retriever import MockSearchBackend, set_search_backend

        # Set up search backend with results that will survive chunking.
        set_search_backend(
            MockSearchBackend(
                [
                    {
                        # Use a primary-source domain so this test verifies Tier 1
                        # assignment end-to-end through the full pipeline.
                        "url": "https://nih.gov/batteries",
                        "content": (
                            "Solid-state batteries replace liquid electrolytes with solid ceramics. "
                            "This substitution significantly improves thermal stability and energy density."
                        ),
                        "title": "Solid-State Battery Review",
                    },
                ]
            )
        )

        synth_json = json.dumps(
            {
                "is_answerable": True,
                "sentences": [
                    {
                        "sentence_id": "s_01",
                        "text": "Solid-state batteries replace liquid electrolytes with solid ceramics.",
                        "is_cited": True,
                        "citations": [
                            {
                                "citation_id": "cite_1",
                                "chunk_id": "doc_1_chunk_A",
                                "exact_source_quote": (
                                    "Solid-state batteries replace liquid electrolytes with solid ceramics."
                                ),
                            }
                        ],
                    }
                ],
            }
        )
        semantic_json = json.dumps(
            {
                "semantic_check": "passed",
                "failure_reason": None,
                "reasoning": "Faithfully represents the source.",
            }
        )
        mock_llm.side_effect = _make_model_router([synth_json], [semantic_json])

        graph = build_axiom_graph()
        # Start from a clean state — no pre-injected chunks.
        state = _base_state()
        result = await graph.ainvoke(state)

        assert result["is_answerable"] is True
        assert len(result["final_sentences"]) == 1
        assert result["final_sentences"][0]["verification"]["tier"] == 1

        # Verify all nodes contributed to the audit trail.
        trail = result.get("audit_trail", [])
        nodes_in_trail = {e["node"] for e in trail}
        assert "retriever" in nodes_in_trail
        assert "scorer" in nodes_in_trail
        assert "ranker" in nodes_in_trail
        assert "synthesizer" in nodes_in_trail
        assert "verifier" in nodes_in_trail

    @patch("litellm.acompletion", new_callable=AsyncMock)
    async def test_full_pipeline_with_banned_domain(self, mock_llm: AsyncMock) -> None:
        """Banned domain results are filtered before reaching the synthesizer."""
        from axiom_engine.nodes.retriever import MockSearchBackend, set_search_backend

        set_search_backend(
            MockSearchBackend(
                [
                    {
                        "url": "https://spam.com/fake",
                        "content": "Spam content that should be filtered out by the retriever node.",
                        "title": "Spam",
                    },
                    {
                        "url": "https://nature.com/battery-review",
                        "content": (
                            "Solid-state batteries replace liquid electrolytes with solid ceramics. "
                            "This substitution significantly improves thermal stability."
                        ),
                        "title": "Nature Review",
                    },
                ]
            )
        )

        synth_json = json.dumps(
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
                                "exact_source_quote": (
                                    "Solid-state batteries replace liquid electrolytes with solid ceramics."
                                ),
                            }
                        ],
                    }
                ],
            }
        )
        semantic_json = json.dumps(
            {
                "semantic_check": "passed",
                "failure_reason": None,
                "reasoning": "Consensus source.",
            }
        )
        mock_llm.side_effect = _make_model_router([synth_json], [semantic_json])

        graph = build_axiom_graph()
        state = _base_state(
            app_config={
                "banned_domains": ["spam.com"],
                "expertise_level": "intermediate",
            },
        )
        result = await graph.ainvoke(state)

        # The spam.com chunk should have been filtered out.
        banned_events = [
            e for e in result["audit_trail"] if e.get("event_type") == "retriever_banned_domain"
        ]
        assert len(banned_events) >= 1

    @patch("litellm.acompletion", new_callable=AsyncMock)
    async def test_full_pipeline_empty_search_results_unanswerable(
        self, mock_llm: AsyncMock
    ) -> None:
        """No search results → no chunks → synthesizer says unanswerable."""
        from axiom_engine.nodes.retriever import MockSearchBackend, set_search_backend

        set_search_backend(MockSearchBackend([]))

        unanswerable_json = json.dumps(
            {
                "is_answerable": False,
                "sentences": [],
            }
        )
        mock_llm.side_effect = _make_model_router([unanswerable_json], [])

        graph = build_axiom_graph()
        result = await graph.ainvoke(_base_state())

        assert result["is_answerable"] is False


# ===========================================================================
# D. Loop-exhaustion monitoring
# ===========================================================================


class TestLoopExhaustionMonitoring:
    async def test_tier5_audit_event_emitted_on_loop_exhaustion(self) -> None:
        """verification_node emits loop_exhausted_unresolved_tier5 when all budget is spent."""
        # last possible iteration: loop_count = max_loops-1, retry_count = max_retries
        state = _base_state(loop_count=2, retrieval_retry_count=1)
        state["pipeline_config"]["stages"]["max_rewrite_loops"] = 3
        state["pipeline_config"]["stages"]["max_retrieval_retries"] = 1
        # Citation pointing to a non-existent chunk → mechanical Tier 5 failure.
        # indexed_chunks is empty (default), so the chunk lookup will miss.
        state["draft_sentences"] = [
            {
                "sentence_id": "s_01",
                "text": "Solid-state batteries are 100% efficient.",
                "is_cited": True,
                "citations": [
                    {
                        "citation_id": "cite_1",
                        "chunk_id": "doc_1_chunk_A",
                        "exact_source_quote": "Solid-state batteries are 100% efficient.",
                    }
                ],
            }
        ]
        state["mechanical_results"] = {}

        result = await verification_node(state)

        event_types = [e["event_type"] for e in result["audit_trail"]]
        assert "loop_exhausted_unresolved_tier5" in event_types

    async def test_no_exhaustion_event_when_retries_remain(self) -> None:
        """No audit event emitted when retrieval retries are still available."""
        state = _base_state(loop_count=2, retrieval_retry_count=0)
        state["pipeline_config"]["stages"]["max_rewrite_loops"] = 3
        state["pipeline_config"]["stages"]["max_retrieval_retries"] = 1
        # Same mechanical failure setup — but retry_count=0 < max_retries=1, so
        # is_final_attempt is False and the exhaustion event must not fire.
        state["draft_sentences"] = [
            {
                "sentence_id": "s_01",
                "text": "Solid-state batteries are 100% efficient.",
                "is_cited": True,
                "citations": [
                    {
                        "citation_id": "cite_1",
                        "chunk_id": "doc_1_chunk_A",
                        "exact_source_quote": "Solid-state batteries are 100% efficient.",
                    }
                ],
            }
        ]
        state["mechanical_results"] = {}

        result = await verification_node(state)

        event_types = [e["event_type"] for e in result["audit_trail"]]
        assert "loop_exhausted_unresolved_tier5" not in event_types
