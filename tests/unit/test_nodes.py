"""
Phase 3 — TDD: Synthesizer and Semantic Verifier node tests.

All LLM calls are mocked via unittest.mock.patch — no real API keys required.

Test categories:
  A. Synthesizer — happy path, escape hatch, rewrite injection, API failure
  B. Synthesizer — malformed LLM response handling
  C. Semantic Verifier — tier assignment (1, 2, 3, 4, 6), disabled mode
  D. Semantic Verifier — state updates (final_sentences, rewrite_requests, loop_count)
  E. Semantic Verifier — API failure graceful degradation to Tier 3
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from axiom_engine.nodes.semantic import _parse_semantic_response, semantic_verifier_node
from axiom_engine.nodes.synthesizer import (
    _parse_llm_response,
    synthesizer_node,
)
from axiom_engine.state import make_initial_state

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mech_result(check: str = "passed") -> dict:
    """Build a VerificationResult-shaped dict for mechanical_results test data.

    In production, verification_node stores ``VerificationResult.model_dump()``
    dicts — never bare strings.
    """
    return {
        "tier": 5 if check == "failed" else 3,
        "tier_label": "hallucinated" if check == "failed" else "model_assisted",
        "mechanical_check": check,
        "semantic_check": "skipped",
        "failure_reason": "Citation not found." if check == "failed" else None,
    }


def _make_state(
    user_query: str = "What is a solid-state battery?",
    indexed_chunks: list[dict] | None = None,
    ranked_chunks: list[dict] | None = None,
    rewrite_requests: list[str] | None = None,
    loop_count: int = 0,
    draft_sentences: list[dict] | None = None,
    mechanical_results: dict[str, dict] | None = None,
    semantic_enabled: bool = True,
    synthesizer_model: str = "claude-3-5-sonnet-20241022",
    verifier_model: str = "gpt-4o-mini",
) -> dict[str, Any]:
    state = make_initial_state(
        request_id="req_test",
        user_query=user_query,
        app_config={"expertise_level": "intermediate", "banned_domains": []},
        models_config={"synthesizer": synthesizer_model, "verifier": verifier_model},
        pipeline_config={
            "stages": {
                "semantic_verification_enabled": semantic_enabled,
                "max_ranked_chunks": 10,
                "max_rewrite_loops": 3,
            }
        },
    )
    if indexed_chunks is not None:
        state["indexed_chunks"] = indexed_chunks
    if ranked_chunks is not None:
        state["ranked_chunks"] = ranked_chunks
    if rewrite_requests is not None:
        state["rewrite_requests"] = rewrite_requests
    state["loop_count"] = loop_count
    if draft_sentences is not None:
        state["draft_sentences"] = draft_sentences
    if mechanical_results is not None:
        state["mechanical_results"] = mechanical_results
    return state


def _mock_litellm_response(content: str) -> MagicMock:
    """Build a mock that mirrors litellm.acompletion() return structure."""
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


_SAMPLE_CHUNKS = [
    {
        "chunk_id": "doc_1_chunk_A",
        "text": (
            "Solid-state batteries replace liquid electrolytes with solid ceramics. "
            "This substitution significantly improves thermal stability and energy density."
        ),
        "source_url": "https://example.com/batteries",
        "domain": "example.com",
        "is_authoritative": True,
    },
    {
        "chunk_id": "doc_2_chunk_B",
        "text": "Multiple studies confirm higher energy density in solid-state designs.",
        # Use a primary-source domain so the Tier 1 test correctly exercises
        # the authoritative path (nih.gov is in _DEFAULT_PRIMARY_DOMAINS).
        "source_url": "https://nih.gov/article",
        "domain": "nih.gov",
        "is_authoritative": True,
    },
    {
        "chunk_id": "doc_3_chunk_C",
        "text": "Independent reporting also confirms higher energy density in solid-state cells.",
        "source_url": "https://example.net/article",
        "domain": "example.net",
        "is_authoritative": False,
    },
]

_VALID_SYNTHESIZER_JSON = json.dumps(
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
                        "exact_source_quote": "Solid-state batteries replace liquid electrolytes with solid ceramics.",
                    }
                ],
            }
        ],
    }
)

_UNANSWERABLE_JSON = json.dumps({"is_answerable": False, "sentences": []})


# ===========================================================================
# A. Synthesizer — happy path, escape hatch, rewrite injection
# ===========================================================================


class TestSynthesizerNode:
    @patch("axiom_engine.nodes.synthesizer.litellm.acompletion", new_callable=AsyncMock)
    async def test_happy_path_returns_draft_sentences(self, mock_completion: AsyncMock) -> None:
        mock_completion.return_value = _mock_litellm_response(_VALID_SYNTHESIZER_JSON)
        state = _make_state(indexed_chunks=_SAMPLE_CHUNKS)

        result = await synthesizer_node(state)

        assert result["is_answerable"] is True
        assert len(result["draft_sentences"]) == 1
        assert result["draft_sentences"][0]["sentence_id"] == "s_01"

    @patch("axiom_engine.nodes.synthesizer.litellm.acompletion", new_callable=AsyncMock)
    async def test_escape_hatch_returns_unanswerable(self, mock_completion: AsyncMock) -> None:
        mock_completion.return_value = _mock_litellm_response(_UNANSWERABLE_JSON)
        state = _make_state(indexed_chunks=_SAMPLE_CHUNKS)

        result = await synthesizer_node(state)

        assert result["is_answerable"] is False
        assert result["draft_sentences"] == []

    @patch("axiom_engine.nodes.synthesizer.litellm.acompletion", new_callable=AsyncMock)
    async def test_rewrite_section_injected_when_rewrite_requests_present(
        self, mock_completion: AsyncMock
    ) -> None:
        mock_completion.return_value = _mock_litellm_response(_VALID_SYNTHESIZER_JSON)
        state = _make_state(
            indexed_chunks=_SAMPLE_CHUNKS,
            rewrite_requests=["Tier 5 failure on cite_1: quote not in doc_1_chunk_A."],
            loop_count=1,
        )

        await synthesizer_node(state)

        # Verify the prompt that was actually sent contains the rewrite instruction.
        call_args = mock_completion.call_args
        messages = call_args[1]["messages"] if call_args[1] else call_args[0][1]
        user_message = next(m for m in messages if m["role"] == "user")
        assert "CORRECTION INSTRUCTIONS" in user_message["content"]
        assert "Rewrite Pass 1" in user_message["content"]

    @patch("axiom_engine.nodes.synthesizer.litellm.acompletion", new_callable=AsyncMock)
    async def test_no_rewrite_section_on_first_pass(self, mock_completion: AsyncMock) -> None:
        mock_completion.return_value = _mock_litellm_response(_VALID_SYNTHESIZER_JSON)
        state = _make_state(indexed_chunks=_SAMPLE_CHUNKS, rewrite_requests=[])

        await synthesizer_node(state)

        call_args = mock_completion.call_args
        messages = call_args[1]["messages"] if call_args[1] else call_args[0][1]
        user_message = next(m for m in messages if m["role"] == "user")
        assert "CORRECTION INSTRUCTIONS" not in user_message["content"]

    @patch("axiom_engine.nodes.synthesizer.litellm.acompletion", new_callable=AsyncMock)
    async def test_empty_ranked_chunks_emits_fallback_audit_event(
        self, mock_completion: AsyncMock
    ) -> None:
        """When ranked_chunks is absent the synthesizer must emit a fallback audit event."""
        mock_completion.return_value = _mock_litellm_response(_VALID_SYNTHESIZER_JSON)
        # ranked_chunks not set — state only has indexed_chunks.
        state = _make_state(indexed_chunks=_SAMPLE_CHUNKS, ranked_chunks=None)

        result = await synthesizer_node(state)

        event_types = [e["event_type"] for e in result["audit_trail"]]
        assert "synthesizer_ranked_empty_fallback" in event_types

    @patch("axiom_engine.nodes.synthesizer.litellm.acompletion", new_callable=AsyncMock)
    async def test_ranked_chunks_preferred_over_indexed_chunks(
        self, mock_completion: AsyncMock
    ) -> None:
        mock_completion.return_value = _mock_litellm_response(_VALID_SYNTHESIZER_JSON)
        ranked = [{"chunk_id": "doc_3_chunk_X", "text": "Ranked chunk text."}]
        state = _make_state(indexed_chunks=_SAMPLE_CHUNKS, ranked_chunks=ranked)

        await synthesizer_node(state)

        call_args = mock_completion.call_args
        messages = call_args[1]["messages"] if call_args[1] else call_args[0][1]
        user_message = next(m for m in messages if m["role"] == "user")
        assert "doc_3_chunk_X" in user_message["content"]
        assert "doc_1_chunk_A" not in user_message["content"]

    @patch("axiom_engine.nodes.synthesizer.litellm.acompletion", new_callable=AsyncMock)
    async def test_api_failure_raises_runtime_error(self, mock_completion: AsyncMock) -> None:
        mock_completion.side_effect = Exception("LLM API timeout")
        state = _make_state(indexed_chunks=_SAMPLE_CHUNKS)

        with pytest.raises(RuntimeError, match="Synthesizer stage failed"):
            await synthesizer_node(state)

    @patch("axiom_engine.nodes.synthesizer.litellm.acompletion", new_callable=AsyncMock)
    async def test_audit_trail_populated(self, mock_completion: AsyncMock) -> None:
        mock_completion.return_value = _mock_litellm_response(_VALID_SYNTHESIZER_JSON)
        state = _make_state(indexed_chunks=_SAMPLE_CHUNKS)

        result = await synthesizer_node(state)

        assert len(result["audit_trail"]) >= 2
        event_types = [e["event_type"] for e in result["audit_trail"]]
        assert "synthesizer_start" in event_types
        assert "synthesizer_complete" in event_types

    @patch("axiom_engine.nodes.synthesizer.litellm.acompletion", new_callable=AsyncMock)
    async def test_audit_event_unanswerable_escape_hatch(self, mock_completion: AsyncMock) -> None:
        mock_completion.return_value = _mock_litellm_response(_UNANSWERABLE_JSON)
        state = _make_state(indexed_chunks=_SAMPLE_CHUNKS)

        result = await synthesizer_node(state)

        event_types = [e["event_type"] for e in result["audit_trail"]]
        assert "synthesizer_unanswerable" in event_types

    @patch("axiom_engine.nodes.synthesizer.litellm.acompletion", new_callable=AsyncMock)
    async def test_uses_model_from_state(self, mock_completion: AsyncMock) -> None:
        mock_completion.return_value = _mock_litellm_response(_VALID_SYNTHESIZER_JSON)
        state = _make_state(indexed_chunks=_SAMPLE_CHUNKS, synthesizer_model="gpt-4o")

        await synthesizer_node(state)

        call_kwargs = mock_completion.call_args[1]
        assert call_kwargs["model"] == "gpt-4o"

    @patch("axiom_engine.nodes.synthesizer.litellm.acompletion", new_callable=AsyncMock)
    async def test_temperature_is_zero(self, mock_completion: AsyncMock) -> None:
        """Deterministic output is mandatory for citation integrity."""
        mock_completion.return_value = _mock_litellm_response(_VALID_SYNTHESIZER_JSON)
        state = _make_state(indexed_chunks=_SAMPLE_CHUNKS)

        await synthesizer_node(state)

        call_kwargs = mock_completion.call_args[1]
        assert call_kwargs["temperature"] == 0.0


# ===========================================================================
# B. Synthesizer — malformed LLM response handling
# ===========================================================================


class TestSynthesizerParsing:
    def test_parse_valid_json(self) -> None:
        output = _parse_llm_response(_VALID_SYNTHESIZER_JSON)
        assert output.is_answerable is True
        assert len(output.sentences) == 1

    def test_parse_strips_markdown_fences(self) -> None:
        fenced = f"```json\n{_VALID_SYNTHESIZER_JSON}\n```"
        output = _parse_llm_response(fenced)
        assert output.is_answerable is True

    def test_parse_strips_plain_fences(self) -> None:
        fenced = f"```\n{_VALID_SYNTHESIZER_JSON}\n```"
        output = _parse_llm_response(fenced)
        assert output.is_answerable is True

    def test_parse_invalid_json_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="not valid JSON"):
            _parse_llm_response("this is not json")

    def test_parse_schema_violation_raises_value_error(self) -> None:
        bad = json.dumps({"is_answerable": "yes", "sentences": []})  # bool expected
        with pytest.raises(ValueError, match="schema"):
            _parse_llm_response(bad)

    @patch("axiom_engine.nodes.synthesizer.litellm.acompletion", new_callable=AsyncMock)
    async def test_malformed_response_retried(self, mock_completion: AsyncMock) -> None:
        """First call returns garbage; second returns valid JSON — should succeed."""
        valid_response = _mock_litellm_response(_VALID_SYNTHESIZER_JSON)
        bad_response = _mock_litellm_response("not json at all %%")
        mock_completion.side_effect = [bad_response, valid_response]
        state = _make_state(indexed_chunks=_SAMPLE_CHUNKS)

        result = await synthesizer_node(state)

        assert result["is_answerable"] is True
        assert mock_completion.call_count == 2

    @patch("axiom_engine.nodes.synthesizer.litellm.acompletion", new_callable=AsyncMock)
    async def test_exhausted_retries_raise_runtime_error(self, mock_completion: AsyncMock) -> None:
        """Both attempts return garbage — malformed output is a stage failure."""
        bad = _mock_litellm_response("not json")
        mock_completion.return_value = bad
        state = _make_state(indexed_chunks=_SAMPLE_CHUNKS)

        with pytest.raises(RuntimeError, match="Synthesizer stage failed"):
            await synthesizer_node(state)


# ===========================================================================
# C. Semantic Verifier — strict verification outcomes
# ===========================================================================


def _semantic_pass_json() -> str:
    return json.dumps(
        {
            "semantic_check": "passed",
            "failure_reason": None,
            "reasoning": "Faithfully represents the cited source.",
        }
    )


def _semantic_fail_json(reason: str) -> str:
    return json.dumps(
        {
            "semantic_check": "failed",
            "failure_reason": reason,
            "reasoning": "The claim distorts the cited source.",
        }
    )


class TestSemanticVerifierTierAssignment:
    async def test_authoritative_sentence_gets_tier_1(self) -> None:
        draft = [
            {
                "sentence_id": "s_01",
                "text": "Multiple studies confirm higher energy density in solid-state designs.",
                "is_cited": True,
                "citations": [
                    {
                        "citation_id": "cite_1",
                        "chunk_id": "doc_2_chunk_B",
                        "exact_source_quote": "Multiple studies confirm higher energy density in solid-state designs.",
                    }
                ],
            }
        ]
        state = _make_state(
            indexed_chunks=_SAMPLE_CHUNKS,
            draft_sentences=draft,
            mechanical_results={"cite_1": _mech_result("passed")},
        )
        with patch(
            "axiom_engine.nodes.semantic.litellm.acompletion", new_callable=AsyncMock
        ) as mock_comp:
            mock_comp.return_value = _mock_litellm_response(_semantic_pass_json())
            result = await semantic_verifier_node(state)

        vr = result["final_sentences"][0]["verification"]
        assert vr["tier"] == 1
        assert vr["tier_label"] == "authoritative"

    async def test_multi_source_sentence_gets_tier_2(self) -> None:
        draft = [
            {
                "sentence_id": "s_01",
                "text": "Independent sources report higher energy density in solid-state cells.",
                "is_cited": True,
                "citations": [
                    {
                        "citation_id": "cite_1",
                        "chunk_id": "doc_1_chunk_A",
                        "exact_source_quote": "This substitution significantly improves thermal stability and energy density.",
                    },
                    {
                        "citation_id": "cite_2",
                        "chunk_id": "doc_3_chunk_C",
                        "exact_source_quote": "Independent reporting also confirms higher energy density in solid-state cells.",
                    },
                ],
            }
        ]
        state = _make_state(
            indexed_chunks=_SAMPLE_CHUNKS,
            draft_sentences=draft,
            mechanical_results={"cite_1": _mech_result("passed"), "cite_2": _mech_result("passed")},
        )
        with patch(
            "axiom_engine.nodes.semantic.litellm.acompletion", new_callable=AsyncMock
        ) as mock_comp:
            mock_comp.side_effect = [
                _mock_litellm_response(_semantic_pass_json()),
                _mock_litellm_response(_semantic_pass_json()),
            ]
            result = await semantic_verifier_node(state)

        vr = result["final_sentences"][0]["verification"]
        assert vr["tier"] == 2
        assert vr["tier_label"] == "multi_source"

    async def test_non_authoritative_single_source_gets_tier_3(self) -> None:
        draft = [
            {
                "sentence_id": "s_01",
                "text": "Solid-state batteries replace liquid electrolytes with solid ceramics.",
                "is_cited": True,
                "citations": [
                    {
                        "citation_id": "cite_1",
                        "chunk_id": "doc_1_chunk_A",
                        "exact_source_quote": "Solid-state batteries replace liquid electrolytes with solid ceramics.",
                    }
                ],
            }
        ]
        state = _make_state(
            indexed_chunks=_SAMPLE_CHUNKS,
            draft_sentences=draft,
            mechanical_results={"cite_1": _mech_result("passed")},
        )
        with patch(
            "axiom_engine.nodes.semantic.litellm.acompletion", new_callable=AsyncMock
        ) as mock_comp:
            mock_comp.return_value = _mock_litellm_response(_semantic_pass_json())
            result = await semantic_verifier_node(state)

        vr = result["final_sentences"][0]["verification"]
        assert vr["tier"] == 3
        assert vr["tier_label"] == "model_assisted"

    async def test_misrepresented_citation_triggers_tier_4_and_rewrite(self) -> None:
        draft = [
            {
                "sentence_id": "s_01",
                "text": "Solid-state batteries have perfect thermal stability.",
                "is_cited": True,
                "citations": [
                    {
                        "citation_id": "cite_1",
                        "chunk_id": "doc_1_chunk_A",
                        "exact_source_quote": "This substitution significantly improves thermal stability and energy density.",
                    }
                ],
            }
        ]
        state = _make_state(
            indexed_chunks=_SAMPLE_CHUNKS,
            draft_sentences=draft,
            mechanical_results={"cite_1": _mech_result("passed")},
        )
        with patch(
            "axiom_engine.nodes.semantic.litellm.acompletion", new_callable=AsyncMock
        ) as mock_comp:
            mock_comp.return_value = _mock_litellm_response(
                _semantic_fail_json("Claim overstates 'improves' as 'perfect'.")
            )
            result = await semantic_verifier_node(state)

        vr = result["final_sentences"][0]["verification"]
        assert vr["tier"] == 4
        assert len(result["rewrite_requests"]) == 1
        assert "Tier 4" in result["rewrite_requests"][0]


# ===========================================================================
# D. Semantic Verifier — state update correctness
# ===========================================================================


class TestSemanticVerifierStateUpdates:
    async def test_loop_count_not_in_semantic_result(self) -> None:
        # M7 fix: loop_count is now incremented in verification_node (the
        # orchestrator), not here. The semantic node must NOT return loop_count
        # so it doesn't accidentally overwrite the counter.
        draft = [
            {
                "sentence_id": "s_01",
                "text": "A sentence.",
                "is_cited": False,
                "citations": [],
            }
        ]
        state = _make_state(draft_sentences=draft, loop_count=1)
        result = await semantic_verifier_node(state)
        assert "loop_count" not in result

    @patch("axiom_engine.nodes.semantic.litellm.acompletion", new_callable=AsyncMock)
    async def test_uncited_transition_sentence_gets_tier_3_no_rewrite(
        self, mock_comp: AsyncMock
    ) -> None:
        """Uncited transition sentences are allowed by the synthesizer prompt.
        They are skipped (Tier 3) rather than penalised (Tier 5)."""
        draft = [
            {
                "sentence_id": "s_01",
                "text": "In summary, solid-state designs show promise.",
                "is_cited": False,
                "citations": [],
            }
        ]
        state = _make_state(draft_sentences=draft)
        result = await semantic_verifier_node(state)
        vr = result["final_sentences"][0]["verification"]
        assert vr["tier"] == 3
        assert vr["mechanical_check"] == "skipped"
        assert vr["semantic_check"] == "skipped"
        assert not result["rewrite_requests"]
        mock_comp.assert_not_called()

    @patch("axiom_engine.nodes.semantic.litellm.acompletion", new_callable=AsyncMock)
    async def test_mechanical_failed_citation_skipped_by_semantic(
        self, mock_comp: AsyncMock
    ) -> None:
        draft = [
            {
                "sentence_id": "s_01",
                "text": "Some sentence.",
                "is_cited": True,
                "citations": [
                    {
                        "citation_id": "cite_1",
                        "chunk_id": "doc_1_chunk_A",
                        "exact_source_quote": "fabricated quote",
                    }
                ],
            }
        ]
        state = _make_state(
            indexed_chunks=_SAMPLE_CHUNKS,
            draft_sentences=draft,
            mechanical_results={"cite_1": _mech_result("failed")},
        )
        result = await semantic_verifier_node(state)
        mock_comp.assert_not_called()
        assert result["final_sentences"][0]["verification"]["tier"] == 5

    @patch("axiom_engine.nodes.semantic.litellm.acompletion", new_callable=AsyncMock)
    async def test_final_sentences_include_citation_verification(
        self, mock_comp: AsyncMock
    ) -> None:
        mock_comp.return_value = _mock_litellm_response(_semantic_pass_json())
        draft = [
            {
                "sentence_id": "s_01",
                "text": "Sentence 1.",
                "is_cited": True,
                "citations": [
                    {
                        "citation_id": "cite_1",
                        "chunk_id": "doc_1_chunk_A",
                        "exact_source_quote": "Solid-state batteries replace liquid electrolytes with solid ceramics.",
                    }
                ],
            }
        ]
        state = _make_state(
            indexed_chunks=_SAMPLE_CHUNKS,
            draft_sentences=draft,
            mechanical_results={"cite_1": _mech_result("passed")},
        )
        result = await semantic_verifier_node(state)
        citation = result["final_sentences"][0]["citations"][0]
        assert citation["verification"]["mechanical_check"] == "passed"
        assert "verification" in citation

    @patch("axiom_engine.nodes.semantic.litellm.acompletion", new_callable=AsyncMock)
    async def test_audit_trail_populated(self, mock_comp: AsyncMock) -> None:
        mock_comp.return_value = _mock_litellm_response(_semantic_pass_json())
        draft = [
            {
                "sentence_id": "s_01",
                "text": "Sentence.",
                "is_cited": True,
                "citations": [
                    {
                        "citation_id": "cite_1",
                        "chunk_id": "doc_1_chunk_A",
                        "exact_source_quote": "Solid-state batteries replace liquid electrolytes with solid ceramics.",
                    }
                ],
            }
        ]
        state = _make_state(
            indexed_chunks=_SAMPLE_CHUNKS,
            draft_sentences=draft,
            mechanical_results={"cite_1": _mech_result("passed")},
        )
        result = await semantic_verifier_node(state)
        event_types = [e["event_type"] for e in result["audit_trail"]]
        assert "semantic_verifier_start" in event_types
        assert "semantic_verifier_complete" in event_types


# ===========================================================================
# E. Semantic Verifier — disabled mode and API failure degradation
# ===========================================================================


class TestSemanticVerifierDegradation:
    async def test_semantic_disabled_all_citations_become_tier_3(self) -> None:
        draft = [
            {
                "sentence_id": "s_01",
                "text": "A cited sentence.",
                "is_cited": True,
                "citations": [
                    {
                        "citation_id": "cite_1",
                        "chunk_id": "doc_1_chunk_A",
                        "exact_source_quote": "Solid-state batteries replace liquid electrolytes with solid ceramics.",
                    }
                ],
            }
        ]
        state = _make_state(
            indexed_chunks=_SAMPLE_CHUNKS,
            draft_sentences=draft,
            mechanical_results={"cite_1": _mech_result("passed")},
            semantic_enabled=False,
        )
        with patch(
            "axiom_engine.nodes.semantic.litellm.acompletion", new_callable=AsyncMock
        ) as mock_comp:
            result = await semantic_verifier_node(state)
            mock_comp.assert_not_called()

        vr = result["final_sentences"][0]["verification"]
        assert vr["tier"] == 3
        assert vr["semantic_check"] == "skipped"

    @patch("axiom_engine.nodes.semantic.litellm.acompletion", new_callable=AsyncMock)
    async def test_api_failure_degrades_to_tier_3(self, mock_comp: AsyncMock) -> None:
        """LLM errors on a single citation degrade it to Tier 3 (not tank the whole pass)."""
        mock_comp.side_effect = Exception("Semantic LLM unavailable")
        draft = [
            {
                "sentence_id": "s_01",
                "text": "A cited sentence.",
                "is_cited": True,
                "citations": [
                    {
                        "citation_id": "cite_1",
                        "chunk_id": "doc_1_chunk_A",
                        "exact_source_quote": "Solid-state batteries replace liquid electrolytes with solid ceramics.",
                    }
                ],
            }
        ]
        state = _make_state(
            indexed_chunks=_SAMPLE_CHUNKS,
            draft_sentences=draft,
            mechanical_results={"cite_1": _mech_result("passed")},
        )
        # Should not raise — infrastructure errors degrade the affected citation only.
        result = await semantic_verifier_node(state)

        vr = result["final_sentences"][0]["verification"]
        assert vr["tier"] == 3
        assert vr["semantic_check"] == "skipped"
        event_types = [e["event_type"] for e in result["audit_trail"]]
        assert "semantic_check_error" in event_types

    def test_parse_semantic_response_rejects_tier_field(self) -> None:
        with pytest.raises(ValueError, match="must not include a tier field"):
            _parse_semantic_response(
                json.dumps(
                    {
                        "tier": 5,
                        "semantic_check": "failed",
                        "failure_reason": "x",
                        "reasoning": "x",
                    }
                )
            )

    def test_parse_semantic_response_rejects_invalid_semantic_check(self) -> None:
        with pytest.raises(ValueError, match="semantic_check"):
            _parse_semantic_response(
                json.dumps(
                    {
                        "semantic_check": "maybe",
                        "failure_reason": None,
                        "reasoning": "x",
                    }
                )
            )

    def test_parse_semantic_response_requires_failure_reason(self) -> None:
        with pytest.raises(ValueError, match="failure_reason"):
            _parse_semantic_response(
                json.dumps(
                    {
                        "semantic_check": "failed",
                        "failure_reason": None,
                        "reasoning": "x",
                    }
                )
            )
