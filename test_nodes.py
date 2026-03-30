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
from unittest.mock import MagicMock, patch

import pytest

from nodes.synthesizer import synthesizer_node, _parse_llm_response, _build_chunks_block
from nodes.semantic import semantic_verifier_node, _parse_semantic_response
from state import make_initial_state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(
    user_query: str = "What is a solid-state battery?",
    indexed_chunks: list[dict] | None = None,
    ranked_chunks: list[dict] | None = None,
    rewrite_requests: list[str] | None = None,
    loop_count: int = 0,
    draft_sentences: list[dict] | None = None,
    mechanical_results: dict[str, str] | None = None,
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
    """Build a mock that mirrors litellm.completion() return structure."""
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
        "source_url": "https://science.org/article",
        "domain": "science.org",
        "is_authoritative": False,
    },
]

_VALID_SYNTHESIZER_JSON = json.dumps({
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
})

_UNANSWERABLE_JSON = json.dumps({"is_answerable": False, "sentences": []})


# ===========================================================================
# A. Synthesizer — happy path, escape hatch, rewrite injection
# ===========================================================================


class TestSynthesizerNode:
    @patch("nodes.synthesizer.litellm.completion")
    def test_happy_path_returns_draft_sentences(self, mock_completion: MagicMock) -> None:
        mock_completion.return_value = _mock_litellm_response(_VALID_SYNTHESIZER_JSON)
        state = _make_state(indexed_chunks=_SAMPLE_CHUNKS)

        result = synthesizer_node(state)

        assert result["is_answerable"] is True
        assert len(result["draft_sentences"]) == 1
        assert result["draft_sentences"][0]["sentence_id"] == "s_01"

    @patch("nodes.synthesizer.litellm.completion")
    def test_escape_hatch_returns_unanswerable(self, mock_completion: MagicMock) -> None:
        mock_completion.return_value = _mock_litellm_response(_UNANSWERABLE_JSON)
        state = _make_state(indexed_chunks=_SAMPLE_CHUNKS)

        result = synthesizer_node(state)

        assert result["is_answerable"] is False
        assert result["draft_sentences"] == []

    @patch("nodes.synthesizer.litellm.completion")
    def test_rewrite_section_injected_when_rewrite_requests_present(
        self, mock_completion: MagicMock
    ) -> None:
        mock_completion.return_value = _mock_litellm_response(_VALID_SYNTHESIZER_JSON)
        state = _make_state(
            indexed_chunks=_SAMPLE_CHUNKS,
            rewrite_requests=["Tier 5 failure on cite_1: quote not in doc_1_chunk_A."],
            loop_count=1,
        )

        result = synthesizer_node(state)

        # Verify the prompt that was actually sent contains the rewrite instruction.
        call_args = mock_completion.call_args
        messages = call_args[1]["messages"] if call_args[1] else call_args[0][1]
        user_message = next(m for m in messages if m["role"] == "user")
        assert "CORRECTION INSTRUCTIONS" in user_message["content"]
        assert "Rewrite Pass 1" in user_message["content"]

    @patch("nodes.synthesizer.litellm.completion")
    def test_no_rewrite_section_on_first_pass(self, mock_completion: MagicMock) -> None:
        mock_completion.return_value = _mock_litellm_response(_VALID_SYNTHESIZER_JSON)
        state = _make_state(indexed_chunks=_SAMPLE_CHUNKS, rewrite_requests=[])

        synthesizer_node(state)

        call_args = mock_completion.call_args
        messages = call_args[1]["messages"] if call_args[1] else call_args[0][1]
        user_message = next(m for m in messages if m["role"] == "user")
        assert "CORRECTION INSTRUCTIONS" not in user_message["content"]

    @patch("nodes.synthesizer.litellm.completion")
    def test_ranked_chunks_preferred_over_indexed_chunks(
        self, mock_completion: MagicMock
    ) -> None:
        mock_completion.return_value = _mock_litellm_response(_VALID_SYNTHESIZER_JSON)
        ranked = [{"chunk_id": "doc_3_chunk_X", "text": "Ranked chunk text."}]
        state = _make_state(indexed_chunks=_SAMPLE_CHUNKS, ranked_chunks=ranked)

        synthesizer_node(state)

        call_args = mock_completion.call_args
        messages = call_args[1]["messages"] if call_args[1] else call_args[0][1]
        user_message = next(m for m in messages if m["role"] == "user")
        assert "doc_3_chunk_X" in user_message["content"]
        assert "doc_1_chunk_A" not in user_message["content"]

    @patch("nodes.synthesizer.litellm.completion")
    def test_api_failure_returns_unanswerable(self, mock_completion: MagicMock) -> None:
        mock_completion.side_effect = Exception("LLM API timeout")
        state = _make_state(indexed_chunks=_SAMPLE_CHUNKS)

        result = synthesizer_node(state)

        assert result["is_answerable"] is False
        assert result["draft_sentences"] == []

    @patch("nodes.synthesizer.litellm.completion")
    def test_audit_trail_populated(self, mock_completion: MagicMock) -> None:
        mock_completion.return_value = _mock_litellm_response(_VALID_SYNTHESIZER_JSON)
        state = _make_state(indexed_chunks=_SAMPLE_CHUNKS)

        result = synthesizer_node(state)

        assert len(result["audit_trail"]) >= 2
        event_types = [e["event_type"] for e in result["audit_trail"]]
        assert "synthesizer_start" in event_types
        assert "synthesizer_complete" in event_types

    @patch("nodes.synthesizer.litellm.completion")
    def test_audit_event_unanswerable_escape_hatch(self, mock_completion: MagicMock) -> None:
        mock_completion.return_value = _mock_litellm_response(_UNANSWERABLE_JSON)
        state = _make_state(indexed_chunks=_SAMPLE_CHUNKS)

        result = synthesizer_node(state)

        event_types = [e["event_type"] for e in result["audit_trail"]]
        assert "synthesizer_unanswerable" in event_types

    @patch("nodes.synthesizer.litellm.completion")
    def test_uses_model_from_state(self, mock_completion: MagicMock) -> None:
        mock_completion.return_value = _mock_litellm_response(_VALID_SYNTHESIZER_JSON)
        state = _make_state(
            indexed_chunks=_SAMPLE_CHUNKS, synthesizer_model="gpt-4o"
        )

        synthesizer_node(state)

        call_kwargs = mock_completion.call_args[1]
        assert call_kwargs["model"] == "gpt-4o"

    @patch("nodes.synthesizer.litellm.completion")
    def test_temperature_is_zero(self, mock_completion: MagicMock) -> None:
        """Deterministic output is mandatory for citation integrity."""
        mock_completion.return_value = _mock_litellm_response(_VALID_SYNTHESIZER_JSON)
        state = _make_state(indexed_chunks=_SAMPLE_CHUNKS)

        synthesizer_node(state)

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

    @patch("nodes.synthesizer.litellm.completion")
    def test_malformed_response_retried(self, mock_completion: MagicMock) -> None:
        """First call returns garbage; second returns valid JSON — should succeed."""
        valid_response = _mock_litellm_response(_VALID_SYNTHESIZER_JSON)
        bad_response = _mock_litellm_response("not json at all %%")
        mock_completion.side_effect = [bad_response, valid_response]
        state = _make_state(indexed_chunks=_SAMPLE_CHUNKS)

        result = synthesizer_node(state)

        assert result["is_answerable"] is True
        assert mock_completion.call_count == 2

    @patch("nodes.synthesizer.litellm.completion")
    def test_exhausted_retries_returns_unanswerable(self, mock_completion: MagicMock) -> None:
        """Both attempts return garbage — node degrades gracefully."""
        bad = _mock_litellm_response("not json")
        mock_completion.return_value = bad
        state = _make_state(indexed_chunks=_SAMPLE_CHUNKS)

        result = synthesizer_node(state)

        assert result["is_answerable"] is False


# ===========================================================================
# C. Semantic Verifier — tier assignment
# ===========================================================================


class TestSemanticVerifierTierAssignment:
    def _run(
        self,
        tier: int,
        semantic_check: str = "passed",
        failure_reason: str | None = None,
        mechanical_results: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        llm_json = json.dumps({
            "tier": tier,
            "semantic_check": semantic_check,
            "failure_reason": failure_reason,
            "reasoning": "Test reasoning.",
        })
        draft = [
            {
                "sentence_id": "s_01",
                "text": "Solid-state batteries replace liquid electrolytes.",
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
            mechanical_results=mechanical_results or {"cite_1": "passed"},
        )
        with patch("nodes.semantic.litellm.completion") as mock_comp:
            mock_comp.return_value = _mock_litellm_response(llm_json)
            return semantic_verifier_node(state)

    def test_tier_1_authoritative(self) -> None:
        result = self._run(1)
        vr = result["final_sentences"][0]["verification"]
        assert vr["tier"] == 1
        assert vr["tier_label"] == "authoritative"
        assert vr["semantic_check"] == "passed"

    def test_tier_2_consensus(self) -> None:
        result = self._run(2)
        vr = result["final_sentences"][0]["verification"]
        assert vr["tier"] == 2
        assert vr["tier_label"] == "consensus"

    def test_tier_3_model_assisted(self) -> None:
        result = self._run(3)
        vr = result["final_sentences"][0]["verification"]
        assert vr["tier"] == 3
        assert vr["tier_label"] == "model_assisted"

    def test_tier_4_misrepresented_triggers_rewrite(self) -> None:
        result = self._run(
            4,
            semantic_check="failed",
            failure_reason="Claim overstates the temperature benefit.",
        )
        vr = result["final_sentences"][0]["verification"]
        assert vr["tier"] == 4
        assert vr["semantic_check"] == "failed"
        assert len(result["rewrite_requests"]) == 1
        assert "Tier 4" in result["rewrite_requests"][0]

    def test_tier_6_conflicted(self) -> None:
        result = self._run(6)
        vr = result["final_sentences"][0]["verification"]
        assert vr["tier"] == 6
        assert vr["tier_label"] == "conflicted"

    def test_tier_5_rejected_by_parser(self) -> None:
        """Semantic verifier must never assign Tier 5 — that belongs to Mechanical."""
        llm_json = json.dumps({
            "tier": 5,
            "semantic_check": "failed",
            "failure_reason": "Quote not found.",
            "reasoning": "Should not happen.",
        })
        draft = [
            {
                "sentence_id": "s_01",
                "text": "Some sentence.",
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
            mechanical_results={"cite_1": "passed"},
        )
        with patch("nodes.semantic.litellm.completion") as mock_comp:
            mock_comp.return_value = _mock_litellm_response(llm_json)
            result = semantic_verifier_node(state)
        # Should degrade to Tier 3, not propagate Tier 5
        vr = result["final_sentences"][0]["verification"]
        assert vr["tier"] == 3


# ===========================================================================
# D. Semantic Verifier — state update correctness
# ===========================================================================


class TestSemanticVerifierStateUpdates:
    @patch("nodes.semantic.litellm.completion")
    def test_loop_count_incremented(self, mock_comp: MagicMock) -> None:
        mock_comp.return_value = _mock_litellm_response(
            json.dumps({"tier": 1, "semantic_check": "passed", "failure_reason": None, "reasoning": "ok"})
        )
        draft = [
            {
                "sentence_id": "s_01",
                "text": "A sentence.",
                "is_cited": False,
                "citations": [],
            }
        ]
        state = _make_state(draft_sentences=draft, loop_count=1)
        result = semantic_verifier_node(state)
        assert result["loop_count"] == 2

    @patch("nodes.semantic.litellm.completion")
    def test_uncited_sentence_gets_tier_3_skipped(self, mock_comp: MagicMock) -> None:
        draft = [
            {
                "sentence_id": "s_01",
                "text": "This is a transition sentence.",
                "is_cited": False,
                "citations": [],
            }
        ]
        state = _make_state(draft_sentences=draft)
        result = semantic_verifier_node(state)
        vr = result["final_sentences"][0]["verification"]
        assert vr["tier"] == 3
        assert vr["mechanical_check"] == "skipped"
        assert vr["semantic_check"] == "skipped"
        mock_comp.assert_not_called()

    @patch("nodes.semantic.litellm.completion")
    def test_mechanical_failed_citation_skipped_by_semantic(
        self, mock_comp: MagicMock
    ) -> None:
        """Citations that failed mechanical check are skipped by semantic."""
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
            mechanical_results={"cite_1": "failed"},
        )
        result = semantic_verifier_node(state)
        mock_comp.assert_not_called()

    @patch("nodes.semantic.litellm.completion")
    def test_final_sentences_count_matches_draft(self, mock_comp: MagicMock) -> None:
        mock_comp.return_value = _mock_litellm_response(
            json.dumps({"tier": 2, "semantic_check": "passed", "failure_reason": None, "reasoning": "ok"})
        )
        draft = [
            {
                "sentence_id": f"s_0{i}",
                "text": f"Sentence {i}.",
                "is_cited": True,
                "citations": [
                    {
                        "citation_id": f"cite_{i}",
                        "chunk_id": "doc_1_chunk_A",
                        "exact_source_quote": "Solid-state batteries replace liquid electrolytes with solid ceramics.",
                    }
                ],
            }
            for i in range(1, 4)
        ]
        state = _make_state(
            indexed_chunks=_SAMPLE_CHUNKS,
            draft_sentences=draft,
            mechanical_results={"cite_1": "passed", "cite_2": "passed", "cite_3": "passed"},
        )
        result = semantic_verifier_node(state)
        assert len(result["final_sentences"]) == 3

    @patch("nodes.semantic.litellm.completion")
    def test_audit_trail_populated(self, mock_comp: MagicMock) -> None:
        mock_comp.return_value = _mock_litellm_response(
            json.dumps({"tier": 1, "semantic_check": "passed", "failure_reason": None, "reasoning": "ok"})
        )
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
            mechanical_results={"cite_1": "passed"},
        )
        result = semantic_verifier_node(state)
        event_types = [e["event_type"] for e in result["audit_trail"]]
        assert "semantic_verifier_start" in event_types
        assert "semantic_verifier_complete" in event_types


# ===========================================================================
# E. Semantic Verifier — disabled mode and API failure degradation
# ===========================================================================


class TestSemanticVerifierDegradation:
    def test_semantic_disabled_all_citations_become_tier_3(self) -> None:
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
            mechanical_results={"cite_1": "passed"},
            semantic_enabled=False,
        )
        with patch("nodes.semantic.litellm.completion") as mock_comp:
            result = semantic_verifier_node(state)
            mock_comp.assert_not_called()

        vr = result["final_sentences"][0]["verification"]
        assert vr["tier"] == 3
        assert vr["semantic_check"] == "skipped"

    @patch("nodes.semantic.litellm.completion")
    def test_api_failure_degrades_to_tier_3(self, mock_comp: MagicMock) -> None:
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
            mechanical_results={"cite_1": "passed"},
        )
        result = semantic_verifier_node(state)
        vr = result["final_sentences"][0]["verification"]
        assert vr["tier"] == 3
        assert vr["semantic_check"] == "skipped"
        assert "degraded to Tier 3" in (vr["failure_reason"] or "")

    def test_parse_semantic_response_rejects_tier_5(self) -> None:
        with pytest.raises(ValueError, match="invalid tier"):
            _parse_semantic_response(
                json.dumps({"tier": 5, "semantic_check": "failed", "failure_reason": None, "reasoning": "x"})
            )

    def test_parse_semantic_response_rejects_tier_7(self) -> None:
        with pytest.raises(ValueError, match="invalid tier"):
            _parse_semantic_response(
                json.dumps({"tier": 7, "semantic_check": "passed", "failure_reason": None, "reasoning": "x"})
            )

    def test_parse_semantic_response_rejects_invalid_semantic_check(self) -> None:
        with pytest.raises(ValueError, match="semantic_check"):
            _parse_semantic_response(
                json.dumps({"tier": 1, "semantic_check": "maybe", "failure_reason": None, "reasoning": "x"})
            )
