"""
Phase 1 — TDD: Pydantic model and GraphState validation tests.
Run with:  pytest axiom_engine/test_models.py -v
"""

from __future__ import annotations

import operator
from typing import get_type_hints

import pytest
from pydantic import ValidationError

from axiom_engine.models import (
    AppConfig,
    AxiomRequest,
    AxiomResponse,
    Citation,
    ConfidenceSummary,
    DraftSentence,
    FinalSentence,
    MechanicalVerificationStageConfig,
    ModelConfig,
    PipelineConfig,
    SynthesizerOutput,
    TierBreakdown,
    VerificationResult,
    VerifiedCitation,
)
from axiom_engine.state import GraphState, make_initial_state

# ===========================================================================
# AppConfig
# ===========================================================================


class TestAppConfig:
    def test_defaults(self) -> None:
        cfg = AppConfig()
        assert cfg.expertise_level == "intermediate"
        assert cfg.banned_domains == []
        assert cfg.authoritative_domains == []
        assert cfg.low_quality_domains == []

    def test_valid_expertise_levels(self) -> None:
        for level in ("beginner", "intermediate", "expert"):
            cfg = AppConfig(expertise_level=level)
            assert cfg.expertise_level == level

    def test_invalid_expertise_level_raises(self) -> None:
        with pytest.raises(ValidationError):
            AppConfig(expertise_level="genius")  # type: ignore[arg-type]

    def test_banned_domains_stored(self) -> None:
        cfg = AppConfig(banned_domains=["reddit.com", "quora.com"])
        assert "reddit.com" in cfg.banned_domains


# ===========================================================================
# AxiomRequest
# ===========================================================================


class TestAxiomRequest:
    def test_minimal_valid_request(self) -> None:
        req = AxiomRequest(request_id="req_001", user_query="What is RAG?")
        assert req.request_id == "req_001"
        assert isinstance(req.app_config, AppConfig)
        assert isinstance(req.models, ModelConfig)
        assert isinstance(req.pipeline_config, PipelineConfig)

    def test_empty_request_id_raises(self) -> None:
        with pytest.raises(ValidationError):
            AxiomRequest(request_id="", user_query="What is RAG?")

    def test_empty_user_query_raises(self) -> None:
        with pytest.raises(ValidationError):
            AxiomRequest(request_id="req_001", user_query="")

    def test_full_payload_parses(self) -> None:
        payload = {
            "request_id": "req_123",
            "user_query": "Explain solid-state batteries.",
            "app_config": {
                "expertise_level": "intermediate",
                "banned_domains": ["reddit.com"],
                "authoritative_domains": ["internal.example.com"],
                "low_quality_domains": ["spam.example.com"],
            },
            "models": {"synthesizer": "claude-3-opus", "verifier": "llama-3-8b-local"},
            "pipeline_config": {
                "stages": {
                    "mechanical_verification": {"enabled": True, "configurable": False},
                    "semantic_verification_enabled": True,
                    "max_ranked_chunks": 10,
                }
            },
        }
        req = AxiomRequest(**payload)
        assert req.app_config.banned_domains == ["reddit.com"]
        assert req.app_config.authoritative_domains == ["internal.example.com"]
        assert req.models.synthesizer == "claude-3-opus"


# ===========================================================================
# MechanicalVerificationStageConfig — non-negotiable floor
# ===========================================================================


class TestMechanicalVerificationStageConfig:
    def test_defaults_always_enabled(self) -> None:
        cfg = MechanicalVerificationStageConfig()
        assert cfg.enabled is True
        assert cfg.configurable is False

    def test_cannot_disable_mechanical_verification(self) -> None:
        """enabled is frozen — callers cannot turn off the mechanical floor."""
        with pytest.raises((ValidationError, TypeError)):
            cfg = MechanicalVerificationStageConfig()
            cfg.enabled = False  # type: ignore[misc]


# ===========================================================================
# Citation
# ===========================================================================


class TestCitation:
    def test_valid_citation(self) -> None:
        c = Citation(
            citation_id="cite_1",
            chunk_id="doc_4_chunk_C",
            exact_source_quote="The defining characteristic is the substitution.",
        )
        assert c.chunk_id == "doc_4_chunk_C"

    def test_chunk_id_pattern_valid_variants(self) -> None:
        valid_ids = [
            "doc_1_chunk_A",
            "doc_10_chunk_Z",
            "doc_99_chunk_AB",
            "doc_1_chunk_1",
        ]
        for cid in valid_ids:
            c = Citation(citation_id="c", chunk_id=cid, exact_source_quote="quote")
            assert c.chunk_id == cid

    def test_chunk_id_pattern_rejects_malformed(self) -> None:
        bad_ids = [
            "doc1_chunk_A",  # missing underscore after doc
            "doc_1_chunkA",  # missing underscore before chunk label
            "1_chunk_A",  # missing 'doc' prefix
            "doc_1_chunk_a",  # lowercase chunk label
            "doc__chunk_A",  # empty doc number
            "",  # empty
        ]
        for bad in bad_ids:
            with pytest.raises(ValidationError, match="chunk_id"):
                Citation(citation_id="c", chunk_id=bad, exact_source_quote="q")

    def test_empty_exact_source_quote_raises(self) -> None:
        with pytest.raises(ValidationError):
            Citation(citation_id="c", chunk_id="doc_1_chunk_A", exact_source_quote="")


# ===========================================================================
# VerificationResult — the 6-tier system
# ===========================================================================


class TestVerificationResult:
    def test_valid_tiers(self) -> None:
        tier_label_map = {
            1: "authoritative",
            2: "multi_source",
            3: "model_assisted",
            4: "misrepresented",
            5: "hallucinated",
            6: "conflicted",
        }
        for tier, label in tier_label_map.items():
            vr = VerificationResult(
                tier=tier,
                tier_label=label,
                mechanical_check="failed" if tier == 5 else "passed",
                semantic_check="passed" if tier not in (4, 5) else "failed",
            )
            assert vr.tier == tier

    def test_tier_below_1_raises(self) -> None:
        with pytest.raises(ValidationError):
            VerificationResult(
                tier=0,
                tier_label="authoritative",
                mechanical_check="passed",
                semantic_check="passed",
            )

    def test_tier_above_6_raises(self) -> None:
        with pytest.raises(ValidationError):
            VerificationResult(
                tier=7,
                tier_label="authoritative",
                mechanical_check="passed",
                semantic_check="passed",
            )

    def test_invalid_mechanical_check_value_raises(self) -> None:
        with pytest.raises(ValidationError):
            VerificationResult(
                tier=1,
                tier_label="authoritative",
                mechanical_check="maybe",  # type: ignore[arg-type]
                semantic_check="passed",
            )

    def test_invalid_semantic_check_value_raises(self) -> None:
        with pytest.raises(ValidationError):
            VerificationResult(
                tier=1,
                tier_label="authoritative",
                mechanical_check="passed",
                semantic_check="unknown",  # type: ignore[arg-type]
            )

    def test_semantic_check_skipped_is_valid(self) -> None:
        vr = VerificationResult(
            tier=5,
            tier_label="hallucinated",
            mechanical_check="failed",
            semantic_check="skipped",
            failure_reason="Quote does not exist in source chunk.",
        )
        assert vr.semantic_check == "skipped"
        assert vr.failure_reason is not None

    def test_tier_5_hallucinated_mechanical_failed(self) -> None:
        """Tier 5 must have mechanical_check='failed'."""
        vr = VerificationResult(
            tier=5,
            tier_label="hallucinated",
            mechanical_check="failed",
            semantic_check="skipped",
        )
        assert vr.tier == 5
        assert vr.mechanical_check == "failed"

    def test_immutability(self) -> None:
        """VerificationResult is frozen — fields cannot be mutated post-init."""
        vr = VerificationResult(
            tier=1,
            tier_label="authoritative",
            mechanical_check="passed",
            semantic_check="passed",
        )
        with pytest.raises((ValidationError, TypeError)):
            vr.tier = 3  # type: ignore[misc]

    def test_tier_5_requires_mechanical_failure(self) -> None:
        with pytest.raises(ValidationError, match="Tier 5 requires"):
            VerificationResult(
                tier=5,
                tier_label="hallucinated",
                mechanical_check="passed",
                semantic_check="skipped",
            )


# ===========================================================================
# DraftSentence & SynthesizerOutput
# ===========================================================================


class TestDraftSentence:
    def test_uncited_sentence_has_empty_citations(self) -> None:
        s = DraftSentence(sentence_id="s_01", text="This is a fact.", is_cited=False)
        assert s.citations == []

    def test_cited_sentence_requires_citations(self) -> None:
        citation = Citation(
            citation_id="cite_1",
            chunk_id="doc_1_chunk_A",
            exact_source_quote="A verbatim excerpt.",
        )
        s = DraftSentence(
            sentence_id="s_01",
            text="A sentence with a citation.",
            is_cited=True,
            citations=[citation],
        )
        assert len(s.citations) == 1

    def test_empty_text_raises(self) -> None:
        with pytest.raises(ValidationError):
            DraftSentence(sentence_id="s_01", text="", is_cited=False)

    def test_cited_sentence_without_citations_raises(self) -> None:
        with pytest.raises(ValidationError, match="at least one citation"):
            DraftSentence(
                sentence_id="s_01",
                text="A cited sentence missing citations.",
                is_cited=True,
                citations=[],
            )


class TestSynthesizerOutput:
    def test_unanswerable_escape_hatch(self) -> None:
        out = SynthesizerOutput(is_answerable=False, sentences=[])
        assert out.is_answerable is False
        assert out.sentences == []

    def test_answerable_with_sentences(self) -> None:
        s = DraftSentence(sentence_id="s_01", text="The answer.", is_cited=False)
        out = SynthesizerOutput(is_answerable=True, sentences=[s])
        assert len(out.sentences) == 1


# ===========================================================================
# FinalSentence
# ===========================================================================


class TestFinalSentence:
    def _make_verification(self, tier: int = 1) -> VerificationResult:
        label_map = {
            1: "authoritative",
            2: "multi_source",
            3: "model_assisted",
            4: "misrepresented",
            5: "hallucinated",
            6: "conflicted",
        }
        return VerificationResult(
            tier=tier,
            tier_label=label_map[tier],
            mechanical_check="passed" if tier != 5 else "failed",
            semantic_check="passed" if tier not in (4, 5) else "failed",
        )

    def test_final_sentence_inherits_draft_fields(self) -> None:
        fs = FinalSentence(
            sentence_id="s_01",
            text="Verified sentence.",
            is_cited=False,
            verification=self._make_verification(1),
        )
        assert fs.sentence_id == "s_01"
        assert fs.verification.tier == 1

    def test_final_sentence_tier_4_misrepresented(self) -> None:
        fs = FinalSentence(
            sentence_id="s_02",
            text="A misrepresented sentence.",
            is_cited=True,
            citations=[
                VerifiedCitation(
                    citation_id="c1",
                    chunk_id="doc_1_chunk_B",
                    exact_source_quote="Some quote.",
                    verification=self._make_verification(4),
                )
            ],
            verification=self._make_verification(4),
        )
        assert fs.verification.tier == 4
        assert fs.verification.mechanical_check == "passed"
        assert fs.verification.semantic_check == "failed"

    def test_cited_final_sentence_requires_verified_citations(self) -> None:
        with pytest.raises(ValidationError, match="verified citations"):
            FinalSentence(
                sentence_id="s_03",
                text="A cited sentence missing citations.",
                is_cited=True,
                citations=[],
                verification=self._make_verification(3),
            )


# ===========================================================================
# AxiomResponse
# ===========================================================================


class TestAxiomResponse:
    def _make_final_sentence(self, tier: int = 1) -> FinalSentence:
        label_map = {
            1: "authoritative",
            2: "multi_source",
            3: "model_assisted",
            4: "misrepresented",
            5: "hallucinated",
            6: "conflicted",
        }
        vr = VerificationResult(
            tier=tier,
            tier_label=label_map[tier],
            mechanical_check="passed" if tier != 5 else "failed",
            semantic_check="passed" if tier not in (4, 5) else "skipped",
        )
        return FinalSentence(
            sentence_id=f"s_0{tier}",
            text="A sentence.",
            is_cited=False,
            verification=vr,
        )

    def test_success_response(self) -> None:
        resp = AxiomResponse(
            request_id="req_123",
            status="success",
            is_answerable=True,
            confidence_summary=ConfidenceSummary(
                overall_score=0.84,
                tier_breakdown=TierBreakdown(tier_1_claims=6),
            ),
            final_response=[self._make_final_sentence(1)],
        )
        assert resp.status == "success"
        assert resp.confidence_summary.overall_score == 0.84

    def test_overall_score_out_of_range_raises(self) -> None:
        with pytest.raises(ValidationError):
            ConfidenceSummary(
                overall_score=1.5,  # > 1.0
                tier_breakdown=TierBreakdown(),
            )

    def test_invalid_status_raises(self) -> None:
        with pytest.raises(ValidationError):
            AxiomResponse(
                request_id="req_123",
                status="unknown_status",  # type: ignore[arg-type]
                is_answerable=True,
                confidence_summary=ConfidenceSummary(
                    overall_score=0.5,
                    tier_breakdown=TierBreakdown(),
                ),
            )


# ===========================================================================
# GraphState & make_initial_state
# ===========================================================================


class TestGraphState:
    def test_make_initial_state_populates_all_keys(self) -> None:
        state = make_initial_state(
            request_id="req_001",
            user_query="What is a solid-state battery?",
            app_config={},
            models_config={},
            pipeline_config={},
        )
        required_keys = {
            "request_id",
            "user_query",
            "app_config",
            "models_config",
            "pipeline_config",
            "search_queries",
            "indexed_chunks",
            "next_doc_index",
            "is_answerable",
            "draft_sentences",
            "rewrite_requests",
            "pending_rewrite_count",
            "loop_count",
            "retrieval_retry_count",
            "mechanical_results",
            "final_sentences",
            "audit_trail",
        }
        assert required_keys.issubset(state.keys())

    def test_initial_state_zero_values(self) -> None:
        state = make_initial_state("r", "q", {}, {}, {})
        assert state["indexed_chunks"] == []
        assert state["next_doc_index"] == 1
        assert state["rewrite_requests"] == []
        assert state["audit_trail"] == []
        assert state["loop_count"] == 0
        assert state["retrieval_retry_count"] == 0
        assert state["mechanical_results"] == {}
        assert state["is_answerable"] is True
        assert state["draft_sentences"] == []
        assert state["final_sentences"] == []

    def test_indexed_chunks_is_plain_list_not_append_reducer(self) -> None:
        annotation = get_type_hints(GraphState)["indexed_chunks"]
        assert annotation == list[dict]

    def test_operator_add_reducer_appends_audit_trail(self) -> None:
        """Audit events from multiple nodes accumulate; no event is ever lost."""
        trail_after_retriever: list[dict] = [{"node": "retriever", "event": "search_complete"}]
        trail_after_synthesizer: list[dict] = [{"node": "synthesizer", "event": "draft_ready"}]
        full_trail = operator.add(trail_after_retriever, trail_after_synthesizer)
        assert len(full_trail) == 2
        assert full_trail[0]["node"] == "retriever"
        assert full_trail[1]["node"] == "synthesizer"

    def test_rewrite_requests_is_plain_list_not_append_reducer(self) -> None:
        annotation = get_type_hints(GraphState)["rewrite_requests"]
        assert annotation == list[str]

    def test_loop_count_initial_value(self) -> None:
        state = make_initial_state("r", "q", {}, {}, {})
        assert state["loop_count"] == 0

    def test_state_is_typeddict(self) -> None:
        """GraphState must be a TypedDict, not a Pydantic model or dataclass."""

        hints = GraphState.__annotations__
        assert "user_query" in hints
        assert "indexed_chunks" in hints
        assert "audit_trail" in hints
