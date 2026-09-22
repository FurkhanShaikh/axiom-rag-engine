"""
Verification contract tests — the README tier table, enforced.

These tests pin what each tier *promises*, not how the code currently happens to
behave. In particular they cover the paths where verification did not actually
run (verifier error, unparseable verifier output, exhausted budget, uncited
sentences): such sentences must be labelled ``unverified`` and must never let a
response report ``status="success"``.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from axiom_rag_engine.models import VerificationResult
from axiom_rag_engine.nodes.semantic import semantic_verifier_node
from axiom_rag_engine.scoring import compute_confidence_summary, determine_status
from axiom_rag_engine.state import make_initial_state
from axiom_rag_engine.utils.llm import LLMBudgetExceededError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CHUNK = {
    "chunk_id": "doc_1_chunk_A",
    "text": (
        "Solid-state batteries replace liquid electrolytes with solid ceramics. "
        "This substitution significantly improves thermal stability and energy density."
    ),
    "source_url": "https://example.com/batteries",
    "domain": "example.com",
}

_QUOTE = "Solid-state batteries replace liquid electrolytes with solid ceramics."


def _response(content: str) -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content=content))]
    return response


def _cited_draft(sentence_id: str = "s_01", citation_id: str = "cite_1") -> dict[str, Any]:
    return {
        "sentence_id": sentence_id,
        "text": "Solid-state batteries use solid ceramic electrolytes.",
        "is_cited": True,
        "citations": [
            {"citation_id": citation_id, "chunk_id": "doc_1_chunk_A", "exact_source_quote": _QUOTE}
        ],
    }


def _uncited_draft(sentence_id: str = "s_02") -> dict[str, Any]:
    return {
        "sentence_id": sentence_id,
        "text": "In summary, the technology shows promise.",
        "is_cited": False,
        "citations": [],
    }


def _mech_passed() -> dict[str, Any]:
    return {
        "tier": 3,
        "tier_label": "model_assisted",
        "mechanical_check": "passed",
        "semantic_check": "skipped",
        "failure_reason": None,
    }


def _state(draft: list[dict[str, Any]], semantic_enabled: bool = True) -> dict[str, Any]:
    state = make_initial_state(
        request_id="req_contract",
        user_query="What is a solid-state battery?",
        app_config={},
        models_config={"synthesizer": "synth-model", "verifier": "verifier-model"},
        pipeline_config={"stages": {"semantic_verification_enabled": semantic_enabled}},
    )
    state["indexed_chunks"] = [_CHUNK]
    state["draft_sentences"] = draft
    state["mechanical_results"] = {
        c["citation_id"]: _mech_passed() for s in draft for c in s["citations"]
    }
    return dict(state)


def _sentence(tier: int, label: str, mech: str, sem: str, *, cited: bool = True) -> dict:
    return {
        "sentence_id": "s_x",
        "text": "x",
        "is_cited": cited,
        "citations": [],
        "verification": {
            "tier": tier,
            "tier_label": label,
            "mechanical_check": mech,
            "semantic_check": sem,
            "failure_reason": None,
        },
    }


VERIFIED_T1 = _sentence(1, "authoritative", "passed", "passed")
VERIFIED_T3 = _sentence(3, "model_assisted", "passed", "passed")
UNVERIFIED_CITED = _sentence(3, "unverified", "passed", "skipped")
UNCITED = _sentence(3, "unverified", "skipped", "skipped", cited=False)


# ---------------------------------------------------------------------------
# Schema: the tier contract is enforced by VerificationResult itself
# ---------------------------------------------------------------------------


class TestUnverifiedLabelContract:
    def test_unverified_is_a_valid_tier_3_label(self) -> None:
        vr = VerificationResult(
            tier=3,
            tier_label="unverified",
            mechanical_check="passed",
            semantic_check="skipped",
        )
        assert vr.tier_label == "unverified"

    def test_unverified_requires_tier_3(self) -> None:
        with pytest.raises(ValidationError, match="unverified"):
            VerificationResult(
                tier=4,
                tier_label="unverified",
                mechanical_check="passed",
                semantic_check="failed",
            )

    def test_unverified_cannot_claim_both_checks_passed(self) -> None:
        with pytest.raises(ValidationError, match="unverified"):
            VerificationResult(
                tier=3,
                tier_label="unverified",
                mechanical_check="passed",
                semantic_check="passed",
            )

    def test_model_assisted_requires_mechanical_pass(self) -> None:
        # An uncited sentence (no quote was ever checked) must not wear the
        # "model_assisted" label, which promises a verbatim quote.
        with pytest.raises(ValidationError, match="model_assisted"):
            VerificationResult(
                tier=3,
                tier_label="model_assisted",
                mechanical_check="skipped",
                semantic_check="skipped",
            )


# ---------------------------------------------------------------------------
# Semantic node: verification that did not run is labelled, not hidden
# ---------------------------------------------------------------------------


class TestVerifierFailuresAreUnverified:
    @pytest.mark.parametrize(
        "failure",
        [
            Exception("provider down"),
            TimeoutError("verifier timed out"),
            LLMBudgetExceededError("budget exhausted"),
        ],
        ids=["api_error", "timeout", "budget_exhausted"],
    )
    async def test_verifier_exception_marks_sentence_unverified(self, failure) -> None:
        with patch(
            "axiom_rag_engine.nodes.semantic.litellm.acompletion",
            new_callable=AsyncMock,
            side_effect=failure,
        ):
            result = await semantic_verifier_node(_state([_cited_draft()]))
        sentence = result["final_sentences"][0]
        assert sentence["verification"]["tier"] == 3
        assert sentence["verification"]["tier_label"] == "unverified"
        assert sentence["citations"][0]["verification"]["tier_label"] == "unverified"

    async def test_non_json_verifier_output_marks_sentence_unverified(self) -> None:
        with patch(
            "axiom_rag_engine.nodes.semantic.litellm.acompletion",
            new_callable=AsyncMock,
            return_value=_response("Sure! The claim looks faithful to me."),
        ):
            result = await semantic_verifier_node(_state([_cited_draft()]))
        assert result["final_sentences"][0]["verification"]["tier_label"] == "unverified"

    async def test_semantic_disabled_by_policy_stays_model_assisted(self) -> None:
        # An operator *choosing* to skip Stage 2 is not a failure: Tier 3 is
        # defined as "mechanical pass; semantic passed or disabled".
        result = await semantic_verifier_node(_state([_cited_draft()], semantic_enabled=False))
        assert result["final_sentences"][0]["verification"]["tier_label"] == "model_assisted"

    async def test_uncited_sentence_is_unverified_without_rewrite(self) -> None:
        with patch(
            "axiom_rag_engine.nodes.semantic.litellm.acompletion", new_callable=AsyncMock
        ) as mock_llm:
            result = await semantic_verifier_node(_state([_uncited_draft()]))
        vr = result["final_sentences"][0]["verification"]
        assert (vr["tier"], vr["tier_label"]) == (3, "unverified")
        assert vr["mechanical_check"] == "skipped"
        assert result["rewrite_requests"] == []
        mock_llm.assert_not_called()


# ---------------------------------------------------------------------------
# Response status + confidence score
# ---------------------------------------------------------------------------


class TestStatusHonesty:
    def test_unverified_cited_sentence_makes_response_partial(self) -> None:
        assert determine_status(True, [VERIFIED_T1, UNVERIFIED_CITED]) == "partial"

    def test_uncited_sentences_do_not_block_success(self) -> None:
        assert determine_status(True, [VERIFIED_T1, UNCITED]) == "success"

    def test_only_uncited_sentences_is_not_success(self) -> None:
        assert determine_status(True, [UNCITED, UNCITED]) == "partial"

    def test_fully_verified_answer_is_success(self) -> None:
        assert determine_status(True, [VERIFIED_T1, VERIFIED_T3]) == "success"


class TestConfidenceHonesty:
    def test_uncited_sentences_are_not_counted_as_claims(self) -> None:
        summary = compute_confidence_summary([VERIFIED_T1, UNCITED])
        assert summary.tier_breakdown.tier_3_claims == 0
        assert summary.overall_score == 1.0

    def test_unverified_claim_scores_below_model_assisted(self) -> None:
        verified = compute_confidence_summary([VERIFIED_T3]).overall_score
        unverified = compute_confidence_summary([UNVERIFIED_CITED]).overall_score
        assert unverified < verified


# ---------------------------------------------------------------------------
# End to end: a broken verifier can no longer produce status="success"
# ---------------------------------------------------------------------------


class TestBrokenVerifierEndToEnd:
    def test_verifier_returning_prose_yields_partial_not_success(self, client) -> None:
        from axiom_rag_engine.nodes.retriever import MockSearchBackend, set_search_backend

        set_search_backend(
            MockSearchBackend(
                [{"url": _CHUNK["source_url"], "content": _CHUNK["text"], "title": "Batteries"}]
            )
        )
        synth = json.dumps({"is_answerable": True, "sentences": [_cited_draft()]})

        async def fake_llm(**kwargs: Any) -> MagicMock:
            system = kwargs["messages"][0]["content"]
            if "Cognitive Synthesizer" in system:
                return _response(synth)
            return _response("Looks faithful to me!")  # verifier ignores the JSON contract

        with patch("litellm.acompletion", side_effect=fake_llm):
            resp = client.post(
                "/v1/synthesize",
                json={"request_id": "broken-verifier", "user_query": "solid-state batteries"},
            )
        body = resp.json()
        assert resp.status_code == 200
        assert body["status"] == "partial"
        assert body["final_response"][0]["verification"]["tier_label"] == "unverified"


# ---------------------------------------------------------------------------
# Citations carry the exact source text, not only the model's rendering
# ---------------------------------------------------------------------------


class TestMatchedSourceTextSurfaced:
    async def test_verified_citation_exposes_matched_source_text(self) -> None:
        from axiom_rag_engine.nodes.verification import verification_node

        draft = _cited_draft()
        draft["citations"][0]["exact_source_quote"] = (
            "solid state batteries replace liquid electrolytes"  # re-cased, hyphen dropped
        )
        state = _state([draft], semantic_enabled=False)
        state["mechanical_results"] = {}
        result = await verification_node(state)
        citation = result["final_sentences"][0]["citations"][0]
        assert (
            citation["matched_source_text"] == "Solid-state batteries replace liquid electrolytes"
        )

    async def test_failed_citation_has_no_matched_source_text(self) -> None:
        from axiom_rag_engine.nodes.verification import verification_node

        draft = _cited_draft()
        draft["citations"][0]["exact_source_quote"] = "liquid batteries are always unsafe at home"
        state = _state([draft], semantic_enabled=False)
        state["mechanical_results"] = {}
        result = await verification_node(state)
        assert result["final_sentences"][0]["citations"][0]["matched_source_text"] is None
