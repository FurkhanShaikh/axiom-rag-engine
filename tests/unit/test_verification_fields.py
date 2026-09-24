"""
The tier answers three questions; separate fields now answer each (VER-5).

Tiers 1-2 describe the source, 3-5 whether the claim matches its source, and 6
disagreement between sources. ``faithfulness``, ``source_class`` and
``agreement`` split them; ``tier`` stays for compatibility, and the confidence
summary's ``overall_score`` gains its honest name, ``grounding_score``.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from axiom_rag_engine.models import ConfidenceSummary, TierBreakdown, VerificationResult
from axiom_rag_engine.nodes.semantic import _source_class


def _vr(**fields: Any) -> VerificationResult:
    return VerificationResult(**fields)


@pytest.mark.parametrize(
    ("tier", "label", "mechanical", "semantic", "faithfulness"),
    [
        (1, "authoritative", "passed", "passed", "verified"),
        (2, "multi_source", "passed", "passed", "verified"),
        (3, "model_assisted", "passed", "passed", "verified"),
        (3, "model_assisted", "passed", "skipped", "not_checked"),  # semantic disabled
        (3, "unverified", "passed", "failed", "misrepresented"),
        (3, "unverified", "skipped", "skipped", "not_checked"),  # uncited
        (4, "misrepresented", "passed", "failed", "misrepresented"),
        (5, "hallucinated", "failed", "skipped", "not_found"),
        (6, "conflicted", "passed", "passed", "verified"),
    ],
)
def test_faithfulness_follows_the_checks(
    tier: int, label: str, mechanical: str, semantic: str, faithfulness: str
) -> None:
    vr = _vr(tier=tier, tier_label=label, mechanical_check=mechanical, semantic_check=semantic)
    assert vr.faithfulness == faithfulness


def test_tier_implied_fields_are_filled_in() -> None:
    t1 = _vr(tier=1, tier_label="authoritative", mechanical_check="passed", semantic_check="passed")
    t6 = _vr(tier=6, tier_label="conflicted", mechanical_check="passed", semantic_check="passed")
    t3 = _vr(
        tier=3, tier_label="model_assisted", mechanical_check="passed", semantic_check="passed"
    )
    assert (t1.source_class, t1.agreement) == ("primary", "not_checked")
    assert (t6.source_class, t6.agreement) == ("other", "conflicted")
    assert (t3.source_class, t3.agreement) == ("other", "not_checked")


def test_contradictory_fields_are_rejected() -> None:
    with pytest.raises(ValidationError, match="primary"):
        _vr(
            tier=1,
            tier_label="authoritative",
            mechanical_check="passed",
            semantic_check="passed",
            source_class="other",
        )
    with pytest.raises(ValidationError, match="Tier 6"):
        _vr(
            tier=3,
            tier_label="model_assisted",
            mechanical_check="passed",
            semantic_check="passed",
            agreement="conflicted",
        )


def test_fields_survive_a_round_trip() -> None:
    vr = _vr(
        tier=2,
        tier_label="multi_source",
        mechanical_check="passed",
        semantic_check="passed",
        agreement="corroborated",
    )
    dumped = vr.model_dump()
    assert dumped["faithfulness"] == "verified"
    assert VerificationResult.model_validate(dumped) == vr


def test_grounding_score_is_the_overall_score() -> None:
    summary = ConfidenceSummary(overall_score=0.72, tier_breakdown=TierBreakdown())
    dumped = summary.model_dump()
    assert dumped["grounding_score"] == dumped["overall_score"] == 0.72
    assert ConfidenceSummary.model_validate(dumped) == summary


def test_source_class_reflects_what_was_cited_whatever_the_verdict() -> None:
    class _Cite:
        def __init__(self, chunk_id: str) -> None:
            self.chunk_id = chunk_id

    lookup = {
        "gov": {"domain": "nih.gov", "source_url": "https://nih.gov/report"},
        "blog": {"domain": "blog.example.com", "source_url": "https://blog.example.com/p"},
    }
    assert _source_class([_Cite("blog"), _Cite("gov")], lookup, set()) == "primary"  # type: ignore[list-item]
    assert _source_class([_Cite("blog")], lookup, set()) == "other"  # type: ignore[list-item]


async def test_uncited_sentences_cite_nothing() -> None:
    from axiom_rag_engine.nodes.semantic import semantic_verifier_node
    from axiom_rag_engine.state import make_initial_state

    state = make_initial_state("r", "q", {}, {"synthesizer": "m/s", "verifier": "m/v"}, {})
    state["draft_sentences"] = [{"sentence_id": "s_01", "text": "In short:", "is_cited": False}]
    result = await semantic_verifier_node(state)
    vr = result["final_sentences"][0]["verification"]
    assert (vr["source_class"], vr["faithfulness"], vr["agreement"]) == (
        "none",
        "not_checked",
        "not_checked",
    )
