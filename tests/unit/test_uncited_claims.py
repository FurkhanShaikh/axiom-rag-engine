"""
Uncited sentences that read as claims make the response partial.

The synthesizer may leave only transitional sentences uncited, but nothing
enforced it: an answer with verified sentences plus an uncited "Tesla sold 1.8
million cars in 2023" came back ``status: "success"`` with a high score. An
uncited sentence carrying numbers or names now makes the response ``partial``,
and the response reports how many sentences were uncited.
"""

from __future__ import annotations

from typing import Any

import pytest

from axiom_rag_engine.scoring import (
    compute_confidence_summary,
    determine_status,
    has_checkable_content,
)
from tests.conftest import make_final_sentence_dict


def _uncited(text: str) -> dict[str, Any]:
    return {
        "sentence_id": "s_99",
        "text": text,
        "is_cited": False,
        "citations": [],
        "verification": {
            "tier": 3,
            "tier_label": "unverified",
            "mechanical_check": "skipped",
            "semantic_check": "skipped",
            "failure_reason": "Uncited sentence — no source quote was checked.",
        },
    }


@pytest.mark.parametrize(
    "text",
    [
        "Tesla sold 1.8 million cars in 2023.",
        "The trial enrolled 2,686 patients.",
        "This was confirmed by the World Health Organization.",
        "Sales rose sharply after Ford entered the market.",
    ],
)
def test_numbers_and_names_are_checkable(text: str) -> None:
    assert has_checkable_content(text)


@pytest.mark.parametrize(
    "text",
    [
        "In summary, the evidence points in one direction.",
        "Overall, these findings are consistent with each other.",
        "However, the picture is more nuanced than it first appears.",
        "I would treat these conclusions with some caution.",
    ],
)
def test_transitional_sentences_are_not_checkable(text: str) -> None:
    assert not has_checkable_content(text)


def test_uncited_claim_makes_the_response_partial() -> None:
    sentences = [make_final_sentence_dict(tier=1), _uncited("Tesla sold 1.8 million cars.")]
    assert determine_status(True, sentences) == "partial"


def test_uncited_transition_does_not_change_status() -> None:
    sentences = [make_final_sentence_dict(tier=1), _uncited("In summary, both sources agree.")]
    assert determine_status(True, sentences) == "success"


def test_summary_reports_uncited_sentences() -> None:
    sentences = [
        make_final_sentence_dict(tier=1),
        _uncited("In summary, both sources agree."),
        _uncited("Tesla sold 1.8 million cars."),
    ]
    summary = compute_confidence_summary(sentences)
    assert summary.uncited_sentences == 2
    assert summary.uncited_checkable_sentences == 1
    # Uncited sentences still do not enter the score or the tier breakdown.
    assert summary.overall_score == 1.0
    assert summary.tier_breakdown.tier_3_claims == 0
