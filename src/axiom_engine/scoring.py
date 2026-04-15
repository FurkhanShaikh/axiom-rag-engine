"""
Axiom Engine — Confidence scoring and status determination.

Extracted from main.py — these are pure domain functions with no HTTP dependency.
"""

from __future__ import annotations

from typing import Any, Literal

from axiom_engine.models import ConfidenceSummary, TierBreakdown

# Tier weights for the overall confidence score (architecture §4):
#   Tier 1 (Authoritative)    → 1.0
#   Tier 2 (Multi-Source)     → 0.85
#   Tier 3 (Model Assisted)   → 0.60
#   Tier 4 (Misrepresented)   → 0.20  (should rarely survive to final output)
#   Tier 5 (Hallucinated)     → 0.00  (should never survive to final output)
#   Tier 6 (Conflicted)       → 0.40
_TIER_WEIGHTS: dict[int, float] = {
    1: 1.0,
    2: 0.85,
    3: 0.60,
    4: 0.20,
    5: 0.00,
    6: 0.40,
}


def compute_confidence_summary(
    final_sentences: list[dict[str, Any]],
) -> ConfidenceSummary:
    """
    Compute tier breakdown and weighted overall confidence score from
    the verified final_sentences produced by the graph.
    """
    breakdown = TierBreakdown()
    weighted_sum = 0.0
    total_claims = 0

    for sentence in final_sentences:
        vr = sentence.get("verification", {})
        tier: int = vr.get("tier", 3)

        attr = f"tier_{tier}_claims"
        setattr(breakdown, attr, getattr(breakdown, attr, 0) + 1)

        weighted_sum += _TIER_WEIGHTS.get(tier, 0.0)
        total_claims += 1

    overall = round(weighted_sum / total_claims, 4) if total_claims > 0 else 0.0

    return ConfidenceSummary(
        overall_score=overall,
        tier_breakdown=breakdown,
    )


def determine_status(
    is_answerable: bool,
    final_sentences: list[dict[str, Any]],
) -> Literal["success", "partial", "unanswerable", "error"]:
    """
    Determine the response status string.

    Rules:
      - "unanswerable" if escape hatch fired OR if the pipeline produced no
        sentences despite is_answerable=True (the answer could not be grounded).
      - "success" if all sentences are Tier 1–3.
      - "partial" if any sentence is Tier 4, 5, or 6 (mixed quality).
      - "error" comes only from exception handling, not here.

    M8 fix: empty final_sentences with is_answerable=True previously returned
    "partial", conflating "something verified, something not" with "nothing at all".
    It now returns "unanswerable" so callers can distinguish the two cases.
    """
    if not is_answerable:
        return "unanswerable"

    # Pipeline ran but produced no verifiable output — treat as unanswerable,
    # not partial.  Partial means some sentences exist but some failed.
    if not final_sentences:
        return "unanswerable"

    for s in final_sentences:
        tier = s.get("verification", {}).get("tier", 3)
        if tier in (4, 5, 6):
            return "partial"

    return "success"
