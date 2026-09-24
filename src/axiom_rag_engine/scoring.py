"""
Axiom Engine — Confidence scoring and status determination.

Extracted from main.py — these are pure domain functions with no HTTP dependency.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from axiom_rag_engine.models import ConfidenceSummary, TierBreakdown

# Tier weights for the overall confidence score (architecture §4):
#   Tier 1 (Authoritative)    → 1.0
#   Tier 2 (Multi-Domain)     → 0.85
#   Tier 3 (Model Assisted)   → 0.60
#   Tier 3 (Unverified)       → 0.30  (cited, semantic check could not run)
#   Tier 4 (Misrepresented)   → 0.20  (should rarely survive to final output)
#   Tier 5 (Hallucinated)     → 0.00  (should never survive to final output)
#   Tier 6 (Conflicted)       → 0.40  (opt-in; below Tier 3 — conflicting sources
#                                       are worse than a single faithful one, but
#                                       above a misrepresentation)
_TIER_WEIGHTS: dict[int, float] = {
    1: 1.0,
    2: 0.85,
    3: 0.60,
    4: 0.20,
    5: 0.00,
    6: 0.40,
}

# A cited claim whose semantic check could not run (tier 3, tier_label
# "unverified"): the quote is verbatim but faithfulness is unknown, so it scores
# well below a checked Tier 3 claim.
_UNVERIFIED_WEIGHT = 0.30


# A digit anywhere, or a capitalised word after the first (a name, a place, an
# organisation). Sentence-initial capitals and the pronoun "I" do not count.
_DIGIT_RE = re.compile(r"\d")
_WORD_RE = re.compile(r"[^\W\d_][\w'’-]*")


def has_checkable_content(text: str) -> bool:
    """Whether an uncited sentence carries content a source could confirm or refute.

    The synthesizer may leave only transitional or summary sentences uncited, but
    nothing enforced that: an uncited "Tesla sold 1.8 million cars in 2023" was
    ignored by the status and the score. Numbers and names are the cheap,
    deterministic signal that a sentence states a fact rather than connects two.
    """
    if _DIGIT_RE.search(text):
        return True
    words = _WORD_RE.findall(text)
    return any(w[0].isupper() and w != "I" for w in words[1:])


def _is_uncited_checkable(sentence: dict[str, Any]) -> bool:
    return not sentence.get("is_cited") and has_checkable_content(str(sentence.get("text", "")))


def _is_claim(sentence: dict[str, Any]) -> bool:
    """Only cited sentences are claims. Uncited (transitional) sentences carry no
    checked quote, so they are excluded from the tier breakdown, the confidence
    score, and the success decision — they neither inflate nor deflate it."""
    return bool(sentence.get("is_cited"))


def _is_unverified(sentence: dict[str, Any]) -> bool:
    return bool(sentence.get("verification", {}).get("tier_label") == "unverified")


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
        if not _is_claim(sentence):
            continue
        vr = sentence.get("verification", {})
        tier: int = vr.get("tier", 3)

        attr = f"tier_{tier}_claims"
        setattr(breakdown, attr, getattr(breakdown, attr, 0) + 1)

        weight = _UNVERIFIED_WEIGHT if _is_unverified(sentence) else _TIER_WEIGHTS.get(tier, 0.0)
        weighted_sum += weight
        total_claims += 1

    overall = round(weighted_sum / total_claims, 4) if total_claims > 0 else 0.0

    return ConfidenceSummary(
        overall_score=overall,
        tier_breakdown=breakdown,
        uncited_sentences=sum(1 for s in final_sentences if not _is_claim(s)),
        uncited_checkable_sentences=sum(1 for s in final_sentences if _is_uncited_checkable(s)),
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
      - "success" if every cited sentence is Tier 1–3 and was fully verified.
      - "partial" if any cited sentence is Tier 4, 5, or 6, or is labelled
        "unverified" (its semantic check could not run), or if the answer has
        no cited sentence at all (nothing in it was checked).
      - "partial" if an uncited sentence carries checkable content (numbers or
        names): it reads as a claim, yet nothing in it was checked.
      - Other uncited (transitional) sentences are ignored for the decision.
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

    claims = [s for s in final_sentences if _is_claim(s)]
    if not claims:
        # Text was produced but none of it was checked against a source.
        return "partial"

    if any(_is_uncited_checkable(s) for s in final_sentences):
        return "partial"

    for s in claims:
        tier = s.get("verification", {}).get("tier", 3)
        if tier in (4, 5, 6) or _is_unverified(s):
            return "partial"

    return "success"
