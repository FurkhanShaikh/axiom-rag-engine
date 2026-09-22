"""Calibration eval harness: the pure pieces (no network, no LLM)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import ClassVar

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evals"))

import calibration_eval as cal


def _v(tier: int, label: str) -> dict:
    return {"tier": tier, "tier_label": label}


class TestStrEm:
    """ALCE's STR-EM: share of gold QA pairs with any short answer in the text."""

    QA: ClassVar[list[dict]] = [
        {"short_answers": ["Ali Daei", "Daei"]},
        {"short_answers": ["Josef Bican"]},
        {"short_answers": ["Christine Sinclair"]},
    ]

    def test_counts_each_answered_interpretation(self) -> None:
        text = "Ali Daei scored 109 international goals; Christine Sinclair leads women's football."
        assert cal.str_em(text, self.QA) == pytest.approx(2 / 3)

    def test_matching_ignores_case_punctuation_and_articles(self) -> None:
        assert (
            cal.str_em("the record belongs to JOSEF BICAN.", [{"short_answers": ["Josef Bican"]}])
            == 1.0
        )

    def test_requires_whole_words(self) -> None:
        assert cal.str_em("Daeiology is not a person.", [{"short_answers": ["Daei"]}]) == 0.0

    def test_no_gold_answers_scores_zero(self) -> None:
        assert cal.str_em("anything", []) == 0.0


class TestWilson:
    def test_interval_brackets_the_rate(self) -> None:
        low, high = cal.wilson_interval(8, 10)
        assert low < 0.8 < high
        assert low >= 0.0 and high <= 1.0

    def test_empty_bucket_is_uninformative(self) -> None:
        assert cal.wilson_interval(0, 0) == (0.0, 1.0)


class TestBucket:
    @pytest.mark.parametrize(
        ("sentence", "bucket"),
        [
            ({"is_cited": True, "verification": _v(1, "authoritative")}, "T1"),
            ({"is_cited": True, "verification": _v(2, "multi_source")}, "T2"),
            ({"is_cited": True, "verification": _v(3, "model_assisted")}, "T3"),
            ({"is_cited": True, "verification": _v(3, "unverified")}, "T3-unverified"),
            ({"is_cited": False, "verification": _v(3, "unverified")}, "uncited"),
            ({"is_cited": True, "verification": _v(4, "misrepresented")}, "T4"),
            ({"is_cited": True, "verification": _v(5, "hallucinated")}, "T5"),
            ({"is_cited": True, "verification": _v(6, "conflicted")}, "T6"),
        ],
    )
    def test_bucket_for_sentence(self, sentence: dict, bucket: str) -> None:
        assert cal.bucket_of(sentence) == bucket


class TestSummary:
    def test_supported_rate_and_weight_per_bucket(self) -> None:
        judged = [
            {"bucket": "T3", "verdict": "supported"},
            {"bucket": "T3", "verdict": "supported"},
            {"bucket": "T3", "verdict": "partial"},
            {"bucket": "T3", "verdict": "unsupported"},
            {"bucket": "T5", "verdict": "unsupported"},
        ]
        rows = {r["bucket"]: r for r in cal.summarize_buckets(judged)}
        assert rows["T3"]["n"] == 4
        assert rows["T3"]["supported_rate"] == pytest.approx(0.5)  # partial is not supported
        assert rows["T3"]["weight"] == pytest.approx(0.60)
        assert rows["T5"]["supported_rate"] == 0.0
        assert rows["T5"]["weight"] == 0.0

    def test_bucket_order_follows_tier_order(self) -> None:
        judged = [
            {"bucket": "T5", "verdict": "unsupported"},
            {"bucket": "uncited", "verdict": "supported"},
            {"bucket": "T1", "verdict": "supported"},
        ]
        assert [r["bucket"] for r in cal.summarize_buckets(judged)] == ["T1", "uncited", "T5"]


class TestSpearman:
    def test_perfect_monotone_relationship(self) -> None:
        assert cal.spearman([0.1, 0.4, 0.9], [1, 2, 3]) == pytest.approx(1.0)

    def test_inverse_relationship(self) -> None:
        assert cal.spearman([0.1, 0.4, 0.9], [3, 2, 1]) == pytest.approx(-1.0)

    def test_constant_input_is_undefined(self) -> None:
        assert cal.spearman([0.5, 0.5, 0.5], [1, 2, 3]) is None
