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


class TestErrorClassification:
    def test_provider_connection_error_is_infrastructure(self) -> None:
        import litellm

        cause = litellm.APIConnectionError(
            message="Cannot connect", llm_provider="ollama", model="m"
        )
        try:
            raise RuntimeError("Synthesizer stage failed") from cause
        except RuntimeError as exc:
            assert cal.classify_error(exc) == "infra"

    def test_plain_connection_and_timeout_errors_are_infrastructure(self) -> None:
        assert cal.classify_error(ConnectionError("refused")) == "infra"
        assert cal.classify_error(TimeoutError()) == "infra"

    def test_malformed_model_output_is_a_pipeline_failure(self) -> None:
        try:
            raise RuntimeError("Synthesizer stage failed") from ValueError("not valid JSON")
        except RuntimeError as exc:
            assert cal.classify_error(exc) == "pipeline"


class TestResume:
    def test_latest_record_per_question_wins(self) -> None:
        records = [
            {"sample_id": "a", "error": "x", "error_kind": "infra"},
            {"sample_id": "b", "status": "success"},
            {"sample_id": "a", "status": "success"},
        ]
        latest = cal.latest_records(records)
        assert [r["sample_id"] for r in latest] == ["a", "b"]
        assert latest[0]["status"] == "success"

    @pytest.mark.parametrize(
        ("record", "retry"),
        [
            ({"sample_id": "a", "status": "success"}, False),
            ({"sample_id": "a", "error": "e", "error_kind": "infra"}, True),
            ({"sample_id": "a", "error": "e", "error_kind": "pipeline"}, False),
            # Records written before error_kind existed are classified by message.
            (
                {
                    "sample_id": "a",
                    "error": "RuntimeError: Synthesizer stage failed: litellm.APIConnectionError: "
                    "OllamaException - Cannot connect",
                },
                True,
            ),
            (
                {
                    "sample_id": "a",
                    "error": "RuntimeError: Synthesizer stage failed: LLM response is not valid JSON",
                },
                False,
            ),
        ],
    )
    def test_only_infrastructure_failures_are_retried(self, record: dict, retry: bool) -> None:
        assert cal.needs_retry(record) is retry

    def test_report_ignores_infrastructure_failures_but_counts_pipeline_ones(self) -> None:
        records = [
            {"sample_id": "a", "error": "e", "error_kind": "infra"},
            {"sample_id": "b", "error": "e", "error_kind": "pipeline"},
            {
                "sample_id": "c",
                "status": "success",
                "overall_score": 0.6,
                "answer_text": "Ali Daei",
                "qa_pairs": [{"short_answers": ["Ali Daei"]}],
                "sentences": [],
            },
        ]
        report = cal.build_report(records, [])
        assert report["questions"] == 2  # the infra failure is not a result
        assert report["errors"] == 1
        assert report["pending_infra"] == 1


class TestRunPending:
    async def test_stops_after_consecutive_infrastructure_failures(self, tmp_path) -> None:
        calls: list[str] = []

        async def runner(row: dict) -> dict:
            calls.append(row["sample_id"])
            return {"sample_id": row["sample_id"], "error": "down", "error_kind": "infra"}

        rows = [{"sample_id": str(i)} for i in range(10)]
        stopped = await cal.run_pending(
            rows, runner, tmp_path / "runs.jsonl", max_consecutive_infra=3
        )
        assert calls == ["0", "1", "2"]
        assert stopped is True

    async def test_pipeline_failures_do_not_stop_the_run(self, tmp_path) -> None:
        async def runner(row: dict) -> dict:
            return {"sample_id": row["sample_id"], "error": "bad json", "error_kind": "pipeline"}

        rows = [{"sample_id": str(i)} for i in range(5)]
        stopped = await cal.run_pending(
            rows, runner, tmp_path / "runs.jsonl", max_consecutive_infra=3
        )
        assert stopped is False
        assert len(cal._read_jsonl(tmp_path / "runs.jsonl")) == 5
