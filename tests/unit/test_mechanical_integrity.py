"""
Mechanical verifier — verbatim integrity.

Normalization may forgive *formatting* (case, smart quotes, whitespace, a hyphen
rendered as a space), but it must never make a *different* quote match. Every
failing case below altered the meaning of the source and previously passed,
because punctuation was deleted (fusing tokens) instead of treated as a word
boundary.
"""

from __future__ import annotations

import pytest

from axiom_rag_engine.verifiers.mechanical import MechanicalVerifier


@pytest.fixture()
def verifier() -> MechanicalVerifier:
    return MechanicalVerifier()


class TestVerbatimIntegrity:
    def test_decimal_point_shift_fails(self, verifier: MechanicalVerifier) -> None:
        chunk = "Revenue rose 15 percent in 2023."
        result = verifier.verify("doc_1_chunk_A", chunk, "Revenue rose 1.5 percent in 2023")
        assert result.status == "failed"

    def test_decimal_suffix_does_not_match_larger_number(
        self, verifier: MechanicalVerifier
    ) -> None:
        chunk = "Inflation reached 21.5 percent last year."
        result = verifier.verify("doc_1_chunk_A", chunk, "1.5 percent last year")
        assert result.status == "failed"

    def test_dropped_negative_sign_fails(self, verifier: MechanicalVerifier) -> None:
        chunk = "The temperature was -5 degrees overnight."
        result = verifier.verify("doc_1_chunk_A", chunk, "The temperature was 5 degrees overnight")
        assert result.status == "failed"

    def test_unicode_minus_matches_ascii_minus(self, verifier: MechanicalVerifier) -> None:
        chunk = "The temperature was \u22125 degrees overnight."  # U+2212 MINUS SIGN
        result = verifier.verify("doc_1_chunk_A", chunk, "The temperature was -5 degrees overnight")
        assert result.status == "passed"

    def test_numeric_range_is_not_a_sign(self, verifier: MechanicalVerifier) -> None:
        chunk = "Adults need 7-9 hours of sleep per night."
        result = verifier.verify("doc_1_chunk_A", chunk, "Adults need 7-9 hours of sleep")
        assert result.status == "passed"

    def test_single_long_latin_word_is_too_short(self, verifier: MechanicalVerifier) -> None:
        # The 12-character floor exists for unspaced scripts (CJK); applied to
        # Latin text it let a single word stand in for a quote.
        chunk = "Cardiovascular disease is common in adults."
        result = verifier.verify("doc_1_chunk_A", chunk, "cardiovascular")
        assert result.status == "failed"
        assert "too short" in result.audit_proof["failure_reason"].lower()

    def test_partial_word_at_edge_fails(self, verifier: MechanicalVerifier) -> None:
        chunk = "Then the cat sat on the mat quietly."
        result = verifier.verify("doc_1_chunk_A", chunk, "hen the cat sat on the mat")
        assert result.status == "failed"

    def test_hyphen_vs_space_is_formatting_only(self, verifier: MechanicalVerifier) -> None:
        chunk = "It is a well-known result in thermodynamics."
        result = verifier.verify("doc_1_chunk_A", chunk, "a well known result in thermodynamics")
        assert result.status == "passed"

    def test_cjk_quote_inside_unspaced_text_passes(self, verifier: MechanicalVerifier) -> None:
        # Word boundaries do not apply to unspaced scripts: a quote may start
        # mid-run.
        chunk = "据报道巴黎是法国的首都也是法国最大的城市位于塞纳河畔"
        result = verifier.verify("doc_1_chunk_A", chunk, "巴黎是法国的首都也是法国最大的城市")
        assert result.status == "passed"


class TestMatchedSourceText:
    """A pass returns the exact source text the quote matched, so the response
    can show what the source says rather than the model's rendering of it."""

    def test_matched_text_is_the_raw_source_span(self, verifier: MechanicalVerifier) -> None:
        chunk = "Per the report, Revenue rose 15.5% in 2023, driven by exports."
        result = verifier.verify("doc_1_chunk_A", chunk, "revenue rose 15.5 in 2023")
        assert result.status == "passed"
        assert result.matched_source_text == "Revenue rose 15.5% in 2023"
        assert result.audit_proof["matched_source_text"] == "Revenue rose 15.5% in 2023"

    def test_matched_text_for_unspaced_script(self, verifier: MechanicalVerifier) -> None:
        chunk = "巴黎是法国的首都，也是法国最大的城市，位于塞纳河畔。"
        result = verifier.verify("doc_1_chunk_A", chunk, "巴黎是法国的首都，也是法国最大的城市")
        assert result.matched_source_text == "巴黎是法国的首都，也是法国最大的城市"

    def test_matched_text_spans_accents_and_smart_quotes(
        self, verifier: MechanicalVerifier
    ) -> None:
        chunk = "She said “the café opens at nine” every day."
        result = verifier.verify("doc_1_chunk_A", chunk, '"the cafe opens at nine"')
        assert result.matched_source_text == "the café opens at nine"

    def test_failed_result_has_no_matched_text(self, verifier: MechanicalVerifier) -> None:
        result = verifier.verify(
            "doc_1_chunk_A", "Water boils at 100 degrees.", "ice melts at zero degrees"
        )
        assert result.matched_source_text is None


class TestNormalizationMapConsistency:
    """The offset-tracking normalizer must produce exactly the same text as the
    plain one — the span it reports is only trustworthy if they agree."""

    @pytest.mark.parametrize(
        "text",
        [
            "Hello, World!",
            "  leading and trailing  ",
            "state\u2013of\u2013the\u2013art",
            "“Smart” quotes… and café",
            "Straße is German",  # casefold expands ß -> ss
            "ال\u0652ع\u0650ل\u0652م\u064f",  # Arabic harakat
            "a \u22125 and 7-9 and COVID-19",
            "hel\u200blo wor\u00adld",
        ],
    )
    def test_mapped_normalization_matches_plain(self, text: str) -> None:
        norm, index = MechanicalVerifier._normalize_with_map(text)
        assert norm == MechanicalVerifier._normalize_text(text)
        assert len(index) == len(norm)
