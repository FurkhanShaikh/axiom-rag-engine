"""
Phase 2 — TDD: MechanicalVerifier tests.

Level 3 (Verification Integrity) — 100% coverage mandated per architecture §8.

Test categories:
  A. Normalization unit tests (_normalize_text in isolation)
  B. Happy-path pass cases
  C. Tokenization artifact cases — must PASS via normalization (v2.3 patch)
  D. Hallucination cases — must FAIL (Tier 5)
  E. Audit proof structure validation
  F. Result immutability
"""

from __future__ import annotations

import pytest

from axiom_engine.verifiers.mechanical import (
    MechanicalVerifier,
)


@pytest.fixture(scope="module")
def verifier() -> MechanicalVerifier:
    return MechanicalVerifier()


# ===========================================================================
# A. _normalize_text — normalization unit tests
# ===========================================================================


class TestNormalizeText:
    def test_lowercases(self) -> None:
        assert MechanicalVerifier._normalize_text("Hello World") == "hello world"

    def test_strips_punctuation(self) -> None:
        assert MechanicalVerifier._normalize_text("Hello, World!") == "hello world"

    def test_collapses_internal_whitespace(self) -> None:
        assert MechanicalVerifier._normalize_text("hello   world") == "hello world"

    def test_strips_leading_trailing_whitespace(self) -> None:
        assert MechanicalVerifier._normalize_text("  hello  ") == "hello"

    def test_collapses_tabs_and_newlines(self) -> None:
        assert MechanicalVerifier._normalize_text("hello\t\nworld") == "hello world"

    def test_smart_quotes_left_single(self) -> None:
        # U+2018 LEFT SINGLE QUOTATION MARK → stripped as punctuation
        result = MechanicalVerifier._normalize_text("\u2018hello\u2019")
        assert result == "hello"

    def test_smart_quotes_left_double(self) -> None:
        # U+201C / U+201D → converted to " then stripped
        result = MechanicalVerifier._normalize_text("\u201chello world\u201d")
        assert result == "hello world"

    def test_non_breaking_space_treated_as_space(self) -> None:
        # U+00A0 → regular space → collapsed
        result = MechanicalVerifier._normalize_text("hello\u00a0world")
        assert result == "hello world"

    def test_en_dash_converted(self) -> None:
        # U+2013 → hyphen → then stripped as punctuation
        result = MechanicalVerifier._normalize_text("state\u2013of\u2013the\u2013art")
        assert result == "stateoftheart"

    def test_em_dash_converted(self) -> None:
        result = MechanicalVerifier._normalize_text("cutting\u2014edge")
        assert result == "cuttingedge"

    def test_zero_width_space_removed(self) -> None:
        result = MechanicalVerifier._normalize_text("hel\u200blo")
        assert result == "hello"

    def test_soft_hyphen_removed(self) -> None:
        result = MechanicalVerifier._normalize_text("hy\u00adphen")
        assert result == "hyphen"

    def test_ellipsis_char_expanded(self) -> None:
        # U+2026 → "..." → stripped as punctuation
        result = MechanicalVerifier._normalize_text("and so on\u2026")
        assert result == "and so on"

    def test_empty_string_returns_empty(self) -> None:
        assert MechanicalVerifier._normalize_text("") == ""

    def test_whitespace_only_returns_empty(self) -> None:
        assert MechanicalVerifier._normalize_text("   \t\n  ") == ""

    def test_punctuation_only_returns_empty(self) -> None:
        assert MechanicalVerifier._normalize_text("!@#$%^&*()") == ""

    def test_idempotent(self) -> None:
        text = "The sky is blue on a clear day."
        once = MechanicalVerifier._normalize_text(text)
        twice = MechanicalVerifier._normalize_text(once)
        assert once == twice


# ===========================================================================
# B. Happy-path PASS cases
# ===========================================================================


class TestPassCases:
    def test_exact_verbatim_match(self, verifier: MechanicalVerifier) -> None:
        chunk = "Solid-state batteries replace liquid electrolytes with solid ceramics."
        quote = "Solid-state batteries replace liquid electrolytes with solid ceramics."
        result = verifier.verify("doc_1_chunk_A", chunk, quote)
        assert result.status == "passed"
        assert result.tier is None

    def test_quote_is_substring_of_chunk(self, verifier: MechanicalVerifier) -> None:
        chunk = (
            "The defining characteristic is the substitution of liquid electrolytes "
            "with solid ceramic or polymer materials. This improves safety significantly."
        )
        quote = "substitution of liquid electrolytes with solid ceramic or polymer materials"
        result = verifier.verify("doc_4_chunk_C", chunk, quote)
        assert result.status == "passed"

    def test_quote_at_start_of_chunk(self, verifier: MechanicalVerifier) -> None:
        chunk = "Python is a high-level programming language."
        quote = "Python is a high-level"
        result = verifier.verify("doc_2_chunk_A", chunk, quote)
        assert result.status == "passed"

    def test_quote_at_end_of_chunk(self, verifier: MechanicalVerifier) -> None:
        # Quote must meet the minimum 4-token floor to pass.
        chunk = "Python is a high-level programming language."
        quote = "Python is a high-level programming language"
        result = verifier.verify("doc_2_chunk_A", chunk, quote)
        assert result.status == "passed"

    def test_short_quote_below_min_tokens_fails(self, verifier: MechanicalVerifier) -> None:
        # C1 fix: quotes shorter than 4 normalized tokens should fail regardless
        # of whether the tokens appear in the chunk — they are too ambiguous to
        # serve as meaningful citations.
        chunk = "The quick brown fox jumps over the lazy dog."
        result = verifier.verify("doc_1_chunk_A", chunk, "fox")
        assert result.status == "failed"
        assert "too short" in result.audit_proof["failure_reason"].lower()


# ===========================================================================
# C. Tokenization artifact cases — must PASS via normalization (v2.3 patch)
# ===========================================================================


class TestTokenizationArtifactsCausePass:
    """
    These cases simulate real LLM output artifacts. Without normalization they
    would produce false-positive Tier 5 failures. The v2.3 patch must handle all.
    """

    def test_smart_double_quotes_around_quote(self, verifier: MechanicalVerifier) -> None:
        chunk = "The sky is blue on a clear day."
        # LLM wrapped the quote in curly quotes
        llm_quote = "\u201cThe sky is blue on a clear day.\u201d"
        result = verifier.verify("doc_1_chunk_A", chunk, llm_quote)
        assert result.status == "passed", "Smart quotes must not cause a false Tier 5."

    def test_smart_single_quotes_in_quote(self, verifier: MechanicalVerifier) -> None:
        chunk = "It's a well-known fact that water boils at 100 degrees Celsius."
        # LLM used U+2019 RIGHT SINGLE QUOTATION MARK instead of apostrophe
        llm_quote = "It\u2019s a well-known fact that water boils at 100 degrees Celsius."
        result = verifier.verify("doc_1_chunk_A", chunk, llm_quote)
        assert result.status == "passed"

    def test_extra_internal_whitespace_in_quote(self, verifier: MechanicalVerifier) -> None:
        chunk = "Solid-state batteries have higher energy density."
        # LLM injected double-spaces
        llm_quote = "Solid-state  batteries  have  higher  energy  density."
        result = verifier.verify("doc_1_chunk_A", chunk, llm_quote)
        assert result.status == "passed", "Whitespace artifacts must not cause a false Tier 5."

    def test_non_breaking_space_in_quote(self, verifier: MechanicalVerifier) -> None:
        chunk = "The reaction rate increases with temperature."
        # LLM emitted U+00A0 non-breaking spaces
        llm_quote = "The\u00a0reaction\u00a0rate\u00a0increases\u00a0with\u00a0temperature."
        result = verifier.verify("doc_2_chunk_B", chunk, llm_quote)
        assert result.status == "passed"

    def test_trailing_newline_in_quote(self, verifier: MechanicalVerifier) -> None:
        chunk = "LangGraph enables stateful multi-agent workflows."
        llm_quote = "LangGraph enables stateful multi-agent workflows.\n"
        result = verifier.verify("doc_3_chunk_A", chunk, llm_quote)
        assert result.status == "passed"

    def test_leading_whitespace_in_quote(self, verifier: MechanicalVerifier) -> None:
        chunk = "FastAPI provides native async support for Python."
        llm_quote = "  FastAPI provides native async support for Python."
        result = verifier.verify("doc_3_chunk_B", chunk, llm_quote)
        assert result.status == "passed"

    def test_mixed_whitespace_types_in_quote(self, verifier: MechanicalVerifier) -> None:
        chunk = "The model outputs structured JSON with citation references."
        llm_quote = "The model\t outputs\n structured JSON\u00a0with citation references."
        result = verifier.verify("doc_5_chunk_A", chunk, llm_quote)
        assert result.status == "passed"

    def test_en_dash_vs_hyphen_in_quote(self, verifier: MechanicalVerifier) -> None:
        chunk = "Solid-state batteries are a state-of-the-art technology."
        # LLM used en-dashes instead of hyphens
        llm_quote = "Solid\u2013state batteries are a state\u2013of\u2013the\u2013art technology."
        result = verifier.verify("doc_1_chunk_A", chunk, llm_quote)
        assert result.status == "passed"

    def test_zero_width_space_in_quote(self, verifier: MechanicalVerifier) -> None:
        chunk = "Retrieval-augmented generation grounds the model in facts."
        llm_quote = "Retrieval\u200b-augmented generation grounds the model in facts."
        result = verifier.verify("doc_6_chunk_A", chunk, llm_quote)
        assert result.status == "passed"


# ===========================================================================
# D. Hallucination cases — must FAIL (Tier 5)
# ===========================================================================


class TestHallucinationCasesFail:
    """
    These are genuine hallucinations — the LLM fabricated or materially altered
    the quote. All must return status="failed" and tier=5.
    """

    def test_completely_fabricated_quote(self, verifier: MechanicalVerifier) -> None:
        chunk = "Water boils at 100 degrees Celsius at standard pressure."
        llm_quote = "Water freezes at 0 degrees Fahrenheit at sea level."
        result = verifier.verify("doc_1_chunk_A", chunk, llm_quote)
        assert result.status == "failed"
        assert result.tier == 5

    def test_paraphrase_fails(self, verifier: MechanicalVerifier) -> None:
        """A paraphrase is NOT a verbatim quote — must fail."""
        chunk = "Solid-state batteries replace liquid electrolytes with solid ceramics."
        # LLM paraphrased instead of quoting verbatim
        llm_quote = "Solid-state batteries use solid ceramic electrolytes instead of liquid ones."
        result = verifier.verify("doc_4_chunk_C", chunk, llm_quote)
        assert result.status == "failed"
        assert result.tier == 5

    def test_word_substitution_fails(self, verifier: MechanicalVerifier) -> None:
        chunk = "The quick brown fox jumps over the lazy dog."
        llm_quote = "The quick brown fox leaps over the lazy dog."  # 'leaps' ≠ 'jumps'
        result = verifier.verify("doc_1_chunk_A", chunk, llm_quote)
        assert result.status == "failed"
        assert result.tier == 5

    def test_word_insertion_fails(self, verifier: MechanicalVerifier) -> None:
        chunk = "Python is a programming language."
        llm_quote = "Python is a very popular programming language."  # inserted 'very popular'
        result = verifier.verify("doc_2_chunk_A", chunk, llm_quote)
        assert result.status == "failed"
        assert result.tier == 5

    def test_word_omission_fails(self, verifier: MechanicalVerifier) -> None:
        chunk = "Python is a high-level programming language."
        llm_quote = "Python is a programming language."  # dropped 'high-level'
        result = verifier.verify("doc_2_chunk_A", chunk, llm_quote)
        assert result.status == "failed"
        assert result.tier == 5

    def test_word_reordering_fails(self, verifier: MechanicalVerifier) -> None:
        chunk = "Energy density is higher in solid-state batteries."
        llm_quote = "In solid-state batteries, energy density is higher."
        result = verifier.verify("doc_4_chunk_D", chunk, llm_quote)
        assert result.status == "failed"
        assert result.tier == 5

    def test_quote_from_different_chunk(self, verifier: MechanicalVerifier) -> None:
        """Quote exists in another document but not in this chunk — must fail."""
        chunk = "LangGraph supports stateful directed acyclic graphs."
        llm_quote = "FastAPI provides native async support for Python."
        result = verifier.verify("doc_3_chunk_A", chunk, llm_quote)
        assert result.status == "failed"
        assert result.tier == 5

    def test_empty_quote_fails(self, verifier: MechanicalVerifier) -> None:
        chunk = "Some valid source text."
        result = verifier.verify("doc_1_chunk_A", chunk, "")
        assert result.status == "failed"
        assert result.tier == 5

    def test_whitespace_only_quote_fails(self, verifier: MechanicalVerifier) -> None:
        chunk = "Some valid source text."
        result = verifier.verify("doc_1_chunk_A", chunk, "   \t\n  ")
        assert result.status == "failed"
        assert result.tier == 5

    def test_punctuation_only_quote_fails(self, verifier: MechanicalVerifier) -> None:
        chunk = "Some valid source text."
        result = verifier.verify("doc_1_chunk_A", chunk, "!!!")
        assert result.status == "failed"
        assert result.tier == 5


# ===========================================================================
# E. Audit proof structure validation
# ===========================================================================


class TestAuditProof:
    def test_pass_audit_proof_keys(self, verifier: MechanicalVerifier) -> None:
        result = verifier.verify(
            "doc_1_chunk_A",
            "The sky is blue.",
            "The sky is blue.",
        )
        proof = result.audit_proof
        assert proof["check"] == "mechanical_verification"
        assert proof["status"] == "passed"
        assert proof["chunk_id"] == "doc_1_chunk_A"
        assert "norm_quote" in proof
        # v2.4: audit proof now records token count and sentences checked
        # instead of the old norm_chunk_length field.
        assert "norm_quote_tokens" in proof
        assert isinstance(proof["norm_quote_tokens"], int)

    def test_fail_audit_proof_keys(self, verifier: MechanicalVerifier) -> None:
        result = verifier.verify(
            "doc_1_chunk_A",
            "The sky is blue.",
            "The grass is green.",
        )
        proof = result.audit_proof
        assert proof["check"] == "mechanical_verification"
        assert proof["status"] == "failed"
        assert proof["tier"] == 5
        assert proof["tier_label"] == "hallucinated"
        assert proof["chunk_id"] == "doc_1_chunk_A"
        assert "failure_reason" in proof
        assert "raw_quote" in proof
        assert "norm_quote" in proof
        # v2.4: renamed from norm_chunk_snippet → norm_quote_snippet
        assert "norm_quote_snippet" in proof

    def test_fail_audit_proof_preserves_raw_quote(self, verifier: MechanicalVerifier) -> None:
        raw_quote = "This quote does not exist in the chunk."
        result = verifier.verify("doc_1_chunk_A", "Some chunk text.", raw_quote)
        assert result.audit_proof["raw_quote"] == raw_quote

    def test_pass_audit_proof_has_no_tier_key(self, verifier: MechanicalVerifier) -> None:
        result = verifier.verify("doc_1_chunk_A", "The sky is blue.", "The sky is blue.")
        assert "tier" not in result.audit_proof


# ===========================================================================
# F. Result immutability
# ===========================================================================


class TestResultImmutability:
    def test_result_is_frozen_dataclass(self, verifier: MechanicalVerifier) -> None:
        result = verifier.verify("doc_1_chunk_A", "The sky is blue.", "The sky is blue.")
        with pytest.raises((AttributeError, TypeError)):
            result.status = "failed"  # type: ignore[misc]

    def test_audit_proof_returned_as_dict(self, verifier: MechanicalVerifier) -> None:
        result = verifier.verify("doc_1_chunk_A", "The sky is blue.", "The sky is blue.")
        assert isinstance(result.audit_proof, dict)
