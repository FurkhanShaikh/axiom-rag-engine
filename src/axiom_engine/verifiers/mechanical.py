"""
Axiom Engine v2.3 — Mechanical Verifier (Stage 1, Non-Negotiable Floor)

Deterministic citation integrity checker. No LLM involved.

Algorithm:
  1. Normalize both the full chunk and the LLM-supplied quote:
       - Expand/replace Unicode punctuation and smart quotes to ASCII equivalents
       - Lowercase
       - Strip all remaining punctuation
       - Collapse all whitespace to a single space and strip edges
  2. Require the normalized quote to have at least _MIN_NORMALIZED_TOKENS
     significant tokens (post-normalization word count). Short fragments like
     "the sky" match too liberally and provide no citation integrity guarantee.
  3. Check whether the normalized quote is a substring of the normalized chunk.
  4. If YES → passed
     If NO  → failed  (tier=5, Hallucinated Citation)
"""

from __future__ import annotations

import re
import string
import unicodedata
from dataclasses import dataclass
from typing import Literal

# ---------------------------------------------------------------------------
# Minimum quote length (in normalized tokens) to accept a "passed" verdict.
# A quote normalized to fewer tokens is too short to meaningfully verify.
# ---------------------------------------------------------------------------
_MIN_NORMALIZED_TOKENS = 4

# ---------------------------------------------------------------------------
# Unicode → ASCII mapping for common LLM tokenization artifacts
# ---------------------------------------------------------------------------
_UNICODE_SUBSTITUTIONS: dict[str, str] = {
    # Smart / curly quotes → straight quotes
    "\u2018": "'",  # LEFT SINGLE QUOTATION MARK
    "\u2019": "'",  # RIGHT SINGLE QUOTATION MARK
    "\u201a": "'",  # SINGLE LOW-9 QUOTATION MARK
    "\u201b": "'",  # SINGLE HIGH-REVERSED-9 QUOTATION MARK
    "\u201c": '"',  # LEFT DOUBLE QUOTATION MARK
    "\u201d": '"',  # RIGHT DOUBLE QUOTATION MARK
    "\u201e": '"',  # DOUBLE LOW-9 QUOTATION MARK
    "\u201f": '"',  # DOUBLE HIGH-REVERSED-9 QUOTATION MARK
    # Dashes → hyphen
    "\u2013": "-",  # EN DASH
    "\u2014": "-",  # EM DASH
    "\u2015": "-",  # HORIZONTAL BAR
    # Non-standard spaces → regular space
    "\u00a0": " ",  # NON-BREAKING SPACE
    "\u202f": " ",  # NARROW NO-BREAK SPACE
    "\u2009": " ",  # THIN SPACE
    "\u2008": " ",  # PUNCTUATION SPACE
    "\u2007": " ",  # FIGURE SPACE
    "\u2006": " ",  # SIX-PER-EM SPACE
    "\u2005": " ",  # FOUR-PER-EM SPACE
    "\u2004": " ",  # THREE-PER-EM SPACE
    "\u2003": " ",  # EM SPACE
    "\u2002": " ",  # EN SPACE
    "\u200b": "",  # ZERO WIDTH SPACE (remove entirely)
    "\u00ad": "",  # SOFT HYPHEN (remove entirely)
    # Ellipsis
    "\u2026": "...",  # HORIZONTAL ELLIPSIS
}

_UNICODE_SUBSTITUTION_TABLE = str.maketrans(_UNICODE_SUBSTITUTIONS)

# Pre-compiled punctuation stripper: removes every character in string.punctuation
_PUNCTUATION_RE = re.compile(r"[" + re.escape(string.punctuation) + r"]")

# Pre-compiled whitespace collapser
_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class MechanicalVerificationResult:
    """
    Immutable result returned by MechanicalVerifier.verify().

    Attributes:
        status:      "passed" if the quote exists within a single source
                     sentence; "failed" otherwise.
        tier:        None on pass. 5 (Hallucinated) on failure.
        audit_proof: Dict suitable for direct insertion into the audit_trail state.
    """

    status: Literal["passed", "failed"]
    tier: Literal[5] | None
    audit_proof: dict


class MechanicalVerifier:
    """
    Deterministic, LLM-free citation integrity checker.

    Usage:
        verifier = MechanicalVerifier()
        result = verifier.verify(
            chunk_id="doc_1_chunk_A",
            chunk_text="The sky is blue on a clear day.",
            llm_quote="The sky is blue on a clear day.",
        )
        assert result.status == "passed"
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def verify(
        self,
        chunk_id: str,
        chunk_text: str,
        llm_quote: str,
    ) -> MechanicalVerificationResult:
        """
        Verify that `llm_quote` genuinely exists inside the `chunk_text`.
        Allows multi-sentence quoting by validating against the whole chunk.

        Algorithm:
          1. Normalize the full chunk and the quote independently.
          2. Require the normalized quote to meet the minimum token floor.
          3. Return "passed" if the normalized quote is a substring of the
             normalized chunk; "failed" otherwise.

        Args:
            chunk_id:   The unique chunk identifier (e.g. "doc_1_chunk_A").
            chunk_text: The full raw text of the source chunk.
            llm_quote:  The verbatim quote the Synthesizer claims to have taken
                        from `chunk_text`.

        Returns:
            MechanicalVerificationResult with status, tier, and audit_proof.
        """
        norm_quote = self._normalize_text(llm_quote)

        # Empty quote after normalization → always fail.
        if not norm_quote:
            return self._failure(
                chunk_id=chunk_id,
                raw_quote=llm_quote,
                norm_quote=norm_quote,
                failure_reason="Quote is empty after normalization.",
            )

        # Minimum token guard — reject trivially short quotes.
        quote_token_count = len(norm_quote.split())
        if quote_token_count < _MIN_NORMALIZED_TOKENS:
            return self._failure(
                chunk_id=chunk_id,
                raw_quote=llm_quote,
                norm_quote=norm_quote,
                failure_reason=(
                    f"Quote is too short after normalization "
                    f"({quote_token_count} tokens < {_MIN_NORMALIZED_TOKENS} required)."
                ),
            )

        norm_chunk = self._normalize_text(chunk_text)

        if norm_quote in norm_chunk:
            return MechanicalVerificationResult(
                status="passed",
                tier=None,
                audit_proof={
                    "check": "mechanical_verification",
                    "status": "passed",
                    "chunk_id": chunk_id,
                    "norm_quote": norm_quote,
                    "norm_quote_tokens": quote_token_count,
                    "verification_scope": "full_chunk",
                },
            )

        return self._failure(
            chunk_id=chunk_id,
            raw_quote=llm_quote,
            norm_quote=norm_quote,
            failure_reason="Normalized quote not found in the chunk.",
        )

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_text(text: str) -> str:
        """
        Canonical normalization applied identically to source sentences and the
        LLM-supplied quote before substring comparison.

        Steps:
          1. Apply Unicode → ASCII substitution table (smart quotes, dashes, etc.)
          2. NFKD decomposition + ASCII encoding to strip combining diacritics and
             homoglyph variants not covered by the substitution table.
          3. Lowercase.
          4. Strip all ASCII punctuation.
          5. Collapse all whitespace sequences to a single space and strip edges.
        """
        text = text.translate(_UNICODE_SUBSTITUTION_TABLE)
        text = unicodedata.normalize("NFKD", text).encode("ascii", errors="ignore").decode("ascii")
        text = text.lower()
        text = _PUNCTUATION_RE.sub("", text)
        text = _WHITESPACE_RE.sub(" ", text).strip()
        return text

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _failure(
        chunk_id: str,
        raw_quote: str,
        norm_quote: str,
        failure_reason: str,
    ) -> MechanicalVerificationResult:
        return MechanicalVerificationResult(
            status="failed",
            tier=5,
            audit_proof={
                "check": "mechanical_verification",
                "status": "failed",
                "tier": 5,
                "tier_label": "hallucinated",
                "chunk_id": chunk_id,
                "failure_reason": failure_reason,
                "raw_quote": raw_quote,
                "norm_quote": norm_quote,
                "norm_quote_snippet": norm_quote[:200],
                "verification_scope": "full_chunk",
            },
        )
