"""
Axiom Engine — Mechanical Verifier (Stage 1, Non-Negotiable Floor)

Deterministic citation integrity checker. No LLM involved.

Algorithm:
  1. Normalize both the full chunk and the LLM-supplied quote:
       - Expand/replace common Unicode punctuation and smart quotes
       - NFKD-decompose, then drop combining marks (accents, Arabic harakat),
         punctuation, symbols, and invisible format characters
       - Casefold
       - Collapse all whitespace to a single space and strip edges
     Letters and digits from every script are preserved, so non-Latin content
     (Arabic, CJK, Cyrillic, ...) remains verifiable.
  2. Require the normalized quote to have at least _MIN_NORMALIZED_TOKENS
     significant tokens, or _MIN_NORMALIZED_CHARS characters for scripts that
     do not use spaces (CJK). Short fragments like "the sky" match too
     liberally and provide no citation integrity guarantee.
  3. Check whether the normalized quote is a substring of the normalized chunk.
  4. If YES → passed
     If NO  → failed  (tier=5, Hallucinated Citation)
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Literal

# ---------------------------------------------------------------------------
# Minimum quote length to accept a "passed" verdict. The token floor governs
# space-separated scripts; the character floor is the fallback for unspaced
# scripts (CJK), where whitespace tokenization would always yield one token.
# A quote clears the guard by meeting EITHER floor.
# ---------------------------------------------------------------------------
_MIN_NORMALIZED_TOKENS = 4
_MIN_NORMALIZED_CHARS = 12

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

# Unicode general categories removed during normalization:
#   P* — punctuation (all scripts, superset of string.punctuation's P entries)
#   S* — symbols ($, +, <, =, >, ^, `, |, ~ are category S, not P)
#   Mn — nonspacing combining marks (Latin accents, Arabic harakat) after NFKD
#   Cf — invisible format characters (ZWJ/ZWNJ, directional marks)
# Letters (L*), digits (N*), and whitespace survive; whitespace is collapsed
# in a later step. Cc (control) is NOT removed here because \t and \n must
# survive until whitespace collapsing.
_REMOVED_CATEGORY_PREFIXES = ("P", "S")
_REMOVED_CATEGORIES = ("Mn", "Cf")

# Pre-compiled whitespace collapser
_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class MechanicalVerificationResult:
    """
    Immutable result returned by MechanicalVerifier.verify().

    Attributes:
        status:      "passed" if the normalized quote is a substring of the
                     normalized chunk; "failed" otherwise.
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

        # Minimum length guard — reject trivially short quotes. Either floor
        # suffices: the character floor keeps unspaced scripts (CJK) verifiable,
        # since whitespace tokenization collapses them to a single "token".
        quote_token_count = len(norm_quote.split())
        if quote_token_count < _MIN_NORMALIZED_TOKENS and len(norm_quote) < _MIN_NORMALIZED_CHARS:
            return self._failure(
                chunk_id=chunk_id,
                raw_quote=llm_quote,
                norm_quote=norm_quote,
                failure_reason=(
                    f"Quote is too short after normalization "
                    f"({quote_token_count} tokens < {_MIN_NORMALIZED_TOKENS} required "
                    f"and {len(norm_quote)} chars < {_MIN_NORMALIZED_CHARS} required)."
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
          1. Apply Unicode substitution table (smart quotes, dashes, etc.)
          2. NFKD decomposition, then drop combining marks (Latin accents,
             Arabic harakat), punctuation, symbols, and format characters by
             Unicode category. Base letters and digits from every script are
             preserved — non-Latin text stays verifiable (the previous ASCII
             coercion deleted Arabic/CJK/Cyrillic content entirely, making
             every citation from such sources fail as Tier 5).
          3. Casefold.
          4. Collapse all whitespace sequences to a single space and strip edges.
        """
        text = text.translate(_UNICODE_SUBSTITUTION_TABLE)
        text = unicodedata.normalize("NFKD", text)
        text = "".join(
            ch
            for ch in text
            if not (
                (cat := unicodedata.category(ch)).startswith(_REMOVED_CATEGORY_PREFIXES)
                or cat in _REMOVED_CATEGORIES
            )
        )
        text = text.casefold()
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
