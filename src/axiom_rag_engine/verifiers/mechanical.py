"""
Axiom Engine — Mechanical Verifier (Stage 1, Non-Negotiable Floor)

Deterministic citation integrity checker. No LLM involved.

Algorithm:
  1. Normalize both the full chunk and the LLM-supplied quote:
       - Expand/replace common Unicode punctuation and smart quotes
       - NFKD-decompose, then drop combining marks (accents, Arabic harakat)
         and invisible format characters
       - Turn punctuation and symbols into word boundaries (a space) — never
         delete them, which would fuse tokens and let "1.5" match "15". A
         minus sign directly before a number is kept, so "-5" never matches "5".
       - Casefold
       - Collapse all whitespace to a single space and strip edges
     Letters and digits from every script are preserved, so non-Latin content
     (Arabic, CJK, Cyrillic, ...) remains verifiable.
  2. Require the normalized quote to have at least _MIN_NORMALIZED_TOKENS
     tokens — or, for scripts written without spaces (CJK, Thai, ...), at least
     _MIN_NORMALIZED_CHARS characters. Short fragments like "the sky" match too
     liberally and provide no citation integrity guarantee.
  3. Search for the normalized quote in the normalized chunk, aligned to word
     boundaries in spaced scripts (so "hen the cat" never matches "then the
     cat"). Unspaced scripts have no word boundaries and match anywhere.
  4. If found → passed, and the exact raw source text that matched is returned
     (``matched_source_text``) so callers can show what the source actually says.
     If not   → failed  (tier=5, Hallucinated Citation)
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Literal

from axiom_rag_engine.utils.text import is_unspaced_char

# ---------------------------------------------------------------------------
# Minimum quote length to accept a "passed" verdict. The token floor governs
# space-separated scripts; the character floor applies only to quotes written
# in unspaced scripts (CJK, Thai, ...), where whitespace tokenization would
# always yield a single "token".
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
    # Dashes and minus → hyphen-minus (a sign when it precedes a number)
    "\u2013": "-",  # EN DASH
    "\u2014": "-",  # EM DASH
    "\u2015": "-",  # HORIZONTAL BAR
    "\u2212": "-",  # MINUS SIGN
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

# Unicode general categories handled during normalization:
#   Mn — nonspacing combining marks (Latin accents, Arabic harakat) → dropped
#   Cf — invisible format characters (ZWJ/ZWNJ, directional marks) → dropped
#   P* — punctuation → word boundary (space)
#   S* — symbols ($, %, +, <, =, >, ...) → word boundary (space)
# Letters (L*) and digits (N*) survive; whitespace is collapsed.
_DROPPED_CATEGORIES = ("Mn", "Cf")
_BOUNDARY_CATEGORY_PREFIXES = ("P", "S")


@dataclass(frozen=True)
class MechanicalVerificationResult:
    """
    Immutable result returned by MechanicalVerifier.verify().

    Attributes:
        status:      "passed" if the normalized quote occurs in the normalized
                     chunk (on word boundaries); "failed" otherwise.
        tier:        None on pass. 5 (Hallucinated) on failure.
        audit_proof: Dict suitable for direct insertion into the audit_trail state.
        matched_source_text: On pass, the exact raw chunk text the quote
                     matched (original casing, punctuation, accents). None on
                     failure.
    """

    status: Literal["passed", "failed"]
    tier: Literal[5] | None
    audit_proof: dict
    matched_source_text: str | None = None


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

        Args:
            chunk_id:   The unique chunk identifier (e.g. "doc_1_chunk_A").
            chunk_text: The full raw text of the source chunk.
            llm_quote:  The verbatim quote the Synthesizer claims to have taken
                        from `chunk_text`.

        Returns:
            MechanicalVerificationResult with status, tier, audit_proof, and —
            on pass — the matched raw source text.
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

        # Minimum length guard — reject trivially short quotes.
        quote_token_count = len(norm_quote.split())
        if self._is_unspaced_text(norm_quote):
            char_count = len(norm_quote.replace(" ", ""))
            if char_count < _MIN_NORMALIZED_CHARS:
                return self._failure(
                    chunk_id=chunk_id,
                    raw_quote=llm_quote,
                    norm_quote=norm_quote,
                    failure_reason=(
                        f"Quote is too short after normalization ({char_count} chars < "
                        f"{_MIN_NORMALIZED_CHARS} required for unspaced scripts)."
                    ),
                )
        elif quote_token_count < _MIN_NORMALIZED_TOKENS:
            return self._failure(
                chunk_id=chunk_id,
                raw_quote=llm_quote,
                norm_quote=norm_quote,
                failure_reason=(
                    f"Quote is too short after normalization "
                    f"({quote_token_count} tokens < {_MIN_NORMALIZED_TOKENS} required)."
                ),
            )

        norm_chunk, index = self._normalize_with_map(chunk_text)
        start = self._find_on_boundaries(norm_chunk, norm_quote)

        if start is not None:
            matched = self._raw_span(chunk_text, index, start, start + len(norm_quote))
            return MechanicalVerificationResult(
                status="passed",
                tier=None,
                audit_proof={
                    "check": "mechanical_verification",
                    "status": "passed",
                    "chunk_id": chunk_id,
                    "norm_quote": norm_quote,
                    "norm_quote_tokens": quote_token_count,
                    "matched_source_text": matched,
                    "verification_scope": "full_chunk",
                },
                matched_source_text=matched,
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
        Canonical normalization applied identically to the chunk and the
        LLM-supplied quote before comparison. See the module docstring for the
        steps; :meth:`_normalize_with_map` is the single implementation.
        """
        return MechanicalVerifier._normalize_with_map(text)[0]

    @staticmethod
    def _normalize_with_map(text: str) -> tuple[str, list[int]]:
        """Normalize ``text`` and map every output character to its raw index.

        Returns ``(normalized, index)`` where ``index[i]`` is the position in
        ``text`` of the raw character that produced ``normalized[i]``. The map
        lets a match in normalized space be reported as the exact raw source
        span. Every step is per-character (substitution, NFKD, category filter,
        casefold), so processing character by character is equivalent to
        normalizing the whole string.
        """
        out: list[str] = []
        index: list[int] = []

        def emit(ch: str, raw_i: int) -> None:
            if ch == " " and (not out or out[-1] == " "):
                return  # collapse runs, drop leading space
            out.append(ch)
            index.append(raw_i)

        for raw_i, raw_ch in enumerate(text):
            for sub_ch in raw_ch.translate(_UNICODE_SUBSTITUTION_TABLE):
                for ch in unicodedata.normalize("NFKD", sub_ch):
                    cat = unicodedata.category(ch)
                    if cat in _DROPPED_CATEGORIES:
                        continue
                    if ch == "-" and MechanicalVerifier._is_sign(text, raw_i):
                        emit("-", raw_i)
                    elif ch.isspace() or cat.startswith(_BOUNDARY_CATEGORY_PREFIXES):
                        emit(" ", raw_i)
                    else:
                        for folded in ch.casefold():
                            emit(folded, raw_i)

        if out and out[-1] == " ":
            out.pop()
            index.pop()
        return "".join(out), index

    @staticmethod
    def _is_sign(text: str, i: int) -> bool:
        """True when the dash at ``text[i]`` is a minus sign on a number: it is
        followed by a digit and not preceded by a letter or digit (so "7-9" and
        "COVID-19" are separators, while "-5" and " −5" are signs)."""
        nxt = text[i + 1] if i + 1 < len(text) else ""
        prev = text[i - 1] if i > 0 else ""
        return nxt.isdigit() and not prev.isalnum()

    @staticmethod
    def _is_unspaced_text(norm_text: str) -> bool:
        """True when most letters in ``norm_text`` belong to an unspaced script."""
        letters = [ch for ch in norm_text if ch.isalnum()]
        if not letters:
            return False
        unspaced = sum(1 for ch in letters if is_unspaced_char(ch))
        return unspaced * 2 >= len(letters)

    @staticmethod
    def _find_on_boundaries(norm_chunk: str, norm_quote: str) -> int | None:
        """Return the start of the first occurrence of ``norm_quote`` in
        ``norm_chunk`` that sits on word boundaries, or None.

        A boundary is the text edge, a space, or an unspaced-script character on
        either side (unspaced scripts have no word delimiters).
        """
        start = norm_chunk.find(norm_quote)
        while start != -1:
            end = start + len(norm_quote)
            left_ok = (
                start == 0
                or norm_chunk[start - 1] == " "
                or is_unspaced_char(norm_quote[0])
                or is_unspaced_char(norm_chunk[start - 1])
            )
            right_ok = (
                end == len(norm_chunk)
                or norm_chunk[end] == " "
                or is_unspaced_char(norm_quote[-1])
                or is_unspaced_char(norm_chunk[end])
            )
            if left_ok and right_ok:
                return start
            start = norm_chunk.find(norm_quote, start + 1)
        return None

    @staticmethod
    def _raw_span(raw: str, index: list[int], norm_start: int, norm_end: int) -> str:
        """Map a normalized ``[norm_start, norm_end)`` span back to raw text,
        extending over trailing combining marks the normalizer dropped."""
        raw_start = index[norm_start]
        raw_end = index[norm_end - 1] + 1
        while raw_end < len(raw) and unicodedata.category(raw[raw_end]) in _DROPPED_CATEGORIES:
            raw_end += 1
        return raw[raw_start:raw_end]

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
