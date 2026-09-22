"""Script-aware text helpers shared by the mechanical verifier and the ranker."""

from __future__ import annotations

# Scripts conventionally written without spaces between words. The mechanical
# verifier length-checks quotes in these scripts by characters and matches them
# without word boundaries; the ranker tokenizes them into character bigrams.
_UNSPACED_RANGES: tuple[tuple[int, int], ...] = (
    (0x0E00, 0x0EFF),  # Thai, Lao
    (0x0F00, 0x0FFF),  # Tibetan
    (0x1000, 0x109F),  # Myanmar
    (0x1780, 0x17FF),  # Khmer
    (0x3040, 0x30FF),  # Hiragana, Katakana
    (0x31F0, 0x31FF),  # Katakana phonetic extensions
    (0x3400, 0x4DBF),  # CJK Extension A
    (0x4E00, 0x9FFF),  # CJK Unified Ideographs
    (0xF900, 0xFAFF),  # CJK Compatibility Ideographs
    (0x20000, 0x2FA1F),  # CJK Extensions B+ and supplements
)


def is_unspaced_char(ch: str) -> bool:
    """True when ``ch`` belongs to a script written without word spaces."""
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _UNSPACED_RANGES)
