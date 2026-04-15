"""
Axiom Engine v2.3 — Source & Chunk Quality Scorer (Modules 3–4)

Responsibilities:
  - Assigns a source_quality_score (0.0–1.0) to each chunk based on domain
    authority heuristics. Domain lists are configurable via app_config and
    fall back to built-in defaults.
  - Assigns a chunk_quality_score (0.0–1.0) based on content quality signals
    (length, information density, presence of data markers).
  - Computes a combined quality_score as a weighted blend of both signals.
  - Filters out chunks below a minimum quality threshold.
  - Updates GraphState keys: scored_chunks, audit_trail.

Both scorers are deterministic — no LLM calls required.
"""

from __future__ import annotations

import logging
import re
from functools import partial
from typing import Any

from axiom_engine.state import GraphState
from axiom_engine.utils.audit import make_audit_event

_audit = partial(make_audit_event, "scorer")
logger = logging.getLogger("axiom_engine.scorer")

# ---------------------------------------------------------------------------
# Source quality — built-in defaults (overridable via app_config)
#
# TIER TAXONOMY note (C4 fix):
#   _DEFAULT_PRIMARY_DOMAINS  — official / primary sources.  Citations from
#       these domains are eligible for Tier 1 ("Authoritative").  Tertiary
#       sources such as Wikipedia must NOT appear here.
#   _DEFAULT_REFERENCE_DOMAINS — academic papers, encyclopedias, and curated
#       reference works.  These are high-quality but not "official primary
#       sources"; they cap out at Tier 2 ("Consensus") or Tier 3.
#   _DEFAULT_AUTHORITATIVE_DOMAINS — union of the two sets above; used by the
#       scorer for the quality score (both tiers receive 0.9 quality boost).
# ---------------------------------------------------------------------------

_DEFAULT_PRIMARY_DOMAINS: set[str] = {
    # Official government / intergovernmental health bodies
    "nih.gov",
    "cdc.gov",
    "who.int",
    "europa.eu",
    # Official standards and specifications
    "w3.org",
    "ietf.org",
    "rfc-editor.org",
    # Official language / platform documentation
    "docs.python.org",
    "developer.mozilla.org",
}

_DEFAULT_REFERENCE_DOMAINS: set[str] = {
    # Peer-reviewed academic publishers / preprint servers
    "arxiv.org",
    "pubmed.ncbi.nlm.nih.gov",
    "nature.com",
    "science.org",
    "ieee.org",
    "acm.org",
    "scholar.google.com",
    # Curated encyclopedic reference (tertiary — NOT primary)
    "en.wikipedia.org",
    "britannica.com",
}

# Union used for source quality scoring — both sets receive the high-authority boost.
_DEFAULT_AUTHORITATIVE_DOMAINS: set[str] = _DEFAULT_PRIMARY_DOMAINS | _DEFAULT_REFERENCE_DOMAINS

_DEFAULT_LOW_QUALITY_DOMAINS: set[str] = {
    "reddit.com",
    "quora.com",
    "answers.yahoo.com",
    "medium.com",
    "blogspot.com",
}

_DEFAULT_SOURCE_SCORE = 0.5


def build_domain_sets(
    app_config: dict[str, Any],
) -> tuple[set[str], set[str]]:
    """
    Build authoritative and low-quality domain sets from app_config.

    Config keys:
      authoritative_domains  — added on top of defaults (additive).
      low_quality_domains    — added on top of defaults (additive).
      exclude_default_domains — list of domain strings to remove from the
          built-in default sets (allows callers to demote e.g. Wikipedia).

    Returns:
        (authoritative_set, low_quality_set)
    """
    caller_authoritative = set(app_config.get("authoritative_domains", []))
    caller_low_quality = set(app_config.get("low_quality_domains", []))
    excluded = {d.lower().strip() for d in app_config.get("exclude_default_domains", [])}

    authoritative = (_DEFAULT_AUTHORITATIVE_DOMAINS - excluded) | caller_authoritative
    low_quality = (_DEFAULT_LOW_QUALITY_DOMAINS - excluded) | caller_low_quality

    return authoritative, low_quality


def build_primary_domain_set(app_config: dict[str, Any]) -> set[str]:
    """
    Build the *primary* domain set — sources eligible for Tier 1
    ("Authoritative").  Reference/encyclopedic domains are excluded.

    Caller-supplied authoritative_domains are treated as primary unless
    they appear in exclude_default_domains.
    """
    excluded = {d.lower().strip() for d in app_config.get("exclude_default_domains", [])}
    caller_primary = set(app_config.get("authoritative_domains", []))
    return (_DEFAULT_PRIMARY_DOMAINS - excluded) | caller_primary


def _normalize_domain(domain: str) -> str:
    """
    Canonical domain normalization: lowercase, strip, and decode punycode
    (ACE/xn--) labels to their Unicode equivalents.

    This prevents lookalike-domain bypasses where an attacker uses a punycode
    variant of an authoritative domain (e.g. ``xn--googl-dua.com``) to pass
    through domain-authority checks unchanged.
    """
    domain = domain.lower().strip()
    parts = domain.split(".")
    normalized: list[str] = []
    for part in parts:
        if part.startswith("xn--"):
            try:
                normalized.append(part.encode("ascii").decode("idna"))
            except (UnicodeError, UnicodeDecodeError):
                normalized.append(part)
        else:
            normalized.append(part)
    return ".".join(normalized)


def is_authoritative_domain(domain: str, authoritative: set[str]) -> bool:
    """Return True when the domain is in the authoritative set or a subdomain of one."""
    domain_norm = _normalize_domain(domain)
    if domain_norm in authoritative:
        return True
    return any(domain_norm.endswith("." + auth) for auth in authoritative)


def is_primary_domain(domain: str, primary: set[str]) -> bool:
    """Return True when the domain is a primary (Tier-1-eligible) source."""
    return is_authoritative_domain(domain, primary)


def score_source_quality(
    domain: str,
    authoritative: set[str] | None = None,
    low_quality: set[str] | None = None,
) -> float:
    """
    Score a domain's authority on a 0.0–1.0 scale.
      - Authoritative (primary or reference) → 0.9
      - Low quality                          → 0.3
      - Unknown                              → 0.5

    Domain matching is punycode/IDN-aware: ACE labels (``xn--…``) are decoded
    to Unicode before comparison so lookalike domains cannot bypass authority
    checks.
    """
    if authoritative is None:
        authoritative = _DEFAULT_AUTHORITATIVE_DOMAINS
    if low_quality is None:
        low_quality = _DEFAULT_LOW_QUALITY_DOMAINS

    domain_norm = _normalize_domain(domain)
    if domain_norm in authoritative:
        return 0.9
    if domain_norm in low_quality:
        return 0.3
    # Check subdomain membership.
    for auth in authoritative:
        if domain_norm.endswith("." + auth):
            return 0.85
    for low in low_quality:
        if domain_norm.endswith("." + low):
            return 0.3
    return _DEFAULT_SOURCE_SCORE


# ---------------------------------------------------------------------------
# Chunk quality heuristics
# ---------------------------------------------------------------------------

# Regex patterns that signal information-rich content.
_DATA_MARKERS = re.compile(
    r"\d+\.?\d*\s*%"  # Percentages
    r"|\d{4}"  # Years / large numbers
    r"|(?:fig(?:ure)?|table)\s*\d"  # Figure/table references
    r"|https?://",  # Embedded URLs (citations within text)
    re.IGNORECASE,
)

_MIN_QUALITY_THRESHOLD = 0.2


def score_chunk_quality(text: str) -> float:
    """
    Score a chunk's content quality on 0.0–1.0 based on:
      - Length (longer paragraphs tend to carry more information)
      - Information density (presence of numbers, data markers)
    """
    if not text or not text.strip():
        return 0.0

    length = len(text.strip())

    # Length score: ramps from 0.2 at 40 chars to 1.0 at 500+ chars.
    length_score = min(1.0, 0.2 + (length - 40) * 0.8 / 460) if length >= 40 else 0.1

    # Data marker density bonus.
    markers = _DATA_MARKERS.findall(text)
    density_bonus = min(0.3, len(markers) * 0.1)

    return round(min(1.0, length_score + density_bonus), 4)


# ---------------------------------------------------------------------------
# Combined scoring
# ---------------------------------------------------------------------------

_SOURCE_WEIGHT = 0.4
_CHUNK_WEIGHT = 0.6


def compute_combined_score(
    source_score: float,
    chunk_score: float,
    source_weight: float = _SOURCE_WEIGHT,
    chunk_weight: float = _CHUNK_WEIGHT,
) -> float:
    """Weighted combination of source and chunk quality scores.

    Weights are configurable at call-time; defaults match the module constants.
    Configure via ``app_config.source_weight`` and ``app_config.chunk_weight``.
    """
    return round(source_weight * source_score + chunk_weight * chunk_score, 4)


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


async def scorer_node(state: GraphState) -> dict[str, Any]:
    """
    LangGraph node — Source & Chunk Quality Scoring.

    Reads indexed_chunks, assigns source and chunk quality scores,
    computes a combined quality_score, and filters below threshold.
    Domain lists are configurable via app_config keys:
      - authoritative_domains: list[str]
      - low_quality_domains: list[str]

    Returns keys: scored_chunks, audit_trail
    """
    audit: list[dict[str, Any]] = []
    indexed_chunks: list[dict] = list(state.get("indexed_chunks") or [])
    app_cfg: dict = state.get("app_config") or {}

    authoritative, low_quality = build_domain_sets(app_cfg)
    source_weight = float(app_cfg.get("source_weight", _SOURCE_WEIGHT))
    chunk_weight = float(app_cfg.get("chunk_weight", _CHUNK_WEIGHT))

    audit.append(
        _audit(
            "scorer_start",
            {"input_chunk_count": len(indexed_chunks)},
        )
    )

    scored: list[dict[str, Any]] = []
    filtered_count = 0

    for chunk in indexed_chunks:
        domain: str = chunk.get("domain", "")
        text: str = chunk.get("text", "")

        source_score = score_source_quality(domain, authoritative, low_quality)
        chunk_score = score_chunk_quality(text)
        combined = compute_combined_score(source_score, chunk_score, source_weight, chunk_weight)

        if combined < _MIN_QUALITY_THRESHOLD:
            filtered_count += 1
            continue

        scored_chunk = {
            **chunk,
            "source_quality_score": source_score,
            "chunk_quality_score": chunk_score,
            "quality_score": combined,
        }
        scored.append(scored_chunk)

    # Sort by quality_score descending for downstream consumption.
    scored.sort(key=lambda c: c["quality_score"], reverse=True)

    audit.append(
        _audit(
            "scorer_complete",
            {
                "input_chunks": len(indexed_chunks),
                "scored_chunks": len(scored),
                "filtered_below_threshold": filtered_count,
            },
        )
    )

    return {
        "scored_chunks": scored,
        "audit_trail": audit,
    }
