"""
Axiom Engine v2.3 — Relevance Ranker (Module 5)

Responsibilities:
  - Ranks scored_chunks by relevance to the user query using BM25-inspired
    scoring (term frequency × inverse document frequency).
  - Combines relevance score with the upstream quality_score
    for a final ranking_score.
  - Trims to top-N chunks (max_ranked_chunks from pipeline config).
  - Updates GraphState keys: ranked_chunks, audit_trail.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from functools import partial
from typing import Any

from axiom_engine.state import GraphState
from axiom_engine.utils.audit import make_audit_event

_audit = partial(make_audit_event, "ranker")
logger = logging.getLogger("axiom_engine.ranker")

# ---------------------------------------------------------------------------
# Text tokenization for keyword matching
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Common English stopwords to exclude from relevance scoring.
_STOPWORDS: set[str] = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "can",
    "shall",
    "to",
    "of",
    "in",
    "for",
    "on",
    "with",
    "at",
    "by",
    "from",
    "as",
    "into",
    "about",
    "between",
    "through",
    "after",
    "before",
    "during",
    "without",
    "and",
    "or",
    "but",
    "not",
    "no",
    "if",
    "then",
    "than",
    "that",
    "this",
    "it",
    "its",
    "what",
    "which",
    "who",
    "whom",
    "how",
    "when",
    "where",
    "why",
    "all",
    "each",
    "every",
    "both",
    "few",
    "more",
    "most",
    "some",
    "such",
    "only",
    "very",
    "just",
    "so",
    "also",
}


def _tokenize(text: str) -> list[str]:
    """Lowercase tokenization with stopword removal."""
    tokens = _TOKEN_RE.findall(text.lower())
    return [t for t in tokens if t not in _STOPWORDS]


# ---------------------------------------------------------------------------
# BM25 scoring
# ---------------------------------------------------------------------------

# BM25 tuning parameters.
_BM25_K1 = 1.2  # Term frequency saturation
_BM25_B = 0.75  # Length normalization strength


def compute_corpus_idf(chunks: list[dict]) -> dict[str, float]:
    """
    Compute Robertson-Walker BM25 IDF across the retrieved chunk corpus.

    IDF(t) = log((N - df(t) + 0.5) / (df(t) + 0.5) + 1)

    where N = total chunks and df(t) = number of chunks containing term t.
    This is always positive and approaches zero for near-universal terms,
    providing meaningful discrimination even on small corpora (10-200 chunks).
    """
    n_docs = len(chunks)
    if n_docs == 0:
        return {}
    df: Counter[str] = Counter()
    for chunk in chunks:
        tokens = set(_tokenize(chunk.get("text", "")))
        df.update(tokens)
    return {term: math.log((n_docs - freq + 0.5) / (freq + 0.5) + 1) for term, freq in df.items()}


def compute_relevance_score(
    query: str,
    chunk_text: str,
    avg_doc_len: float = 1.0,
    idf: dict[str, float] | None = None,
) -> float:
    """
    Compute BM25 relevance between a query and a chunk.

    When ``idf`` is supplied (corpus-level Robertson-Walker IDF), terms that
    appear in every retrieved chunk are down-weighted relative to rare,
    discriminating terms.  Without ``idf``, all query terms are treated
    equally (TF-only fallback, suitable for unit tests).

    Returns a score in [0.0, 1.0] (normalized against the theoretical maximum).
    """
    query_tokens = _tokenize(query)
    if not query_tokens:
        return 0.0

    chunk_tokens = _tokenize(chunk_text)
    if not chunk_tokens:
        return 0.0

    chunk_tf = Counter(chunk_tokens)
    doc_len = len(chunk_tokens)

    score = 0.0
    max_possible = 0.0
    query_term_set = set(query_tokens)

    for term in query_term_set:
        term_idf = idf.get(term, 1.0) if idf else 1.0
        max_possible += term_idf * (_BM25_K1 + 1) / (1 + _BM25_K1)

        tf = chunk_tf.get(term, 0)
        if tf == 0:
            continue

        # BM25 term frequency component with length normalization.
        numerator = tf * (_BM25_K1 + 1)
        denominator = tf + _BM25_K1 * (1 - _BM25_B + _BM25_B * doc_len / max(avg_doc_len, 1.0))
        score += term_idf * numerator / denominator

    if max_possible > 0:
        score = min(1.0, score / max_possible)

    return round(score, 4)


# ---------------------------------------------------------------------------
# Combined ranking score
# ---------------------------------------------------------------------------

_RELEVANCE_WEIGHT = 0.6
_QUALITY_WEIGHT = 0.4


def compute_ranking_score(
    relevance: float,
    quality: float,
    relevance_weight: float = _RELEVANCE_WEIGHT,
    quality_weight: float = _QUALITY_WEIGHT,
) -> float:
    """Weighted combination of relevance and quality for final ranking.

    Weights are configurable at call-time; defaults match the module constants.
    Configure via ``pipeline_config.stages.relevance_weight`` and
    ``pipeline_config.stages.quality_weight``.
    """
    return round(relevance_weight * relevance + quality_weight * quality, 4)


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

_DEFAULT_MAX_RANKED = 10


async def ranker_node(state: GraphState) -> dict[str, Any]:
    """
    LangGraph node — Relevance Ranking.

    Reads scored_chunks and user_query, computes BM25 relevance scores
    with IDF weighting across the chunk corpus, combines with quality_score,
    ranks, and trims to top-N.

    Returns keys: ranked_chunks, audit_trail
    """
    audit: list[dict[str, Any]] = []
    scored_chunks: list[dict] = state.get("scored_chunks") or []
    user_query: str = state.get("user_query", "")

    pipeline_cfg: dict = state.get("pipeline_config") or {}
    stages_cfg: dict = pipeline_cfg.get("stages") or {}
    max_ranked: int = stages_cfg.get("max_ranked_chunks", _DEFAULT_MAX_RANKED)

    audit.append(
        _audit(
            "ranker_start",
            {
                "input_chunk_count": len(scored_chunks),
                "max_ranked_chunks": max_ranked,
            },
        )
    )

    relevance_weight = float(stages_cfg.get("relevance_weight", _RELEVANCE_WEIGHT))
    quality_weight = float(stages_cfg.get("quality_weight", _QUALITY_WEIGHT))

    doc_lengths: list[int] = []
    for chunk in scored_chunks:
        tokens = _tokenize(chunk.get("text", ""))
        doc_lengths.append(len(tokens))

    n_docs = len(scored_chunks)
    avg_doc_len = sum(doc_lengths) / n_docs if n_docs > 0 else 1.0
    corpus_idf = compute_corpus_idf(scored_chunks)

    ranked: list[dict[str, Any]] = []
    for chunk in scored_chunks:
        text: str = chunk.get("text", "")
        quality: float = chunk.get("quality_score", 0.5)

        relevance = compute_relevance_score(
            user_query, text, avg_doc_len=avg_doc_len, idf=corpus_idf
        )
        ranking_score = compute_ranking_score(
            relevance, quality, relevance_weight=relevance_weight, quality_weight=quality_weight
        )

        ranked_chunk = {
            **chunk,
            "relevance_score": relevance,
            "ranking_score": ranking_score,
        }
        ranked.append(ranked_chunk)

    # Sort by ranking_score descending, then trim to top-N.
    ranked.sort(key=lambda c: c["ranking_score"], reverse=True)
    trimmed = ranked[:max_ranked]

    audit.append(
        _audit(
            "ranker_complete",
            {
                "total_scored": len(ranked),
                "returned_top_n": len(trimmed),
                "max_ranked_chunks": max_ranked,
            },
        )
    )

    return {
        "ranked_chunks": trimmed,
        "audit_trail": audit,
    }
