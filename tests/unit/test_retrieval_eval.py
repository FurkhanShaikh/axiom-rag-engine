"""Retrieval eval — fidelity of the fast BM25 path and correctness of IR metrics.

The eval's pre-tokenized ``BM25Ranker._score`` must produce the *same* number as
the production ``compute_relevance_score`` it stands in for; otherwise the
retrieval benchmark measures a lookalike, not the ranker that ships. This test
pins the two together. It also checks the IR metric implementations against
hand-worked cases.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from axiom_rag_engine.nodes.ranker import compute_corpus_idf, compute_relevance_score

# evals/ is a script directory, not a package — load by path.
_EVALS = Path(__file__).resolve().parents[2] / "evals"
sys.path.insert(0, str(_EVALS))
_spec = importlib.util.spec_from_file_location("axiom_retrieval_eval", _EVALS / "retrieval_eval.py")
assert _spec and _spec.loader
reval = importlib.util.module_from_spec(_spec)
sys.modules["axiom_retrieval_eval"] = reval
_spec.loader.exec_module(reval)


_DOCS = [
    "Solid-state batteries replace liquid electrolytes with solid ceramics for safety.",
    "The gravitational pull of the Moon is the primary cause of ocean tides on Earth.",
    "Photosynthesis converts light energy into chemical energy stored as glucose in plants.",
    "Vaccines train the immune system to recognize and fight specific pathogens quickly.",
    "The speed of light in a vacuum is a fundamental constant of the universe.",
]
_QUERIES = [
    "how do solid-state batteries improve safety",
    "what causes ocean tides",
    "gravitational pull moon tides earth water",
    "immune system pathogens",
    "a query with no overlapping terms zzzzz",
    "",
]


class TestBM25Fidelity:
    """The eval's fast ranker must match production scoring exactly."""

    def _corpus(self) -> object:
        return reval.Corpus(doc_ids=[str(i) for i in range(len(_DOCS))], texts=list(_DOCS))

    def test_fast_score_matches_production_per_pair(self) -> None:
        corpus = self._corpus()
        ranker = reval.BM25Ranker(corpus)

        # Production reference: same IDF and avg_doc_len the ranker computed.
        idf = compute_corpus_idf([{"text": t} for t in _DOCS])
        avg_len = ranker._avg_len

        from axiom_rag_engine.nodes.ranker import _tokenize

        for q in _QUERIES:
            qtokens = _tokenize(q)
            for i, doc in enumerate(_DOCS):
                fast = ranker._score(qtokens, i) if qtokens else 0.0
                prod = compute_relevance_score(q, doc, avg_doc_len=avg_len, idf=idf)
                # Empty query: production returns 0.0; the ranker skips scoring.
                expected = prod if qtokens else 0.0
                assert fast == pytest.approx(expected, abs=1e-9), (
                    f"query={q!r} doc={i}: fast={fast} prod={expected}"
                )

    def test_ranking_order_is_sensible(self) -> None:
        ranker = reval.BM25Ranker(self._corpus())
        ranked = ranker.rank("what causes ocean tides from the moon")
        # The tides document (index 1) should rank first.
        assert ranked[0] == "1"

    def test_empty_query_returns_all_docs_stably(self) -> None:
        ranker = reval.BM25Ranker(self._corpus())
        ranked = ranker.rank("")
        assert sorted(ranked) == sorted(str(i) for i in range(len(_DOCS)))

    def test_rank_matches_bruteforce_score(self) -> None:
        """The fast inverted-index rank() must equal a full _score sort.

        _score is pinned to production by test_fast_score_matches_production;
        this pins the optimized rank() to _score, so the speedup can never change
        the measured ranking.
        """
        ranker = reval.BM25Ranker(self._corpus())
        from axiom_rag_engine.nodes.ranker import _tokenize

        for q in _QUERIES:
            fast = ranker.rank(q)
            qtokens = _tokenize(q)
            if not qtokens:
                continue
            brute = [
                ranker.doc_ids[i]
                for _, i in sorted(
                    ((ranker._score(qtokens, i), i) for i in range(len(ranker.doc_ids))),
                    key=lambda t: (-t[0], t[1]),
                )
            ]
            assert fast == brute, f"query={q!r}: fast rank != brute-force _score rank"


class TestIRMetrics:
    def test_recall_at_k(self) -> None:
        ranked = ["a", "b", "c", "d", "e"]
        assert reval.recall_at_k(ranked, {"a", "c"}, 5) == 1.0
        assert reval.recall_at_k(ranked, {"a", "c"}, 1) == 0.5
        assert reval.recall_at_k(ranked, {"z"}, 5) == 0.0
        assert reval.recall_at_k(ranked, set(), 5) == 0.0

    def test_ndcg_rewards_higher_placement(self) -> None:
        top = reval.ndcg_at_k(["gold", "x", "y"], {"gold"}, 10)
        low = reval.ndcg_at_k(["x", "y", "gold"], {"gold"}, 10)
        assert top == 1.0  # single gold doc at rank 1 is ideal
        assert low < top

    def test_ndcg_perfect_ordering_is_one(self) -> None:
        assert reval.ndcg_at_k(["a", "b", "c"], {"a", "b"}, 10) == pytest.approx(1.0)

    def test_reciprocal_rank(self) -> None:
        assert reval.reciprocal_rank(["a", "b", "c"], {"b"}) == pytest.approx(0.5)
        assert reval.reciprocal_rank(["a", "b", "c"], {"a"}) == 1.0
        assert reval.reciprocal_rank(["a", "b"], {"z"}) == 0.0


class TestSummary:
    def test_summary_averages_over_queries(self) -> None:
        results = [
            reval.QueryResult("1", 1, {1: 1.0, 5: 1.0, 10: 1.0, 20: 1.0}, 1.0, 1.0, 1),
            reval.QueryResult("2", 1, {1: 0.0, 5: 1.0, 10: 1.0, 20: 1.0}, 0.5, 0.5, 2),
        ]
        s = reval._summarize(results)
        assert s["queries"] == 2
        assert s["recall_at_1"] == 0.5
        assert s["recall_at_10"] == 1.0
        assert s["mrr"] == 0.75

    def test_gate_metrics_cover_baseline_keys(self) -> None:
        s = {
            "recall_at_1": 0.4,
            "recall_at_5": 0.7,
            "recall_at_10": 0.8,
            "recall_at_20": 0.9,
            "ndcg_at_10": 0.6,
            "mrr": 0.5,
        }
        metrics = reval._gate_metrics(s)
        assert set(metrics) == {"recall_at_10", "recall_at_20", "ndcg_at_10", "mrr"}
