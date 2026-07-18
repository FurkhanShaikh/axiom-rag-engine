"""Dense + hybrid ranker correctness (evals/retrieval_eval.py).

Uses a fake embedder with hand-chosen vectors so the cosine ranking and the
reciprocal-rank fusion are checked deterministically, without Ollama. The point
is to prove the fusion math is right — the *quality* question (does hybrid beat
BM25) is answered by the eval on real data, not here.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_EVALS = Path(__file__).resolve().parents[2] / "evals"
sys.path.insert(0, str(_EVALS))
_spec = importlib.util.spec_from_file_location(
    "axiom_retrieval_eval2", _EVALS / "retrieval_eval.py"
)
assert _spec and _spec.loader
reval = importlib.util.module_from_spec(_spec)
sys.modules["axiom_retrieval_eval2"] = reval
_spec.loader.exec_module(reval)


class FakeEmbedder:
    """Returns pre-assigned unit vectors per text; unknown texts embed to zero."""

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self._v = {k: _unit(np.asarray(val, dtype=np.float32)) for k, val in vectors.items()}
        self._dim = len(next(iter(vectors.values())))

    def _lookup(self, texts: list[str]) -> np.ndarray:
        rows = [self._v.get(t, np.zeros(self._dim, dtype=np.float32)) for t in texts]
        return np.asarray(rows, dtype=np.float32)

    def embed_corpus(self, doc_ids: list[str], texts: list[str]) -> np.ndarray:
        return self._lookup(texts)

    def embed_queries(self, texts: list[str], *, label: str = "") -> np.ndarray:
        return self._lookup(texts)


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n else v


class TestDenseRanker:
    def test_ranks_by_cosine_similarity(self) -> None:
        # Three orthogonal-ish docs; a query aligned with doc B must rank B first.
        corpus = reval.Corpus(
            doc_ids=["A", "B", "C"],
            texts=["doc a", "doc b", "doc c"],
        )
        embedder = FakeEmbedder(
            {
                "doc a": [1.0, 0.0, 0.0],
                "doc b": [0.0, 1.0, 0.0],
                "doc c": [0.0, 0.0, 1.0],
                "find b": [0.1, 0.9, 0.0],
            }
        )
        ranked = reval.DenseRanker(corpus, embedder).rank("find b")
        assert ranked[0] == "B"

    def test_prewarm_populates_query_cache(self) -> None:
        corpus = reval.Corpus(doc_ids=["A"], texts=["doc a"])
        embedder = FakeEmbedder({"doc a": [1.0, 0.0], "q": [1.0, 0.0]})
        ranker = reval.DenseRanker(corpus, embedder)
        ranker.prewarm_queries(["q", "q"])  # dedups
        assert "q" in ranker._qcache


class TestHybridRRF:
    def _corpus(self) -> object:
        return reval.Corpus(
            doc_ids=["A", "B", "C", "D"],
            texts=["ta", "tb", "tc", "td"],
        )

    def test_document_ranked_high_by_either_ranker_surfaces(self) -> None:
        """A doc BM25 buries but dense loves (or vice versa) should rise under RRF."""
        corpus = self._corpus()
        # Dense vectors: query aligns with D, then C — the reverse of a
        # hypothetical BM25 order — so fusion should pull D and C up.
        embedder = FakeEmbedder(
            {
                "ta": [1.0, 0.0],
                "tb": [0.9, 0.1],
                "tc": [0.2, 0.9],
                "td": [0.0, 1.0],
                "q": [0.0, 1.0],
            }
        )
        hybrid = reval.HybridRanker(corpus, embedder)
        ranked = hybrid.rank("q")
        # D is dense-rank 1; it must not land last after fusion.
        assert ranked.index("D") < len(ranked) - 1
        assert set(ranked) == {"A", "B", "C", "D"}

    def test_rrf_score_formula(self) -> None:
        """Fusion score is sum(1/(k+rank+1)) across both rankers."""
        corpus = self._corpus()
        embedder = FakeEmbedder(
            {
                "ta": [1.0, 0.0],
                "tb": [0.0, 1.0],
                "tc": [1.0, 1.0],
                "td": [1.0, -1.0],
                "q": [1.0, 0.0],
            }
        )
        hybrid = reval.HybridRanker(corpus, embedder, k=60)
        bm = hybrid._bm25.rank("q")
        dn = hybrid._dense.rank("q")
        expected: dict[str, float] = {}
        for ranking in (bm, dn):
            for rank, doc in enumerate(ranking):
                expected[doc] = expected.get(doc, 0.0) + 1.0 / (60 + rank + 1)
        want = sorted(expected, key=lambda d: (-expected[d], d))
        assert hybrid.rank("q") == want

    def test_output_is_a_permutation_of_the_corpus(self) -> None:
        corpus = self._corpus()
        embedder = FakeEmbedder(
            {
                "ta": [1.0, 0.0],
                "tb": [0.0, 1.0],
                "tc": [1.0, 1.0],
                "td": [0.5, 0.5],
                "q": [1.0, 1.0],
            }
        )
        ranked = reval.HybridRanker(corpus, embedder).rank("q")
        assert sorted(ranked) == ["A", "B", "C", "D"]


class TestBuildRanker:
    def test_bm25_needs_no_embedder(self) -> None:
        corpus = reval.Corpus(doc_ids=["A"], texts=["some words here for tokens"])
        assert isinstance(reval.build_ranker("bm25", corpus, None), reval.BM25Ranker)

    def test_dense_without_embedder_raises(self) -> None:
        corpus = reval.Corpus(doc_ids=["A"], texts=["x"])
        with pytest.raises(ValueError, match="requires an embedder"):
            reval.build_ranker("dense", corpus, None)

    def test_unknown_method_raises(self) -> None:
        corpus = reval.Corpus(doc_ids=["A"], texts=["x"])
        with pytest.raises(ValueError, match="unknown method"):
            reval.build_ranker("bogus", corpus, None)
