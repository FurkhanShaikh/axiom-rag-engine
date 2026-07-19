"""Pin the corpus-retrieval eval's scoring logic.

Like `test_retrieval_eval`, the eval script is loaded by path (evals/ is a script
directory, not a package). A deterministic bag-of-words embedder is injected so
the ingest → store → search → score round trip runs with no live model, and the
recall it computes can be asserted exactly.
"""

from __future__ import annotations

import asyncio
import importlib.util
import math
import sys
from pathlib import Path

_EVALS = Path(__file__).resolve().parents[2] / "evals"
sys.path.insert(0, str(_EVALS))
_spec = importlib.util.spec_from_file_location("axiom_corpus_eval", _EVALS / "corpus_eval.py")
assert _spec and _spec.loader
ceval = importlib.util.module_from_spec(_spec)
sys.modules["axiom_corpus_eval"] = ceval
_spec.loader.exec_module(ceval)

_VOCAB = ["alpha", "beta", "gamma", "delta", "epsilon"]


def _vectorize(text: str) -> list[float]:
    counts = [float(text.lower().count(w)) for w in _VOCAB]
    if sum(counts) == 0.0:
        counts = [1.0] * len(_VOCAB)
    norm = math.sqrt(sum(c * c for c in counts))
    return [c / norm for c in counts]


async def _fake_docs(model: str, texts: list[str]) -> list[list[float]]:
    return [_vectorize(t) for t in texts]


async def _fake_query(model: str, query: str) -> list[float]:
    return _vectorize(query)


def _corpus() -> object:
    return ceval.Corpus(
        doc_ids=["alpha", "beta", "gamma"],
        texts=[
            "Alpha alpha alpha energy storage chemistry overview written here.",
            "Beta beta capacitor discharge characteristics summarized here today.",
            "Gamma gamma radiation shielding material engineering notes here.",
        ],
    )


def _queries() -> list[object]:
    return [
        ceval.Query(claim_id="q1", text="alpha", relevant_doc_ids={"alpha"}),
        ceval.Query(claim_id="q2", text="beta", relevant_doc_ids={"beta"}),
        ceval.Query(claim_id="q3", text="gamma", relevant_doc_ids={"gamma"}),
    ]


def test_evaluate_scores_perfect_recall_on_separable_corpus(tmp_path) -> None:
    summary, ingested = asyncio.run(
        ceval.evaluate(
            _corpus(),
            _queries(),
            embedding_model="fake",
            db_path=str(tmp_path / "corpus_eval.db"),
            embed_docs=_fake_docs,
            embed_q=_fake_query,
            candidate_depth=20,
        )
    )
    assert ingested == 3
    assert summary["queries"] == 3
    # Each query's gold document is the clear cosine winner → recall@1 == 1.0.
    assert summary["recall_at_1"] == 1.0
    assert summary["mrr"] == 1.0
    assert summary["ndcg_at_10"] == 1.0


def test_evaluate_misses_when_gold_is_absent(tmp_path) -> None:
    # Query points at a document that was never ingested → zero recall.
    queries = [ceval.Query(claim_id="q", text="alpha", relevant_doc_ids={"not-ingested"})]
    summary, _ = asyncio.run(
        ceval.evaluate(
            _corpus(),
            queries,
            embedding_model="fake",
            db_path=str(tmp_path / "corpus_eval.db"),
            embed_docs=_fake_docs,
            embed_q=_fake_query,
            candidate_depth=20,
        )
    )
    assert summary["recall_at_10"] == 0.0
    assert summary["mrr"] == 0.0
