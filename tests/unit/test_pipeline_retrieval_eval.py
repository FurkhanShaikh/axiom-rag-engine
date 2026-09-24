"""
Production-shaped retrieval eval — scores the shipped retriever/scorer/ranker.

``retrieval_eval.py`` ranks the whole SciFact corpus with BM25, which is not
what production does; this eval feeds a small search pool through the shipped
nodes. These tests pin its pool construction and scoring on a toy corpus.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_EVALS = Path(__file__).resolve().parents[2] / "evals"
sys.path.insert(0, str(_EVALS))


def _load(name: str, filename: str) -> object:
    spec = importlib.util.spec_from_file_location(name, _EVALS / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


reval = _load("retrieval_eval", "retrieval_eval.py")
peval = _load("axiom_pipeline_retrieval_eval", "pipeline_retrieval_eval.py")

_TEXTS = {
    "gold": (
        "Vitamin D supplementation reduces the risk of fractures in elderly patients. "
        "A randomized trial of 2,686 people over 5 years found 22% fewer fractures."
    ),
    "near": (
        "Vitamin D levels in elderly patients vary with season and sun exposure, "
        "according to a cohort of 1,200 participants followed for 3 years."
    ),
    "far1": "Deep learning improves protein structure prediction accuracy in benchmarks.",
    "far2": "Soil microbiomes shift after wildfires, altering nitrogen cycling in forests.",
    "far3": "Quantum dots emit light at wavelengths set by their size and composition.",
}
_CORPUS = reval.Corpus(doc_ids=list(_TEXTS), texts=list(_TEXTS.values()))
_QUERY = reval.Query(
    claim_id="c1",
    text="Vitamin D supplementation reduces fracture risk in the elderly",
    relevant_doc_ids={"gold"},
)


def _urls(docs: list[dict]) -> list[str]:
    return [d["url"].rsplit("/", 1)[1] for d in docs]


def test_pool_has_gold_plus_hardest_distractors_in_search_order() -> None:
    bm25 = reval.BM25Ranker(_CORPUS)
    docs = peval.search_pool(bm25, _TEXTS, _QUERY, pool=3)
    ids = _urls(docs)
    assert "gold" in ids and "near" in ids  # the hardest distractor is included
    assert len(ids) == 3
    assert ids == [d for d in bm25.rank(_QUERY.text) if d in ids]  # search-rank order


async def test_ranks_with_the_production_nodes_and_scores_the_gold_chunk() -> None:
    results = await peval.evaluate([_QUERY], _CORPUS, pool=3, max_ranked=10, variant="production")
    (result,) = results
    assert result.pool_docs == 3
    assert result.ranked_chunks >= 3
    assert result.first_gold_rank == 1
    assert result.p_at_1 == 1.0
    assert result.evidence_recall == 1.0

    summary = peval.summarize(results)
    assert summary["p_at_1"] == 1.0
    assert summary["claims"] == 1


async def test_evidence_recall_counts_only_chunks_that_survive_the_trim() -> None:
    gold_first = reval.Query(claim_id="c2", text=_QUERY.text, relevant_doc_ids={"gold"})
    elsewhere = reval.Query(claim_id="c3", text=_QUERY.text, relevant_doc_ids={"far3"})
    kept, dropped = await peval.evaluate(
        [gold_first, elsewhere], _CORPUS, pool=5, max_ranked=1, variant="production"
    )
    # With room for one chunk, the best match survives and the off-topic
    # "gold" document is trimmed out of the context.
    assert kept.ranked_chunks == dropped.ranked_chunks == 1
    assert (kept.first_gold_rank, kept.evidence_recall) == (1, 1.0)
    assert (dropped.first_gold_rank, dropped.evidence_recall, dropped.rr) == (None, 0.0, 0.0)


def test_variants_are_the_production_default_and_bm25_only() -> None:
    assert peval._VARIANTS["production"] == {}
    assert peval._VARIANTS["bm25_only"] == {"relevance_weight": 1.0, "quality_weight": 0.0}
