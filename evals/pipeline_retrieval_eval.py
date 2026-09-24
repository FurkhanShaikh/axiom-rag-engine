"""Production-shaped retrieval eval — the shipped retriever/scorer/ranker on SciFact.

``retrieval_eval.py`` ranks all ~5,000 SciFact abstracts with BM25: it measures
the ranking *method* as a first-stage retriever. Production does something else:
a search engine returns a handful of pages, and the retriever chunks them, the
scorer adds quality scores, and the ranker re-ranks the chunks with BM25 blended
40% with those quality heuristics before trimming to ``max_ranked_chunks``. None
of that was measured.

This eval reproduces the production shape with no keys and no network:

1. For each claim, a stand-in "search engine" (BM25 over the whole corpus)
   returns ``--pool`` documents: the claim's gold documents plus the hardest
   non-gold ones (the top BM25 distractors), in search-rank order.
2. Those documents go through the shipped ``retriever_node`` (chunking),
   ``scorer_node`` (quality scores) and ``ranker_node`` (ranking + trim) —
   the production code path, under explicit settings so an environment's
   embedding or reranker model cannot change the result.
3. Metrics are about the evidence the synthesizer would see:
   - ``p_at_1``: the top-ranked chunk comes from a gold document;
   - ``mrr``: reciprocal rank of the first gold chunk;
   - ``evidence_recall``: a gold chunk survives the trim into the context.

``--variant`` compares the production ranking blend with BM25 alone (quality
weight 0) — the measurement the quality heuristics never had.

Usage:
    uv run python evals/pipeline_retrieval_eval.py --limit 0
    uv run python evals/pipeline_retrieval_eval.py --limit 0 --variant bm25_only
    uv run python evals/pipeline_retrieval_eval.py --limit 0 --gate
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import gate
import retrieval_eval as reval

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from axiom_rag_engine.config.settings import Settings, use_settings
from axiom_rag_engine.nodes.ranker import ranker_node
from axiom_rag_engine.nodes.retriever import MockSearchBackend, retriever_node
from axiom_rag_engine.nodes.scorer import scorer_node
from axiom_rag_engine.state import make_initial_state

EVALS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVALS_DIR / "results"
BASELINE_PATH = EVALS_DIR / "baselines" / "retrieval-pipeline.json"

_URL = "https://scifact.example/doc/{doc_id}"
_VARIANTS: dict[str, dict[str, float]] = {
    # The shipped defaults (no override).
    "production": {},
    # Relevance only: what the quality blend is measured against.
    "bm25_only": {"relevance_weight": 1.0, "quality_weight": 0.0},
}


def _echo(message: str = "") -> None:
    sys.stdout.write(f"{message}\n")


@dataclass
class PipelineResult:
    claim_id: str
    gold_docs: int
    pool_docs: int
    ranked_chunks: int
    first_gold_rank: int | None  # 1-based among ranked chunks; None = trimmed out
    p_at_1: float
    rr: float
    evidence_recall: float


def search_pool(
    bm25: reval.BM25Ranker, corpus_text: dict[str, str], query: reval.Query, pool: int
) -> list[dict[str, Any]]:
    """The stand-in search engine: gold documents plus the hardest distractors,
    in corpus-BM25 order (gold documents BM25 missed come last)."""
    order = bm25.rank(query.text)
    distractors = [d for d in order if d not in query.relevant_doc_ids]
    chosen = set(query.relevant_doc_ids) | set(
        distractors[: max(1, pool - len(query.relevant_doc_ids))]
    )
    position = {doc_id: i for i, doc_id in enumerate(order)}
    ranked = sorted(chosen, key=lambda d: (position.get(d, len(order)), d))
    return [
        {
            "url": _URL.format(doc_id=d),
            "title": "",
            "content": corpus_text[d],
            "content_mode": "raw",
        }
        for d in ranked
    ]


async def rank_with_production(
    query: str, results: list[dict[str, Any]], max_ranked: int, weights: dict[str, float]
) -> list[dict[str, Any]]:
    """Run the shipped retriever → scorer → ranker over ``results``."""
    state: dict[str, Any] = dict(
        make_initial_state(
            request_id="pipeline-eval",
            user_query=query,
            app_config={},
            models_config={},
            pipeline_config={"stages": {"max_ranked_chunks": max_ranked, **weights}},
        )
    )
    config = {"configurable": {"search_backend": MockSearchBackend(results)}}
    state.update(await retriever_node(state, config))  # type: ignore[arg-type]
    state.update(await scorer_node(state))  # type: ignore[arg-type]
    state.update(await ranker_node(state))  # type: ignore[arg-type]
    return list(state["ranked_chunks"])


def score(query: reval.Query, ranked: list[dict[str, Any]], pool_docs: int) -> PipelineResult:
    gold_urls = {_URL.format(doc_id=d) for d in query.relevant_doc_ids}
    first = next(
        (i + 1 for i, chunk in enumerate(ranked) if chunk["source_url"] in gold_urls), None
    )
    return PipelineResult(
        claim_id=query.claim_id,
        gold_docs=len(query.relevant_doc_ids),
        pool_docs=pool_docs,
        ranked_chunks=len(ranked),
        first_gold_rank=first,
        p_at_1=1.0 if first == 1 else 0.0,
        rr=1.0 / first if first else 0.0,
        evidence_recall=1.0 if first else 0.0,
    )


def summarize(results: list[PipelineResult]) -> dict[str, Any]:
    n = len(results)
    if not n:
        return {"claims": 0}
    return {
        "claims": n,
        "p_at_1": round(sum(r.p_at_1 for r in results) / n, 4),
        "mrr": round(sum(r.rr for r in results) / n, 4),
        "evidence_recall": round(sum(r.evidence_recall for r in results) / n, 4),
        "avg_ranked_chunks": round(sum(r.ranked_chunks for r in results) / n, 2),
        "ci95": {
            "p_at_1": [
                round(x, 4) for x in gate.wilson_interval(int(sum(r.p_at_1 for r in results)), n)
            ],
            "evidence_recall": [
                round(x, 4)
                for x in gate.wilson_interval(int(sum(r.evidence_recall for r in results)), n)
            ],
        },
    }


def _gate_metrics(summary: dict[str, Any]) -> dict[str, float]:
    return {k: summary[k] for k in ("p_at_1", "mrr", "evidence_recall")}


async def evaluate(
    queries: list[reval.Query],
    corpus: reval.Corpus,
    pool: int,
    max_ranked: int,
    variant: str,
) -> list[PipelineResult]:
    # Explicit settings: an environment's embedding / reranker model must not
    # turn this deterministic BM25 measurement into a hybrid or LLM one.
    use_settings(Settings(embedding_model=None, reranker_model=None))
    bm25 = reval.BM25Ranker(corpus)
    corpus_text = dict(zip(corpus.doc_ids, corpus.texts, strict=True))
    results: list[PipelineResult] = []
    for query in queries:
        docs = search_pool(bm25, corpus_text, query, pool)
        ranked = await rank_with_production(query.text, docs, max_ranked, _VARIANTS[variant])
        results.append(score(query, ranked, len(docs)))
    return results


def run(
    limit: int,
    seed: int,
    pool: int,
    max_ranked: int,
    variant: str,
    gate_baseline: Path | None,
    ratchet: bool = False,
) -> int:
    corpus = reval.load_corpus()
    queries = reval.load_queries("dev")
    if limit:
        rng = random.Random(seed)  # noqa: S311 - reproducible sampling, not crypto
        queries = rng.sample(queries, min(limit, len(queries)))
    _echo(
        f"Scoring {len(queries)} claims: pool={pool} docs, max_ranked={max_ranked}, "
        f"variant={variant} ..."
    )
    start = time.monotonic()
    results = asyncio.run(evaluate(queries, corpus, pool, max_ranked, variant))
    summary = summarize(results)
    _echo(f"  ranked {len(results)} claims in {time.monotonic() - start:.1f}s")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"retrieval-pipeline-{variant}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(
        json.dumps(
            {
                "eval": "retrieval_pipeline",
                "dataset": "scifact/dev",
                "variant": variant,
                "pool": pool,
                "max_ranked": max_ranked,
                "summary": summary,
                "results": [asdict(r) for r in results],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    ci = summary["ci95"]
    _echo()
    _echo(f"  p@1 (top chunk gold)   : {summary['p_at_1']}  (95% CI {ci['p_at_1']})")
    _echo(f"  MRR (first gold chunk) : {summary['mrr']}")
    _echo(
        f"  evidence recall        : {summary['evidence_recall']}  (95% CI {ci['evidence_recall']})"
    )
    _echo(f"  avg ranked chunks      : {summary['avg_ranked_chunks']}")
    _echo(f"Full records: {out}")

    if gate_baseline is not None:
        report = gate.evaluate_gate(_gate_metrics(summary), gate.load_baseline(gate_baseline))
        _echo()
        _echo(report.render())
        if report.gating_failed:
            return 1
        if ratchet:
            moved = gate.ratchet_baseline(
                gate_baseline, _gate_metrics(summary), time.strftime("%Y-%m-%d")
            )
            _echo(f"Ratcheted {gate_baseline.name}: {', '.join(moved) or 'nothing to raise'}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0, help="Max claims (0 = all labeled)")
    parser.add_argument("--seed", type=int, default=13, help="Sampling seed")
    parser.add_argument("--pool", type=int, default=10, help="Documents per search (gold + hard)")
    parser.add_argument("--max-ranked", type=int, default=10, help="Chunks kept for synthesis")
    parser.add_argument("--variant", default="production", choices=tuple(_VARIANTS))
    parser.add_argument(
        "--gate",
        nargs="?",
        const=str(BASELINE_PATH),
        default=None,
        metavar="BASELINE",
        help=f"Fail (exit 1) on regression against a baseline. Defaults to {BASELINE_PATH.name}.",
    )
    parser.add_argument(
        "--ratchet",
        action="store_true",
        help="With --gate: raise the baseline's floors to this run's values where it did better.",
    )
    args = parser.parse_args()
    baseline = Path(args.gate) if args.gate else None
    sys.exit(
        run(
            args.limit,
            args.seed,
            args.pool,
            args.max_ranked,
            args.variant,
            baseline,
            ratchet=args.ratchet,
        )
    )


if __name__ == "__main__":
    main()
