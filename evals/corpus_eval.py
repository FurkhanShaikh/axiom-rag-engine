"""Corpus-retrieval eval — measure the *shipped* ingest → store → search path.

`retrieval_eval.py` scores a ranker over an in-memory candidate set. This eval is
different: it exercises the production bring-your-own-corpus code end to end —
ingesting a labeled dataset through `corpus.ingest.ingest_text` (the real chunker
+ embedder), persisting to a real `CorpusStore`, and retrieving with
`CorpusStore.search`. The numbers therefore reflect what a user actually gets,
including chunking and the SQLite round trip, not a lookalike implementation.

It reuses the eval harness for datasets and metrics (`retrieval_eval.load_dataset`
and the recall/nDCG/MRR functions), so corpus numbers are directly comparable to
the web-retrieval numbers.

Needs a live LiteLLM embedding model, e.g.:

    uv run python evals/corpus_eval.py --dataset scifact --model ollama/nomic-embed-text --limit 100

``--bench-search N`` instead times ``CorpusStore.search`` over N synthetic chunks
(no model needed): uncached pure Python (every query re-reads and decodes every
vector, as search did before the vector cache), cached pure Python, and cached
numpy.

Recall is measured at the *document* level: a document is chunked on ingest, and a
retrieved chunk is mapped back to its document before scoring.
"""

from __future__ import annotations

import argparse
import asyncio
import random
import sys
import tempfile
from collections.abc import Awaitable, Callable
from pathlib import Path

# Sibling eval modules (datasets, metrics, gate) resolve from the evals/ dir.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gate
from retrieval_eval import (
    _DEEP_K,
    _K_VALUES,
    Corpus,
    Query,
    QueryResult,
    _gate_metrics,
    _summarize,
    load_dataset,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)

# Production corpus code — the whole point is to score *this*, not a copy.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from axiom_rag_engine.corpus.ingest import IngestionError, ingest_text
from axiom_rag_engine.corpus.store import CorpusStore
from axiom_rag_engine.embeddings import embed_documents, embed_query

EmbedDocs = Callable[[str, list[str]], Awaitable[list[list[float]]]]
EmbedQuery = Callable[[str, str], Awaitable[list[float]]]

EVALS_DIR = Path(__file__).resolve().parent
BASELINE_PATH = EVALS_DIR / "baselines" / "corpus-dense.json"


def _echo(message: str = "") -> None:
    sys.stdout.write(f"{message}\n")


async def _ingest_corpus(
    store: CorpusStore,
    corpus: Corpus,
    *,
    embedding_model: str,
    embed_docs: EmbedDocs,
    max_chunks: int | None,
) -> int:
    """Ingest every corpus document through the production path. Returns the
    number successfully ingested (documents that yield no chunks are skipped)."""
    ingested = 0
    for doc_id, text in zip(corpus.doc_ids, corpus.texts, strict=True):
        try:
            await ingest_text(
                store,
                doc_id=doc_id,
                text=text,
                embedding_model=embedding_model,
                embedder=embed_docs,
                max_chunks=max_chunks,
            )
            ingested += 1
        except IngestionError:
            continue
    return ingested


async def _ranked_doc_ids(
    store: CorpusStore,
    query_text: str,
    *,
    embedding_model: str,
    embed_q: EmbedQuery,
    candidate_depth: int,
) -> list[str]:
    """Retrieve chunks for a query and collapse them to a ranked list of unique
    document ids (first occurrence wins, preserving similarity order)."""
    query_vec = await embed_q(embedding_model, query_text)
    hits = store.search(query_vec, embedding_model=embedding_model, k=candidate_depth)
    ordered: list[str] = []
    seen: set[str] = set()
    for hit in hits:
        if hit.doc_id not in seen:
            seen.add(hit.doc_id)
            ordered.append(hit.doc_id)
    return ordered


async def evaluate(
    corpus: Corpus,
    queries: list[Query],
    *,
    embedding_model: str,
    db_path: str,
    embed_docs: EmbedDocs = embed_documents,
    embed_q: EmbedQuery = embed_query,
    candidate_depth: int = _DEEP_K * 10,
    max_chunks: int | None = None,
) -> tuple[dict, int]:
    """Ingest ``corpus`` and score ``queries`` through the production store.

    Embedders are injectable so this core is unit-testable without a live model;
    the CLI wires the real LiteLLM embedders. Returns ``(summary, num_ingested)``.
    """
    store = CorpusStore(db_path)
    ingested = await _ingest_corpus(
        store, corpus, embedding_model=embedding_model, embed_docs=embed_docs, max_chunks=max_chunks
    )

    results: list[QueryResult] = []
    for q in queries:
        ranked = await _ranked_doc_ids(
            store,
            q.text,
            embedding_model=embedding_model,
            embed_q=embed_q,
            candidate_depth=candidate_depth,
        )
        results.append(
            QueryResult(
                claim_id=q.claim_id,
                relevant=len(q.relevant_doc_ids),
                recall={k: recall_at_k(ranked, q.relevant_doc_ids, k) for k in _K_VALUES},
                ndcg_at_10=ndcg_at_k(ranked, q.relevant_doc_ids, 10),
                rr=reciprocal_rank(ranked, q.relevant_doc_ids),
                first_hit_rank=None,
            )
        )
    return _summarize(results), ingested


def run(
    *,
    dataset: str,
    embedding_model: str,
    limit: int,
    seed: int,
    candidate_depth: int,
    max_chunks: int | None,
    gate_baseline: Path | None,
) -> int:
    corpus, queries = load_dataset(dataset, paraphrased=False)
    _echo(f"Loaded corpus={len(corpus.doc_ids)} docs, labeled queries={len(queries)} ({dataset})")

    rng = random.Random(seed)  # noqa: S311 - reproducible sampling, not crypto
    rng.shuffle(queries)
    sample = queries[:limit] if limit else queries

    _echo(
        f"Ingesting {len(corpus.doc_ids)} docs via the production pipeline "
        f"(model {embedding_model}) — this embeds every document and is not cached."
    )
    with tempfile.TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "corpus_eval.db")
        summary, ingested = asyncio.run(
            evaluate(
                corpus,
                sample,
                embedding_model=embedding_model,
                db_path=db_path,
                candidate_depth=candidate_depth,
                max_chunks=max_chunks,
            )
        )

    _echo()
    _echo(f"Corpus retrieval — {dataset} (model {embedding_model}, {ingested} docs ingested)")
    _echo(f"  queries scored : {summary.get('queries', 0)}")
    for k in _K_VALUES:
        _echo(f"  recall@{k:<2}      : {summary.get(f'recall_at_{k}', 0.0):.4f}")
    _echo(f"  nDCG@10        : {summary.get('ndcg_at_10', 0.0):.4f}")
    _echo(f"  MRR            : {summary.get('mrr', 0.0):.4f}")

    if gate_baseline is not None:
        report = gate.evaluate_gate(_gate_metrics(summary), gate.load_baseline(gate_baseline))
        _echo()
        _echo(report.render())
        return 1 if report.gating_failed else 0
    return 0


def _unit(vec: list[float]) -> list[float]:
    norm = sum(x * x for x in vec) ** 0.5 or 1.0
    return [x / norm for x in vec]


def bench_search(
    n_chunks: int, dim: int = 768, queries: int = 30, k: int = 50, seed: int = 13
) -> dict[str, dict[str, float]]:
    """Time ``CorpusStore.search`` over ``n_chunks`` random unit vectors.

    Returns mean and p95 milliseconds per query for each mode. The vectors are
    synthetic, so this measures speed only, not retrieval quality.
    """
    import statistics
    import time

    rng = random.Random(seed)  # noqa: S311 - benchmark data, not crypto
    per_doc = 100
    results: dict[str, dict[str, float]] = {}
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bench.db"
        writer = CorpusStore(path)
        for d in range(0, n_chunks, per_doc):
            size = min(per_doc, n_chunks - d)
            writer.add_document(
                doc_id=f"d{d}",
                title=f"doc {d}",
                source="bench",
                embedding_model="bench",
                chunks=[
                    (f"chunk {d + i}", _unit([rng.gauss(0, 1) for _ in range(dim)]))
                    for i in range(size)
                ],
            )
        probes = [_unit([rng.gauss(0, 1) for _ in range(dim)]) for _ in range(queries)]

        modes = {
            "uncached, pure Python": (False, True),
            "cached, pure Python": (False, False),
            "cached, numpy": (True, False),
        }
        for label, (use_numpy, cold) in modes.items():
            store = CorpusStore(path, use_numpy=use_numpy)
            store.search(probes[0], embedding_model="bench", k=k)  # build the cache
            timings = []
            for probe in probes:
                if cold:
                    store._indexes.clear()
                start = time.perf_counter()
                store.search(probe, embedding_model="bench", k=k)
                timings.append((time.perf_counter() - start) * 1000)
            timings.sort()
            results[label] = {
                "mean_ms": round(statistics.fmean(timings), 2),
                "p95_ms": round(timings[int(0.95 * (len(timings) - 1))], 2),
            }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="scifact", help="scifact | arguana | <beir name>")
    parser.add_argument(
        "--model",
        default="ollama/nomic-embed-text",
        help="LiteLLM embedding model (ollama/nomic-embed-text, text-embedding-3-small, ...)",
    )
    parser.add_argument("--limit", type=int, default=100, help="Max queries to score (0 = all)")
    parser.add_argument("--seed", type=int, default=13, help="Sampling seed")
    parser.add_argument(
        "--candidate-depth",
        type=int,
        default=_DEEP_K * 10,
        help="Chunks retrieved per query before collapsing to unique documents",
    )
    parser.add_argument(
        "--max-chunks", type=int, default=None, help="Cap chunks per document (default: unbounded)"
    )
    parser.add_argument(
        "--gate",
        action="store_true",
        help=f"Compare against the committed baseline ({BASELINE_PATH.name}) and exit nonzero on regression",
    )
    parser.add_argument(
        "--bench-search",
        type=int,
        metavar="N",
        default=0,
        help="Time CorpusStore.search over N synthetic chunks instead (no model needed)",
    )
    args = parser.parse_args()

    if args.bench_search:
        _echo(f"CorpusStore.search over {args.bench_search} chunks (dim 768, k=50):")
        for label, row in bench_search(args.bench_search).items():
            _echo(f"  {label:<24} mean {row['mean_ms']:>8.2f} ms   p95 {row['p95_ms']:>8.2f} ms")
        raise SystemExit(0)

    baseline = BASELINE_PATH if args.gate else None
    if args.gate and not BASELINE_PATH.exists():
        _echo(f"No baseline at {BASELINE_PATH}; run without --gate first and save the numbers.")
        raise SystemExit(2)

    raise SystemExit(
        run(
            dataset=args.dataset,
            embedding_model=args.model,
            limit=args.limit,
            seed=args.seed,
            candidate_depth=args.candidate_depth,
            max_chunks=args.max_chunks,
            gate_baseline=baseline,
        )
    )


if __name__ == "__main__":
    main()
