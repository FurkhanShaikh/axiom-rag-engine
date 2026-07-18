"""Layer 3 eval — retrieval quality on SciFact.

The semantic eval measures whether the verifier judges a claim correctly; the
golden eval measures end-to-end pipeline behavior. Neither measures whether
retrieval surfaces the *right sources* in the first place — and verification can
only bless what retrieval finds. This layer closes that gap.

SciFact gives us relevance labels for free: each claim's ``evidence`` names the
corpus document(s) that actually support or refute it. So for a claim we can
rank the whole corpus and ask a standard IR question — do the gold documents
land near the top? — and score it with recall@k, nDCG@k, and MRR.

This is the measurement Phase 1 needs: the BM25 number here is the baseline that
hybrid retrieval and a reranker must beat. It runs with no LLM keys.

Fidelity: the BM25 ranker below reuses the production tokenizer, corpus IDF, and
BM25 constants from ``nodes/ranker.py``. It pre-tokenizes the corpus once for
speed (the production ``compute_relevance_score`` re-tokenizes per call, which is
fine in the pipeline's ~200-chunk context but far too slow over 5k docs x
hundreds of claims). ``tests/unit/test_retrieval_eval.py`` pins the fast path to
the production function so it can never silently diverge.

Usage:
    python tasks.py evals retrieval                          # BM25, default sample
    uv run python evals/retrieval_eval.py --limit 0          # all labeled claims
    uv run python evals/retrieval_eval.py --gate             # regress-gate the run
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import gate
import numpy as np
from embeddings import OllamaEmbedder

# Production ranker internals — reused so the eval scores the real BM25, not a
# lookalike. The fast pre-tokenized path replicates this math exactly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from axiom_rag_engine.nodes.ranker import (
    _BM25_B,
    _BM25_K1,
    _tokenize,
    compute_corpus_idf,
)

EVALS_DIR = Path(__file__).resolve().parent
SCIFACT_DIR = EVALS_DIR / "data" / "scifact"
BEIR_DIR = EVALS_DIR / "data" / "beir"
RESULTS_DIR = EVALS_DIR / "results"
BASELINE_PATH = EVALS_DIR / "baselines" / "retrieval-bm25.json"

# Cutoffs reported for recall@k. nDCG and MRR use the deepest cutoff.
_K_VALUES = (1, 5, 10, 20)
_DEEP_K = max(_K_VALUES)


def _echo(message: str = "") -> None:
    sys.stdout.write(f"{message}\n")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


@dataclass
class Corpus:
    doc_ids: list[str]
    texts: list[str]


@dataclass
class Query:
    claim_id: str
    text: str
    relevant_doc_ids: set[str]


def load_corpus() -> Corpus:
    path = SCIFACT_DIR / "corpus.jsonl"
    if not path.exists():
        _echo(f"SciFact corpus not found in {SCIFACT_DIR}. Run: python tasks.py evals download")
        sys.exit(1)
    doc_ids: list[str] = []
    texts: list[str] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            doc = json.loads(line)
            doc_ids.append(str(doc["doc_id"]))
            # Title + abstract mirrors how a retrieved page's text reaches the
            # ranker: a heading followed by body prose.
            title = doc.get("title", "") or ""
            abstract = " ".join(doc.get("abstract") or [])
            texts.append(f"{title}. {abstract}".strip())
    return Corpus(doc_ids=doc_ids, texts=texts)


_PARAPHRASE_PATH = SCIFACT_DIR / "paraphrases_dev.jsonl"


def _load_paraphrases() -> dict[str, str]:
    """claim_id -> paraphrased query text, if the cache exists."""
    if not _PARAPHRASE_PATH.exists():
        _echo(
            f"Paraphrase cache not found at {_PARAPHRASE_PATH}. "
            "Run: uv run python evals/make_paraphrases.py"
        )
        sys.exit(1)
    mapping: dict[str, str] = {}
    for line in _PARAPHRASE_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("paraphrase"):
                mapping[str(row["claim_id"])] = row["paraphrase"]
    return mapping


def load_queries(split: str = "dev", *, paraphrased: bool = False) -> list[Query]:
    path = SCIFACT_DIR / f"claims_{split}.jsonl"
    if not path.exists():
        _echo(f"SciFact claims not found in {SCIFACT_DIR}. Run: python tasks.py evals download")
        sys.exit(1)
    # In paraphrased mode, keep only claims that have a cached paraphrase, and
    # swap in the reworded text — so the original vs. paraphrased A/B compares
    # the exact same claim set with lexical overlap as the only difference.
    paraphrases = _load_paraphrases() if paraphrased else {}
    queries: list[Query] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            claim = json.loads(line)
            evidence: dict = claim.get("evidence") or {}
            relevant = {str(doc_id) for doc_id in evidence}
            if not relevant:
                # Only labeled claims can be scored for retrieval.
                continue
            claim_id = str(claim["id"])
            if paraphrased:
                if claim_id not in paraphrases:
                    continue
                text = paraphrases[claim_id]
            else:
                text = claim["claim"]
            queries.append(Query(claim_id=claim_id, text=text, relevant_doc_ids=relevant))
    return queries


# ---------------------------------------------------------------------------
# BEIR-format datasets (corpus.jsonl / queries.jsonl / qrels/test.tsv)
# ---------------------------------------------------------------------------


def load_beir(name: str) -> tuple[Corpus, list[Query]]:
    """Load a BEIR-format dataset (e.g. ArguAna) into the eval's shapes.

    BEIR is the standard IR benchmark layout: ``corpus.jsonl`` (``_id``/``title``
    /``text``), ``queries.jsonl`` (``_id``/``text``), and ``qrels/test.tsv``
    (``query-id  corpus-id  score``). Only queries with at least one positive
    qrel are kept, mirroring the SciFact loader.
    """
    root = BEIR_DIR / name
    if not root.exists():
        _echo(
            f"BEIR dataset not found at {root}. Run: python tasks.py evals download-beir -- {name}"
        )
        sys.exit(1)

    doc_ids: list[str] = []
    texts: list[str] = []
    for line in (root / "corpus.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        doc = json.loads(line)
        doc_ids.append(str(doc["_id"]))
        title = doc.get("title", "") or ""
        body = doc.get("text", "") or ""
        texts.append(f"{title}. {body}".strip())
    corpus = Corpus(doc_ids=doc_ids, texts=texts)

    # qrels: query_id -> set of relevant doc_ids (score > 0)
    relevant: dict[str, set[str]] = {}
    qrels_path = root / "qrels" / "test.tsv"
    for i, line in enumerate(qrels_path.read_text(encoding="utf-8").splitlines()):
        if i == 0 or not line.strip():  # header row
            continue
        qid, docid, score = line.split("\t")[:3]
        if int(score) > 0:
            relevant.setdefault(qid, set()).add(docid)

    query_text: dict[str, str] = {}
    for line in (root / "queries.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            q = json.loads(line)
            query_text[str(q["_id"])] = q["text"]

    queries = [
        Query(claim_id=qid, text=query_text[qid], relevant_doc_ids=docs)
        for qid, docs in relevant.items()
        if qid in query_text
    ]
    return corpus, queries


def load_dataset(dataset: str, *, paraphrased: bool) -> tuple[Corpus, list[Query]]:
    """Dispatch to the SciFact-native or BEIR-format loader."""
    if dataset == "scifact":
        return load_corpus(), load_queries("dev", paraphrased=paraphrased)
    if paraphrased:
        _echo("--paraphrased is only supported for the scifact dataset.")
        sys.exit(1)
    return load_beir(dataset)


# ---------------------------------------------------------------------------
# Rankers
# ---------------------------------------------------------------------------


class Ranker(Protocol):
    """A retrieval method: score the whole corpus for a query, best first."""

    def rank(self, query: str) -> list[str]:
        """Return corpus doc_ids ordered by descending relevance to ``query``."""
        ...


class BM25Ranker:
    """Pre-tokenized BM25 that replicates ``ranker.compute_relevance_score``.

    Same tokenizer, same corpus IDF, same K1/B constants, same max-possible
    normalization and 4-dp rounding as the production function — only the per-doc
    tokenization is hoisted out of the inner loop. The fidelity test guarantees
    the two agree.
    """

    def __init__(self, corpus: Corpus) -> None:
        self.doc_ids = corpus.doc_ids
        self._tfs: list[Counter[str]] = []
        self._lens: list[int] = []
        # Inverted index: term -> doc indices containing it. Lets rank() score
        # only docs that share a query term instead of the whole corpus.
        self._postings: dict[str, list[int]] = {}
        for idx, text in enumerate(corpus.texts):
            tokens = _tokenize(text)
            tf = Counter(tokens)
            self._tfs.append(tf)
            self._lens.append(len(tokens))
            for term in tf:
                self._postings.setdefault(term, []).append(idx)
        n = len(self._lens)
        self._avg_len = (sum(self._lens) / n) if n else 1.0
        # compute_corpus_idf expects chunk dicts with a "text" field.
        self._idf = compute_corpus_idf([{"text": t} for t in corpus.texts])

    def _score(self, query_tokens: list[str], doc_idx: int) -> float:
        """Per-pair BM25, kept as the reference the fidelity test pins to
        production. ``rank`` uses the faster inverted-index path below, which
        produces identical scores (see test_rank_matches_bruteforce_score)."""
        tf = self._tfs[doc_idx]
        doc_len = self._lens[doc_idx]
        score = 0.0
        max_possible = 0.0
        for term in set(query_tokens):
            term_idf = self._idf.get(term, 1.0)
            max_possible += term_idf * (_BM25_K1 + 1) / (1 + _BM25_K1)
            count = tf.get(term, 0)
            if count == 0:
                continue
            numerator = count * (_BM25_K1 + 1)
            denominator = count + _BM25_K1 * (
                1 - _BM25_B + _BM25_B * doc_len / max(self._avg_len, 1.0)
            )
            score += term_idf * numerator / denominator
        if max_possible > 0:
            score = min(1.0, score / max_possible)
        return round(score, 4)

    def rank(self, query: str) -> list[str]:
        query_tokens = _tokenize(query)
        if not query_tokens:
            return list(self.doc_ids)
        query_terms = set(query_tokens)
        # max_possible is query-dependent but doc-independent — compute it once,
        # not per doc (the previous per-doc recomputation was the hot spot).
        max_possible = sum(
            self._idf.get(t, 1.0) * (_BM25_K1 + 1) / (1 + _BM25_K1) for t in query_terms
        )
        # Accumulate the raw score for each doc that contains a query term.
        # Docs sharing no term score exactly 0.0 (as in _score), so they are
        # never visited — only their default 0.0 rank matters.
        raw: dict[int, float] = {}
        for term in query_terms:
            term_idf = self._idf.get(term, 1.0)
            for idx in self._postings.get(term, ()):
                count = self._tfs[idx][term]
                doc_len = self._lens[idx]
                numerator = count * (_BM25_K1 + 1)
                denominator = count + _BM25_K1 * (
                    1 - _BM25_B + _BM25_B * doc_len / max(self._avg_len, 1.0)
                )
                raw[idx] = raw.get(idx, 0.0) + term_idf * numerator / denominator
        scores: dict[int, float] = {}
        if max_possible > 0:
            scores = {idx: round(min(1.0, s / max_possible), 4) for idx, s in raw.items()}
        order = sorted(range(len(self.doc_ids)), key=lambda i: (-scores.get(i, 0.0), i))
        return [self.doc_ids[i] for i in order]


class DenseRanker:
    """Dense retrieval by cosine similarity over cached corpus embeddings.

    The corpus matrix is L2-normalized once (cached to disk by the embedder), so
    ranking a query is a single matrix-vector product. Query vectors are cached
    in memory and can be pre-warmed in one batched pass to avoid 188 serial
    embed calls.
    """

    def __init__(self, corpus: Corpus, embedder: OllamaEmbedder) -> None:
        self.doc_ids = corpus.doc_ids
        self._embedder = embedder
        self._matrix = embedder.embed_corpus(corpus.doc_ids, corpus.texts)  # (n, dim), normalized
        self._qcache: dict[str, np.ndarray] = {}

    def prewarm_queries(self, queries: list[str]) -> None:
        unique = [q for q in dict.fromkeys(queries) if q not in self._qcache]
        if not unique:
            return
        vecs = self._embedder.embed_queries(unique, label="queries")
        for q, v in zip(unique, vecs, strict=True):
            self._qcache[q] = v

    def _query_vec(self, query: str) -> np.ndarray:
        cached = self._qcache.get(query)
        if cached is None:
            cached = self._embedder.embed_queries([query])[0]
            self._qcache[query] = cached
        return cached

    def rank(self, query: str) -> list[str]:
        sims = self._matrix @ self._query_vec(query)  # cosine (both normalized)
        order = np.argsort(-sims, kind="stable")
        return [self.doc_ids[i] for i in order]


# Reciprocal-rank-fusion constant. 60 is the value from the original RRF paper
# and the de-facto default; it damps the influence of very deep ranks.
_RRF_K = 60


class HybridRanker:
    """Reciprocal-rank fusion of BM25 (lexical) and dense (semantic) rankings.

    RRF scores each document by ``sum(1 / (k + rank))`` across both rankers, so a
    document ranked highly by *either* method surfaces — dense can rescue a
    lexically-dissimilar match BM25 misses, and BM25 anchors exact-term queries
    the embedder blurs. Fusion on ranks (not raw scores) needs no score
    calibration between the two very different scales.
    """

    def __init__(self, corpus: Corpus, embedder: OllamaEmbedder, k: int = _RRF_K) -> None:
        self.doc_ids = corpus.doc_ids
        self._bm25 = BM25Ranker(corpus)
        self._dense = DenseRanker(corpus, embedder)
        self._k = k

    def prewarm_queries(self, queries: list[str]) -> None:
        self._dense.prewarm_queries(queries)

    def rank(self, query: str) -> list[str]:
        scores: dict[str, float] = {}
        for ranking in (self._bm25.rank(query), self._dense.rank(query)):
            for rank, doc_id in enumerate(ranking):
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (self._k + rank + 1)
        # Highest fused score first; doc_id tiebreak keeps the order reproducible.
        return sorted(scores, key=lambda d: (-scores[d], d))


_METHODS = ("bm25", "dense", "hybrid")
_NEEDS_EMBEDDER = frozenset({"dense", "hybrid"})


def _embed_prefixes(model: str) -> tuple[str, str]:
    """Return (doc_prefix, query_prefix) for instructed embedders.

    nomic-embed-text is trained with task prefixes and loses substantial
    retrieval quality without them. Other models default to no prefix.
    """
    if "nomic" in model.lower():
        return "search_document: ", "search_query: "
    return "", ""


def build_ranker(method: str, corpus: Corpus, embedder: OllamaEmbedder | None) -> Ranker:
    if method == "bm25":
        return BM25Ranker(corpus)
    if method not in _NEEDS_EMBEDDER:
        raise ValueError(f"unknown method {method!r}")
    if embedder is None:
        raise ValueError(f"method {method!r} requires an embedder")
    return DenseRanker(corpus, embedder) if method == "dense" else HybridRanker(corpus, embedder)


# ---------------------------------------------------------------------------
# IR metrics
# ---------------------------------------------------------------------------


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Fraction of gold documents found in the top k."""
    if not relevant:
        return 0.0
    hits = sum(1 for doc_id in ranked[:k] if doc_id in relevant)
    return hits / len(relevant)


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Binary-relevance nDCG@k (gain 1 for a gold doc, 0 otherwise)."""
    dcg = 0.0
    for i, doc_id in enumerate(ranked[:k]):
        if doc_id in relevant:
            dcg += 1.0 / math.log2(i + 2)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return (dcg / idcg) if idcg > 0 else 0.0


def reciprocal_rank(ranked: list[str], relevant: set[str]) -> float:
    """1 / rank of the first gold document, else 0."""
    for i, doc_id in enumerate(ranked):
        if doc_id in relevant:
            return 1.0 / (i + 1)
    return 0.0


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


@dataclass
class QueryResult:
    claim_id: str
    relevant: int
    recall: dict[int, float]
    ndcg_at_10: float
    rr: float
    first_hit_rank: int | None


def _summarize(results: list[QueryResult]) -> dict[str, Any]:
    n = len(results)
    if n == 0:
        return {"queries": 0}
    summary: dict[str, Any] = {"queries": n}
    for k in _K_VALUES:
        summary[f"recall_at_{k}"] = round(sum(r.recall[k] for r in results) / n, 4)
    summary["ndcg_at_10"] = round(sum(r.ndcg_at_10 for r in results) / n, 4)
    summary["mrr"] = round(sum(r.rr for r in results) / n, 4)
    return summary


def _gate_metrics(summary: dict[str, Any]) -> dict[str, float]:
    """Flatten the summary into the metric names the baseline gates on."""
    return {
        "recall_at_10": summary["recall_at_10"],
        "recall_at_20": summary["recall_at_20"],
        "ndcg_at_10": summary["ndcg_at_10"],
        "mrr": summary["mrr"],
    }


def run(
    method: str,
    limit: int,
    seed: int,
    dataset: str,
    gate_baseline: Path | None,
    embed_model: str,
    paraphrased: bool = False,
) -> int:
    corpus, queries = load_dataset(dataset, paraphrased=paraphrased)
    query_kind = "paraphrased" if paraphrased else "original"
    _echo(
        f"Loaded corpus={len(corpus.doc_ids)} docs, "
        f"labeled queries={len(queries)} ({dataset}, {query_kind} queries)"
    )

    rng = random.Random(seed)  # noqa: S311 - reproducible sampling, not crypto
    rng.shuffle(queries)
    sample = queries[:limit] if limit else queries

    embedder: OllamaEmbedder | None = None
    if method in _NEEDS_EMBEDDER:
        from _env import load_dotenv

        load_dotenv()
        base = os.environ.get("OLLAMA_API_BASE", "http://localhost:11434")
        doc_prefix, query_prefix = _embed_prefixes(embed_model)
        embedder = OllamaEmbedder(
            model=embed_model,
            base_url=base,
            doc_prefix=doc_prefix,
            query_prefix=query_prefix,
        )
        _echo(f"Embedder: {embed_model} @ {base}" + (" (nomic prefixes)" if doc_prefix else ""))

    _echo(f"Building {method} index over {len(corpus.doc_ids)} docs ...")
    ranker: Ranker = build_ranker(method, corpus, embedder)

    # Pre-embed all query vectors in one batched pass rather than 188 serial
    # calls; a no-op for BM25.
    prewarm = getattr(ranker, "prewarm_queries", None)
    if prewarm is not None:
        prewarm([q.text for q in sample])

    _echo(f"Ranking {len(sample)} claims ...")
    start = time.monotonic()
    results: list[QueryResult] = []
    for q in sample:
        ranked = ranker.rank(q.text)
        first_hit = next((i for i, d in enumerate(ranked) if d in q.relevant_doc_ids), None)
        results.append(
            QueryResult(
                claim_id=q.claim_id,
                relevant=len(q.relevant_doc_ids),
                recall={k: recall_at_k(ranked, q.relevant_doc_ids, k) for k in _K_VALUES},
                ndcg_at_10=ndcg_at_k(ranked, q.relevant_doc_ids, 10),
                rr=reciprocal_rank(ranked, q.relevant_doc_ids),
                first_hit_rank=(first_hit + 1) if first_hit is not None else None,
            )
        )
    elapsed = time.monotonic() - start
    summary = _summarize(results)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = "paraphrased" if paraphrased else "original"
    out_path = (
        RESULTS_DIR / f"retrieval-{dataset}-{method}-{tag}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    )
    out_path.write_text(
        json.dumps(
            {
                "eval": "retrieval",
                "method": method,
                "dataset": dataset,
                "query_kind": tag,
                "seed": seed,
                "elapsed_s": round(elapsed, 1),
                "summary": summary,
                "records": [
                    {
                        "claim_id": r.claim_id,
                        "relevant": r.relevant,
                        "recall_at_10": r.recall[10],
                        "ndcg_at_10": r.ndcg_at_10,
                        "first_hit_rank": r.first_hit_rank,
                    }
                    for r in results
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    _echo()
    _echo(f"  method       : {method}")
    for k in _K_VALUES:
        _echo(f"  recall@{k:<2}    : {summary[f'recall_at_{k}']}")
    _echo(f"  nDCG@10      : {summary['ndcg_at_10']}")
    _echo(f"  MRR          : {summary['mrr']}")
    _echo(f"  ranked {len(sample)} claims in {elapsed:.1f}s")
    _echo(f"Full records: {out_path}")

    if gate_baseline is not None:
        baseline = gate.load_baseline(gate_baseline)
        report = gate.evaluate_gate(_gate_metrics(summary), baseline)
        _echo()
        _echo(report.render())
        return 1 if report.gating_failed else 0
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", default="bm25", choices=_METHODS, help="Retrieval method")
    parser.add_argument("--limit", type=int, default=100, help="Max queries to score (0 = all)")
    parser.add_argument("--seed", type=int, default=13, help="Sampling seed")
    parser.add_argument(
        "--dataset",
        default="scifact",
        help="Dataset: 'scifact' (native) or a BEIR dataset name under evals/data/beir/ "
        "(e.g. 'arguana').",
    )
    parser.add_argument(
        "--embed-model",
        default="nomic-embed-text",
        help="Ollama embedding model for dense/hybrid methods.",
    )
    parser.add_argument(
        "--paraphrased",
        action="store_true",
        help="Query with cached paraphrases (evals/make_paraphrases.py) instead of "
        "original claims — the controlled vocabulary-mismatch A/B.",
    )
    parser.add_argument(
        "--gate",
        nargs="?",
        const=str(BASELINE_PATH),
        default=None,
        metavar="BASELINE",
        help=f"Fail (exit 1) on regression against a baseline. Defaults to {BASELINE_PATH.name}.",
    )
    args = parser.parse_args()
    baseline = Path(args.gate) if args.gate else None
    sys.exit(
        run(
            args.method,
            args.limit,
            args.seed,
            args.dataset,
            baseline,
            args.embed_model,
            paraphrased=args.paraphrased,
        )
    )


if __name__ == "__main__":
    main()
