"""Query-expansion eval — do the retriever's reformulated searches earn their cost?

The retriever used to send three web searches per question: the original query
plus "What is <q>" and "Explain <q>". This eval measures whether extra searches
improve the context the synthesizer actually sees. Configurations:

  original   — the original query only (1 search)
  legacy     — original + "What is" + "Explain" (3 searches; production until
               this eval showed no relevance gain — see BENCHMARKS.md)
  rewrite    — original + one LLM keyword rewrite (2 searches)

Each configuration runs the production retriever → scorer → ranker over real
Tavily results. There are no relevance labels for live web results, so the top
chunks are graded 0-3 by an LLM judge using the reranker's grading prompt
(pooled: every chunk that reaches any configuration's top-k is graded once).

Metrics per configuration (mean over questions):
  grade@k      mean judge grade of the top-k chunks
  ndcg@k       nDCG@k against the pooled ideal ordering
  prec@k       share of top-k chunks graded >= 2 ("substantive")
  domains@10   distinct domains among the top-10 ranked chunks (Tier 2 needs >= 2)
  searches     web searches spent per question

Tavily responses are cached in evals/data/query_expansion_cache.json, so reruns
(different judge, k, or configs) cost no search credits.

Usage:
    uv run python evals/query_expansion_eval.py --judge ollama/qwen3.5:9b
    uv run python evals/query_expansion_eval.py --judge ollama/qwen3.5:9b --limit 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

EVALS_DIR = Path(__file__).resolve().parent
SEED_PATH = EVALS_DIR / "golden" / "seed.jsonl"
CACHE_PATH = EVALS_DIR / "data" / "query_expansion_cache.json"
RESULTS_DIR = EVALS_DIR / "results"

_REWRITE_PROMPT = (
    "Rewrite the question as a short web-search query of 3-8 keywords that would "
    "find authoritative pages answering it. Keep the question's language. Reply "
    'with JSON only: {"query": "<keywords>"}'
)


def _echo(message: str = "") -> None:
    # Questions include Arabic and CJK; don't depend on the console's codepage.
    sys.stdout.buffer.write(f"{message}\n".encode())
    sys.stdout.flush()


def load_questions(limit: int | None) -> list[str]:
    questions: list[str] = []
    with SEED_PATH.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                case = json.loads(line)
                if case["expect"].get("answerable"):
                    questions.append(case["query"])
    return questions[:limit] if limit else questions


class _Cache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.data: dict[str, Any] = (
            json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        )

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, ensure_ascii=False), encoding="utf-8")


async def llm_rewrite(question: str, model: str, cache: _Cache) -> str:
    key = f"rewrite::{model}::{question}"
    if key not in cache.data:
        from axiom_rag_engine.utils.llm import call_llm, parse_json_object, reset_llm_budget

        reset_llm_budget(max_calls=5)
        raw = await call_llm(
            "query_rewrite",
            model,
            [
                {"role": "system", "content": _REWRITE_PROMPT},
                {"role": "user", "content": question},
            ],
        )
        try:
            cache.data[key] = str(parse_json_object(raw).get("query") or question).strip()
        except ValueError:
            cache.data[key] = question
        cache.save()
    return str(cache.data[key])


def search(query: str, backend: Any, cache: _Cache) -> list[dict[str, Any]]:
    key = f"search::{query}"
    if key not in cache.data:
        cache.data[key] = backend.search(query)
        cache.save()
    return list(cache.data[key])


async def ranked_context(question: str, queries: list[str], results: dict[str, list]) -> list[dict]:
    """Run the production retriever → scorer → ranker over cached results."""
    from axiom_rag_engine.nodes import retriever as retriever_mod
    from axiom_rag_engine.nodes.ranker import ranker_node
    from axiom_rag_engine.nodes.scorer import scorer_node
    from axiom_rag_engine.state import make_initial_state

    class _Backend:
        def search(self, query: str) -> list[dict[str, Any]]:
            return results[query]

    original = retriever_mod.generate_search_queries
    retriever_mod.generate_search_queries = lambda q, rewrite_requests=None: list(queries)  # type: ignore[assignment]
    try:
        state: dict[str, Any] = dict(
            make_initial_state(
                request_id="qx",
                user_query=question,
                app_config={},
                models_config={},
                pipeline_config={"stages": {"max_ranked_chunks": 10}},
            )
        )
        state.update(
            await retriever_mod.retriever_node(
                state, {"configurable": {"search_backend": _Backend()}}
            )
        )
        state.update(await scorer_node(state))  # type: ignore[arg-type]
        state.update(await ranker_node(state))  # type: ignore[arg-type]
    finally:
        retriever_mod.generate_search_queries = original  # type: ignore[assignment]
    return list(state.get("ranked_chunks") or [])


async def grade(question: str, text: str, judge: str, cache: _Cache) -> int:
    key = f"grade::{judge}::{question}::{text[:500]}"
    if key not in cache.data:
        from axiom_rag_engine.nodes.ranker import _grade_chunk
        from axiom_rag_engine.utils.llm import reset_llm_budget

        reset_llm_budget(max_calls=5)
        try:
            cache.data[key] = await _grade_chunk(question, text, judge)
        except Exception as exc:  # judge failure: count as irrelevant, but say so
            _echo(f"    judge error ({type(exc).__name__}); grading 0")
            cache.data[key] = 0
        cache.save()
    return int(cache.data[key])


def ndcg(grades: list[int], ideal: list[int], k: int) -> float:
    def dcg(values: list[int]) -> float:
        return sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(values[:k]))

    best = dcg(sorted(ideal, reverse=True))
    return dcg(grades) / best if best > 0 else 0.0


async def run(judge: str, rewrite_model: str, k: int, limit: int | None) -> int:
    from axiom_rag_engine.config.settings import get_settings
    from axiom_rag_engine.search.tavily import TavilySearchBackend

    settings = get_settings()
    if not settings.tavily_api_key:
        _echo("TAVILY_API_KEY is required (results are cached after the first run).")
        return 2
    backend = TavilySearchBackend(
        api_key=settings.tavily_api_key,
        fetch_full_pages=settings.fetch_full_pages,
        max_raw_content_chars=settings.max_raw_content_chars,
    )
    cache = _Cache(CACHE_PATH)
    questions = load_questions(limit)
    configs = ("original", "legacy", "rewrite")
    per_config: dict[str, list[dict[str, float]]] = {c: [] for c in configs}
    started = time.monotonic()

    for n, question in enumerate(questions, 1):
        rewrite = await llm_rewrite(question, rewrite_model, cache)
        variants = {
            "original": [question],
            "legacy": [question, f"What is {question}", f"Explain {question}"],
            "rewrite": [question, rewrite] if rewrite != question else [question],
        }
        # "What is" is skipped by production for questions already starting with it.
        lower = question.lower().strip()
        if lower.startswith(("what is", "what are")):
            variants["legacy"] = [question, f"Explain {question}"]

        results = {q: search(q, backend, cache) for qs in variants.values() for q in qs}
        contexts = {c: await ranked_context(question, variants[c], results) for c in configs}

        pooled: dict[str, int] = {}
        for chunks in contexts.values():
            for chunk in chunks[:k]:
                if chunk["text"] not in pooled:
                    pooled[chunk["text"]] = await grade(question, chunk["text"], judge, cache)
        ideal = list(pooled.values())

        _echo(f"[{n}/{len(questions)}] {question[:60]}  (rewrite: {rewrite[:40]!r})")
        for c in configs:
            top = contexts[c][:k]
            grades = [pooled[ch["text"]] for ch in top]
            row = {
                "grade": sum(grades) / len(grades) if grades else 0.0,
                "ndcg": ndcg(grades, ideal, k),
                "prec": sum(g >= 2 for g in grades) / k,
                "domains": float(len({ch["domain"] for ch in contexts[c][:10]})),
                "searches": float(len(variants[c])),
            }
            per_config[c].append(row)
            _echo(
                f"    {c:9} grade={row['grade']:.2f} ndcg={row['ndcg']:.3f} "
                f"prec={row['prec']:.2f} domains={int(row['domains'])} searches={int(row['searches'])}"
            )

    summary = {
        c: {m: round(sum(r[m] for r in rows) / len(rows), 4) for m in rows[0]}
        for c, rows in per_config.items()
        if rows
    }
    _echo(f"\nSummary over {len(questions)} questions (judge={judge}, k={k}):")
    _echo(
        f"  {'config':9} {'grade@k':>8} {'ndcg@k':>8} {'prec@k':>8} {'domains':>8} {'searches':>9}"
    )
    for c, m in summary.items():
        _echo(
            f"  {c:9} {m['grade']:8.3f} {m['ndcg']:8.3f} {m['prec']:8.3f} "
            f"{m['domains']:8.2f} {m['searches']:9.2f}"
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"query-expansion-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(
        json.dumps(
            {
                "judge": judge,
                "rewrite_model": rewrite_model,
                "k": k,
                "questions": questions,
                "summary": summary,
                "per_question": per_config,
                "elapsed_s": round(time.monotonic() - started, 1),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    _echo(f"Full records: {out}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--judge", default="ollama/qwen3.5:9b", help="LiteLLM model grading 0-3")
    parser.add_argument(
        "--rewrite-model", default="ollama/qwen3.5:9b", help="LiteLLM model for the rewrite arm"
    )
    parser.add_argument("--k", type=int, default=5, help="Top-k chunks graded per configuration")
    parser.add_argument("--limit", type=int, default=None, help="Only the first N questions")
    args = parser.parse_args()
    sys.exit(asyncio.run(run(args.judge, args.rewrite_model, args.k, args.limit)))


if __name__ == "__main__":
    main()
