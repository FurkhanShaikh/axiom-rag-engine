"""Query-expansion eval harness: metric math and query routing, no network."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evals"))

import query_expansion_eval as qx


class TestNdcg:
    def test_ideal_order_scores_one(self) -> None:
        assert qx.ndcg([3, 2, 1], [1, 2, 3], k=3) == pytest.approx(1.0)

    def test_reversed_order_scores_below_one(self) -> None:
        assert qx.ndcg([1, 2, 3], [1, 2, 3], k=3) < 1.0

    def test_no_relevant_chunks_scores_zero(self) -> None:
        assert qx.ndcg([0, 0], [0, 0], k=2) == 0.0


class TestRankedContext:
    async def test_each_configuration_searches_exactly_its_queries(self) -> None:
        text = "Water boils at 100 degrees Celsius at sea level under standard pressure."
        results = {
            "q": [{"url": "https://a.example.com", "title": "A", "content": text}],
            "What is q": [{"url": "https://b.example.org", "title": "B", "content": text + " B"}],
        }
        only_original = await qx.ranked_context("water boiling", ["q"], results)
        both = await qx.ranked_context("water boiling", ["q", "What is q"], results)
        assert {c["domain"] for c in only_original} == {"a.example.com"}
        assert {c["domain"] for c in both} == {"a.example.com", "b.example.org"}

    async def test_production_query_generator_is_restored(self) -> None:
        from axiom_rag_engine.nodes import retriever

        original = retriever.generate_search_queries
        await qx.ranked_context("x", ["q"], {"q": []})
        assert retriever.generate_search_queries is original
