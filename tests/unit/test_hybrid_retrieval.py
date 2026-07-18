"""Production hybrid retrieval in the ranker node.

Hybrid retrieval is opt-in (AXIOM_EMBEDDING_MODEL). These tests mock the
embedding call so they need no Ollama, and check the three behaviors that
matter: off by default, applied when configured, and graceful fallback when the
embedder fails. The quality question (does hybrid rank better) is answered by the
retrieval eval on real data, not here.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from axiom_rag_engine.config.settings import get_settings
from axiom_rag_engine.nodes.ranker import ranker_node
from axiom_rag_engine.state import make_initial_state


def _chunk(chunk_id: str, text: str, quality: float = 0.5) -> dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "text": text,
        "source_url": f"https://example.com/{chunk_id}",
        "domain": "example.com",
        "title": "T",
        "doc_index": int(chunk_id.split("_")[1]),
        "chunk_index": 0,
        "quality_score": quality,
    }


def _state(chunks: list[dict], query: str = "ocean tides") -> dict[str, Any]:
    s = make_initial_state(
        request_id="req",
        user_query=query,
        app_config={},
        models_config={},
        pipeline_config={},
    )
    s["scored_chunks"] = chunks
    return s


def _embedding_response(vectors: list[list[float]]) -> dict[str, Any]:
    """LiteLLM-shaped embedding response."""
    return {"data": [{"embedding": v} for v in vectors]}


def _enable_hybrid(monkeypatch, model: str = "ollama/nomic-embed-text") -> None:
    monkeypatch.setenv("AXIOM_EMBEDDING_MODEL", model)
    get_settings.cache_clear()


class TestOptIn:
    async def test_bm25_only_by_default(self) -> None:
        """With no embedding model configured, the ranker never embeds."""
        chunks = [
            _chunk("doc_1_chunk_A", "the moon causes ocean tides"),
            _chunk("doc_2_chunk_A", "gardening tips for tomatoes"),
        ]
        with patch("litellm.aembedding", new_callable=AsyncMock) as mock_embed:
            result = await ranker_node(_state(chunks))
        mock_embed.assert_not_called()
        ranked = result["ranked_chunks"]
        assert all("dense_score" not in c for c in ranked)
        complete = next(e for e in result["audit_trail"] if e["event_type"] == "ranker_complete")
        assert complete["payload"]["ranking_mode"] == "bm25"


class TestHybridApplied:
    async def test_fusion_attaches_scores_and_reorders(self, monkeypatch) -> None:
        _enable_hybrid(monkeypatch)
        # Two chunks. BM25 favors chunk A (it contains the query terms); the
        # embedder is rigged so chunk B is the query's semantic match. RRF should
        # pull B up from BM25-last.
        chunks = [
            _chunk("doc_1_chunk_A", "ocean tides ocean tides ocean tides"),
            _chunk("doc_2_chunk_B", "lunar gravitational pull on seawater"),
        ]
        # inputs order: [query, chunkA_text, chunkB_text]; make B align with query.
        vectors = [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
        with patch("litellm.aembedding", new_callable=AsyncMock) as mock_embed:
            mock_embed.return_value = _embedding_response(vectors)
            result = await ranker_node(_state(chunks))

        ranked = result["ranked_chunks"]
        assert all("dense_score" in c and "fused_score" in c for c in ranked)
        # B is dense-aligned with the query (cosine 1.0), A is orthogonal (0.0).
        by_id = {c["chunk_id"]: c for c in ranked}
        assert by_id["doc_2_chunk_B"]["dense_score"] == pytest.approx(1.0)
        assert by_id["doc_1_chunk_A"]["dense_score"] == pytest.approx(0.0)
        complete = next(e for e in result["audit_trail"] if e["event_type"] == "ranker_complete")
        assert complete["payload"]["ranking_mode"] == "hybrid"

    async def test_ranking_score_untouched_by_fusion(self, monkeypatch) -> None:
        """The pre-LLM answerability gate reads ranking_score; fusion must not
        change it (it only reorders)."""
        _enable_hybrid(monkeypatch)
        chunks = [
            _chunk("doc_1_chunk_A", "the moon causes ocean tides on earth"),
            _chunk("doc_2_chunk_B", "the sun also contributes to tidal forces"),
        ]
        # Compute BM25-only ranking_scores first (hybrid disabled).
        get_settings.cache_clear()
        monkeypatch.delenv("AXIOM_EMBEDDING_MODEL", raising=False)
        get_settings.cache_clear()
        bm25_result = await ranker_node(_state([dict(c) for c in chunks]))
        bm25_scores = {c["chunk_id"]: c["ranking_score"] for c in bm25_result["ranked_chunks"]}

        _enable_hybrid(monkeypatch)
        with patch("litellm.aembedding", new_callable=AsyncMock) as mock_embed:
            mock_embed.return_value = _embedding_response([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
            hybrid_result = await ranker_node(_state([dict(c) for c in chunks]))
        hybrid_scores = {c["chunk_id"]: c["ranking_score"] for c in hybrid_result["ranked_chunks"]}
        assert bm25_scores == hybrid_scores

    async def test_single_chunk_skips_embedding(self, monkeypatch) -> None:
        """Fusing one chunk is a no-op — don't spend an embed call on it."""
        _enable_hybrid(monkeypatch)
        with patch("litellm.aembedding", new_callable=AsyncMock) as mock_embed:
            result = await ranker_node(_state([_chunk("doc_1_chunk_A", "only one chunk here")]))
        mock_embed.assert_not_called()
        assert len(result["ranked_chunks"]) == 1


class TestGracefulFallback:
    async def test_embedding_failure_falls_back_to_bm25(self, monkeypatch) -> None:
        """A dead/slow embedder must never break ranking — fall back to BM25."""
        _enable_hybrid(monkeypatch)
        chunks = [
            _chunk("doc_1_chunk_A", "ocean tides caused by the moon"),
            _chunk("doc_2_chunk_B", "unrelated content about cooking"),
        ]
        with patch("litellm.aembedding", new_callable=AsyncMock) as mock_embed:
            mock_embed.side_effect = RuntimeError("connection refused")
            result = await ranker_node(_state(chunks))

        ranked = result["ranked_chunks"]
        assert len(ranked) == 2  # ranking still produced
        assert all("dense_score" not in c for c in ranked)
        events = {e["event_type"] for e in result["audit_trail"]}
        assert "ranker_dense_error" in events
        complete = next(e for e in result["audit_trail"] if e["event_type"] == "ranker_complete")
        assert complete["payload"]["ranking_mode"] == "bm25"
