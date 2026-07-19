"""Corpus search backends and their integration with the retriever.

The query embedder is faked (a deterministic bag-of-words vectorizer), so these
run without a live embedding backend. The end-to-end test proves the payoff: a
document ingested into the store is retrieved by ``retriever_node`` through the
exact same path a web result takes.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest

from axiom_rag_engine.corpus.ingest import ingest_text
from axiom_rag_engine.corpus.store import CorpusStore
from axiom_rag_engine.nodes.retriever import (
    MockSearchBackend,
    is_safe_public_url,
    retriever_node,
    set_search_backend,
)
from axiom_rag_engine.search import corpus_backend as cb
from axiom_rag_engine.search.corpus_backend import (
    CompositeSearchBackend,
    CorpusSearchBackend,
    corpus_chunk_url,
)
from axiom_rag_engine.state import make_initial_state

_VOCAB = ["alpha", "beta", "gamma", "delta", "epsilon"]


def _vectorize(text: str) -> list[float]:
    counts = [float(text.lower().count(w)) for w in _VOCAB]
    if sum(counts) == 0.0:
        counts = [1.0] * len(_VOCAB)
    norm = math.sqrt(sum(c * c for c in counts))
    return [c / norm for c in counts]


async def _fake_embed_documents(model: str, texts: list[str]) -> list[list[float]]:
    return [_vectorize(t) for t in texts]


@pytest.fixture
def store(tmp_path: Path) -> CorpusStore:
    return CorpusStore(tmp_path / "corpus.db")


@pytest.fixture(autouse=True)
def _patch_query_embedder(monkeypatch: pytest.MonkeyPatch) -> None:
    """Route the backend's async query embedding to the deterministic fake."""

    async def _fake_embed_query(model: str, query: str) -> list[float]:
        return _vectorize(query)

    monkeypatch.setattr(cb, "embed_query", _fake_embed_query)


@pytest.fixture(autouse=True)
def _reset_backend() -> Any:
    """Never leak a corpus backend into other tests' module-level state."""
    yield
    set_search_backend(MockSearchBackend())


def _seed(store: CorpusStore, doc_id: str, chunks: list[str], model: str = "m") -> None:
    store.add_document(
        doc_id=doc_id,
        title=f"Title {doc_id}",
        source=f"upload://{doc_id}",
        embedding_model=model,
        chunks=[(c, _vectorize(c)) for c in chunks],
    )


# ---------------------------------------------------------------------------
# Citation URL
# ---------------------------------------------------------------------------


class TestCorpusChunkUrl:
    def test_passes_retriever_url_filter(self) -> None:
        ok, _ = is_safe_public_url(corpus_chunk_url("doc1", 0))
        assert ok is True

    def test_distinct_per_chunk(self) -> None:
        assert corpus_chunk_url("d", 0) != corpus_chunk_url("d", 1)

    def test_escapes_odd_doc_ids(self) -> None:
        url = corpus_chunk_url("weird id/with?chars", 3)
        ok, _ = is_safe_public_url(url)
        assert ok is True
        assert " " not in url


# ---------------------------------------------------------------------------
# CorpusSearchBackend
# ---------------------------------------------------------------------------


class TestCorpusSearchBackend:
    def test_returns_retriever_shaped_results(self, store: CorpusStore) -> None:
        _seed(store, "d1", ["alpha alpha alpha", "beta beta"])
        backend = CorpusSearchBackend(store, embedding_model="m", max_results=5)
        results = backend.search("alpha")
        assert results
        top = results[0]
        assert set(top) == {"url", "content", "title", "content_mode"}
        assert top["content"] == "alpha alpha alpha"
        assert top["content_mode"] == "raw"
        assert top["title"] == "Title d1"
        assert top["url"].startswith("https://corpus.local/doc/d1")

    def test_respects_max_results(self, store: CorpusStore) -> None:
        _seed(store, "d1", ["alpha one", "beta two", "gamma three", "delta four"])
        backend = CorpusSearchBackend(store, embedding_model="m", max_results=2)
        assert len(backend.search("alpha")) == 2

    def test_empty_query_returns_nothing(self, store: CorpusStore) -> None:
        _seed(store, "d1", ["alpha one"])
        backend = CorpusSearchBackend(store, embedding_model="m")
        assert backend.search("   ") == []

    def test_model_mismatch_returns_nothing(self, store: CorpusStore) -> None:
        _seed(store, "d1", ["alpha one"], model="model-a")
        backend = CorpusSearchBackend(store, embedding_model="model-b")
        assert backend.search("alpha") == []

    def test_embedding_error_fails_soft(
        self, store: CorpusStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _seed(store, "d1", ["alpha one"])

        async def _boom(model: str, query: str) -> list[float]:
            raise RuntimeError("embedder down")

        monkeypatch.setattr(cb, "embed_query", _boom)
        backend = CorpusSearchBackend(store, embedding_model="m")
        assert backend.search("alpha") == []  # soft failure, not an exception


# ---------------------------------------------------------------------------
# CompositeSearchBackend
# ---------------------------------------------------------------------------


class TestCompositeSearchBackend:
    def test_merges_in_order(self) -> None:
        a = MockSearchBackend([{"url": "https://a.com/1", "content": "x", "title": "A"}])
        b = MockSearchBackend([{"url": "https://b.com/1", "content": "y", "title": "B"}])
        merged = CompositeSearchBackend([a, b]).search("q")
        assert [r["url"] for r in merged] == ["https://a.com/1", "https://b.com/1"]

    def test_one_backend_failing_does_not_sink_others(self) -> None:
        class _Broken:
            def search(self, query: str) -> list[dict[str, Any]]:
                raise RuntimeError("boom")

        good = MockSearchBackend([{"url": "https://ok.com/1", "content": "z", "title": "OK"}])
        merged = CompositeSearchBackend([_Broken(), good]).search("q")
        assert [r["url"] for r in merged] == ["https://ok.com/1"]


# ---------------------------------------------------------------------------
# End to end: ingest → retrieve through the real retriever node
# ---------------------------------------------------------------------------


class TestCorpusRetrievalE2E:
    async def test_ingested_document_is_retrieved(self, store: CorpusStore) -> None:
        # Ingest a document whose distinctive fact we will query for.
        await ingest_text(
            store,
            doc_id="battery-doc",
            text=(
                "Alpha cells use a lithium iron phosphate chemistry that tolerates "
                "thousands of charge cycles with minimal capacity loss.\n\n"
                "Beta cells prioritize energy density over cycle life for portable use."
            ),
            embedding_model="m",
            title="Battery Chemistry",
            source="upload://battery-doc",
            embedder=_fake_embed_documents,
        )

        set_search_backend(CorpusSearchBackend(store, embedding_model="m", max_results=5))

        state = make_initial_state(
            request_id="req-1",
            user_query="alpha",
            app_config={"banned_domains": []},
            models_config={"synthesizer": "mock/s", "verifier": "mock/v"},
            pipeline_config={},
        )
        result = await retriever_node(state)

        chunks = result["indexed_chunks"]
        assert chunks, "corpus retrieval produced no chunks"
        # The alpha fact was ingested and comes back through the normal pipeline.
        assert any("lithium iron phosphate" in c["text"] for c in chunks)
        # Provenance is the synthetic corpus domain, and content is full text.
        alpha_chunk = next(c for c in chunks if "lithium iron phosphate" in c["text"])
        assert alpha_chunk["domain"] == "corpus.local"
        assert alpha_chunk["content_mode"] == "raw"
