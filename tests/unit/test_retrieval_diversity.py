"""
Retrieval keeps its best evidence and its source diversity.

  * The per-request chunk cap is shared round-robin across documents. It used to
    be first-come: one or two long pages (Tavily raw content runs to 200k chars,
    ~130 chunks) could consume the whole cap and push every other domain out —
    which also made multi-domain (Tier 2) answers unreachable.
  * A retrieval retry keeps the previous round's top-ranked chunks. It used to
    discard them (URLs already seen were skipped), leaving the retry pass with
    only lower-ranked search results.
  * A corpus document's ``source`` label reaches the citation (it was dropped).
"""

from __future__ import annotations

from typing import Any

from axiom_rag_engine.graph import retriever_with_retry
from axiom_rag_engine.models import CitationSource
from axiom_rag_engine.nodes.retriever import MockSearchBackend, retriever_node, set_search_backend
from axiom_rag_engine.nodes.semantic import _resolve_citation_source
from axiom_rag_engine.state import make_initial_state


def _state(app_config: dict[str, Any] | None = None) -> dict[str, Any]:
    return dict(
        make_initial_state(
            request_id="req",
            user_query="solid-state batteries",
            app_config=app_config or {},
            models_config={},
            pipeline_config={},
        )
    )


def _paragraphs(tag: str, n: int) -> str:
    return "\n\n".join(
        f"{tag} paragraph {i} explains a distinct aspect of solid-state battery chemistry."
        for i in range(n)
    )


class TestChunkCapIsShared:
    async def test_long_page_cannot_crowd_out_other_documents(self) -> None:
        set_search_backend(
            MockSearchBackend(
                [
                    {
                        "url": "https://long.example.com/a",
                        "title": "Long",
                        "content": _paragraphs("Long", 40),
                    },
                    {
                        "url": "https://short.example.org/b",
                        "title": "Short",
                        "content": _paragraphs("Short", 2),
                    },
                    {
                        "url": "https://third.example.net/c",
                        "title": "Third",
                        "content": _paragraphs("Third", 3),
                    },
                ]
            )
        )
        result = await retriever_node(_state({"max_chunks_per_request": 10}))
        chunks = result["indexed_chunks"]
        domains = {c["domain"] for c in chunks}
        assert len(chunks) == 10
        assert domains == {"long.example.com", "short.example.org", "third.example.net"}
        per_domain = {d: sum(c["domain"] == d for c in chunks) for d in domains}
        assert per_domain["short.example.org"] == 2
        assert per_domain["third.example.net"] == 3

    async def test_cap_keeps_each_documents_earliest_chunks(self) -> None:
        set_search_backend(
            MockSearchBackend(
                [
                    {
                        "url": "https://long.example.com/a",
                        "title": "Long",
                        "content": _paragraphs("Long", 12),
                    }
                ]
            )
        )
        result = await retriever_node(_state({"max_chunks_per_request": 4}))
        ids = [c["chunk_id"] for c in result["indexed_chunks"]]
        assert ids == ["doc_1_chunk_A", "doc_1_chunk_B", "doc_1_chunk_C", "doc_1_chunk_D"]

    async def test_cap_reached_is_audited(self) -> None:
        set_search_backend(
            MockSearchBackend(
                [
                    {
                        "url": "https://long.example.com/a",
                        "title": "Long",
                        "content": _paragraphs("Long", 12),
                    }
                ]
            )
        )
        result = await retriever_node(_state({"max_chunks_per_request": 4}))
        assert any(e["event_type"] == "retriever_chunk_cap_reached" for e in result["audit_trail"])


class TestRetryKeepsBestSources:
    async def test_retry_retains_previous_top_ranked_chunks(self) -> None:
        previous_best = {
            "chunk_id": "doc_1_chunk_A",
            "text": "Solid-state batteries replace liquid electrolytes with solid ceramics.",
            "source_url": "https://best.example.com/a",
            "domain": "best.example.com",
            "title": "Best",
            "ranking_score": 0.91,
            "relevance_score": 0.8,
        }
        state = _state()
        state["ranked_chunks"] = [previous_best]
        state["past_seen_urls"] = ["https://best.example.com/a"]
        state["next_doc_index"] = 2
        state["rewrite_requests"] = ["Sentence s_01 ... Tier 5"]
        set_search_backend(
            MockSearchBackend(
                [
                    {
                        "url": "https://fresh.example.org/b",
                        "title": "Fresh",
                        "content": _paragraphs("Fresh", 2),
                    }
                ]
            )
        )
        result = await retriever_with_retry(state)
        ids = [c["chunk_id"] for c in result["indexed_chunks"]]
        assert "doc_1_chunk_A" in ids  # kept
        assert any(i.startswith("doc_2_") for i in ids)  # plus fresh sources
        kept = next(c for c in result["indexed_chunks"] if c["chunk_id"] == "doc_1_chunk_A")
        # Stale ranking fields are dropped so the scorer/ranker re-derive them.
        assert "ranking_score" not in kept and "relevance_score" not in kept
        assert any(e["event_type"] == "retriever_retained_chunks" for e in result["audit_trail"])


class TestCorpusSourceLabel:
    async def test_source_label_flows_from_search_result_to_citation(self) -> None:
        set_search_backend(
            MockSearchBackend(
                [
                    {
                        "url": "https://corpus.local/doc/handbook#chunk-0",
                        "title": "Employee Handbook",
                        "source": "s3://hr-docs/handbook-2026.pdf",
                        "content": _paragraphs("Handbook", 1),
                        "content_mode": "raw",
                    }
                ]
            )
        )
        result = await retriever_node(_state())
        chunk = result["indexed_chunks"][0]
        assert chunk["source_label"] == "s3://hr-docs/handbook-2026.pdf"

        source = _resolve_citation_source(chunk["chunk_id"], {chunk["chunk_id"]: chunk})
        assert source == CitationSource(
            url="https://corpus.local/doc/handbook#chunk-0",
            title="Employee Handbook",
            domain="corpus.local",
            source_label="s3://hr-docs/handbook-2026.pdf",
        )

    def test_corpus_backend_emits_the_document_source(self, tmp_path) -> None:
        from axiom_rag_engine.corpus.store import CorpusStore
        from axiom_rag_engine.search import corpus_backend as cb

        store = CorpusStore(tmp_path / "c.db")
        store.add_document(
            doc_id="d1",
            title="T",
            source="upload://handbook.pdf",
            embedding_model="m",
            chunks=[("alpha alpha alpha", [1.0, 0.0])],
        )
        original = cb.embed_query_sync
        cb.embed_query_sync = lambda model, query: [1.0, 0.0]  # type: ignore[assignment]
        try:
            hits = cb.CorpusSearchBackend(store, embedding_model="m").search("alpha")
        finally:
            cb.embed_query_sync = original
        assert hits[0]["source"] == "upload://handbook.pdf"
