"""
Corpus changes invalidate cached answers.

The response cache key had no notion of the corpus's contents, and ingest or
delete never cleared the cache, so after ``DELETE /v1/documents/{id}`` identical
queries kept returning answers quoting the deleted document until the entry
expired. The store now keeps a version bumped by every ingest and delete, and
the cache key includes it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from axiom_rag_engine.config.settings import Settings
from axiom_rag_engine.corpus.store import CorpusStore
from axiom_rag_engine.main import create_app
from axiom_rag_engine.nodes.retriever import MockSearchBackend

_TEXT = "Alpha batteries use lithium iron phosphate chemistry for a long cycle life."
_SYNTH = json.dumps(
    {
        "is_answerable": True,
        "sentences": [
            {
                "sentence_id": "s_01",
                "text": "Alpha batteries use LFP chemistry.",
                "is_cited": True,
                "citations": [
                    {
                        "citation_id": "cite_1",
                        "chunk_id": "doc_1_chunk_A",
                        "exact_source_quote": "Alpha batteries use lithium iron phosphate chemistry",
                    }
                ],
            }
        ],
    }
)


def _add(store: CorpusStore, doc_id: str) -> None:
    store.add_document(
        doc_id=doc_id,
        title="T",
        source="s",
        embedding_model="m",
        chunks=[("some chunk text", [1.0, 0.0])],
    )


class TestCorpusVersion:
    def test_ingest_and_delete_bump_the_version(self, tmp_path: Path) -> None:
        store = CorpusStore(tmp_path / "corpus.db")
        v0 = store.version()
        _add(store, "a")
        v1 = store.version()
        _add(store, "a")  # re-ingest replaces the document: still a change
        v2 = store.version()
        assert store.delete_document("a")
        v3 = store.version()
        assert v0 < v1 < v2 < v3

    def test_deleting_a_missing_document_is_not_a_change(self, tmp_path: Path) -> None:
        store = CorpusStore(tmp_path / "corpus.db")
        before = store.version()
        assert not store.delete_document("missing")
        assert store.version() == before

    def test_version_survives_reopening(self, tmp_path: Path) -> None:
        db = tmp_path / "corpus.db"
        _add(CorpusStore(db), "a")
        assert CorpusStore(db).version() == 1


def _reply(content: str) -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content=content))]
    return response


async def _llm(**kwargs: Any) -> MagicMock:
    return _reply(_SYNTH)


def test_deleting_a_document_invalidates_cached_answers(tmp_path: Path) -> None:
    app = create_app(
        Settings(
            env="test",
            semantic_verification_enabled=False,
            corpus_db_path=str(tmp_path / "corpus.db"),
        ),
        search_backend=MockSearchBackend(
            [{"url": "https://example.com/a", "title": "A", "content": _TEXT}]
        ),
    )
    llm = AsyncMock(side_effect=_llm)
    body = {
        "request_id": "r1",
        "user_query": "alpha batteries chemistry",
        "models": {"synthesizer": "mock/s", "verifier": "mock/v"},
    }
    with TestClient(app) as client, patch("litellm.acompletion", llm):
        store = client.app.state.services.corpus_store  # type: ignore[attr-defined]
        _add(store, "doc-to-delete")

        assert client.post("/v1/synthesize", json=body).status_code == 200
        assert client.post("/v1/synthesize", json=body).status_code == 200
        assert llm.await_count == 1  # second request was a cache hit

        assert client.delete("/v1/documents/doc-to-delete").status_code == 200
        assert client.post("/v1/synthesize", json=body).status_code == 200
        assert llm.await_count == 2  # the delete invalidated the cached answer
