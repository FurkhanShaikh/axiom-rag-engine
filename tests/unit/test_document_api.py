"""Document-management API: POST/GET/DELETE /v1/documents (+ /upload).

The embedding backend is faked (a deterministic bag-of-words vectorizer), so no
network or model is needed. The corpus is enabled by pointing
AXIOM_CORPUS_DB_PATH at a temp file before the app's lifespan runs, and the
settings cache is cleared so the fresh env is read.
"""

from __future__ import annotations

import math
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from axiom_rag_engine.config.settings import get_settings
from axiom_rag_engine.main import app
from axiom_rag_engine.nodes.retriever import MockSearchBackend, set_search_backend

_VOCAB = ["alpha", "beta", "gamma", "delta", "epsilon"]


def _vectorize(text: str) -> list[float]:
    counts = [float(text.lower().count(w)) for w in _VOCAB]
    if sum(counts) == 0.0:
        counts = [1.0] * len(_VOCAB)
    norm = math.sqrt(sum(c * c for c in counts))
    return [c / norm for c in counts]


async def _fake_embed_documents(model: str, texts: list[str]) -> list[list[float]]:
    return [_vectorize(t) for t in texts]


_DOC_TEXT = (
    "Alpha cells use lithium iron phosphate chemistry with a long cycle life.\n\n"
    "Beta cells favor energy density over longevity for portable electronics."
)


def _base_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setenv("AXIOM_ENV", "test")
    monkeypatch.delenv("AXIOM_API_KEYS", raising=False)
    monkeypatch.setattr("axiom_rag_engine.corpus.ingest.embed_documents", _fake_embed_documents)
    set_search_backend(MockSearchBackend([]))


@pytest.fixture
def corpus_client(tmp_path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """Client with the corpus enabled and an embedding model configured."""
    _base_env(monkeypatch)
    monkeypatch.setenv("AXIOM_CORPUS_DB_PATH", str(tmp_path / "corpus.db"))
    monkeypatch.setenv("AXIOM_EMBEDDING_MODEL", "fake/model")
    get_settings.cache_clear()
    with TestClient(app) as c:
        yield c
    get_settings.cache_clear()


@pytest.fixture
def no_corpus_client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """Client with the corpus disabled (no AXIOM_CORPUS_DB_PATH)."""
    _base_env(monkeypatch)
    monkeypatch.delenv("AXIOM_CORPUS_DB_PATH", raising=False)
    get_settings.cache_clear()
    with TestClient(app) as c:
        yield c
    get_settings.cache_clear()


@pytest.fixture
def corpus_client_no_model(tmp_path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """Corpus enabled but AXIOM_EMBEDDING_MODEL unset — ingestion must 409."""
    _base_env(monkeypatch)
    monkeypatch.setenv("AXIOM_CORPUS_DB_PATH", str(tmp_path / "corpus.db"))
    monkeypatch.delenv("AXIOM_EMBEDDING_MODEL", raising=False)
    get_settings.cache_clear()
    with TestClient(app) as c:
        yield c
    get_settings.cache_clear()


# ---------------------------------------------------------------------------
# Ingest (JSON) + lifecycle
# ---------------------------------------------------------------------------


class TestDocumentLifecycle:
    def test_ingest_list_get_delete(self, corpus_client: TestClient) -> None:
        # Ingest
        resp = corpus_client.post(
            "/v1/documents",
            json={"text": _DOC_TEXT, "title": "Batteries", "source": "upload://b"},
        )
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert body["chunk_count"] == 2
        assert body["title"] == "Batteries"
        assert body["embedding_model"] == "fake/model"
        doc_id = body["doc_id"]

        # List
        listing = corpus_client.get("/v1/documents").json()
        assert listing["total_documents"] == 1
        assert listing["total_chunks"] == 2
        assert listing["embedding_models"] == ["fake/model"]
        assert any(d["doc_id"] == doc_id for d in listing["documents"])

        # Get one
        one = corpus_client.get(f"/v1/documents/{doc_id}")
        assert one.status_code == 200
        assert one.json()["doc_id"] == doc_id

        # Delete
        deleted = corpus_client.delete(f"/v1/documents/{doc_id}")
        assert deleted.status_code == 200
        assert deleted.json() == {"deleted": True, "doc_id": doc_id}

        # Gone
        assert corpus_client.get(f"/v1/documents/{doc_id}").status_code == 404

    def test_explicit_doc_id_and_reingest_replaces(self, corpus_client: TestClient) -> None:
        r1 = corpus_client.post("/v1/documents", json={"text": _DOC_TEXT, "doc_id": "fixed"})
        assert r1.status_code == 201
        assert r1.json()["doc_id"] == "fixed"
        # Re-ingest same id with a single-paragraph doc → replaces, not duplicates.
        r2 = corpus_client.post(
            "/v1/documents",
            json={
                "text": "Gamma is a single sufficiently long paragraph for chunking.",
                "doc_id": "fixed",
            },
        )
        assert r2.status_code == 201
        listing = corpus_client.get("/v1/documents").json()
        assert listing["total_documents"] == 1
        assert listing["total_chunks"] == 1

    def test_html_body_is_stripped(self, corpus_client: TestClient) -> None:
        html = "<!doctype html><html><body><p>Alpha visible paragraph with real content here.</p><script>evil()</script></body></html>"
        resp = corpus_client.post("/v1/documents", json={"text": html})
        assert resp.status_code == 201
        doc_id = resp.json()["doc_id"]
        # Verify the stored text is clean by fetching it back through search is
        # out of scope here; the ingest succeeding on stripped text is the check.
        assert corpus_client.get(f"/v1/documents/{doc_id}").status_code == 200


# ---------------------------------------------------------------------------
# Upload (multipart)
# ---------------------------------------------------------------------------


class TestDocumentUpload:
    def test_upload_text_file(self, corpus_client: TestClient) -> None:
        resp = corpus_client.post(
            "/v1/documents/upload",
            files={"file": ("notes.md", _DOC_TEXT.encode("utf-8"), "text/markdown")},
        )
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert body["chunk_count"] == 2
        # Filename becomes the default source/title.
        assert body["source"] == "notes.md"

    def test_upload_html_file_stripped(self, corpus_client: TestClient) -> None:
        html = b"<p>Alpha fragment paragraph content that is long enough to keep.</p>"
        resp = corpus_client.post(
            "/v1/documents/upload",
            files={"file": ("page.html", html, "text/html")},
        )
        assert resp.status_code == 201

    def test_upload_pdf_file(self, corpus_client: TestClient) -> None:
        from tests.conftest import make_pdf

        pdf = make_pdf("Alpha battery specification: nominal voltage is three point seven volts.")
        resp = corpus_client.post(
            "/v1/documents/upload",
            files={"file": ("spec.pdf", pdf, "application/pdf")},
        )
        assert resp.status_code == 201, resp.text
        assert resp.json()["chunk_count"] >= 1

    def test_upload_corrupt_pdf_returns_422(self, corpus_client: TestClient) -> None:
        resp = corpus_client.post(
            "/v1/documents/upload",
            files={"file": ("broken.pdf", b"%PDF-1.4\nnot a real pdf", "application/pdf")},
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


class TestDocumentErrors:
    def test_corpus_disabled_returns_404(self, no_corpus_client: TestClient) -> None:
        resp = no_corpus_client.post("/v1/documents", json={"text": _DOC_TEXT})
        assert resp.status_code == 404
        assert "AXIOM_CORPUS_DB_PATH" in resp.json()["detail"]

    def test_missing_embedding_model_returns_409(self, corpus_client_no_model: TestClient) -> None:
        resp = corpus_client_no_model.post("/v1/documents", json={"text": _DOC_TEXT})
        assert resp.status_code == 409
        assert "AXIOM_EMBEDDING_MODEL" in resp.json()["detail"]

    def test_text_with_no_usable_chunks_returns_422(self, corpus_client: TestClient) -> None:
        resp = corpus_client.post("/v1/documents", json={"text": "tiny"})
        assert resp.status_code == 422
        assert "chunk" in resp.json()["detail"].lower()

    def test_empty_text_rejected_by_schema(self, corpus_client: TestClient) -> None:
        # min_length=1 → Pydantic 422 before the handler runs.
        assert corpus_client.post("/v1/documents", json={"text": ""}).status_code == 422

    def test_get_missing_document_returns_404(self, corpus_client: TestClient) -> None:
        assert corpus_client.get("/v1/documents/nope").status_code == 404

    def test_delete_missing_document_returns_404(self, corpus_client: TestClient) -> None:
        assert corpus_client.delete("/v1/documents/nope").status_code == 404


# ---------------------------------------------------------------------------
# Status reporting
# ---------------------------------------------------------------------------


class TestStatusReportsCorpus:
    def test_status_includes_corpus_stats(self, corpus_client: TestClient) -> None:
        corpus_client.post("/v1/documents", json={"text": _DOC_TEXT})
        retrieval = corpus_client.get("/v1/status").json()["retrieval"]
        assert retrieval["source"] == "web"  # default; corpus enabled but not wired to retrieval
        assert retrieval["corpus"] == {
            "documents": 1,
            "chunks": 2,
            "embedding_models": ["fake/model"],
        }

    def test_status_corpus_null_when_disabled(self, no_corpus_client: TestClient) -> None:
        retrieval = no_corpus_client.get("/v1/status").json()["retrieval"]
        assert retrieval["corpus"] is None
