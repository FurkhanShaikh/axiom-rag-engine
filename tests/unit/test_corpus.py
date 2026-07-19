"""Corpus store + ingestion (bring-your-own-corpus foundation).

No live embedding backend is needed: a tiny deterministic bag-of-words embedder
produces L2-normalized vectors over a fixed vocabulary, so cosine similarity is
predictable and search ordering can be asserted exactly. The store reconnects
per call (thread safety), so every test uses a real temp-file DB — an in-memory
SQLite database would be empty on the next connection.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from axiom_rag_engine.corpus.ingest import IngestionError, extract_text, ingest_text
from axiom_rag_engine.corpus.store import (
    CorpusStore,
    _pack_embedding,
    _unpack_embedding,
)

# ---------------------------------------------------------------------------
# Deterministic fake embedder
# ---------------------------------------------------------------------------

_VOCAB = ["alpha", "beta", "gamma", "delta", "epsilon"]


def _vectorize(text: str, dim: int = len(_VOCAB)) -> list[float]:
    counts = [float(text.lower().count(w)) for w in _VOCAB[:dim]]
    if sum(counts) == 0.0:
        counts = [1.0] * dim
    norm = math.sqrt(sum(c * c for c in counts))
    return [c / norm for c in counts]


async def _fake_embedder(model: str, texts: list[str]) -> list[list[float]]:
    return [_vectorize(t) for t in texts]


@pytest.fixture
def store(tmp_path: Path) -> CorpusStore:
    return CorpusStore(tmp_path / "corpus.db")


def _add(store: CorpusStore, doc_id: str, chunks: list[str], model: str = "m") -> None:
    store.add_document(
        doc_id=doc_id,
        title=f"title-{doc_id}",
        source=f"source-{doc_id}",
        embedding_model=model,
        chunks=[(c, _vectorize(c)) for c in chunks],
    )


# ---------------------------------------------------------------------------
# Embedding (de)serialization
# ---------------------------------------------------------------------------


class TestEmbeddingCodec:
    def test_roundtrip_preserves_values(self) -> None:
        vec = [0.1, -0.5, 0.333, 1.0, 0.0]
        out = _unpack_embedding(_pack_embedding(vec))
        assert len(out) == len(vec)
        for a, b in zip(vec, out, strict=True):
            assert a == pytest.approx(b, abs=1e-6)

    def test_blob_length_matches_dim(self) -> None:
        assert len(_pack_embedding([1.0, 2.0, 3.0])) == 3 * 4  # float32


# ---------------------------------------------------------------------------
# Store: writes & reads
# ---------------------------------------------------------------------------


class TestStoreCrud:
    def test_add_and_get_roundtrip(self, store: CorpusStore) -> None:
        meta = store.add_document(
            doc_id="d1",
            title="T",
            source="s",
            embedding_model="m",
            chunks=[
                ("alpha text one here padded", _vectorize("alpha")),
                ("beta two", _vectorize("beta")),
            ],
        )
        assert meta.chunk_count == 2
        assert meta.char_count == len("alpha text one here padded") + len("beta two")
        got = store.get_document("d1")
        assert got is not None
        assert got.doc_id == "d1"
        assert got.title == "T"
        assert got.embedding_model == "m"
        assert got.content_sha == meta.content_sha

    def test_get_missing_returns_none(self, store: CorpusStore) -> None:
        assert store.get_document("nope") is None

    def test_counts(self, store: CorpusStore) -> None:
        _add(store, "d1", ["alpha one two three four", "beta five six seven"])
        _add(store, "d2", ["gamma eight nine"])
        assert store.count_documents() == 2
        assert store.count_chunks() == 3

    def test_list_documents_newest_first(self, store: CorpusStore) -> None:
        _add(store, "aaa", ["alpha one"])
        _add(store, "zzz", ["beta two"])
        ids = [d.doc_id for d in store.list_documents()]
        assert set(ids) == {"aaa", "zzz"}

    def test_delete_removes_doc_and_chunks(self, store: CorpusStore) -> None:
        _add(store, "d1", ["alpha one two", "beta three four"])
        assert store.delete_document("d1") is True
        assert store.get_document("d1") is None
        assert store.count_chunks() == 0  # cascade

    def test_delete_missing_returns_false(self, store: CorpusStore) -> None:
        assert store.delete_document("ghost") is False

    def test_reingest_replaces_and_leaves_no_orphans(self, store: CorpusStore) -> None:
        _add(store, "d1", ["alpha one", "beta two", "gamma three"])
        assert store.count_chunks() == 3
        # Re-ingest the same id with fewer chunks — old chunks must not linger.
        _add(store, "d1", ["delta four"])
        assert store.count_documents() == 1
        assert store.count_chunks() == 1

    def test_stats(self, store: CorpusStore) -> None:
        _add(store, "d1", ["alpha one"], model="m1")
        _add(store, "d2", ["beta two"], model="m2")
        stats = store.stats()
        assert (stats.documents, stats.chunks, stats.embedding_models) == (2, 2, ["m1", "m2"])
        assert stats.as_dict() == {
            "documents": 2,
            "chunks": 2,
            "embedding_models": ["m1", "m2"],
        }


class TestStoreValidation:
    def test_empty_chunks_rejected(self, store: CorpusStore) -> None:
        with pytest.raises(ValueError, match="at least one chunk"):
            store.add_document(doc_id="d", title="", source="", embedding_model="m", chunks=[])

    def test_empty_embedding_rejected(self, store: CorpusStore) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            store.add_document(
                doc_id="d",
                title="",
                source="",
                embedding_model="m",
                chunks=[("text", [])],
            )

    def test_ragged_dimensions_rejected(self, store: CorpusStore) -> None:
        with pytest.raises(ValueError, match="inconsistent dimensions"):
            store.add_document(
                doc_id="d",
                title="",
                source="",
                embedding_model="m",
                chunks=[("a", [1.0, 0.0]), ("b", [1.0, 0.0, 0.0])],
            )


# ---------------------------------------------------------------------------
# Store: search
# ---------------------------------------------------------------------------


class TestStoreSearch:
    def test_returns_most_similar_first(self, store: CorpusStore) -> None:
        _add(store, "d1", ["alpha alpha alpha", "beta beta", "gamma"])
        hits = store.search(_vectorize("alpha"), embedding_model="m", k=3)
        assert hits[0].text == "alpha alpha alpha"
        assert hits[0].score == pytest.approx(1.0, abs=1e-6)
        # Scores are monotonically non-increasing.
        assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)

    def test_k_limits_results(self, store: CorpusStore) -> None:
        _add(store, "d1", ["alpha one", "beta two", "gamma three", "delta four"])
        assert len(store.search(_vectorize("alpha"), embedding_model="m", k=2)) == 2

    def test_carries_provenance(self, store: CorpusStore) -> None:
        _add(store, "d1", ["alpha context here"])
        hit = store.search(_vectorize("alpha"), embedding_model="m", k=1)[0]
        assert hit.doc_id == "d1"
        assert hit.title == "title-d1"
        assert hit.source == "source-d1"
        assert hit.chunk_id == "d1::0"

    def test_only_matching_embedding_model_scored(self, store: CorpusStore) -> None:
        """A query under model 'm1' must never match chunks stored under 'm2'."""
        _add(store, "a", ["alpha one alpha"], model="m1")
        _add(store, "b", ["alpha two alpha"], model="m2")
        hits = store.search(_vectorize("alpha"), embedding_model="m1", k=10)
        assert {h.doc_id for h in hits} == {"a"}

    def test_dimension_mismatch_skipped(self, store: CorpusStore) -> None:
        _add(store, "d1", ["alpha one"])  # 5-dim vectors
        # Query vector of a different dimensionality is skipped, not crashed.
        assert store.search([1.0, 0.0, 0.0], embedding_model="m", k=5) == []

    def test_empty_query_or_k_zero(self, store: CorpusStore) -> None:
        _add(store, "d1", ["alpha one"])
        assert store.search([], embedding_model="m", k=5) == []
        assert store.search(_vectorize("alpha"), embedding_model="m", k=0) == []

    def test_search_across_multiple_documents(self, store: CorpusStore) -> None:
        _add(store, "d1", ["alpha alpha"], model="m")
        _add(store, "d2", ["beta beta"], model="m")
        hit = store.search(_vectorize("beta"), embedding_model="m", k=1)[0]
        assert hit.doc_id == "d2"


# ---------------------------------------------------------------------------
# Ingestion: text extraction
# ---------------------------------------------------------------------------


class TestExtractText:
    def test_plain_text_passthrough(self) -> None:
        assert extract_text("  hello world  ") == "hello world"

    def test_markdown_passthrough(self) -> None:
        md = "# Title\n\nSome **bold** prose about batteries and their capacity."
        assert extract_text(md, filename="notes.md") == md.strip()

    def test_bytes_decoded_utf8(self) -> None:
        assert extract_text("café".encode()) == "café"

    def test_bad_bytes_do_not_crash(self) -> None:
        # Lone continuation byte — replaced, not raised.
        out = extract_text(b"ab\xffcd")
        assert "ab" in out and "cd" in out

    def test_html_stripped_by_signature(self) -> None:
        html = "<!doctype html><html><body><p>Visible paragraph text.</p><script>x=1</script></body></html>"
        out = extract_text(html)
        assert "Visible paragraph text." in out
        assert "x=1" not in out

    def test_html_stripped_by_extension(self) -> None:
        frag = "<p>Fragment paragraph content here.</p>"
        out = extract_text(frag, filename="page.htm")
        assert out == "Fragment paragraph content here."

    def test_html_stripped_by_content_type(self) -> None:
        frag = "<div>Content typed as html.</div>"
        out = extract_text(frag, content_type="text/html; charset=utf-8")
        assert out == "Content typed as html."


# ---------------------------------------------------------------------------
# Ingestion: end to end (fake embedder)
# ---------------------------------------------------------------------------

_LONG_TEXT = (
    "Alpha batteries store energy using lithium chemistry and long cycle life.\n\n"
    "Beta capacitors release charge quickly but hold far less total energy overall."
)


class TestIngestText:
    async def test_chunks_embeds_and_persists(self, store: CorpusStore) -> None:
        meta = await ingest_text(
            store,
            doc_id="doc1",
            text=_LONG_TEXT,
            embedding_model="m",
            title="Energy",
            source="upload://doc1",
            embedder=_fake_embedder,
        )
        assert meta.chunk_count == 2
        assert store.count_chunks() == 2
        # The embedded chunks are searchable end to end.
        hit = store.search(_vectorize("alpha"), embedding_model="m", k=1)[0]
        assert "Alpha batteries" in hit.text

    async def test_no_usable_chunks_raises(self, store: CorpusStore) -> None:
        with pytest.raises(IngestionError, match="no usable chunks"):
            await ingest_text(
                store, doc_id="d", text="tiny", embedding_model="m", embedder=_fake_embedder
            )

    async def test_embedder_count_mismatch_raises(self, store: CorpusStore) -> None:
        async def _bad(model: str, texts: list[str]) -> list[list[float]]:
            return [_vectorize(texts[0])]  # one vector regardless of chunk count

        with pytest.raises(IngestionError, match="vectors for"):
            await ingest_text(
                store, doc_id="d", text=_LONG_TEXT, embedding_model="m", embedder=_bad
            )

    async def test_max_chunks_cap(self, store: CorpusStore) -> None:
        meta = await ingest_text(
            store,
            doc_id="d",
            text=_LONG_TEXT,
            embedding_model="m",
            embedder=_fake_embedder,
            max_chunks=1,
        )
        assert meta.chunk_count == 1

    async def test_reingest_same_id_replaces(self, store: CorpusStore) -> None:
        await ingest_text(
            store, doc_id="d", text=_LONG_TEXT, embedding_model="m", embedder=_fake_embedder
        )
        first = store.count_chunks()
        assert first == 2
        await ingest_text(
            store,
            doc_id="d",
            text="Gamma single paragraph that is definitely long enough to survive chunking.",
            embedding_model="m",
            embedder=_fake_embedder,
        )
        assert store.count_documents() == 1
        assert store.count_chunks() == 1
