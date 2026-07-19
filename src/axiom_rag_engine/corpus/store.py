"""SQLite-backed corpus store for ingested documents (bring-your-own corpus).

Single-node and dependency-light: ``sqlite3`` + ``struct`` from the stdlib, no
vector-DB server and no numpy. Chunk embeddings are stored as packed float32
blobs and searched by brute-force cosine in Python — the same pure-Python
similarity the ranker already uses, and fast enough at single-node corpus scale
(thousands of chunks). If a corpus outgrows brute force, the eval harness will
show it before users feel it.

Embedding-space safety
----------------------
Cosine similarity is only meaningful between vectors produced by the *same*
embedding model. Each document records the model it was embedded with, and
:meth:`CorpusStore.search` scores only chunks whose model matches the query's.
Documents embedded under a different model are inert until re-ingested — never
silently mis-scored against an incompatible query vector.

Concurrency
-----------
The retriever runs search off the event loop (``asyncio.to_thread``), so store
methods must be safe to call from worker threads. Each call opens its own
short-lived SQLite connection (a local file open is cheap) rather than sharing
one across threads, which sidesteps SQLite's per-connection thread affinity.
"""

from __future__ import annotations

import hashlib
import heapq
import sqlite3
import struct
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Value types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DocumentMeta:
    """Metadata for one ingested document (no chunk text or vectors)."""

    doc_id: str
    title: str
    source: str
    embedding_model: str
    content_sha: str
    chunk_count: int
    char_count: int
    created_at: str


@dataclass(frozen=True)
class ScoredChunk:
    """A single search hit: chunk text, similarity, and its document provenance."""

    chunk_id: str
    doc_id: str
    chunk_index: int
    text: str
    score: float
    title: str
    source: str


@dataclass(frozen=True)
class CorpusStats:
    """Operator snapshot of corpus contents."""

    documents: int
    chunks: int
    embedding_models: list[str]

    def as_dict(self) -> dict[str, object]:
        return {
            "documents": self.documents,
            "chunks": self.chunks,
            "embedding_models": self.embedding_models,
        }


# ---------------------------------------------------------------------------
# Embedding (de)serialization — packed little-endian float32
# ---------------------------------------------------------------------------


def _pack_embedding(vec: list[float]) -> bytes:
    return struct.pack(f"<{len(vec)}f", *vec)


def _unpack_embedding(blob: bytes) -> list[float]:
    return list(struct.unpack(f"<{len(blob) // 4}f", blob))


def _dot(a: list[float], b: list[float]) -> float:
    """Dot product; equals cosine because stored/query vectors are L2-normalized."""
    return sum(x * y for x, y in zip(a, b, strict=True))


def _content_sha(texts: list[str]) -> str:
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode("utf-8"))
        h.update(b"\x00")  # boundary so ["ab","c"] != ["a","bc"]
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id          TEXT PRIMARY KEY,
    title           TEXT NOT NULL DEFAULT '',
    source          TEXT NOT NULL DEFAULT '',
    embedding_model TEXT NOT NULL,
    content_sha     TEXT NOT NULL,
    chunk_count     INTEGER NOT NULL,
    char_count      INTEGER NOT NULL,
    created_at      TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id    TEXT PRIMARY KEY,
    doc_id      TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    text        TEXT NOT NULL,
    dim         INTEGER NOT NULL,
    embedding   BLOB NOT NULL,
    FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_docs_model ON documents(embedding_model);
"""


class CorpusStore:
    """Persistent store of ingested documents and their chunk embeddings.

    Args:
        db_path: SQLite file path. Parent directories are created if missing.
            ``":memory:"`` is accepted for tests, but note an in-memory database
            is per-connection — since each call reconnects, use a temp file for
            anything that must persist across calls.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        if self._db_path != ":memory:":
            Path(self._db_path).expanduser().parent.mkdir(parents=True, exist_ok=True)
            self._db_path = str(Path(self._db_path).expanduser())
        with closing(self._connect()) as conn:
            conn.executescript(_SCHEMA)
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    # -- Writes ------------------------------------------------------------

    def add_document(
        self,
        *,
        doc_id: str,
        title: str,
        source: str,
        embedding_model: str,
        chunks: list[tuple[str, list[float]]],
    ) -> DocumentMeta:
        """Insert (or replace) a document and its embedded chunks atomically.

        Re-ingesting an existing ``doc_id`` replaces it wholesale — the old
        chunks are removed first (ON DELETE CASCADE) so a shrunk re-ingest never
        leaves orphaned chunks behind.

        Args:
            chunks: ``(chunk_text, embedding)`` pairs, in document order. Every
                embedding must be non-empty and share one dimensionality.

        Raises:
            ValueError: no chunks, or empty / ragged embedding dimensions.
        """
        if not chunks:
            raise ValueError("add_document requires at least one chunk")

        texts = [t for t, _ in chunks]
        dims = {len(vec) for _, vec in chunks}
        if 0 in dims:
            raise ValueError("chunk embeddings must be non-empty")
        if len(dims) != 1:
            raise ValueError(f"chunk embeddings have inconsistent dimensions: {sorted(dims)}")
        dim = dims.pop()

        meta = DocumentMeta(
            doc_id=doc_id,
            title=title,
            source=source,
            embedding_model=embedding_model,
            content_sha=_content_sha(texts),
            chunk_count=len(chunks),
            char_count=sum(len(t) for t in texts),
            created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        )

        with closing(self._connect()) as conn, conn:
            # Replace semantics: drop the old document (cascades to its chunks).
            conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
            conn.execute(
                "INSERT INTO documents "
                "(doc_id, title, source, embedding_model, content_sha, "
                " chunk_count, char_count, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    meta.doc_id,
                    meta.title,
                    meta.source,
                    meta.embedding_model,
                    meta.content_sha,
                    meta.chunk_count,
                    meta.char_count,
                    meta.created_at,
                ),
            )
            conn.executemany(
                "INSERT INTO chunks (chunk_id, doc_id, chunk_index, text, dim, embedding) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (
                        f"{doc_id}::{idx}",
                        doc_id,
                        idx,
                        text,
                        dim,
                        _pack_embedding(vec),
                    )
                    for idx, (text, vec) in enumerate(chunks)
                ],
            )
        return meta

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document and its chunks. Returns True if it existed."""
        with closing(self._connect()) as conn, conn:
            cur = conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
            return cur.rowcount > 0

    # -- Reads -------------------------------------------------------------

    def get_document(self, doc_id: str) -> DocumentMeta | None:
        with closing(self._connect()) as conn:
            row = conn.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
        return _row_to_meta(row) if row else None

    def list_documents(self) -> list[DocumentMeta]:
        """All documents, newest first."""
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT * FROM documents ORDER BY created_at DESC, doc_id"
            ).fetchall()
        return [_row_to_meta(r) for r in rows]

    def count_documents(self) -> int:
        with closing(self._connect()) as conn:
            return int(conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0])

    def count_chunks(self) -> int:
        with closing(self._connect()) as conn:
            return int(conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])

    def stats(self) -> CorpusStats:
        """Operator snapshot for /v1/status: counts and the models in use."""
        with closing(self._connect()) as conn:
            docs = int(conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0])
            chunks = int(conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
            models = [
                r[0]
                for r in conn.execute(
                    "SELECT DISTINCT embedding_model FROM documents ORDER BY embedding_model"
                ).fetchall()
            ]
        return CorpusStats(documents=docs, chunks=chunks, embedding_models=models)

    def search(
        self,
        query_embedding: list[float],
        *,
        embedding_model: str,
        k: int,
    ) -> list[ScoredChunk]:
        """Return the top-``k`` chunks by cosine similarity to ``query_embedding``.

        Only chunks whose document was embedded with ``embedding_model`` are
        considered — mixing embedding spaces would produce meaningless scores.
        Chunks whose stored dimension does not match the query vector are skipped
        defensively. Returns fewer than ``k`` results when the corpus is smaller.
        """
        if k <= 0 or not query_embedding:
            return []
        qdim = len(query_embedding)

        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT c.chunk_id, c.doc_id, c.chunk_index, c.text, c.dim, c.embedding, "
                "       d.title, d.source "
                "FROM chunks c JOIN documents d ON c.doc_id = d.doc_id "
                "WHERE d.embedding_model = ?",
                (embedding_model,),
            ).fetchall()

        scored: list[ScoredChunk] = []
        for r in rows:
            if r["dim"] != qdim:
                continue
            score = _dot(query_embedding, _unpack_embedding(r["embedding"]))
            scored.append(
                ScoredChunk(
                    chunk_id=r["chunk_id"],
                    doc_id=r["doc_id"],
                    chunk_index=r["chunk_index"],
                    text=r["text"],
                    score=score,
                    title=r["title"],
                    source=r["source"],
                )
            )
        return heapq.nlargest(k, scored, key=lambda c: c.score)


def _row_to_meta(row: sqlite3.Row) -> DocumentMeta:
    return DocumentMeta(
        doc_id=row["doc_id"],
        title=row["title"],
        source=row["source"],
        embedding_model=row["embedding_model"],
        content_sha=row["content_sha"],
        chunk_count=row["chunk_count"],
        char_count=row["char_count"],
        created_at=row["created_at"],
    )
