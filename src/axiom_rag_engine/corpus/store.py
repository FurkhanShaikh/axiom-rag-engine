"""SQLite-backed corpus store for ingested documents (bring-your-own corpus).

Single-node and dependency-light: ``sqlite3`` + ``struct`` from the stdlib, no
vector-DB server. Chunk embeddings are stored as packed float32 blobs and
searched by brute-force cosine similarity.

Search keeps a decoded copy of each embedding model's vectors in memory, keyed
by the corpus version counter (bumped by every ingest and delete), so a query
reads SQLite only for that counter and its top-``k`` rows. Scoring uses numpy
when it is installed (the ``vector`` extra) and pure Python otherwise; results
are identical up to float32 rounding. ``evals/corpus_eval.py --bench-search``
measures both (see BENCHMARKS.md).

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
The database runs in WAL mode so searches never wait on an ingest, and each
connection waits up to ``_BUSY_TIMEOUT_SECONDS`` for another writer's lock.

Schema
------
The schema version lives in ``PRAGMA user_version``; ``_MIGRATIONS`` upgrades
older databases in place when the store opens, and a database written by a
newer build is refused rather than misread.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import sqlite3
import struct
import threading
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:  # optional: vectorised scoring (the "vector" extra)
    import numpy as _np
except ImportError:  # pragma: no cover - exercised when numpy is absent
    _np = None  # type: ignore[assignment]

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


@dataclass
class _VectorIndex:
    """One embedding model's vectors of one dimension, as of a corpus version.

    ``rowids`` identify each vector's chunk; ``matrix`` is an ``(n, dim)``
    float32 array with numpy, else a list of vectors.
    """

    version: int
    rowids: list[int]
    matrix: Any


def _build_matrix(blobs: list[bytes], dim: int, use_numpy: bool) -> Any:
    if use_numpy and _np is not None:
        if not blobs:
            return _np.zeros((0, dim), dtype=_np.float32)
        return _np.frombuffer(b"".join(blobs), dtype="<f4").reshape(len(blobs), dim)
    return [_unpack_embedding(b) for b in blobs]


def _top_k(matrix: Any, query: list[float], k: int, use_numpy: bool) -> list[tuple[int, float]]:
    """(position, score) of the ``k`` best rows, best first; ties keep row order."""
    if use_numpy and _np is not None:
        if len(matrix) == 0:
            return []
        scores = matrix @ _np.asarray(query, dtype=_np.float32)
        if k < len(scores):
            candidates = _np.argpartition(-scores, k - 1)[:k]
        else:
            candidates = _np.arange(len(scores))
        order = sorted(candidates.tolist(), key=lambda i: (-float(scores[i]), i))
        return [(i, float(scores[i])) for i in order]
    scored = ((i, _dot(query, vec)) for i, vec in enumerate(matrix))
    return heapq.nlargest(k, scored, key=lambda pair: pair[1])


def _content_sha(texts: list[str]) -> str:
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode("utf-8"))
        h.update(b"\x00")  # boundary so ["ab","c"] != ["a","bc"]
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

# Schema migrations, applied in order. The database's version lives in
# ``PRAGMA user_version``: a database at version N runs ``_MIGRATIONS[N:]``.
# Version 1 uses IF NOT EXISTS so databases created before versioning (which
# report version 0 but already have the tables) upgrade in place.
_MIGRATIONS: tuple[tuple[str, ...], ...] = (
    # v1 — documents and their embedded chunks.
    (
        """CREATE TABLE IF NOT EXISTS documents (
            doc_id          TEXT PRIMARY KEY,
            title           TEXT NOT NULL DEFAULT '',
            source          TEXT NOT NULL DEFAULT '',
            embedding_model TEXT NOT NULL,
            content_sha     TEXT NOT NULL,
            chunk_count     INTEGER NOT NULL,
            char_count      INTEGER NOT NULL,
            created_at      TEXT NOT NULL
        )""",
        """CREATE TABLE IF NOT EXISTS chunks (
            chunk_id    TEXT PRIMARY KEY,
            doc_id      TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            text        TEXT NOT NULL,
            dim         INTEGER NOT NULL,
            embedding   BLOB NOT NULL,
            FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
        )""",
        "CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id)",
        "CREATE INDEX IF NOT EXISTS idx_docs_model ON documents(embedding_model)",
    ),
    # v2 — a corpus version, bumped on every change, so response caches keyed on
    # it stop serving answers built from deleted or replaced documents.
    (
        "CREATE TABLE IF NOT EXISTS corpus_meta (key TEXT PRIMARY KEY, value INTEGER NOT NULL)",
        "INSERT OR IGNORE INTO corpus_meta (key, value) VALUES ('version', 0)",
    ),
)
_BUMP_VERSION = "UPDATE corpus_meta SET value = value + 1 WHERE key = 'version'"
SCHEMA_VERSION = len(_MIGRATIONS)

# How long a connection waits on another writer's lock before failing with
# "database is locked". WAL mode keeps readers from blocking the writer at all.
_BUSY_TIMEOUT_SECONDS = 10.0


def _migrate(conn: sqlite3.Connection) -> None:
    """Bring the database up to SCHEMA_VERSION, one transaction per step.

    ``BEGIN IMMEDIATE`` takes the write lock before the version is re-read, so
    two processes opening the same file cannot both apply a step.

    Raises:
        RuntimeError: the database was written by a newer build.
    """
    while True:
        conn.execute("BEGIN IMMEDIATE")
        try:
            current = int(conn.execute("PRAGMA user_version").fetchone()[0])
            if current > SCHEMA_VERSION:
                raise RuntimeError(
                    f"Corpus database schema v{current} is newer than this build supports "
                    f"(v{SCHEMA_VERSION}); upgrade axiom-rag-engine."
                )
            if current == SCHEMA_VERSION:
                conn.execute("COMMIT")
                return
            for statement in _MIGRATIONS[current]:
                conn.execute(statement)
            conn.execute(f"PRAGMA user_version = {current + 1}")
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise


class CorpusStore:
    """Persistent store of ingested documents and their chunk embeddings.

    Args:
        db_path: SQLite file path. Parent directories are created if missing.
            ``":memory:"`` is accepted for tests, but note an in-memory database
            is per-connection — since each call reconnects, use a temp file for
            anything that must persist across calls.
    """

    def __init__(self, db_path: str | Path, *, use_numpy: bool = True) -> None:
        self._db_path = str(db_path)
        # Decoded vectors per (embedding model, dim), rebuilt when the corpus
        # version moves. Searches run in worker threads, hence the lock.
        self._use_numpy = use_numpy and _np is not None
        self._indexes: dict[tuple[str, int], _VectorIndex] = {}
        self._index_lock = threading.Lock()
        if self._db_path != ":memory:":
            Path(self._db_path).expanduser().parent.mkdir(parents=True, exist_ok=True)
            self._db_path = str(Path(self._db_path).expanduser())
        with closing(self._connect()) as conn:
            if self._db_path != ":memory:":
                # Persistent per file: readers no longer block the writer, so a
                # search during an ingest does not hit "database is locked".
                conn.execute("PRAGMA journal_mode = WAL")
            conn.isolation_level = None  # explicit transactions in _migrate
            _migrate(conn)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=_BUSY_TIMEOUT_SECONDS)
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
            conn.execute(_BUMP_VERSION)
        return meta

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document and its chunks. Returns True if it existed."""
        with closing(self._connect()) as conn, conn:
            cur = conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
            if cur.rowcount == 0:
                return False
            conn.execute(_BUMP_VERSION)
            return True

    # -- Reads -------------------------------------------------------------

    def version(self) -> int:
        """Monotonic counter bumped by every ingest and delete (in the same
        transaction), for invalidating caches derived from the corpus."""
        with closing(self._connect()) as conn:
            return int(
                conn.execute("SELECT value FROM corpus_meta WHERE key = 'version'").fetchone()[0]
            )

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
        index = self._vector_index(embedding_model, qdim)
        top = _top_k(index.matrix, query_embedding, k, self._use_numpy)
        if not top:
            return []

        rowids = [index.rowids[i] for i, _ in top]
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT c.rowid, c.chunk_id, c.doc_id, c.chunk_index, c.text, "
                "       d.title, d.source "
                "FROM chunks c JOIN documents d ON c.doc_id = d.doc_id "
                "WHERE c.rowid IN (SELECT value FROM json_each(?))",
                (json.dumps(rowids),),
            ).fetchall()
        by_rowid = {r["rowid"]: r for r in rows}

        hits: list[ScoredChunk] = []
        for (_, score), rowid in zip(top, rowids, strict=True):
            r = by_rowid.get(rowid)
            if r is None:  # deleted since the index was read; the next search rebuilds
                continue
            hits.append(
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
        return hits

    def _vector_index(self, embedding_model: str, dim: int) -> _VectorIndex:
        """The decoded vectors for ``embedding_model`` at ``dim``, current as of
        the corpus version (one small query when nothing changed)."""
        key = (embedding_model, dim)
        with closing(self._connect()) as conn:
            version = int(
                conn.execute("SELECT value FROM corpus_meta WHERE key = 'version'").fetchone()[0]
            )
            cached = self._indexes.get(key)
            if cached is not None and cached.version == version:
                return cached
            with self._index_lock:
                cached = self._indexes.get(key)
                if cached is not None and cached.version == version:
                    return cached
                # One read transaction: the version and the rows it describes
                # come from the same snapshot.
                conn.execute("BEGIN")
                version = int(
                    conn.execute("SELECT value FROM corpus_meta WHERE key = 'version'").fetchone()[
                        0
                    ]
                )
                rows = conn.execute(
                    "SELECT c.rowid, c.embedding "
                    "FROM chunks c JOIN documents d ON c.doc_id = d.doc_id "
                    "WHERE d.embedding_model = ? AND c.dim = ? ORDER BY c.rowid",
                    (embedding_model, dim),
                ).fetchall()
                conn.execute("COMMIT")
                index = _VectorIndex(
                    version=version,
                    rowids=[r["rowid"] for r in rows],
                    matrix=_build_matrix([r["embedding"] for r in rows], dim, self._use_numpy),
                )
                self._indexes[key] = index
                return index


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
