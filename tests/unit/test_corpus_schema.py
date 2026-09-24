"""
Corpus SQLite hardening — WAL mode, a busy timeout, and versioned migrations.

The store used the default rollback journal (a writer's lock blocks readers)
and created its tables with ``CREATE TABLE IF NOT EXISTS`` only, with no schema
version and no way to evolve the schema.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from axiom_rag_engine.corpus.store import SCHEMA_VERSION, CorpusStore


def _user_version(path: Path) -> int:
    with closing(sqlite3.connect(path)) as conn:
        return int(conn.execute("PRAGMA user_version").fetchone()[0])


def _add(store: CorpusStore, doc_id: str) -> None:
    store.add_document(
        doc_id=doc_id,
        title="T",
        source="s",
        embedding_model="m",
        chunks=[("some chunk text", [1.0, 0.0])],
    )


class TestMigrations:
    def test_new_database_is_at_the_current_version(self, tmp_path: Path) -> None:
        db = tmp_path / "corpus.db"
        CorpusStore(db)
        assert _user_version(db) == SCHEMA_VERSION

    def test_pre_versioning_database_upgrades_in_place(self, tmp_path: Path) -> None:
        # A database written before versioning: tables present, user_version 0.
        db = tmp_path / "legacy.db"
        _add(CorpusStore(db), "kept")
        with closing(sqlite3.connect(db)) as conn:
            conn.execute("PRAGMA user_version = 0")
            conn.commit()

        store = CorpusStore(db)

        assert _user_version(db) == SCHEMA_VERSION
        assert store.get_document("kept") is not None

    def test_reopening_is_idempotent(self, tmp_path: Path) -> None:
        db = tmp_path / "corpus.db"
        _add(CorpusStore(db), "a")
        store = CorpusStore(db)
        assert store.count_documents() == 1
        assert _user_version(db) == SCHEMA_VERSION

    def test_newer_schema_is_refused(self, tmp_path: Path) -> None:
        db = tmp_path / "future.db"
        CorpusStore(db)
        with closing(sqlite3.connect(db)) as conn:
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
            conn.commit()
        with pytest.raises(RuntimeError, match="newer than this build"):
            CorpusStore(db)


class TestConcurrency:
    def test_database_uses_wal(self, tmp_path: Path) -> None:
        db = tmp_path / "corpus.db"
        CorpusStore(db)
        with closing(sqlite3.connect(db)) as conn:
            assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"

    def test_reads_proceed_while_a_writer_holds_the_lock(self, tmp_path: Path) -> None:
        db = tmp_path / "corpus.db"
        store = CorpusStore(db)
        _add(store, "a")

        writer = sqlite3.connect(db, isolation_level=None)
        try:
            # Under the rollback journal an exclusive lock blocks every reader
            # ("database is locked" after the timeout); under WAL reads go on.
            writer.execute("BEGIN EXCLUSIVE")
            writer.execute("DELETE FROM documents")
            assert store.count_documents() == 1  # sees the last committed state
        finally:
            writer.execute("ROLLBACK")
            writer.close()
