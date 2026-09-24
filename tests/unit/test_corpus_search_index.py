"""
Corpus search reads SQLite for the version counter and its top-k rows only (COR-5).

Every query used to load and decode every chunk embedding for the model. The
decoded vectors are now cached per (model, dim) until the corpus version moves,
and scored with numpy when available — with the same results as pure Python.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from axiom_rag_engine.corpus.store import CorpusStore


def _rng(seed: int) -> random.Random:
    return random.Random(seed)  # noqa: S311 - test vectors, not crypto


def _unit(rng: random.Random, dim: int = 16) -> list[float]:
    vec = [rng.gauss(0, 1) for _ in range(dim)]
    norm = sum(x * x for x in vec) ** 0.5
    return [x / norm for x in vec]


def _fill(store: CorpusStore, rng: random.Random, docs: int = 5, per_doc: int = 20) -> None:
    for d in range(docs):
        store.add_document(
            doc_id=f"d{d}",
            title=f"Doc {d}",
            source=f"https://example.com/{d}",
            embedding_model="m",
            chunks=[(f"doc {d} chunk {i}", _unit(rng)) for i in range(per_doc)],
        )


@pytest.fixture
def db(tmp_path: Path) -> Path:
    path = tmp_path / "corpus.db"
    _fill(CorpusStore(path), _rng(1))
    return path


def test_numpy_and_pure_python_agree(db: Path) -> None:
    query = _unit(_rng(2))
    fast = CorpusStore(db, use_numpy=True).search(query, embedding_model="m", k=10)
    slow = CorpusStore(db, use_numpy=False).search(query, embedding_model="m", k=10)
    assert [h.chunk_id for h in fast] == [h.chunk_id for h in slow]
    assert [h.score for h in fast] == pytest.approx([h.score for h in slow], abs=1e-5)
    assert [h.score for h in fast] == sorted((h.score for h in fast), reverse=True)
    assert fast[0].title.startswith("Doc ") and fast[0].source.startswith("https://")


def test_vectors_are_decoded_once_per_version(db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from axiom_rag_engine.corpus import store as store_module

    builds: list[int] = []
    real = store_module._build_matrix

    def counting(blobs: list[bytes], dim: int, use_numpy: bool) -> object:
        builds.append(len(blobs))
        return real(blobs, dim, use_numpy)

    monkeypatch.setattr(store_module, "_build_matrix", counting)
    store = CorpusStore(db)
    query = _unit(_rng(3))
    for _ in range(3):
        store.search(query, embedding_model="m", k=5)
    assert builds == [100]


@pytest.mark.parametrize("use_numpy", [True, False])
def test_ingest_and_delete_invalidate_the_cache(db: Path, use_numpy: bool) -> None:
    store = CorpusStore(db, use_numpy=use_numpy)
    rng = _rng(4)
    query = _unit(rng)
    store.search(query, embedding_model="m", k=5)  # warm the cache

    store.add_document(
        doc_id="exact",
        title="Exact",
        source="s",
        embedding_model="m",
        chunks=[("the exact match", query)],
    )
    assert store.search(query, embedding_model="m", k=1)[0].chunk_id == "exact::0"

    store.delete_document("exact")
    assert all(h.doc_id != "exact" for h in store.search(query, embedding_model="m", k=100))


def test_another_process_writing_is_seen(db: Path) -> None:
    reader = CorpusStore(db)
    query = _unit(_rng(5))
    reader.search(query, embedding_model="m", k=1)
    CorpusStore(db).add_document(  # a separate store object, as another replica would be
        doc_id="late", title="Late", source="s", embedding_model="m", chunks=[("late", query)]
    )
    assert reader.search(query, embedding_model="m", k=1)[0].doc_id == "late"


def test_other_models_and_dimensions_are_not_scored(db: Path) -> None:
    store = CorpusStore(db)
    rng = _rng(6)
    store.add_document(
        doc_id="other", title="O", source="s", embedding_model="other", chunks=[("x", _unit(rng))]
    )
    hits = store.search(_unit(rng), embedding_model="m", k=500)
    assert len(hits) == 100 and {h.doc_id for h in hits} != {"other"}
    assert store.search(_unit(rng, dim=8), embedding_model="m", k=5) == []


def test_k_larger_than_the_corpus_returns_everything(db: Path) -> None:
    hits = CorpusStore(db).search(_unit(_rng(7)), embedding_model="m", k=10_000)
    assert len(hits) == 100
