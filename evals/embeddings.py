"""Embedding backend for the retrieval eval, with a disk cache.

Dense retrieval needs a vector per corpus document. Embedding 5k docs through
Ollama is a one-time ~5 minute cost when batched, and pointless to repeat every
run, so the corpus matrix is cached to disk keyed by (model, corpus content).
A cache hit makes subsequent runs instant.

This is eval-side machinery on purpose: it proves whether dense retrieval helps
*before* the embedder is wired into the production pipeline (roadmap 1.1). If
hybrid earns its keep here, this batching + caching pattern is the template for
the production embedder.
"""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

_EVALS_DIR = Path(__file__).resolve().parent
_CACHE_DIR = _EVALS_DIR / "data" / "embeddings"

# nomic-embed-text handles ~2k tokens; SciFact abstracts are far shorter, so a
# large batch is safe and keeps throughput high (serial embedding is ~17x slower).
_BATCH_SIZE = 64


def _echo(msg: str = "") -> None:
    sys.stdout.write(f"{msg}\n")


class OllamaEmbedder:
    """Batched Ollama embeddings with an on-disk corpus cache.

    Args:
        model: Ollama embedding model (e.g. ``nomic-embed-text``).
        base_url: Ollama host.
    """

    def __init__(self, model: str, base_url: str) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")

    # -- HTTP ---------------------------------------------------------------

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        body = json.dumps({"model": self.model, "input": texts}).encode()
        req = urllib.request.Request(  # noqa: S310 — operator-controlled Ollama host
            f"{self.base_url}/api/embed",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=300) as resp:  # noqa: S310
            data = json.loads(resp.read())
        embeddings = data.get("embeddings")
        if not embeddings:
            raise RuntimeError(f"Ollama returned no embeddings for a batch of {len(texts)} texts")
        return embeddings

    def embed_texts(self, texts: list[str], *, label: str = "") -> np.ndarray:
        """Embed a list of texts, returning an (n, dim) float32 matrix.

        Rows are L2-normalized so a plain dot product is cosine similarity.
        """
        vectors: list[list[float]] = []
        total = len(texts)
        for start in range(0, total, _BATCH_SIZE):
            batch = texts[start : start + _BATCH_SIZE]
            vectors.extend(self._embed_batch(batch))
            if label and (start // _BATCH_SIZE) % 10 == 0:
                _echo(f"  embedding {label}: {min(start + _BATCH_SIZE, total)}/{total}")
        matrix = np.asarray(vectors, dtype=np.float32)
        return _l2_normalize(matrix)

    # -- Corpus cache -------------------------------------------------------

    def _cache_path(self, doc_ids: list[str], texts: list[str]) -> Path:
        # Key on the model plus a digest of the corpus content and order, so a
        # changed corpus (or model) misses rather than returning stale vectors.
        h = hashlib.sha256()
        h.update(self.model.encode())
        for doc_id, text in zip(doc_ids, texts, strict=True):
            h.update(doc_id.encode())
            h.update(b"\x00")
            h.update(text.encode())
            h.update(b"\x01")
        model_slug = self.model.replace("/", "_").replace(":", "_")
        return _CACHE_DIR / f"{model_slug}-{h.hexdigest()[:16]}.npz"

    def embed_corpus(self, doc_ids: list[str], texts: list[str]) -> np.ndarray:
        """Return the normalized (n, dim) corpus matrix, using the disk cache."""
        cache_path = self._cache_path(doc_ids, texts)
        if cache_path.exists():
            _echo(f"  corpus embeddings: cache hit ({cache_path.name})")
            return np.load(cache_path)["matrix"]

        _echo(f"  corpus embeddings: cache miss — embedding {len(texts)} docs (one-time)")
        matrix = self.embed_texts(texts, label="corpus")
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, matrix=matrix)
        _echo(f"  corpus embeddings: cached to {cache_path.name}")
        return matrix


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization; zero rows are left as zeros (dot product 0)."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms
