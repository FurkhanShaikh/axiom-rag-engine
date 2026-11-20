"""Embedding backend for dense/hybrid retrieval.

All embedding calls go through LiteLLM via ``utils.llm.call_embedding``, so they
share the chat calls' budget, concurrency limit, retries, and usage/cost
accounting. The model
is a config string (``ollama/nomic-embed-text`` for local, ``text-embedding-3-small``
for OpenAI, ...), and Ollama models get their ``api_base`` injected the same way
``utils.llm.build_completion_kwargs`` does.

The retrieval eval established the value of dense retrieval; this is the
production path that lets it run in the pipeline. It is *opt-in* — the ranker
only calls it when ``AXIOM_EMBEDDING_MODEL`` is set, so BM25-only remains the
default and existing deployments are unaffected.
"""

from __future__ import annotations

import math
from typing import Any

from axiom_rag_engine.config.settings import current_settings
from axiom_rag_engine.utils.llm import call_embedding, call_embedding_sync

# Budget/usage label for embedding calls (``node`` in metrics and usage).
_NODE = "embedding"
# Bound each embedding API call so large documents stay under provider per-request limits.
_EMBEDDING_BATCH_SIZE = 64


def embed_prefixes(model: str) -> tuple[str, str]:
    """Return (doc_prefix, query_prefix) for instructed embedders.

    nomic-embed-text is trained with task prefixes and loses substantial
    retrieval quality without them. Other models default to no prefix.
    """
    if "nomic" in model.lower():
        return "search_document: ", "search_query: "
    return "", ""


def _embedding_kwargs(model: str, inputs: list[str]) -> dict[str, Any]:
    settings = current_settings()
    kwargs: dict[str, Any] = {
        "model": model,
        "input": inputs,
        "timeout": settings.llm_timeout_seconds,
    }
    # Ollama models need the api_base pointed at the local server, exactly as
    # the LLM path does. Non-ollama providers read their key from the env.
    if model.startswith("ollama/"):
        kwargs["api_base"] = settings.ollama_api_base
    return kwargs


def _l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return vec
    return [x / norm for x in vec]


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two vectors (0.0 if either is degenerate)."""
    if len(a) != len(b):
        return 0.0
    return sum(x * y for x, y in zip(a, b, strict=True))


async def _embed_batched(
    model: str,
    inputs: list[str],
    *,
    batch_size: int = _EMBEDDING_BATCH_SIZE,
) -> list[list[float]]:
    """Embed ``inputs`` in bounded batches and return L2-normalized vectors."""
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    vectors: list[list[float]] = []
    for start in range(0, len(inputs), batch_size):
        batch = inputs[start : start + batch_size]
        response = await call_embedding(_NODE, model, _embedding_kwargs(model, batch))
        # LiteLLM normalizes the response to OpenAI shape: data[i]["embedding"].
        rows = list(response["data"])
        if rows and all(isinstance(row, dict) and "index" in row for row in rows):
            rows.sort(key=lambda row: row["index"])
        if len(rows) != len(batch):
            raise ValueError(
                f"Embedding response count mismatch: expected {len(batch)}, got {len(rows)}"
            )
        vectors.extend(_l2_normalize(list(row["embedding"])) for row in rows)
    return vectors


async def embed_query_and_chunks(
    model: str,
    query: str,
    chunk_texts: list[str],
    *,
    batch_size: int = _EMBEDDING_BATCH_SIZE,
) -> tuple[list[float], list[list[float]]]:
    """Embed the query and chunk texts; return normalized vectors.

    Rows are L2-normalized so ``cosine`` is a plain dot product. Applies the
    model's task prefixes (query vs document). Returns
    ``(query_vector, [chunk_vector, ...])`` aligned with ``chunk_texts``.
    """
    doc_prefix, query_prefix = embed_prefixes(model)
    inputs = [query_prefix + query] + [doc_prefix + t for t in chunk_texts]
    vectors = await _embed_batched(model, inputs, batch_size=batch_size)
    return vectors[0], vectors[1:]


async def embed_documents(
    model: str,
    texts: list[str],
    *,
    batch_size: int = _EMBEDDING_BATCH_SIZE,
) -> list[list[float]]:
    """Embed passage/document texts with the doc-side task prefix, L2-normalized.

    Used by corpus ingestion to compute stored chunk vectors. Returns one vector
    per input, aligned with ``texts`` (empty input → empty list). Because rows
    are L2-normalized, later cosine scoring against a query vector is a plain
    dot product — the same convention the ranker uses.
    """
    if not texts:
        return []
    doc_prefix, _ = embed_prefixes(model)
    inputs = [doc_prefix + t for t in texts]
    return await _embed_batched(model, inputs, batch_size=batch_size)


async def embed_query(model: str, query: str) -> list[float]:
    """Embed a single query string with the query-side task prefix, L2-normalized.

    The corpus-retrieval counterpart to :func:`embed_documents`: the query
    vector it returns is cosine-comparable to stored chunk vectors *only when
    both were produced by the same model*, which the corpus store enforces.
    """
    _, query_prefix = embed_prefixes(model)
    response = await call_embedding(_NODE, model, _embedding_kwargs(model, [query_prefix + query]))
    return _l2_normalize(list(response["data"][0]["embedding"]))


def embed_query_sync(model: str, query: str) -> list[float]:
    """Synchronous :func:`embed_query` for code already running off the event loop.

    The corpus backend runs inside ``asyncio.to_thread`` worker threads. Driving
    the async embedder there with ``asyncio.run`` spins up a throwaway loop per
    call and leaves LiteLLM's cached aiohttp sessions and logging tasks bound to
    dead loops ("attached to a different loop"). The sync client has no loop
    affinity.
    """
    _, query_prefix = embed_prefixes(model)
    response = call_embedding_sync(_NODE, model, _embedding_kwargs(model, [query_prefix + query]))
    return _l2_normalize(list(response["data"][0]["embedding"]))
