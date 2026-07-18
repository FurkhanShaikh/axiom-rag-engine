"""Embedding backend for dense/hybrid retrieval.

All embedding calls go through LiteLLM, mirroring the LLM integration: the model
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

import litellm

from axiom_rag_engine.config.settings import get_settings


def embed_prefixes(model: str) -> tuple[str, str]:
    """Return (doc_prefix, query_prefix) for instructed embedders.

    nomic-embed-text is trained with task prefixes and loses substantial
    retrieval quality without them. Other models default to no prefix.
    """
    if "nomic" in model.lower():
        return "search_document: ", "search_query: "
    return "", ""


def _embedding_kwargs(model: str, inputs: list[str]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"model": model, "input": inputs}
    # Ollama models need the api_base pointed at the local server, exactly as
    # the LLM path does. Non-ollama providers read their key from the env.
    if model.startswith("ollama/"):
        kwargs["api_base"] = get_settings().ollama_api_base
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


async def embed_query_and_chunks(
    model: str,
    query: str,
    chunk_texts: list[str],
) -> tuple[list[float], list[list[float]]]:
    """Embed the query and chunk texts in one call; return normalized vectors.

    Rows are L2-normalized so ``cosine`` is a plain dot product. Applies the
    model's task prefixes (query vs document). Returns
    ``(query_vector, [chunk_vector, ...])`` aligned with ``chunk_texts``.
    """
    doc_prefix, query_prefix = embed_prefixes(model)
    inputs = [query_prefix + query] + [doc_prefix + t for t in chunk_texts]
    response = await litellm.aembedding(**_embedding_kwargs(model, inputs))
    # LiteLLM normalizes the response to OpenAI shape: data[i]["embedding"].
    vectors = [_l2_normalize(list(row["embedding"])) for row in response["data"]]
    return vectors[0], vectors[1:]
