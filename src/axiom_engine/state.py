"""
Axiom Engine v2.3 — LangGraph GraphState
Uses typing.Annotated + operator.add only where append-only semantics are safe.
Evidence and rewrite state are replaced on retry passes so stale data cannot
pollute fresh retrieval attempts or leak obsolete rewrite instructions.
"""

from __future__ import annotations

import operator
from collections.abc import Sequence
from typing import Annotated, TypedDict


class GraphState(TypedDict):
    """
    Shared mutable state threaded through all LangGraph nodes.

    Fields annotated with `Annotated[Sequence[...], operator.add]` are
    reducers: LangGraph merges node return values by *appending* rather than
    replacing, which is critical for the incremental override logic (§5 of the
    architecture document).
    """

    # ------------------------------------------------------------------
    # INPUT — populated once at graph entry; never mutated by nodes
    # ------------------------------------------------------------------
    request_id: str
    user_query: str
    app_config: dict  # Serialised AppConfig
    models_config: dict  # Serialised ModelConfig
    pipeline_config: dict  # Serialised PipelineConfig

    # ------------------------------------------------------------------
    # RETRIEVAL STATE
    # ------------------------------------------------------------------
    search_queries: list[str]
    indexed_chunks: list[dict]
    # Monotonic doc counter so chunk IDs stay unique across retrieval retries.
    next_doc_index: int
    # Appended-to list of URLs across all retries to prevent duplicate fetching.
    past_seen_urls: Annotated[Sequence[str], operator.add]

    # ------------------------------------------------------------------
    # SCORING & RANKING STATE
    # ------------------------------------------------------------------
    # Chunks after source quality scoring (domain authority + consistency).
    scored_chunks: list[dict]
    # Chunks ranked by relevance to the user query, trimmed to top-N.
    ranked_chunks: list[dict]

    # ------------------------------------------------------------------
    # COGNITIVE STATE
    # ------------------------------------------------------------------
    is_answerable: bool
    # Plain list — Synthesizer replaces its output on each rewrite pass.
    draft_sentences: list[dict]

    # ------------------------------------------------------------------
    # VERIFICATION LOOP STATE
    # ------------------------------------------------------------------
    # Current-pass rewrite requests only. Replaced on every verification pass.
    rewrite_requests: list[str]
    # Overwritten each pass — number of NEW rewrite requests from the
    # most recent verification pass. Used by route_post_verification to
    # decide whether to loop (accumulated list is for correction context).
    pending_rewrite_count: int
    # Incremented by the verification node on every loop iteration.
    loop_count: int
    # Number of times retrieval has been retried due to persistent failures.
    retrieval_retry_count: int
    mechanical_results: dict[str, dict]

    # ------------------------------------------------------------------
    # OUTPUT STATE
    # ------------------------------------------------------------------
    # Plain list — replaced wholesale once verification fully passes.
    final_sentences: list[dict]
    # operator.add — every node appends its own audit events; the audit
    # trail is never overwritten, preserving causality across re-entries.
    audit_trail: Annotated[Sequence[dict], operator.add]


def make_initial_state(
    request_id: str,
    user_query: str,
    app_config: dict,
    models_config: dict,
    pipeline_config: dict,
) -> GraphState:
    """
    Construct a zero-valued GraphState for a fresh pipeline invocation.
    Explicit initialisation of every key prevents KeyError inside nodes.
    """
    return GraphState(
        request_id=request_id,
        user_query=user_query,
        app_config=app_config,
        models_config=models_config,
        pipeline_config=pipeline_config,
        search_queries=[],
        indexed_chunks=[],
        next_doc_index=1,
        past_seen_urls=[],
        scored_chunks=[],
        ranked_chunks=[],
        is_answerable=True,
        draft_sentences=[],
        rewrite_requests=[],
        pending_rewrite_count=0,
        loop_count=0,
        retrieval_retry_count=0,
        mechanical_results={},
        final_sentences=[],
        audit_trail=[],
    )


def reset_verification_state() -> dict:
    """Return the canonical reset dict for verification-related fields.

    Used by ``retriever_with_retry`` (graph.py) to clear stale verification
    state before a fresh retrieval pass.  Centralising this here ensures
    that new verification fields are reset in one place rather than being
    scattered across multiple call sites.
    """
    return {
        "draft_sentences": [],
        "final_sentences": [],
        "mechanical_results": {},
        "rewrite_requests": [],
        "pending_rewrite_count": 0,
        "scored_chunks": [],
        "ranked_chunks": [],
    }
