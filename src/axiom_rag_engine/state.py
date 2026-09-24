"""
Axiom Engine — LangGraph GraphState
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
    # Completed semantic verdicts for this request, keyed by
    # semantic._verdict_key (claim, chunk, quote, model). Reused by later
    # passes so unchanged sentences are not re-judged; never reset.
    semantic_verdicts: dict[str, dict]

    # ------------------------------------------------------------------
    # OUTPUT STATE
    # ------------------------------------------------------------------
    # Plain list — replaced wholesale once verification fully passes.
    final_sentences: list[dict]
    # Best verified pass seen so far in this request (across rewrite passes AND
    # retrieval retries — final sentences are self-contained, citations carry
    # their resolved source). Returned instead of the last pass when every retry
    # is exhausted and the last pass is worse. Deliberately NOT cleared by
    # reset_verification_state.
    best_final_sentences: list[dict]
    # Sort key of best_final_sentences (lower is better); None before any pass.
    best_pass_rank: list[int] | None
    # Set when a node failed (or the synthesizer gave up) after a verified pass
    # already existed: the run ends and returns best_final_sentences instead of
    # failing the request. None while the run is healthy.
    halt_reason: str | None
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
        semantic_verdicts={},
        final_sentences=[],
        best_final_sentences=[],
        best_pass_rank=None,
        halt_reason=None,
        audit_trail=[],
    )


# Defaults mirror PipelineStagesConfig; used when a state carries no stage config
# (direct node calls in tests / evals).
DEFAULT_MAX_REWRITE_LOOPS = 2
DEFAULT_MAX_RETRIEVAL_RETRIES = 1


def loop_limits(state: GraphState | dict) -> tuple[int, int]:
    """Return ``(max_rewrite_loops, max_retrieval_retries)`` for this request.

    ``max_rewrite_loops`` is the number of *rewrite* passes allowed per
    retrieval round (so each round runs at most ``max_rewrite_loops + 1``
    synthesis passes). Shared by the router (graph.py) and the verifier's
    exhaustion check so the two can never disagree.
    """
    stages: dict = (state.get("pipeline_config") or {}).get("stages") or {}
    return (
        int(stages.get("max_rewrite_loops", DEFAULT_MAX_REWRITE_LOOPS)),
        int(stages.get("max_retrieval_retries", DEFAULT_MAX_RETRIEVAL_RETRIES)),
    )


def rewrites_remaining(state: GraphState | dict, loop_count: int) -> bool:
    """True when another rewrite pass is allowed in the current retrieval round.

    ``loop_count`` counts completed verification passes in this round, so
    ``loop_count - 1`` rewrites have already run.
    """
    max_loops, _ = loop_limits(state)
    return loop_count - 1 < max_loops


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
