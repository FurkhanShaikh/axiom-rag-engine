"""
Axiom Engine — LangGraph DAG Compilation (The Engine Core)

Wires the nodes and conditional edges into an executable StateGraph.

DAG topology:
  retriever → scorer → ranker → synthesizer → verifier ─┐
                 ▲                    ▲                   │
                 │                    └── (rewrite loop) ◄┘  (if Tier 4/5 & loop < max)
                 └── (re-retrieve) ◄──────────────────────┘  (if loop exhausted & retries left)
                                                          │
                                                          └──► END
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from collections.abc import Callable
from typing import Any, Literal, cast

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from axiom_rag_engine.config.observability import NODE_DURATION
from axiom_rag_engine.nodes.ranker import ranker_node
from axiom_rag_engine.nodes.retriever import retriever_node
from axiom_rag_engine.nodes.scorer import scorer_node
from axiom_rag_engine.nodes.synthesizer import synthesizer_node
from axiom_rag_engine.nodes.verification import verification_node
from axiom_rag_engine.state import (
    GraphState,
    loop_limits,
    reset_verification_state,
    rewrites_remaining,
)
from axiom_rag_engine.utils.audit import error_fields, make_audit_event

logger = logging.getLogger("axiom_rag_engine.graph")

# ---------------------------------------------------------------------------
# Conditional edge — the verification loop (LLD §4)
# ---------------------------------------------------------------------------


def route_post_verification(
    state: GraphState,
) -> Literal["synthesizer", "re_retriever", "__end__"]:
    """
    Determine whether the graph terminates, loops back to the Synthesizer
    for a rewrite pass, or goes all the way back to retriever for fresh sources.

    Routing rules (architecture §5, LLD §4):
      0. If the run halted (a later pass failed) → END with the best pass.
      1. If is_answerable is False → END (escape hatch or insufficient data).
      2. If pending_rewrite_count == 0 → END (all citations verified).
      3. If fewer than max_rewrite_loops rewrites have run in this retrieval
         round → loop to "synthesizer" (rewrite).
      4. If loop exhausted but retrieval_retry_count < max_retries → "retriever"
         (re-retrieve with fresh sources).
      5. Otherwise → END (exhaustion).
    """
    # Rule 0: a later pass failed; the best verified pass is already in place.
    if state.get("halt_reason"):
        return "__end__"

    # Rule 1: escape hatch
    if not state.get("is_answerable", True):
        return "__end__"

    # Rule 2: all good
    if state.get("pending_rewrite_count", 0) == 0:
        return "__end__"

    # Rule 3: rewrite loop
    if rewrites_remaining(state, state.get("loop_count", 0)):
        return "synthesizer"

    # Rule 4: re-retrieve if rewrites exhausted but retries available
    _, max_retries = loop_limits(state)
    if state.get("retrieval_retry_count", 0) < max_retries:
        return "re_retriever"

    # Rule 5: exhaustion
    return "__end__"


# ---------------------------------------------------------------------------
# Re-retrieve wrapper — increments retry counter
# ---------------------------------------------------------------------------


# Scores the scorer/ranker derive. Dropped from retained chunks so the next
# round re-scores them against the union instead of inheriting stale values.
_DERIVED_CHUNK_FIELDS = (
    "source_quality_score",
    "chunk_quality_score",
    "quality_score",
    "relevance_score",
    "ranking_score",
    "dense_score",
    "fused_score",
    "rerank_grade",
)


def _accepts_config(fn: Callable[..., Any]) -> bool:
    try:
        return "config" in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


async def _call_node(fn: Callable[..., Any], state: GraphState, config: Any) -> Any:
    """Invoke a node, handing it the run config only if it declares one."""
    if _accepts_config(fn):
        return await fn(state, config)
    return await fn(state)


async def retriever_with_retry(state: GraphState, config: RunnableConfig | None = None) -> dict:
    """Re-retrieve after the rewrite budget is spent, keeping the best evidence.

    Fresh search results (URLs not seen before) are merged with the previous
    round's top-ranked chunks, so a retry can only add sources — it never
    throws away the best ones it already had. Increments
    retrieval_retry_count and resets per-round verification state.
    """
    result: dict[str, Any] = await _call_node(retriever_node, state, config)
    retained = [
        {k: v for k, v in chunk.items() if k not in _DERIVED_CHUNK_FIELDS}
        for chunk in (state.get("ranked_chunks") or [])
    ]
    if retained:
        fresh_ids = {c["chunk_id"] for c in result.get("indexed_chunks") or []}
        retained = [c for c in retained if c["chunk_id"] not in fresh_ids]
        result["indexed_chunks"] = retained + list(result.get("indexed_chunks") or [])
        result["audit_trail"] = [
            *result.get("audit_trail", []),
            make_audit_event(
                "retriever",
                "retriever_retained_chunks",
                {"retained_chunk_ids": [c["chunk_id"] for c in retained]},
            ),
        ]
    result["retrieval_retry_count"] = state.get("retrieval_retry_count", 0) + 1
    # Reset loop_count so the synthesizer gets fresh rewrite attempts.
    result["loop_count"] = 0
    # Clear all stale verification state from the previous pass.
    result.update(reset_verification_state())
    return result


# ---------------------------------------------------------------------------
# Fail-soft after the first verified pass
# ---------------------------------------------------------------------------
# Once one pass has been verified, every later node (rewrite, re-retrieval,
# re-verification) is refinement: its failure must not cost the caller the
# answer already in hand. A failing later node — provider error, exhausted LLM
# budget, all searches down, or the synthesizer giving up — ends the run with
# the best verified pass instead of failing the request. Before the first
# verified pass there is nothing to fall back to, so errors still propagate.


def _has_verified_pass(state: GraphState) -> bool:
    return state.get("best_pass_rank") is not None


def _halt_with_best_pass(
    state: GraphState, node: str, reason: str, detail: dict[str, Any]
) -> dict[str, Any]:
    """State update that ends the run and returns the best verified pass."""
    best: list[dict[str, Any]] = list(state.get("best_final_sentences") or [])
    logger.warning(
        "Pipeline halted at %s (%s) for request %s; returning the best verified pass.",
        node,
        reason,
        state.get("request_id"),
    )
    return {
        "halt_reason": reason,
        "is_answerable": bool(best),
        "final_sentences": best,
        "pending_rewrite_count": 0,
        "audit_trail": [
            make_audit_event(
                node,
                "pipeline_halted_best_pass_returned",
                {
                    "reason": reason,
                    "best_pass_rank": state.get("best_pass_rank"),
                    "loop_count": state.get("loop_count", 0),
                    "retrieval_retry_count": state.get("retrieval_retry_count", 0),
                    **detail,
                },
            )
        ],
    }


def _fail_soft(name: str, fn: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap a node so a failure after the first verified pass halts the run
    with that pass rather than failing the whole request."""

    async def _wrapper(state: GraphState, config: RunnableConfig) -> dict:
        try:
            result = cast(dict[str, Any], await _call_node(fn, state, config))
        except Exception as exc:
            if not _has_verified_pass(state):
                raise
            logger.warning("Node %s failed after a verified pass: %r", name, exc)
            return _halt_with_best_pass(state, name, "node_error", error_fields(exc))
        if (
            name == "synthesizer"
            and _has_verified_pass(state)
            and not result.get("is_answerable", True)
        ):
            # A later pass declaring the query unanswerable is a failed repair,
            # not new evidence: an earlier pass already answered from sources.
            halt = _halt_with_best_pass(state, name, "synthesizer_gave_up", {})
            halt["audit_trail"] = [*result.get("audit_trail", []), *halt["audit_trail"]]
            return halt
        return result

    _wrapper.__name__ = fn.__name__
    return _wrapper


def _unless_halted(next_node: str) -> Callable[[GraphState], str]:
    """Edge router: continue to ``next_node`` unless the run halted."""

    def _route(state: GraphState) -> str:
        return END if state.get("halt_reason") else next_node

    return _route


# ---------------------------------------------------------------------------
# Request deadline
# ---------------------------------------------------------------------------


class PipelineDeadlineError(Exception):
    """The request deadline expired before any pass was verified."""


class PipelineProgress:
    """The latest full graph state, as handed to the most recent node.

    LangGraph gives each node the state merged from every completed step, so
    this is the run's last checkpoint: what a deadline can still return.
    """

    def __init__(self, state: GraphState) -> None:
        self.state: GraphState = state


_PROGRESS_KEY = "axiom_progress"


def _record_progress(config: Any, state: GraphState) -> None:
    progress = ((config or {}).get("configurable") or {}).get(_PROGRESS_KEY)
    if isinstance(progress, PipelineProgress):
        progress.state = state


def finish_with_best_pass(state: GraphState, reason: str) -> dict[str, Any] | None:
    """Final state for a run stopped from outside the graph: ``state`` with the
    best verified pass as the answer, or None when no pass was verified yet."""
    if not _has_verified_pass(state):
        return None
    halt = _halt_with_best_pass(state, "pipeline", reason, {})
    return {
        **state,
        **halt,
        "audit_trail": [*(state.get("audit_trail") or []), *halt["audit_trail"]],
    }


async def run_pipeline(
    engine: Any,
    initial_state: GraphState,
    run_config: dict[str, Any] | None,
    deadline_seconds: float,
) -> dict[str, Any]:
    """Run the graph under a wall-clock deadline (0 disables it).

    When the deadline expires after a verified pass, the best pass is returned
    as a halted run (``halt_reason="deadline"``) instead of losing it.

    Raises:
        PipelineDeadlineError: the deadline expired before any verified pass.
    """
    if deadline_seconds <= 0:
        return cast(dict[str, Any], await engine.ainvoke(initial_state, config=run_config))

    progress = PipelineProgress(initial_state)
    config = dict(run_config or {})
    config["configurable"] = {**(config.get("configurable") or {}), _PROGRESS_KEY: progress}
    deadline = asyncio.timeout(deadline_seconds)
    try:
        async with deadline:
            return cast(dict[str, Any], await engine.ainvoke(initial_state, config=config))
    except TimeoutError:
        if not deadline.expired():
            raise  # a timeout raised inside the pipeline, not the deadline
    final = finish_with_best_pass(progress.state, "deadline")
    if final is None:
        raise PipelineDeadlineError(
            f"Request deadline of {deadline_seconds:g}s expired before any verified pass."
        )
    return final


# ---------------------------------------------------------------------------
# Node duration instrumentation
# ---------------------------------------------------------------------------


def _timed_node(name: str, fn: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap an async graph node function to record wall-clock duration."""

    # LangGraph injects the run config into nodes that declare a ``config``
    # parameter; the wrapper declares it and forwards it to nodes that want it
    # (the retriever reads its search backend from it).
    async def _wrapper(state: GraphState, config: RunnableConfig) -> dict:
        _record_progress(config, state)
        start = time.monotonic()
        result = await _call_node(fn, state, config)
        NODE_DURATION.labels(node=name).observe(time.monotonic() - start)
        return cast(dict[str, Any], result)

    _wrapper.__name__ = fn.__name__
    return _wrapper


# ---------------------------------------------------------------------------
# Graph compilation
# ---------------------------------------------------------------------------


def build_axiom_graph() -> CompiledStateGraph:
    """
    Construct and compile the Axiom Engine LangGraph DAG.

    Returns the compiled graph, ready to be invoked with an initial state.
    """
    workflow = StateGraph(GraphState)

    # Add nodes (instrumented with per-node duration metrics). Every node that
    # can run after the first verified pass is fail-soft (see _fail_soft).
    workflow.add_node("retriever", _timed_node("retriever", retriever_node))
    for name, fn in (
        ("re_retriever", retriever_with_retry),
        ("scorer", scorer_node),
        ("ranker", ranker_node),
        ("synthesizer", synthesizer_node),
        ("verifier", verification_node),
    ):
        workflow.add_node(name, _timed_node(name, _fail_soft(name, fn)))

    # Linear edges — full pipeline; a halted run ends at the node that halted.
    workflow.set_entry_point("retriever")
    workflow.add_edge("retriever", "scorer")
    for source, target in (
        ("re_retriever", "scorer"),
        ("scorer", "ranker"),
        ("ranker", "synthesizer"),
        ("synthesizer", "verifier"),
    ):
        workflow.add_conditional_edges(source, _unless_halted(target), {target: target, END: END})

    # Conditional edge — the verification loop + re-retrieve
    workflow.add_conditional_edges(
        "verifier",
        route_post_verification,
        {
            "synthesizer": "synthesizer",
            "re_retriever": "re_retriever",
            "__end__": END,
        },
    )

    return workflow.compile()
