"""
Axiom Engine v2.3 — LangGraph DAG Compilation (The Engine Core)

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

import time
from collections.abc import Callable
from typing import Any, Literal, cast

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from axiom_engine.config.observability import NODE_DURATION
from axiom_engine.nodes.ranker import ranker_node
from axiom_engine.nodes.retriever import retriever_node
from axiom_engine.nodes.scorer import scorer_node
from axiom_engine.nodes.synthesizer import synthesizer_node
from axiom_engine.nodes.verification import verification_node
from axiom_engine.state import GraphState, reset_verification_state

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_MAX_RETRIEVAL_RETRIES = 1  # Fallback when not set in pipeline_config


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
      1. If is_answerable is False → END (escape hatch or insufficient data).
      2. If pending_rewrite_count == 0 → END (all citations verified).
      3. If loop_count < max_rewrite_loops → loop to "synthesizer" (rewrite).
      4. If loop exhausted but retrieval_retry_count < max_retries → "retriever"
         (re-retrieve with fresh sources).
      5. Otherwise → END (exhaustion).
    """
    # Rule 1: escape hatch
    if not state.get("is_answerable", True):
        return "__end__"

    # Rule 2: all good
    if state.get("pending_rewrite_count", 0) == 0:
        return "__end__"

    # Rule 3: rewrite loop
    pipeline_cfg: dict = state.get("pipeline_config") or {}
    stages_cfg: dict = pipeline_cfg.get("stages") or {}
    max_loops: int = stages_cfg.get("max_rewrite_loops", 3)

    if state.get("loop_count", 0) < max_loops:
        return "synthesizer"

    # Rule 4: re-retrieve if rewrites exhausted but retries available
    max_retries: int = stages_cfg.get("max_retrieval_retries", _DEFAULT_MAX_RETRIEVAL_RETRIES)
    if state.get("retrieval_retry_count", 0) < max_retries:
        return "re_retriever"

    # Rule 5: exhaustion
    return "__end__"


# ---------------------------------------------------------------------------
# Re-retrieve wrapper — increments retry counter
# ---------------------------------------------------------------------------


async def retriever_with_retry(state: GraphState) -> dict:
    """Wrapper that runs retriever_node and increments retrieval_retry_count."""
    result = await retriever_node(state)
    result["retrieval_retry_count"] = state.get("retrieval_retry_count", 0) + 1
    # Reset loop_count so the synthesizer gets fresh rewrite attempts.
    result["loop_count"] = 0
    # Clear all stale verification state from the previous pass.
    result.update(reset_verification_state())
    return result


# ---------------------------------------------------------------------------
# Node duration instrumentation
# ---------------------------------------------------------------------------


def _timed_node(name: str, fn: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap an async graph node function to record wall-clock duration."""

    async def _wrapper(state: GraphState) -> dict:
        start = time.monotonic()
        result = await fn(state)
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

    # Add nodes (instrumented with per-node duration metrics)
    workflow.add_node("retriever", _timed_node("retriever", retriever_node))
    workflow.add_node("re_retriever", _timed_node("re_retriever", retriever_with_retry))
    workflow.add_node("scorer", _timed_node("scorer", scorer_node))
    workflow.add_node("ranker", _timed_node("ranker", ranker_node))
    workflow.add_node("synthesizer", _timed_node("synthesizer", synthesizer_node))
    workflow.add_node("verifier", _timed_node("verifier", verification_node))

    # Linear edges — full pipeline
    workflow.set_entry_point("retriever")
    workflow.add_edge("retriever", "scorer")
    workflow.add_edge("re_retriever", "scorer")
    workflow.add_edge("scorer", "ranker")
    workflow.add_edge("ranker", "synthesizer")
    workflow.add_edge("synthesizer", "verifier")

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
