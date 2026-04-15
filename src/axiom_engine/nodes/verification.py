"""
Axiom Engine v2.3 — Unified Verification Node (Module 7)

Orchestrates the two-stage verification pipeline:
  Stage 1: MechanicalVerifier (deterministic, non-negotiable)
  Stage 2: SemanticVerifier (configurable LLM check)

This is the single LangGraph node registered as "verifier" in the DAG.
It runs mechanical verification on every citation first, then passes
mechanically-approved citations through semantic verification.

Tier 5 (Hallucinated) citations generate rewrite_requests for the
Synthesizer loop. Tier 4 (Misrepresented) citations also generate
rewrite requests via the semantic verifier.

Updates GraphState keys: final_sentences, rewrite_requests, loop_count,
mechanical_results, audit_trail
"""

from __future__ import annotations

import logging
from functools import partial
from typing import Any, cast

from axiom_engine.config.observability import LOOP_EXHAUSTED_TIER5, get_tracer
from axiom_engine.models import VerificationResult
from axiom_engine.nodes.semantic import semantic_verifier_node
from axiom_engine.state import GraphState
from axiom_engine.utils.audit import make_audit_event
from axiom_engine.verifiers.mechanical import MechanicalVerifier

# Module-level singleton — stateless, safe to reuse.
_mechanical = MechanicalVerifier()
_audit = partial(make_audit_event, "verifier")
logger = logging.getLogger("axiom_engine.verifier")


def _build_tier5_rewrite_request(
    sentence_id: str,
    citation_id: str,
    chunk_id: str,
    failure_reason: str,
) -> str:
    return (
        f"Sentence {sentence_id}, citation {citation_id} (chunk {chunk_id}): "
        f"Tier 5 (hallucinated) failure — {failure_reason}"
    )


async def verification_node(state: GraphState) -> dict[str, Any]:
    """
    LangGraph node — Unified Verification.

    Stage 1: Runs MechanicalVerifier on every citation.
    Stage 2: Delegates to semantic_verifier_node for mechanically-passed citations.

    Returns keys: final_sentences, rewrite_requests, loop_count,
                  mechanical_results, audit_trail
    """
    tracer = get_tracer()
    with tracer.start_as_current_span(
        "verification", attributes={"loop_count": state.get("loop_count", 0)}
    ):
        return await _run_verification(state)


async def _run_verification(state: GraphState) -> dict[str, Any]:
    """Inner verification logic, wrapped by the OTel span in verification_node."""
    audit: list[dict[str, Any]] = []
    draft_sentences: list[dict] = list(state.get("draft_sentences") or [])
    indexed_chunks: list[dict] = list(state.get("indexed_chunks") or [])
    chunk_lookup: dict[str, dict] = {c["chunk_id"]: c for c in indexed_chunks}

    audit.append(
        _audit(
            "verification_start",
            {
                "sentence_count": len(draft_sentences),
                "loop_count": state.get("loop_count", 0),
            },
        )
    )

    # ------------------------------------------------------------------
    # Stage 1: Mechanical Verification
    # ------------------------------------------------------------------
    mechanical_results: dict[str, dict[str, Any]] = {}
    mechanical_rewrite_requests: list[str] = []

    for sentence_dict in draft_sentences:
        sentence_id: str = sentence_dict["sentence_id"]
        citations: list[dict] = sentence_dict.get("citations") or []

        for citation in citations:
            cit_id: str = citation["citation_id"]
            chunk_id: str = citation["chunk_id"]
            exact_quote: str = citation.get("exact_source_quote", "")

            chunk_data = chunk_lookup.get(chunk_id)
            if chunk_data is None:
                # Chunk not found — treat as Tier 5.
                mechanical_results[cit_id] = VerificationResult(
                    tier=5,
                    tier_label="hallucinated",
                    mechanical_check="failed",
                    semantic_check="skipped",
                    failure_reason=f"Chunk {chunk_id} not found in indexed_chunks.",
                ).model_dump()
                mechanical_rewrite_requests.append(
                    _build_tier5_rewrite_request(
                        sentence_id,
                        cit_id,
                        chunk_id,
                        f"Chunk {chunk_id} not found in indexed_chunks.",
                    )
                )
                audit.append(
                    _audit(
                        "mechanical_chunk_not_found",
                        {"citation_id": cit_id, "chunk_id": chunk_id},
                    )
                )
                continue

            chunk_text: str = chunk_data.get("text", "")
            result = _mechanical.verify(
                chunk_id=chunk_id,
                chunk_text=chunk_text,
                llm_quote=exact_quote,
            )

            if result.status == "failed":
                mechanical_results[cit_id] = VerificationResult(
                    tier=5,
                    tier_label="hallucinated",
                    mechanical_check="failed",
                    semantic_check="skipped",
                    failure_reason=result.audit_proof.get(
                        "failure_reason",
                        "Normalized quote not found in chunk.",
                    ),
                ).model_dump()
            else:
                mechanical_results[cit_id] = VerificationResult(
                    tier=3,
                    tier_label="model_assisted",
                    mechanical_check="passed",
                    semantic_check="skipped",
                    failure_reason=None,
                ).model_dump()
            audit.append(
                _audit(
                    "mechanical_result",
                    result.audit_proof,
                )
            )

            if result.status == "failed":
                mechanical_rewrite_requests.append(
                    _build_tier5_rewrite_request(
                        sentence_id,
                        cit_id,
                        chunk_id,
                        cast(dict[str, Any], mechanical_results[cit_id]).get(
                            "failure_reason",
                            "Normalized quote not found in chunk.",
                        ),
                    )
                )

    audit.append(
        _audit(
            "mechanical_phase_complete",
            {
                "total_citations": len(mechanical_results),
                "passed": sum(
                    1
                    for payload in mechanical_results.values()
                    if payload.get("mechanical_check") == "passed"
                ),
                "failed": sum(
                    1
                    for payload in mechanical_results.values()
                    if payload.get("mechanical_check") == "failed"
                ),
            },
        )
    )

    # ------------------------------------------------------------------
    # Stage 2: Semantic Verification (delegates to semantic_verifier_node)
    # ------------------------------------------------------------------
    # Build an intermediate state with mechanical_results injected so the
    # semantic node knows which citations to skip.
    semantic_input_state = cast(GraphState, {**state, "mechanical_results": mechanical_results})

    semantic_result = await semantic_verifier_node(semantic_input_state)

    # ------------------------------------------------------------------
    # Merge results
    # ------------------------------------------------------------------
    # Combine mechanical rewrite requests (Tier 5) with semantic ones (Tier 4).
    all_rewrite_requests: list[str] = mechanical_rewrite_requests + semantic_result.get(
        "rewrite_requests", []
    )
    pending_count = len(all_rewrite_requests)

    # Merge audit trails.
    all_audit: list[dict] = audit + semantic_result.get("audit_trail", [])

    # M7 fix: loop_count is incremented here (once per verification pass) rather
    # than inside semantic_verifier_node.  The semantic node previously incremented
    # unconditionally — including the first pass — so max_rewrite_loops=3 allowed
    # only 2 actual rewrites.  Incrementing here keeps the counter semantically
    # correct: it counts completed verification cycles.
    new_loop_count = state.get("loop_count", 0) + 1

    # -----------------------------------------------------------------------
    # Loop-exhaustion guard: when ALL retry budget is consumed and unresolved
    # Tier 5 sentences still remain, they will reach the final response verbatim.
    # Emit a metric and an audit event so operators can alert on this condition.
    # -----------------------------------------------------------------------
    stages_cfg: dict = (state.get("pipeline_config") or {}).get("stages") or {}
    max_loops: int = stages_cfg.get("max_rewrite_loops", 3)
    max_retries: int = stages_cfg.get("max_retrieval_retries", 1)
    retry_count: int = state.get("retrieval_retry_count", 0)

    is_final_attempt = (new_loop_count >= max_loops) and (retry_count >= max_retries)
    if is_final_attempt and pending_count > 0:
        final_sents = semantic_result.get("final_sentences", [])
        tier5_count = sum(
            1
            for s in final_sents
            if isinstance(s.get("verification"), dict) and s["verification"].get("tier") == 5
        )
        if tier5_count > 0:
            LOOP_EXHAUSTED_TIER5.inc(tier5_count)
            all_audit.append(
                _audit(
                    "loop_exhausted_unresolved_tier5",
                    {
                        "tier5_sentence_count": tier5_count,
                        "loop_count": new_loop_count,
                        "retrieval_retry_count": retry_count,
                    },
                )
            )

    return {
        "final_sentences": semantic_result.get("final_sentences", []),
        "rewrite_requests": all_rewrite_requests,
        "pending_rewrite_count": pending_count,
        "loop_count": new_loop_count,
        "mechanical_results": mechanical_results,
        "audit_trail": all_audit,
    }
