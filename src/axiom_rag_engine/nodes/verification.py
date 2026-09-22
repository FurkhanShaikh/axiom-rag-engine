"""
Axiom Engine — Unified Verification Node (Module 7)

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

from axiom_rag_engine.config.observability import LOOP_EXHAUSTED_TIER5, get_tracer
from axiom_rag_engine.models import VerificationResult
from axiom_rag_engine.nodes.semantic import semantic_verifier_node
from axiom_rag_engine.state import GraphState, loop_limits, rewrites_remaining
from axiom_rag_engine.utils.audit import make_audit_event
from axiom_rag_engine.verifiers.mechanical import MechanicalVerifier

# Module-level singleton — stateless, safe to reuse.
_mechanical = MechanicalVerifier()
_audit = partial(make_audit_event, "verifier")
logger = logging.getLogger("axiom_rag_engine.verifier")


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


def _pass_rank(final_sentences: list[dict[str, Any]]) -> list[int]:
    """Sort key for a verification pass — lower is better.

    Primary: fewest sentences that still fail verification (Tier 4/5). Then:
    most fully verified claims (Tier 1-3 and not "unverified").
    """
    failed = 0
    verified = 0
    for sentence in final_sentences:
        vr = sentence.get("verification") or {}
        if vr.get("tier") in (4, 5):
            failed += 1
        elif (
            sentence.get("is_cited")
            and vr.get("tier_label") != "unverified"
            and vr.get("tier") in (1, 2, 3)
        ):
            verified += 1
    return [failed, -verified]


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
    logger.debug(
        "_run_verification: draft_sentences=%d indexed_chunks=%d is_answerable=%s",
        len(draft_sentences),
        len(indexed_chunks),
        state.get("is_answerable"),
    )
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
                mechanical_results[cit_id] = {
                    **VerificationResult(
                        tier=3,
                        tier_label="model_assisted",
                        mechanical_check="passed",
                        semantic_check="skipped",
                        failure_reason=None,
                    ).model_dump(),
                    # Carried alongside the verdict so the response can show the
                    # source's own text (VerificationResult ignores the extra key).
                    "matched_source_text": result.matched_source_text,
                }
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

    # loop_count counts completed verification passes in this retrieval round
    # (the orchestrator owns it; the semantic node never touches it).
    new_loop_count = state.get("loop_count", 0) + 1
    final_sentences: list[dict[str, Any]] = semantic_result.get("final_sentences", [])

    # Track the best pass across the whole request (rewrites and re-retrievals).
    rank = _pass_rank(final_sentences)
    best_rank = state.get("best_pass_rank")
    best_sentences: list[dict[str, Any]] = list(state.get("best_final_sentences") or [])
    if best_rank is None or rank < best_rank:
        best_rank, best_sentences = rank, final_sentences

    # Terminal-by-exhaustion: failures remain and neither another rewrite nor
    # another retrieval round is allowed (mirrors route_post_verification).
    _, max_retries = loop_limits(state)
    retry_count: int = state.get("retrieval_retry_count", 0)
    is_final_attempt = (
        pending_count > 0
        and not rewrites_remaining(state, new_loop_count)
        and retry_count >= max_retries
    )

    if is_final_attempt and best_sentences is not final_sentences:
        # A later pass came out worse than an earlier one — return the best.
        all_audit.append(
            _audit(
                "best_pass_selected",
                {
                    "returned_rank": best_rank,
                    "last_pass_rank": rank,
                    "loop_count": new_loop_count,
                    "retrieval_retry_count": retry_count,
                },
            )
        )
        final_sentences = best_sentences

    # -----------------------------------------------------------------------
    # Loop-exhaustion guard: when ALL retry budget is consumed and unresolved
    # Tier 5 sentences still remain, they reach the final response (labelled).
    # Emit a metric and an audit event so operators can alert on this condition.
    # -----------------------------------------------------------------------
    if is_final_attempt:
        tier5_count = sum(
            1
            for s in final_sentences
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
        "final_sentences": final_sentences,
        "best_final_sentences": best_sentences,
        "best_pass_rank": best_rank,
        "rewrite_requests": all_rewrite_requests,
        "pending_rewrite_count": pending_count,
        "loop_count": new_loop_count,
        "mechanical_results": mechanical_results,
        "audit_trail": all_audit,
    }
