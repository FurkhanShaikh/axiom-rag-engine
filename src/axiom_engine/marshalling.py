"""
Axiom Engine — GraphState → AxiomResponse marshalling.

Converts raw graph output into validated API responses, including error
responses. Extracted from main.py to follow SRP.
"""

from __future__ import annotations

import logging
from typing import Any

from axiom_engine.models import (
    AuditEvent,
    AxiomResponse,
    ConfidenceSummary,
    DebugInfo,
    FinalSentence,
    TierBreakdown,
)
from axiom_engine.scoring import compute_confidence_summary, determine_status

logger = logging.getLogger("axiom_engine.marshalling")


def marshal_response(
    request_id: str,
    graph_result: dict[str, Any],
    include_debug: bool = False,
) -> AxiomResponse:
    """
    Convert the raw GraphState dict returned by the compiled graph into
    a validated AxiomResponse.
    """
    is_answerable: bool = graph_result.get("is_answerable", False)
    raw_sentences: list[dict] = graph_result.get("final_sentences", [])

    # Validate each sentence through the Pydantic model to ensure
    # the response contract is fully honoured.
    final_sentences: list[FinalSentence] = [FinalSentence.model_validate(s) for s in raw_sentences]

    status = determine_status(is_answerable, raw_sentences)
    confidence = compute_confidence_summary(raw_sentences)

    debug: DebugInfo | None = None
    if include_debug:
        raw_audit = graph_result.get("audit_trail", [])
        debug = DebugInfo(
            audit_trail=[AuditEvent.model_validate(e) for e in raw_audit],
            pipeline_stats={
                "chunks_retrieved": len(graph_result.get("indexed_chunks", [])),
                "chunks_ranked": len(graph_result.get("ranked_chunks", [])),
                "loop_count": graph_result.get("loop_count", 0),
                "retrieval_retry_count": graph_result.get("retrieval_retry_count", 0),
            },
        )

    return AxiomResponse(
        request_id=request_id,
        status=status,
        is_answerable=is_answerable,
        confidence_summary=confidence,
        final_response=final_sentences,
        debug=debug,
    )


def make_error_response(
    request_id: str,
    error: Exception,
) -> AxiomResponse:
    """
    Build a structured error response matching the AxiomResponse schema.
    Category 1 errors (architecture §7): unrecoverable system failures.
    """
    # Log full detail server-side; return only a generic message to the client.
    logger.error(
        "Pipeline error for request %s: %s: %s",
        request_id,
        type(error).__name__,
        error,
    )

    return AxiomResponse(
        request_id=request_id,
        status="error",
        is_answerable=False,
        confidence_summary=ConfidenceSummary(
            overall_score=0.0,
            tier_breakdown=TierBreakdown(),
        ),
        final_response=[],
        error_message=f"Internal pipeline error — see server logs for request {request_id}.",
    )
