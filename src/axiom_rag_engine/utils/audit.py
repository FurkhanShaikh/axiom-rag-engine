"""
Axiom Engine — Audit trail utilities.

Provides a single factory function for creating audit events, replacing
the identical _make_audit_event() that was copy-pasted across 6 node modules.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4


def make_audit_event(
    node: str,
    event_type: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Create a standardised audit event dict for the graph audit trail.

    Args:
        node:       Name of the emitting LangGraph node (e.g. "retriever").
        event_type: Machine-readable event type (e.g. "retriever_start").
        payload:    Arbitrary data specific to this event.

    Returns:
        Dict matching the AuditEvent Pydantic schema, ready for insertion
        into GraphState.audit_trail.
    """
    return {
        "event_id": str(uuid4()),
        "node": node,
        "event_type": event_type,
        "payload": payload,
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
