"""Audit-trail retention: persistence helper plus the /v1/audits endpoints."""

from __future__ import annotations

import hashlib
import logging
import time
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse, Response

from axiom_rag_engine.api.auth import verify_api_key
from axiom_rag_engine.api.deps import Services
from axiom_rag_engine.services import AppServices

logger = logging.getLogger("axiom_rag_engine")

router = APIRouter()


def audit_owner(api_key: str | None) -> str:
    """Tenant identity for audit retention: a hash of the caller's API key.

    ``""`` when auth is disabled (development) — a single shared namespace. The
    raw key never enters the store.
    """
    if not api_key:
        return ""
    return hashlib.sha256(api_key.encode()).hexdigest()[:32]


def persist_and_emit_audit(
    services: AppServices,
    request_id: str,
    status: str,
    graph_result: dict[str, Any],
    usage_snapshot: dict[str, Any] | None = None,
    owner: str = "",
) -> None:
    """Push the audit trail into the app's store and (optionally) the logs.

    Entries are stored under ``owner`` (see ``audit_owner``) so one tenant can
    never read or overwrite another tenant's trail. Both operations are
    best-effort: an audit failure must never poison the response path.
    """
    audit_trail = list(graph_result.get("audit_trail") or [])

    # Append a synthetic terminal event so downstream consumers (audit CLI,
    # log aggregators) see per-request cost without joining separate streams.
    if usage_snapshot:
        audit_trail.append(
            {
                "event_id": f"{request_id}-usage",
                "node": "engine",
                "event_type": "usage_summary",
                "timestamp_utc": datetime.now(UTC).isoformat(),
                "payload": usage_snapshot,
            }
        )

    store = services.audit_store
    if store.enabled:
        store.put(
            request_id,
            {
                "request_id": request_id,
                "status": status,
                "recorded_at": time.time(),
                "audit_trail": audit_trail,
            },
            owner=owner,
        )

    if services.settings.log_audit_events and audit_trail:
        for event in audit_trail:
            try:
                logger.info(
                    "audit_event",
                    extra={"axiom_audit": {"request_id": request_id, **event}},
                )
            except Exception:
                logger.exception("Failed to emit audit event for %s", request_id)


@router.get("/v1/audits", summary="List retained audit trail IDs.")
async def list_audits(
    services: Services,
    _api_key: str | None = Depends(verify_api_key),
) -> Response:
    """Return the caller's request IDs currently held in the audit store.

    Only trails produced with the caller's own API key are listed. Returns an
    empty list (not 404) when retention is disabled so UI clients can treat the
    response uniformly.
    """
    store = services.audit_store
    return JSONResponse(
        content={
            "retention_enabled": store.enabled,
            "capacity": store.capacity,
            "retained": len(store),
            "request_ids": store.list_ids(owner=audit_owner(_api_key)) if store.enabled else [],
        }
    )


@router.get("/v1/audits/{request_id}", summary="Retrieve the audit trail for a recent request.")
async def get_audit(
    services: Services,
    request_id: str,
    _api_key: str | None = Depends(verify_api_key),
) -> Response:
    """Return the caller's retained audit trail for ``request_id`` or 404.

    Retention is process-local and bounded by ``AXIOM_AUDIT_RETENTION``. Another
    tenant's trail is indistinguishable from a missing one (404).
    """
    store = services.audit_store
    if not store.enabled:
        return JSONResponse(
            status_code=404,
            content={
                "detail": (
                    "Audit retention is disabled. "
                    "Set AXIOM_AUDIT_RETENTION to a positive integer to enable."
                )
            },
        )
    entry = store.get(request_id, owner=audit_owner(_api_key))
    if entry is None:
        return JSONResponse(
            status_code=404,
            content={"detail": f"No audit trail retained for request_id={request_id!r}."},
        )
    return JSONResponse(content=entry)
