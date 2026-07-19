"""``axiom-rag-engine audit <request_id>`` — fetch a retained audit trail."""

from __future__ import annotations

import json
import os
import sys
from typing import Any

import httpx

_DEFAULT_SERVER = "http://localhost:8000"


def run_audit(
    request_id: str,
    server_url: str = _DEFAULT_SERVER,
    api_key: str | None = None,
    pretty: bool = True,
) -> int:
    """Fetch ``/v1/audits/{request_id}`` and render it to stdout."""
    headers: dict[str, str] = {}
    key = api_key or os.environ.get("AXIOM_API_KEY")
    if key:
        headers["X-API-Key"] = key

    url = f"{server_url.rstrip('/')}/v1/audits/{request_id}"
    try:
        response = httpx.get(url, headers=headers, timeout=10.0)
    except httpx.HTTPError as exc:
        sys.stderr.write(f"ERROR: failed to reach {url}: {exc}\n")
        return 2

    if response.status_code == 404:
        sys.stderr.write(f"No audit trail found for request_id={request_id!r}.\n")
        try:
            detail = response.json().get("detail")
        except Exception:
            detail = None
        if detail:
            sys.stderr.write(f"  {detail}\n")
        return 1
    if response.status_code == 401:
        sys.stderr.write("Authentication failed. Set AXIOM_API_KEY or pass --api-key.\n")
        return 1
    if response.status_code >= 400:
        sys.stderr.write(f"ERROR: server returned {response.status_code}: {response.text}\n")
        return 2

    data: dict[str, Any] = response.json()
    if pretty:
        sys.stdout.write(_render_human(data))
    else:
        sys.stdout.write(json.dumps(data, indent=2, default=str) + "\n")
    return 0


def _render_human(entry: dict[str, Any]) -> str:
    """Pretty-print an audit entry as a human-readable event log."""
    lines: list[str] = []
    lines.append(f"Request ID : {entry.get('request_id')}")
    lines.append(f"Status     : {entry.get('status')}")
    recorded_at = entry.get("recorded_at")
    if recorded_at is not None:
        lines.append(f"Recorded at: {recorded_at}")
    lines.append("-" * 60)
    events = entry.get("audit_trail") or []
    if not events:
        lines.append("(no audit events recorded)")
    for i, event in enumerate(events, 1):
        node = event.get("node", "?")
        etype = event.get("event_type", "?")
        ts = event.get("timestamp_utc", "")
        lines.append(f"[{i:03d}] {ts}  {node:>16}  {etype}")
        payload = event.get("payload") or {}
        if payload:
            for k, v in payload.items():
                rendered = json.dumps(v, default=str) if not isinstance(v, str) else v
                lines.append(f"        {k} = {rendered}")
    return "\n".join(lines) + "\n"
