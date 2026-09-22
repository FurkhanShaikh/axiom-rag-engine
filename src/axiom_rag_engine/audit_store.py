"""
Axiom Engine — In-memory audit trail retention.

A small, bounded ring-buffer keyed by ``request_id``. Populated by the
synthesize endpoint after the graph finishes; read by the
``GET /v1/audits/{request_id}`` endpoint and the ``axiom-rag-engine audit``
CLI command.

The store is **process-local** — it does not survive restarts and is not
shared across workers. Operators who need durable audit history should pipe
the structured JSON logs (``AXIOM_LOG_AUDIT_EVENTS=true``) into their log
aggregator.

Retention size is controlled by ``AXIOM_AUDIT_RETENTION``. A value of 0
disables retention entirely and makes :meth:`AuditStore.put` a no-op.

Tenancy: every entry is owned by the caller that produced it (``owner`` — a
hash of the API key; ``""`` when auth is disabled). Reads and listings only see
the caller's own entries, and because ``request_id`` is client-chosen, two
tenants using the same id get two separate entries rather than overwriting
each other.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any


class AuditStore:
    """Thread-safe bounded FIFO of recent audit trails."""

    def __init__(self, maxsize: int) -> None:
        self._maxsize = max(0, int(maxsize))
        self._lock = threading.Lock()
        self._data: OrderedDict[tuple[str, str], dict[str, Any]] = OrderedDict()

    @property
    def enabled(self) -> bool:
        return self._maxsize > 0

    @property
    def capacity(self) -> int:
        return self._maxsize

    def put(self, request_id: str, entry: dict[str, Any], owner: str = "") -> None:
        """Record ``owner``'s audit entry. No-op when retention is disabled."""
        if not self.enabled or not request_id:
            return
        key = (owner, request_id)
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
            self._data[key] = entry
            while len(self._data) > self._maxsize:
                self._data.popitem(last=False)

    def get(self, request_id: str, owner: str = "") -> dict[str, Any] | None:
        """Return ``owner``'s entry for ``request_id`` (never another owner's)."""
        with self._lock:
            entry = self._data.get((owner, request_id))
            return None if entry is None else dict(entry)

    def list_ids(self, owner: str = "") -> list[str]:
        """Return ``owner``'s retained request IDs, most-recent last."""
        with self._lock:
            return [rid for (entry_owner, rid) in self._data if entry_owner == owner]

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)
