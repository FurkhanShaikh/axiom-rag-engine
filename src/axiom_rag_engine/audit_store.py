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
        self._data: OrderedDict[str, dict[str, Any]] = OrderedDict()

    @property
    def enabled(self) -> bool:
        return self._maxsize > 0

    @property
    def capacity(self) -> int:
        return self._maxsize

    def put(self, request_id: str, entry: dict[str, Any]) -> None:
        """Record an audit entry. No-op when retention is disabled."""
        if not self.enabled or not request_id:
            return
        with self._lock:
            if request_id in self._data:
                self._data.move_to_end(request_id)
            self._data[request_id] = entry
            while len(self._data) > self._maxsize:
                self._data.popitem(last=False)

    def get(self, request_id: str) -> dict[str, Any] | None:
        with self._lock:
            entry = self._data.get(request_id)
            return None if entry is None else dict(entry)

    def list_ids(self) -> list[str]:
        """Return retained request IDs, most-recent last."""
        with self._lock:
            return list(self._data.keys())

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)
