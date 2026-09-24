"""
Axiom Engine — per-API-key daily spend.

Rate limits cap requests, not money: one valid key could run up a large bill
(each request may make dozens of LLM calls). With ``AXIOM_KEY_DAILY_BUDGET_USD``
set, each key's LLM cost is summed per UTC day and a key at its cap is refused
(HTTP 429 with ``Retry-After`` until midnight UTC) before any model is called.

Spend is counted in Redis when the response cache uses Redis, so replicas share
one total; otherwise per process. A Redis error falls back to the in-process
ledger for that operation rather than failing the request. Cost comes from
``litellm.completion_cost``, so models LiteLLM cannot price (local Ollama,
unlisted providers) count as $0.
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

from axiom_rag_engine.config.observability import CACHE_ERRORS

logger = logging.getLogger("axiom_rag_engine.spend")

# Two days, so yesterday's total is still readable just after midnight.
_REDIS_TTL_SECONDS = 2 * 24 * 3600


def utc_day(now: float | None = None) -> str:
    return datetime.fromtimestamp(time.time() if now is None else now, UTC).strftime("%Y-%m-%d")


def seconds_until_utc_midnight(now: float | None = None) -> int:
    current = datetime.fromtimestamp(time.time() if now is None else now, UTC)
    midnight = (current + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return max(1, int((midnight - current).total_seconds()))


class SpendLedger(Protocol):
    async def spent(self, key_id: str, day: str) -> float: ...

    async def add(self, key_id: str, day: str, usd: float) -> None: ...


class MemorySpendLedger:
    """Per-process totals; only the current and previous day are kept."""

    def __init__(self) -> None:
        self._totals: dict[tuple[str, str], float] = {}

    async def spent(self, key_id: str, day: str) -> float:
        return self._totals.get((key_id, day), 0.0)

    async def add(self, key_id: str, day: str, usd: float) -> None:
        self._totals[(key_id, day)] = self._totals.get((key_id, day), 0.0) + usd
        if len(self._totals) > 1:
            keep = sorted({d for _, d in self._totals})[-2:]
            self._totals = {k: v for k, v in self._totals.items() if k[1] in keep}


class RedisSpendLedger:
    """Totals shared across replicas (``INCRBYFLOAT`` on ``axiom:spend:<day>:<key>``)."""

    _PREFIX = "axiom:spend:"

    def __init__(self, client: Any) -> None:
        self._redis = client
        self._fallback = MemorySpendLedger()

    def _key(self, key_id: str, day: str) -> str:
        return f"{self._PREFIX}{day}:{key_id}"

    async def spent(self, key_id: str, day: str) -> float:
        try:
            value = await self._redis.get(self._key(key_id, day))
            return float(value or 0.0) + await self._fallback.spent(key_id, day)
        except Exception as exc:
            CACHE_ERRORS.labels(op="spend_get").inc()
            logger.warning("Redis spend read failed (%s); using this process's total.", exc)
            return await self._fallback.spent(key_id, day)

    async def add(self, key_id: str, day: str, usd: float) -> None:
        try:
            key = self._key(key_id, day)
            await self._redis.incrbyfloat(key, usd)
            await self._redis.expire(key, _REDIS_TTL_SECONDS)
        except Exception as exc:
            CACHE_ERRORS.labels(op="spend_add").inc()
            logger.warning("Redis spend write failed (%s); counting in this process.", exc)
            await self._fallback.add(key_id, day, usd)
