"""
Axiom Engine — Pluggable Caching Backend

Provides an async CacheBackend protocol with in-memory and Redis implementations.
Allows Axiom Engine to scale horizontally across multiple instances while
sharing response caches.

Both backends are awaited from request handlers on the event loop, so neither
may block it: the in-memory backend is pure CPU, and the Redis backend uses the
``redis.asyncio`` client. Any Redis failure degrades to a cache miss — the cache
is an optimization, never a dependency.
"""

import json
import logging
import threading
from typing import Any, Protocol, cast, runtime_checkable

from cachetools import TTLCache

from axiom_rag_engine.config.observability import CACHE_ERRORS

try:
    from redis.exceptions import RedisError as _RedisError
except ImportError:  # redis is optional

    class _RedisError(Exception):  # type: ignore[no-redef]
        """Fallback when the redis package is not installed."""


RedisError: type[Exception] = _RedisError

logger = logging.getLogger("axiom_rag_engine.cache")


@runtime_checkable
class CacheBackend(Protocol):
    async def get(self, key: str) -> dict[str, Any] | None:
        """Retrieve a dictionary by key. Return None if missing or expired."""
        ...

    async def set(self, key: str, value: dict[str, Any]) -> None:
        """Store a dictionary by key with the backend-configured TTL."""
        ...

    async def clear(self) -> None:
        """Clear all entries from the cache. Used primarily for testing."""
        ...

    async def aclose(self) -> None:
        """Release backend resources (connections). Called at shutdown."""
        ...

    async def ping(self) -> bool:
        """Whether the backend is reachable. Never raises (readiness probes)."""
        ...


class MemoryCacheBackend:
    """Process-local TTL + LRU cache."""

    def __init__(self, maxsize: int = 256, ttl_seconds: int = 300) -> None:
        self._cache: TTLCache[str, dict[str, Any]] = TTLCache(maxsize=maxsize, ttl=ttl_seconds)
        self._lock = threading.Lock()

    async def get(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            return cast(dict[str, Any] | None, self._cache.get(key))

    async def set(self, key: str, value: dict[str, Any]) -> None:
        with self._lock:
            self._cache[key] = value

    async def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    async def aclose(self) -> None:
        return None

    async def ping(self) -> bool:
        return True


class RedisCacheBackend:
    """Distributed Redis cache with key-prefix namespacing.

    Keys are prefixed with ``axiom:cache:`` so the cache can safely share a
    Redis instance with other applications without risk of collision or
    accidental data loss.

    Args:
        redis_url: Connection URL for a new ``redis.asyncio`` client.
        ttl_seconds: Expiry applied to every entry.
        client: An existing async client (tests inject a fake). Takes
            precedence over ``redis_url``.
    """

    _PREFIX = "axiom:cache:"

    def __init__(
        self,
        redis_url: str | None = None,
        ttl_seconds: int = 300,
        client: Any = None,
    ) -> None:
        if client is None:
            if not redis_url:
                raise ValueError("RedisCacheBackend needs redis_url or client")
            import redis.asyncio as redis_asyncio  # Lazy import: redis is optional

            # Short socket timeouts: a hung Redis must fail fast into a cache
            # miss rather than hold a request open.
            client = redis_asyncio.Redis.from_url(
                redis_url,
                decode_responses=True,
                socket_connect_timeout=1.0,
                socket_timeout=1.0,
            )
        self._redis = client
        self.ttl = ttl_seconds

    def _prefixed(self, key: str) -> str:
        return f"{self._PREFIX}{key}"

    async def get(self, key: str) -> dict[str, Any] | None:
        try:
            val = await self._redis.get(self._prefixed(key))
            if val is not None:
                loaded = json.loads(val)
                if isinstance(loaded, dict):
                    return cast(dict[str, Any], loaded)
                logger.warning("Redis GET returned a non-dict payload for key %s", key)
        except (RedisError, OSError, json.JSONDecodeError) as exc:
            CACHE_ERRORS.labels(op="get").inc()
            logger.warning("Redis GET failed for key %s: %s", key, exc)
        return None

    async def set(self, key: str, value: dict[str, Any]) -> None:
        try:
            await self._redis.setex(self._prefixed(key), self.ttl, json.dumps(value))
        except (RedisError, OSError, TypeError) as exc:
            CACHE_ERRORS.labels(op="set").inc()
            logger.warning("Redis SET failed for key %s: %s", key, exc)

    async def clear(self) -> None:
        """Delete only axiom:cache:* keys — never flushdb."""
        try:
            cursor: int | str = 0
            while True:
                cursor, keys = await self._redis.scan(
                    cursor=cursor, match=f"{self._PREFIX}*", count=100
                )
                if keys:
                    await self._redis.delete(*keys)
                if cursor == 0:
                    break
        except (RedisError, OSError) as exc:
            CACHE_ERRORS.labels(op="clear").inc()
            logger.warning("Redis CLEAR failed: %s", exc)

    async def ping(self) -> bool:
        try:
            return bool(await self._redis.ping())
        except Exception as exc:  # any failure means "not reachable" to a probe
            CACHE_ERRORS.labels(op="ping").inc()
            logger.warning("Redis PING failed: %s", exc)
            return False

    async def aclose(self) -> None:
        try:
            await self._redis.aclose()
        except (RedisError, OSError, AttributeError) as exc:
            logger.warning("Redis close failed: %s", exc)
