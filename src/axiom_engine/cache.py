"""
Axiom Engine — Pluggable Caching Backend

Provides an abstract CacheBackend protocol with in-memory and Redis implementations.
Allows Axiom Engine to scale horizontally across multiple instances while
sharing response caches.
"""

import json
import logging
import threading
from typing import Any, Protocol, cast, runtime_checkable

from cachetools import TTLCache

try:
    from redis.exceptions import RedisError as _RedisError
except ImportError:  # redis is optional

    class _RedisError(Exception):  # type: ignore[no-redef]
        """Fallback when the redis package is not installed."""


RedisError: type[Exception] = _RedisError

logger = logging.getLogger("axiom_engine.cache")


@runtime_checkable
class CacheBackend(Protocol):
    def get(self, key: str) -> dict[str, Any] | None:
        """Retrieve a dictionary by key. Return None if missing or expired."""
        ...

    def set(self, key: str, value: dict[str, Any]) -> None:
        """Store a dictionary by key with the backend-configured TTL."""
        ...

    def clear(self) -> None:
        """Clear all entries from the cache. Used primarily for testing."""
        ...


class MemoryCacheBackend:
    """Process-local LRU cache."""

    def __init__(self, maxsize: int = 256, ttl_seconds: int = 300) -> None:
        self._cache: TTLCache[str, dict[str, Any]] = TTLCache(maxsize=maxsize, ttl=ttl_seconds)
        self._lock = threading.Lock()

    def get(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            return cast(dict[str, Any] | None, self._cache.get(key))

    def set(self, key: str, value: dict[str, Any]) -> None:
        with self._lock:
            self._cache[key] = value

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()


class RedisCacheBackend:
    """Distributed Redis cache with key-prefix namespacing.

    Keys are prefixed with ``axiom:cache:`` so the cache can safely share a
    Redis instance with other applications without risk of collision or
    accidental data loss.
    """

    _PREFIX = "axiom:cache:"

    def __init__(self, redis_url: str, ttl_seconds: int = 300) -> None:
        import redis  # Lazy import to avoid hard dependency

        self._redis = redis.Redis.from_url(redis_url, decode_responses=True)
        self.ttl = ttl_seconds

    def _prefixed(self, key: str) -> str:
        return f"{self._PREFIX}{key}"

    def get(self, key: str) -> dict[str, Any] | None:
        try:
            val = self._redis.get(self._prefixed(key))
            if val is not None:
                loaded = json.loads(val)
                if isinstance(loaded, dict):
                    return cast(dict[str, Any], loaded)
                logger.warning("Redis GET returned a non-dict payload for key %s", key)
        except (RedisError, json.JSONDecodeError) as exc:
            logger.warning("Redis GET failed for key %s: %s", key, exc)
        return None

    def set(self, key: str, value: dict[str, Any]) -> None:
        try:
            self._redis.setex(self._prefixed(key), self.ttl, json.dumps(value))
        except (RedisError, TypeError) as exc:
            logger.warning("Redis SET failed for key %s: %s", key, exc)

    def clear(self) -> None:
        """Delete only axiom:cache:* keys — never flushdb."""
        try:
            cursor: int | str = 0
            while True:
                cursor, keys = self._redis.scan(cursor=cursor, match=f"{self._PREFIX}*", count=100)
                if keys:
                    self._redis.delete(*keys)
                if cursor == 0:
                    break
        except RedisError as exc:
            logger.warning("Redis CLEAR failed: %s", exc)
