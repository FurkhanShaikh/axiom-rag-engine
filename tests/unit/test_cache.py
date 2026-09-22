"""
Response cache backends — async interface, fail-soft Redis.

The Redis backend used a synchronous client on the event loop, so a slow Redis
stalled every request in the process (bounded only by a 1 s socket timeout per
call). It now uses ``redis.asyncio``; failures still degrade to a cache miss.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from axiom_rag_engine.cache import MemoryCacheBackend, RedisCacheBackend, RedisError


class _FakeAsyncRedis:
    """Minimal async stand-in for ``redis.asyncio.Redis``."""

    def __init__(self, fail: bool = False) -> None:
        self.store: dict[str, str] = {}
        self.ttls: dict[str, int] = {}
        self.fail = fail

    def _maybe_fail(self) -> None:
        if self.fail:
            raise RedisError("connection refused")

    async def get(self, key: str) -> str | None:
        self._maybe_fail()
        return self.store.get(key)

    async def setex(self, key: str, ttl: int, value: str) -> None:
        self._maybe_fail()
        self.store[key] = value
        self.ttls[key] = ttl

    async def scan(self, cursor: int = 0, match: str = "*", count: int = 100) -> tuple[int, list]:
        self._maybe_fail()
        prefix = match.rstrip("*")
        return 0, [k for k in self.store if k.startswith(prefix)]

    async def delete(self, *keys: str) -> None:
        self._maybe_fail()
        for k in keys:
            self.store.pop(k, None)

    async def aclose(self) -> None:
        return None


class TestMemoryCache:
    async def test_round_trip(self) -> None:
        cache = MemoryCacheBackend(maxsize=4, ttl_seconds=60)
        await cache.set("k", {"a": 1})
        assert await cache.get("k") == {"a": 1}

    async def test_miss_is_none(self) -> None:
        assert await MemoryCacheBackend().get("nope") is None

    async def test_clear(self) -> None:
        cache = MemoryCacheBackend()
        await cache.set("k", {"a": 1})
        await cache.clear()
        assert await cache.get("k") is None


class TestRedisCache:
    async def test_round_trip_is_prefixed_and_ttl_bound(self) -> None:
        fake = _FakeAsyncRedis()
        cache = RedisCacheBackend(client=fake, ttl_seconds=42)
        await cache.set("k", {"a": 1})
        assert fake.store == {"axiom:cache:k": json.dumps({"a": 1})}
        assert fake.ttls["axiom:cache:k"] == 42
        assert await cache.get("k") == {"a": 1}

    @pytest.mark.parametrize("op", ["get", "set", "clear"])
    async def test_redis_errors_degrade_to_miss(self, op: str) -> None:
        cache = RedisCacheBackend(client=_FakeAsyncRedis(fail=True))
        args: dict[str, Any] = {"get": ("k",), "set": ("k", {"a": 1}), "clear": ()}
        result = await getattr(cache, op)(*args[op])  # must not raise
        assert result is None

    async def test_non_dict_payload_is_a_miss(self) -> None:
        fake = _FakeAsyncRedis()
        fake.store["axiom:cache:k"] = json.dumps([1, 2])
        assert await RedisCacheBackend(client=fake).get("k") is None

    async def test_clear_only_touches_axiom_keys(self) -> None:
        fake = _FakeAsyncRedis()
        fake.store.update({"axiom:cache:a": "{}", "other-app:b": "{}"})
        await RedisCacheBackend(client=fake).clear()
        assert fake.store == {"other-app:b": "{}"}

    async def test_client_is_async(self) -> None:
        pytest.importorskip("redis")
        cache = RedisCacheBackend(redis_url="redis://localhost:6399/0")
        assert type(cache._redis).__module__.startswith("redis.asyncio")
        await cache.aclose()
