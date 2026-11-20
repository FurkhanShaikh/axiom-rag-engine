"""
Pure ASGI async rate limiting, proxy IP extraction, and CORS/body-size ordering.

Rate limits are shared across replicas via ``redis.asyncio`` when
``AXIOM_REDIS_URL`` is set (falling back to process memory on error or missing
``redis`` extra), enforce ``AXIOM_STREAM_RATE_LIMIT`` on ``/v1/synthesize/stream``,
extract the rightmost untrusted ``X-Forwarded-For`` entry, and sit inside
``CORSMiddleware`` so 429/413 responses carry CORS headers.
"""

from __future__ import annotations

import builtins
import sys
from typing import Any

import httpx
import pytest
from fastapi import Request
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

from axiom_rag_engine.api import rate_limit
from axiom_rag_engine.api.rate_limit import (
    RateLimiter,
    build_limiter,
    get_real_ip,
    parse_rate_limit,
)
from axiom_rag_engine.config.settings import Settings
from axiom_rag_engine.main import create_app
from axiom_rag_engine.nodes.retriever import MockSearchBackend


class _FakeAsyncRedis:
    def __init__(self, *, fail_incr: bool = False, fail_expire: bool = False) -> None:
        self.counts: dict[str, int] = {}
        self.expiry: dict[str, int] = {}
        self.fail_incr = fail_incr
        self.fail_expire = fail_expire
        self.closed = False

    async def incr(self, key: str) -> int:
        if self.fail_incr:
            raise ConnectionError("redis down")
        self.counts[key] = self.counts.get(key, 0) + 1
        return self.counts[key]

    async def expire(self, key: str, seconds: int) -> None:
        if self.fail_expire:
            raise ConnectionError("expire failed")
        self.expiry[key] = seconds

    async def aclose(self) -> None:
        self.closed = True


def _metric(name: str, **labels: str) -> float:
    return REGISTRY.get_sample_value(name, labels) or 0.0


def _make_request(
    headers: dict[str, str],
    *,
    client_ip: str = "198.51.100.1",
    trusted_proxy_ips: list[str] | None = None,
) -> Request:
    app = create_app(
        Settings(env="test", trusted_proxy_ips=trusted_proxy_ips or []),
        search_backend=MockSearchBackend([]),
    )
    scope: dict[str, Any] = {
        "type": "http",
        "method": "POST",
        "path": "/v1/synthesize",
        "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
        "client": (client_ip, 54321),
        "query_string": b"",
        "app": app,
    }
    return Request(scope)


class TestParseRateLimit:
    @pytest.mark.parametrize(
        ("spec", "expected"),
        [
            ("20/minute", (20, 60)),
            ("5 per second", (5, 1)),
            ("10/2 minutes", (10, 120)),
            ("100/hr", (100, 3600)),
            ("1000/day", (1000, 86400)),
        ],
    )
    def test_valid_specs(self, spec: str, expected: tuple[int, int]) -> None:
        assert parse_rate_limit(spec) == expected

    @pytest.mark.parametrize("spec", ["", "0/minute", "-1/second", "20/fortnight", "abc"])
    def test_invalid_specs_raise(self, spec: str) -> None:
        with pytest.raises(ValueError, match=r"Invalid rate limit|positive"):
            parse_rate_limit(spec)


class TestGetRealIp:
    def test_ignores_xff_from_untrusted_caller(self) -> None:
        req = _make_request(
            {"X-Forwarded-For": "1.2.3.4"},
            client_ip="198.51.100.9",
            trusted_proxy_ips=["10.0.0.1"],
        )
        assert get_real_ip(req) == "198.51.100.9"

    def test_single_trusted_proxy_uses_rightmost_untrusted_ip(self) -> None:
        req = _make_request(
            {"X-Forwarded-For": "spoofed-client, 203.0.113.55"},
            client_ip="10.0.0.1",
            trusted_proxy_ips=["10.0.0.1"],
        )
        assert get_real_ip(req) == "203.0.113.55"

    def test_multihop_trusted_cidr_walks_right_to_left(self) -> None:
        req = _make_request(
            {"X-Forwarded-For": "1.1.1.1, 203.0.113.88, 10.0.0.5, 10.2.3.4"},
            client_ip="10.0.0.9",
            trusted_proxy_ips=["10.0.0.0/8"],
        )
        assert get_real_ip(req) == "203.0.113.88"

    def test_wildcard_picks_rightmost_entry(self) -> None:
        req = _make_request(
            {"X-Forwarded-For": "1.2.3.4, 203.0.113.99"},
            client_ip="172.16.0.10",
            trusted_proxy_ips=["*"],
        )
        assert get_real_ip(req) == "203.0.113.99"


class TestSharedRedisLimiter:
    async def test_shares_counters_across_replicas_with_prefix_and_ttl(self) -> None:
        redis = _FakeAsyncRedis()
        limiter_a = build_limiter(Settings(env="test", rate_limit="2/minute"), redis_client=redis)
        limiter_b = build_limiter(Settings(env="test", rate_limit="2/minute"), redis_client=redis)

        assert (await limiter_a.allow("default", "ip:1.2.3.4", 2, 60, now=120.0))[0] is True
        assert (await limiter_b.allow("default", "ip:1.2.3.4", 2, 60, now=121.0))[0] is True
        allowed, retry_after = await limiter_a.allow("default", "ip:1.2.3.4", 2, 60, now=122.0)
        assert allowed is False
        assert 1 <= retry_after <= 60
        expected_key = "axiom:ratelimit:default:ip:1.2.3.4:2"
        assert redis.counts[expected_key] == 3
        assert redis.expiry[expected_key] == 61

    def test_without_redis_uses_memory_only(self) -> None:
        limiter = rate_limit.build_limiter(Settings(env="test"))
        assert isinstance(limiter, RateLimiter)
        assert limiter._redis is None

    def test_missing_redis_extra_falls_back_to_memory(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        real_import = builtins.__import__

        def _fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "redis.asyncio" or name.startswith("redis"):
                raise ImportError("No module named 'redis'")
            return real_import(name, *args, **kwargs)

        monkeypatch.delitem(sys.modules, "redis.asyncio", raising=False)
        monkeypatch.setattr(builtins, "__import__", _fake_import)
        limiter = rate_limit.build_limiter(
            Settings(env="test", AXIOM_REDIS_URL="redis://cache:6379/1")
        )
        assert isinstance(limiter, RateLimiter)
        assert limiter._redis is None
        assert "counting per process" in caplog.text

    async def test_runtime_redis_error_falls_back_to_memory_and_counts_metric(self) -> None:
        redis = _FakeAsyncRedis(fail_incr=True)
        limiter = build_limiter(Settings(env="test", rate_limit="1/minute"), redis_client=redis)
        before = _metric("axiom_cache_errors_total", op="ratelimit")

        assert (await limiter.allow("default", "ip:1.2.3.4", 1, 60, now=120.0))[0] is True
        assert (await limiter.allow("default", "ip:1.2.3.4", 1, 60, now=121.0))[0] is False
        assert _metric("axiom_cache_errors_total", op="ratelimit") - before == 2


class TestEndpointRateLimitsAndCors:
    def test_stream_rate_limit_is_enforced_independently(self) -> None:
        app = create_app(
            Settings(
                env="test",
                rate_limit="100/minute",
                stream_rate_limit="1/minute",
                semantic_verification_enabled=False,
            ),
            search_backend=MockSearchBackend([]),
        )
        payload = {
            "request_id": "s",
            "user_query": "solid-state batteries",
            "models": {"synthesizer": "mock/s", "verifier": "mock/v"},
        }
        with TestClient(app) as client:
            assert client.post("/v1/synthesize/stream", json=payload).status_code == 200
            second_stream = client.post("/v1/synthesize/stream", json=payload)
            assert second_stream.status_code == 429
            assert "Retry-After" in second_stream.headers
            # Non-stream endpoint uses its own bucket & limit (100/minute).
            assert client.post("/v1/synthesize", json=payload).status_code == 200

    def test_cors_headers_present_on_429_and_413(self) -> None:
        origin = "https://ui.example.com"
        app = create_app(
            Settings(
                env="test",
                rate_limit="1/minute",
                max_body_bytes=200,
                cors_origins=[origin],
            ),
            search_backend=MockSearchBackend([]),
        )
        with TestClient(app) as client:
            oversized = client.post(
                "/v1/synthesize",
                headers={"Origin": origin},
                json={"request_id": "r1", "user_query": "x" * 500},
            )
            assert oversized.status_code == 413
            assert oversized.headers.get("access-control-allow-origin") == origin

            # Rate-limit middleware sat outside body-size middleware, so the 413
            # consumed the 1/minute budget and the second request hits 429.
            limited = client.post(
                "/v1/synthesize",
                headers={"Origin": origin},
                json={"request_id": "r2", "user_query": "ok"},
            )
            assert limited.status_code == 429
            assert limited.headers.get("access-control-allow-origin") == origin

    async def test_chunked_oversized_body_returns_413_with_cors(self) -> None:
        origin = "https://ui.example.com"
        app = create_app(
            Settings(
                env="test",
                rate_limit="100/minute",
                max_body_bytes=64,
                cors_origins=[origin],
            ),
            search_backend=MockSearchBackend([]),
        )

        async def _chunks():
            yield b'{"request_id":"r","user_query":"'
            yield b"a" * 200
            yield b'"}'

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            resp = await client.post(
                "/v1/synthesize",
                headers={"Origin": origin, "Content-Type": "application/json"},
                content=_chunks(),
            )
        assert resp.status_code == 413
        assert resp.headers.get("access-control-allow-origin") == origin

    def test_spoofed_leftmost_xff_cannot_bypass_rate_limit(self) -> None:
        app = create_app(
            Settings(
                env="test",
                rate_limit="1/minute",
                trusted_proxy_ips=["testclient"],
                semantic_verification_enabled=False,
            ),
            search_backend=MockSearchBackend([]),
        )
        payload = {
            "request_id": "xff",
            "user_query": "solid-state batteries",
            "models": {"synthesizer": "mock/s", "verifier": "mock/v"},
        }
        with TestClient(app) as client:
            first = client.post(
                "/v1/synthesize",
                headers={"X-Forwarded-For": "1.1.1.1, 203.0.113.5"},
                json=payload,
            )
            second = client.post(
                "/v1/synthesize",
                headers={"X-Forwarded-For": "2.2.2.2, 203.0.113.5"},
                json=payload,
            )
        assert first.status_code == 200
        assert second.status_code == 429
