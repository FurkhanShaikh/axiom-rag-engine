"""
Axiom Engine — pure ASGI rate limiting.

One ``RateLimiter`` per app (``build_limiter``), keyed by a hashed valid API key
or, failing that, the real client IP. Keys and trusted proxies are read from the
request's app settings, so apps built with explicit settings rate-limit by their
own configuration without touching global state or blocking the event loop.

Counters live in Redis via ``redis.asyncio`` when ``AXIOM_REDIS_URL`` is set,
so replicas share one limit instead of each allowing the full rate; otherwise
they are kept in process memory.
"""

from __future__ import annotations

import hashlib
import ipaddress
import logging
import re
import time
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from axiom_rag_engine.api.auth import is_valid_api_key, settings_from_request
from axiom_rag_engine.config.observability import CACHE_ERRORS, RATE_LIMIT_REJECTIONS
from axiom_rag_engine.config.settings import Settings

logger = logging.getLogger("axiom_rag_engine.rate_limit")

_UNIT_SECONDS = {
    "s": 1,
    "m": 60,
    "h": 3600,
    "d": 86400,
}

_RATE_LIMIT_RE = re.compile(
    r"^\s*(\d+)\s*(?:/|per)\s*(?:(\d+)\s*)?(s|sec|second|seconds|m|min|minute|minutes|h|hr|hour|hours|d|day|days)\s*$",
    re.IGNORECASE,
)

# Bounded fallback table size before expired buckets are swept.
_MEMORY_PRUNE_THRESHOLD = 4096


def parse_rate_limit(spec: str) -> tuple[int, int]:
    """Parse a rate-limit string like ``'20/minute'`` into ``(limit, window_seconds)``."""
    match = _RATE_LIMIT_RE.fullmatch(spec)
    if match is None:
        raise ValueError(f"Invalid rate limit specification: {spec!r}")
    limit = int(match.group(1))
    multiplier = int(match.group(2) or "1")
    unit = match.group(3)[0].lower()
    if limit < 1 or multiplier < 1:
        raise ValueError(f"Rate limit and window must be positive integers, got: {spec!r}")
    return limit, multiplier * _UNIT_SECONDS[unit]


def get_remote_address(request: Request) -> str:
    """Return the direct peer IP address for ``request``."""
    if request.client and request.client.host:
        return request.client.host
    return "127.0.0.1"


def _is_trusted_ip(ip: str, trusted: set[str]) -> bool:
    """Return True if ``ip`` matches an exact IP or CIDR range in ``trusted``."""
    if ip in trusted:
        return True
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False
    for entry in trusted:
        try:
            if "/" in entry:
                if addr in ipaddress.ip_network(entry, strict=False):
                    return True
            elif addr == ipaddress.ip_address(entry):
                return True
        except ValueError:
            continue
    return False


def get_real_ip(request: Request) -> str:
    """Extract client IP handling proxy ``X-Forwarded-For`` headers safely.

    ``X-Forwarded-For`` is trusted only when the immediate caller is in
    ``AXIOM_TRUSTED_PROXY_IPS`` (exact IPs, CIDRs, or ``*``). Walks right-to-left
    past trusted proxy hops and returns the rightmost untrusted IP so clients
    cannot spoof the leftmost entry to bypass IP rate limits.
    """
    remote_ip = get_remote_address(request)
    forwarded = request.headers.get("X-Forwarded-For")
    trusted_proxies = {
        p.strip() for p in settings_from_request(request).trusted_proxy_ips if p.strip()
    }
    if not forwarded or not trusted_proxies:
        return remote_ip

    wildcard = "*" in trusted_proxies
    explicit = trusted_proxies - {"*"}
    if not wildcard and not _is_trusted_ip(remote_ip, explicit):
        return remote_ip

    entries = [entry.strip() for entry in forwarded.split(",") if entry.strip()]
    if not entries:
        return remote_ip
    if wildcard and not explicit:
        return entries[-1]
    for entry in reversed(entries):
        if not _is_trusted_ip(entry, explicit):
            return entry
    return entries[0]


def rate_limit_key(request: Request) -> str:
    """Rate-limit bucket.

    Prefer a hashed API key so one key shared across many IPs still hits a
    single bucket (the IP-based key let an attacker bypass limits by spraying
    source addresses). Only *valid* keys get a key bucket: bucketing on the raw
    header would let a client mint a fresh bucket per request by rotating random
    ``X-API-Key`` values, bypassing the IP limit entirely. Invalid or missing
    keys fall back to the real client IP.
    """
    api_key = request.headers.get("X-API-Key")
    if api_key and is_valid_api_key(api_key, settings_from_request(request)):
        return "key:" + hashlib.sha256(api_key.encode()).hexdigest()[:32]
    return "ip:" + get_real_ip(request)


class RateLimiter:
    """Fixed-window async rate limiter backed by ``redis.asyncio`` or memory."""

    _PREFIX = "axiom:ratelimit:"

    def __init__(
        self,
        default_limit: str,
        *,
        route_limits: dict[str, str] | None = None,
        exempt_paths: frozenset[str] = frozenset(),
        redis_client: Any = None,
        owns_redis_client: bool = False,
    ) -> None:
        self.default_spec = default_limit
        self.default_parsed = parse_rate_limit(default_limit)
        self.route_limits: dict[str, tuple[str, int, int]] = {
            path: (spec, *parse_rate_limit(spec)) for path, spec in (route_limits or {}).items()
        }
        self.exempt_paths = exempt_paths
        self._redis = redis_client
        self._owns_redis = owns_redis_client
        # (scope_name, bucket) -> (window_id, expires_at_ts, count)
        self._counts: dict[tuple[str, str], tuple[int, int, int]] = {}

    def limit_for_path(self, path: str) -> tuple[str, str, int, int] | None:
        """Return ``(scope_name, spec, limit, window_seconds)`` or None if exempt."""
        if path in self.exempt_paths:
            return None
        override = self.route_limits.get(path)
        if override is not None:
            spec, limit, window_seconds = override
            return path, spec, limit, window_seconds
        limit, window_seconds = self.default_parsed
        return "default", self.default_spec, limit, window_seconds

    def _allow_memory(
        self,
        scope_name: str,
        bucket: str,
        limit: int,
        window_seconds: int,
        ts: float,
    ) -> tuple[bool, int]:
        window_id = int(ts // window_seconds)
        expires_at = (window_id + 1) * window_seconds
        retry_after = max(1, expires_at - int(ts))
        key = (scope_name, bucket)
        prev = self._counts.get(key)
        count = (prev[2] + 1) if (prev is not None and prev[0] == window_id) else 1
        self._counts[key] = (window_id, expires_at, count)
        if len(self._counts) >= _MEMORY_PRUNE_THRESHOLD:
            self._counts = {k: v for k, v in self._counts.items() if v[1] > ts}
        return count <= limit, retry_after

    async def allow(
        self,
        scope_name: str,
        bucket: str,
        limit: int,
        window_seconds: int,
        *,
        now: float | None = None,
    ) -> tuple[bool, int]:
        """Consume one request token and return ``(allowed, retry_after_seconds)``."""
        ts = time.time() if now is None else now
        window_id = int(ts // window_seconds)
        expires_at = (window_id + 1) * window_seconds
        retry_after = max(1, expires_at - int(ts))

        if self._redis is not None:
            redis_key = f"{self._PREFIX}{scope_name}:{bucket}:{window_id}"
            try:
                count = int(await self._redis.incr(redis_key))
            except Exception as exc:
                CACHE_ERRORS.labels(op="ratelimit").inc()
                logger.warning("Redis rate limit failed (%s); falling back to memory.", exc)
            else:
                if count == 1:
                    try:
                        await self._redis.expire(redis_key, window_seconds + 1)
                    except Exception as exc:
                        CACHE_ERRORS.labels(op="ratelimit").inc()
                        logger.warning("Redis rate limit expire failed (%s).", exc)
                return count <= limit, retry_after

        return self._allow_memory(scope_name, bucket, limit, window_seconds, ts)

    def reset(self) -> None:
        """Clear in-process rate-limit counters (used between tests)."""
        self._counts.clear()

    async def aclose(self) -> None:
        """Close the owned ``redis.asyncio`` client, if any."""
        if self._redis is not None and self._owns_redis:
            try:
                await self._redis.aclose()
            except Exception as exc:
                logger.warning("Redis rate-limiter close failed: %s", exc)


def build_limiter(
    settings: Settings,
    *,
    exempt_paths: frozenset[str] = frozenset(),
    route_limits: dict[str, str] | None = None,
    redis_client: Any = None,
) -> RateLimiter:
    """Build a per-app :class:`RateLimiter`.

    With ``AXIOM_REDIS_URL`` the counters are shared through ``redis.asyncio``;
    if Redis is unreachable at request time the limiter falls back to in-memory
    counters rather than failing requests or blocking the event loop.
    """
    resolved_routes = (
        route_limits
        if route_limits is not None
        else {"/v1/synthesize/stream": settings.stream_rate_limit}
    )
    if redis_client is not None:
        return RateLimiter(
            settings.rate_limit,
            route_limits=resolved_routes,
            exempt_paths=exempt_paths,
            redis_client=redis_client,
            owns_redis_client=False,
        )
    if settings.redis_url:
        try:
            import redis.asyncio as redis_asyncio  # Lazy import: redis is optional

            client = redis_asyncio.Redis.from_url(
                settings.redis_url,
                decode_responses=True,
                socket_connect_timeout=0.5,
                socket_timeout=0.5,
            )
            logger.info("Rate limits: shared through Redis.")
            return RateLimiter(
                settings.rate_limit,
                route_limits=resolved_routes,
                exempt_paths=exempt_paths,
                redis_client=client,
                owns_redis_client=True,
            )
        except Exception as exc:
            logger.warning(
                "AXIOM_REDIS_URL is set but rate limits cannot use it (%s); counting "
                "per process. Install the 'redis' extra to share limits across replicas.",
                exc,
            )
    return RateLimiter(
        settings.rate_limit,
        route_limits=resolved_routes,
        exempt_paths=exempt_paths,
    )


class RateLimitMiddleware:
    """Pure ASGI rate-limiting middleware (safe for SSE streams and CORS errors)."""

    def __init__(self, app: ASGIApp, *, limiter: RateLimiter) -> None:
        self.app = app
        self.limiter = limiter

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        limit_info = self.limiter.limit_for_path(scope.get("path", ""))
        if limit_info is None:
            await self.app(scope, receive, send)
            return

        scope_name, spec, limit, window_seconds = limit_info
        bucket = rate_limit_key(Request(scope))
        allowed, retry_after = await self.limiter.allow(scope_name, bucket, limit, window_seconds)
        if not allowed:
            RATE_LIMIT_REJECTIONS.inc()
            response = JSONResponse(
                status_code=429,
                content={"error": f"Rate limit exceeded: {spec}"},
                headers={"Retry-After": str(retry_after)},
            )
            await response(scope, receive, send)
            return

        await self.app(scope, receive, send)
