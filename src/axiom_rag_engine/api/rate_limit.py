"""
Axiom Engine — rate limiting.

One slowapi ``Limiter`` per app (``build_limiter``), keyed by a hashed valid API
key or, failing that, the real client IP. Keys and trusted proxies are read from
the request's app settings, so apps built with explicit settings rate-limit by
their own configuration.

Counters live in Redis when ``AXIOM_REDIS_URL`` is set, so replicas share one
limit instead of each allowing the full rate; otherwise they are per process.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from axiom_rag_engine.api.auth import is_valid_api_key, settings_from_request
from axiom_rag_engine.config.settings import Settings

logger = logging.getLogger("axiom_rag_engine.rate_limit")

# The limiter checks Redis synchronously on the request path, so a slow Redis
# must fail fast (and fall back to in-memory counters) rather than stall it.
# (slowapi annotates the options as str, but they reach redis-py as-is, which
# needs numbers.)
_REDIS_OPTIONS: dict[str, Any] = {"socket_timeout": 0.5, "socket_connect_timeout": 0.5}


def get_real_ip(request: Request) -> str:
    """Extract client IP handling proxy X-Forwarded-For headers.

    X-Forwarded-For is trusted only when the immediate caller is in
    AXIOM_TRUSTED_PROXY_IPS — clients can spoof the header otherwise.
    """
    forwarded = request.headers.get("X-Forwarded-For")
    trusted_proxies = set(settings_from_request(request).trusted_proxy_ips)
    if forwarded and trusted_proxies:
        remote_ip = get_remote_address(request)
        if "*" in trusted_proxies or remote_ip in trusted_proxies:
            # Some proxies add spaces after commas — strip all entries.
            return forwarded.split(",")[0].strip()
    return get_remote_address(request)


def rate_limit_key(request: Request) -> str:
    """
    Rate-limit bucket.

    Prefer a hashed API key so one key shared across many IPs still hits a
    single bucket (the IP-based key let an attacker bypass limits by spraying
    source addresses). Only *valid* keys get a key bucket: bucketing on the raw
    header would let a client mint a fresh bucket per request by rotating random
    X-API-Key values, bypassing the IP limit entirely. Invalid or missing keys
    fall back to the real client IP.
    """
    api_key = request.headers.get("X-API-Key")
    if api_key and is_valid_api_key(api_key, settings_from_request(request)):
        return "key:" + hashlib.sha256(api_key.encode()).hexdigest()[:32]
    return "ip:" + get_real_ip(request)


def build_limiter(settings: Settings) -> Limiter:
    """A fresh limiter with the default limit.

    With ``AXIOM_REDIS_URL`` the counters are shared through Redis; if Redis is
    unreachable at request time the limiter falls back to in-memory counters
    (per process) until it recovers, rather than failing requests. Without it,
    or if the redis package is missing, counters are in memory.
    """
    if settings.redis_url:
        try:
            limiter = Limiter(
                key_func=rate_limit_key,
                default_limits=[settings.rate_limit],
                storage_uri=settings.redis_url,
                storage_options=_REDIS_OPTIONS,
                key_prefix="axiom:ratelimit",
                in_memory_fallback_enabled=True,
            )
            logger.info("Rate limits: shared through Redis.")
            return limiter
        except Exception as exc:  # e.g. limits' ConfigurationError: redis not installed
            logger.warning(
                "AXIOM_REDIS_URL is set but rate limits cannot use it (%s); counting "
                "per process. Install the 'redis' extra to share limits across replicas.",
                exc,
            )
    return Limiter(key_func=rate_limit_key, default_limits=[settings.rate_limit])
