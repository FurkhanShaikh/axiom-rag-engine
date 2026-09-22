"""
Axiom Engine — rate limiting.

One slowapi ``Limiter`` per app (``build_limiter``), keyed by a hashed valid API
key or, failing that, the real client IP. Keys and trusted proxies are read from
the request's app settings, so apps built with explicit settings rate-limit by
their own configuration.
"""

from __future__ import annotations

import hashlib

from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from axiom_rag_engine.api.auth import is_valid_api_key, settings_from_request
from axiom_rag_engine.config.settings import Settings


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
    """A fresh limiter (its own in-memory counters) with the default limit."""
    return Limiter(key_func=rate_limit_key, default_limits=[settings.rate_limit])
