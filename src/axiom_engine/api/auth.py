"""
Axiom Engine — API key authentication.

Extracted from main.py to follow SRP. Provides the ``verify_api_key``
FastAPI dependency for protecting endpoints.
"""

from __future__ import annotations

import functools
import hashlib
import hmac
import os

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# H1: Fail-closed by default — auth is required unless AXIOM_ENV is explicitly set
# to a non-production value.  Forgetting to set AXIOM_ENV in prod no longer ships
# an open endpoint.
_NON_PROD_ENVS = {"development", "dev", "local", "test"}


def _app_env() -> str:
    """Return the current runtime environment (defaults to 'production')."""
    return os.environ.get("AXIOM_ENV", "production").strip().lower()


def _api_keys() -> set[str]:
    """Read valid API keys from the current environment."""
    return {k.strip() for k in os.environ.get("AXIOM_API_KEYS", "").split(",") if k.strip()}


def _auth_required() -> bool:
    """Return True unless AXIOM_ENV is explicitly set to a non-production value."""
    return _app_env() not in _NON_PROD_ENVS


@functools.cache
def _pre_hashed_keys(keys_tuple: tuple[str, ...]) -> frozenset[str]:
    """
    Return the pre-computed SHA-256 hex digests of all valid keys.

    Cached by key-tuple identity so the O(N) hash work happens once at the
    first auth check after startup (or after a key rotation), not once per
    request.  The ``@functools.cache`` LRU is keyed on the sorted tuple of
    raw key values; a key rotation produces a new tuple and a fresh hash set.
    """
    return frozenset(hashlib.sha256(k.encode()).hexdigest() for k in keys_tuple)


def _hashed_key_check(presented: str, valid_keys: set[str]) -> bool:
    """
    Constant-time API key verification.

    Compares the SHA-256 hash of the presented key against the pre-computed
    hashes of all valid keys using ``hmac.compare_digest``.  The loop is NOT
    short-circuited so timing does not leak which key (or how many characters)
    matched.  Pre-hashing valid keys at startup eliminates the O(N × hash)
    cost that existed when every request hashed every configured key.
    """
    presented_hash = hashlib.sha256(presented.encode()).hexdigest()
    hashed_valid = _pre_hashed_keys(tuple(sorted(valid_keys)))
    matched = False
    for h in hashed_valid:
        if hmac.compare_digest(presented_hash, h):
            matched = True
    return matched


async def verify_api_key(
    api_key: str | None = Security(_api_key_header),
) -> str | None:
    """Validate the API key if authentication is enabled."""
    valid_keys = _api_keys()
    if not valid_keys:
        if _auth_required():
            # Misconfigured production: auth is required but no keys are defined.
            # 503 (Service Unavailable) — the server is not ready to handle requests
            # until AXIOM_API_KEYS is configured. Clients and load-balancers that
            # retry on 503 will recover automatically once keys are supplied, whereas
            # 500 would be treated as a permanent crash by most orchestrators.
            raise HTTPException(status_code=503, detail="Server authentication is misconfigured.")
        return None
    if not api_key or not _hashed_key_check(api_key, valid_keys):
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")
    return api_key
