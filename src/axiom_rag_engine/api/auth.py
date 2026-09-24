"""
Axiom Engine — API key authentication.

Provides the ``verify_api_key`` FastAPI dependency for protecting endpoints.
Keys and the auth mode come from the *app's* settings (``AppServices.settings``)
when the request belongs to a running app, so apps built by ``create_app`` with
explicit settings authenticate against their own configuration. Outside an app
(direct calls, tests) the process settings are used.
"""

from __future__ import annotations

import functools
import hashlib
import hmac

from fastapi import HTTPException, Request, Security
from fastapi.security import APIKeyHeader

from axiom_rag_engine.config.settings import Settings, get_settings

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def settings_from_request(request: Request | None) -> Settings:
    """The settings governing ``request``: its app's, else the process settings."""
    app = request.scope.get("app") if request is not None else None
    services = getattr(getattr(app, "state", None), "services", None)
    if services is not None:
        return services.settings  # type: ignore[no-any-return]
    return get_settings()


def _api_keys(settings: Settings | None = None) -> set[str]:
    """Valid API keys from ``settings`` (process settings by default).

    Admin keys are valid API keys too, so an operator can use one key for both.
    """
    resolved = settings or get_settings()
    return {k for k in (*resolved.api_keys, *resolved.admin_api_keys) if k}


def _auth_required(settings: Settings | None = None) -> bool:
    """True unless the runtime env is explicitly a non-production alias."""
    return (settings or get_settings()).auth_required()


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
    matched.
    """
    presented_hash = hashlib.sha256(presented.encode()).hexdigest()
    hashed_valid = _pre_hashed_keys(tuple(sorted(valid_keys)))
    matched = False
    for h in hashed_valid:
        if hmac.compare_digest(presented_hash, h):
            matched = True
    return matched


def is_valid_api_key(presented: str, settings: Settings | None = None) -> bool:
    """Return True when ``presented`` matches a configured API key.

    Non-raising variant of the ``verify_api_key`` check, for callers that need
    a boolean (e.g. rate-limit bucketing) rather than an HTTP 401. Returns
    False when no keys are configured.
    """
    valid_keys = _api_keys(settings)
    if not valid_keys:
        return False
    return _hashed_key_check(presented, valid_keys)


def check_api_key(api_key: str | None, settings: Settings) -> str | None:
    """Validate ``api_key`` against ``settings``; raise HTTPException if refused.

    Returns the key (or None when auth is disabled).
    """
    valid_keys = _api_keys(settings)
    if not valid_keys:
        if _auth_required(settings):
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


async def verify_api_key(
    request: Request,
    api_key: str | None = Security(_api_key_header),
) -> str | None:
    """FastAPI dependency: validate the API key against the app's settings."""
    return check_api_key(api_key, settings_from_request(request))


async def verify_admin_key(
    request: Request,
    api_key: str | None = Security(_api_key_header),
) -> str | None:
    """FastAPI dependency for operations that change shared state (the corpus).

    Every tenant's answers are built from the one shared corpus, so letting any
    key ingest or delete documents let one tenant poison or erase what the
    others retrieve. When auth is required the key must be listed in
    AXIOM_ADMIN_API_KEYS (403 otherwise; refused outright if none are set).
    With auth disabled there are no tenants, so writes stay open.
    """
    settings = settings_from_request(request)
    key = check_api_key(api_key, settings)
    if not _auth_required(settings):
        return key
    admin_keys = {k for k in settings.admin_api_keys if k}
    if not admin_keys or not api_key or not _hashed_key_check(api_key, admin_keys):
        raise HTTPException(
            status_code=403,
            detail="This operation requires an admin API key (AXIOM_ADMIN_API_KEYS).",
        )
    return key
