"""
Rate limits are shared across replicas when Redis is configured.

slowapi kept its counters in process memory, so N replicas allowed N times the
configured rate. With ``AXIOM_REDIS_URL`` the counters now live in Redis
(falling back to memory while Redis is unreachable); without it, or without
the redis package, they stay per process (DEP-1).
"""

from __future__ import annotations

from typing import Any

import pytest
from slowapi import Limiter

from axiom_rag_engine.api import rate_limit
from axiom_rag_engine.config.settings import Settings


class _Recorder:
    def __init__(self, fail_on_storage: bool = False) -> None:
        self.calls: list[dict[str, Any]] = []
        self.fail_on_storage = fail_on_storage

    def __call__(self, **kwargs: Any) -> Limiter:
        self.calls.append(kwargs)
        if self.fail_on_storage and "storage_uri" in kwargs:
            raise RuntimeError("'redis' prerequisite not available")
        return Limiter(key_func=kwargs["key_func"], default_limits=kwargs["default_limits"])


def test_redis_url_shares_the_counters(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _Recorder()
    monkeypatch.setattr(rate_limit, "Limiter", recorder)
    rate_limit.build_limiter(Settings(env="test", AXIOM_REDIS_URL="redis://cache:6379/1"))
    (kwargs,) = recorder.calls
    assert kwargs["storage_uri"] == "redis://cache:6379/1"
    assert kwargs["in_memory_fallback_enabled"] is True  # Redis outage != failed requests
    assert kwargs["key_prefix"] == "axiom:ratelimit"
    assert float(kwargs["storage_options"]["socket_timeout"]) <= 1.0


def test_without_redis_counters_are_in_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _Recorder()
    monkeypatch.setattr(rate_limit, "Limiter", recorder)
    rate_limit.build_limiter(Settings(env="test"))
    (kwargs,) = recorder.calls
    assert "storage_uri" not in kwargs


def test_unusable_redis_falls_back_to_memory(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    recorder = _Recorder(fail_on_storage=True)
    monkeypatch.setattr(rate_limit, "Limiter", recorder)
    limiter = rate_limit.build_limiter(Settings(env="test", AXIOM_REDIS_URL="redis://cache:6379"))
    assert isinstance(limiter, Limiter)
    assert [("storage_uri" in c) for c in recorder.calls] == [True, False]
    assert "counting per process" in caplog.text
