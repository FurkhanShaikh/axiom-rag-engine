"""
Readiness checks the app's dependencies, not just its configuration.

``/health/ready`` returned 200 whenever configuration looked right, so a pod
whose corpus database could not be read kept receiving traffic that failed. The
corpus is now a hard dependency (503); the response cache is optional by design
(failures degrade to misses), so an unreachable Redis reports ``degraded`` but
stays ready. Checks are cached briefly so probes do not hammer the backends.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from axiom_rag_engine.api.routes import ops
from axiom_rag_engine.config.settings import Settings
from axiom_rag_engine.main import create_app
from axiom_rag_engine.nodes.retriever import MockSearchBackend


def _client(tmp_path: Path, **overrides: Any) -> TestClient:
    settings = Settings(env="test", corpus_db_path=str(tmp_path / "corpus.db"), **overrides)
    return TestClient(create_app(settings, search_backend=MockSearchBackend([])))


class _Pings:
    def __init__(self, reachable: bool) -> None:
        self.reachable = reachable
        self.calls = 0

    async def __call__(self) -> bool:
        self.calls += 1
        return self.reachable


@pytest.fixture(autouse=True)
def _no_ttl(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ops, "_READINESS_TTL_SECONDS", 0.0)


def test_ready_reports_each_dependency(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        body = client.get("/health/ready").json()
    assert body == {"status": "ok", "checks": {"cache": "ok", "corpus": "ok"}}


def test_unreachable_cache_is_degraded_but_ready(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        client.app.state.services.cache.ping = _Pings(reachable=False)  # type: ignore[attr-defined]
        resp = client.get("/health/ready")
    assert resp.status_code == 200
    assert resp.json()["status"] == "degraded"
    assert resp.json()["checks"]["cache"] == "unavailable"


def test_unreadable_corpus_is_not_ready(tmp_path: Path) -> None:
    def _broken() -> int:
        raise OSError("disk gone")

    with _client(tmp_path) as client:
        client.app.state.services.corpus_store.count_documents = _broken  # type: ignore[attr-defined]
        resp = client.get("/health/ready")
    assert resp.status_code == 503
    assert resp.json()["checks"]["corpus"] == "unavailable"


def test_checks_are_cached_between_probes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ops, "_READINESS_TTL_SECONDS", 60.0)
    with _client(tmp_path) as client:
        pings = _Pings(reachable=True)
        client.app.state.services.cache.ping = pings  # type: ignore[attr-defined]
        for _ in range(3):
            assert client.get("/health/ready").status_code == 200
    assert pings.calls == 1


class _FakeRedis:
    def __init__(self, fail: bool) -> None:
        self.fail = fail

    async def ping(self) -> bool:
        if self.fail:
            raise ConnectionError("redis down")
        return True


async def test_redis_ping_reports_reachability_without_raising() -> None:
    from axiom_rag_engine.cache import RedisCacheBackend

    assert await RedisCacheBackend(client=_FakeRedis(fail=False)).ping() is True
    assert await RedisCacheBackend(client=_FakeRedis(fail=True)).ping() is False
