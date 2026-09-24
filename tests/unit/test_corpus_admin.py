"""
Corpus writes require an admin key when auth is required.

The corpus is shared by every tenant, yet any valid API key could ingest or
delete any document, so one tenant could poison (including with text written to
inject instructions) or erase what every other tenant's answers are built from.
Ingest and delete now require a key listed in AXIOM_ADMIN_API_KEYS; reads stay
open to any valid key, and admin keys are valid API keys too.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import axiom_rag_engine.bootstrap as bootstrap_module
from axiom_rag_engine.config.settings import Settings
from axiom_rag_engine.main import create_app
from axiom_rag_engine.nodes.retriever import MockSearchBackend

TENANT = "tenant-key-0123456789"
ADMIN = "admin-key-0123456789"


async def _fake_embed(model: str, texts: list[str]) -> list[list[float]]:
    return [[1.0 / math.sqrt(2), 1.0 / math.sqrt(2)] for _ in texts]


def _app(tmp_path: Path, **overrides: Any) -> Any:
    settings = Settings(
        env="production",
        api_keys=[TENANT],
        allow_mock_search=True,
        corpus_db_path=str(tmp_path / "corpus.db"),
        embedding_model="fake/embed",
        default_synthesizer_model="server/synth",
        default_verifier_model="server/verifier",
        **overrides,
    )
    return create_app(settings, search_backend=MockSearchBackend([]))


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bootstrap_module, "_list_ollama_models", lambda _base: [])
    monkeypatch.setattr("axiom_rag_engine.corpus.ingest.embed_documents", _fake_embed)


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    with TestClient(_app(tmp_path, admin_api_keys=[ADMIN])) as c:
        yield c


def _ingest(client: TestClient, key: str, doc_id: str = "d1") -> Any:
    return client.post(
        "/v1/documents",
        headers={"X-API-Key": key},
        json={"doc_id": doc_id, "text": "Alpha cells use lithium iron phosphate chemistry."},
    )


class TestWrites:
    def test_tenant_key_cannot_ingest(self, client: TestClient) -> None:
        resp = _ingest(client, TENANT)
        assert resp.status_code == 403
        assert "admin" in resp.json()["detail"]

    def test_admin_key_can_ingest_and_delete(self, client: TestClient) -> None:
        assert _ingest(client, ADMIN).status_code == 201
        resp = client.delete("/v1/documents/d1", headers={"X-API-Key": ADMIN})
        assert resp.status_code == 200

    def test_tenant_key_cannot_delete(self, client: TestClient) -> None:
        assert _ingest(client, ADMIN).status_code == 201
        resp = client.delete("/v1/documents/d1", headers={"X-API-Key": TENANT})
        assert resp.status_code == 403
        listed = client.get("/v1/documents", headers={"X-API-Key": ADMIN}).json()
        assert [d["doc_id"] for d in listed["documents"]] == ["d1"]

    def test_tenant_key_cannot_upload(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/documents/upload",
            headers={"X-API-Key": TENANT},
            files={"file": ("a.txt", b"Alpha cells use lithium iron phosphate.", "text/plain")},
        )
        assert resp.status_code == 403

    def test_writes_refused_when_no_admin_keys_are_configured(self, tmp_path: Path) -> None:
        with TestClient(_app(tmp_path)) as c:
            assert _ingest(c, TENANT).status_code == 403

    def test_invalid_key_is_still_401(self, client: TestClient) -> None:
        assert _ingest(client, "not-a-key").status_code == 401


class TestReads:
    def test_tenant_key_can_read(self, client: TestClient) -> None:
        assert _ingest(client, ADMIN).status_code == 201
        assert client.get("/v1/documents", headers={"X-API-Key": TENANT}).status_code == 200
        assert client.get("/v1/documents/d1", headers={"X-API-Key": TENANT}).status_code == 200

    def test_admin_key_is_a_valid_api_key(self, client: TestClient) -> None:
        assert client.get("/v1/status", headers={"X-API-Key": ADMIN}).status_code == 200


def test_auth_disabled_keeps_writes_open(tmp_path: Path) -> None:
    app = create_app(
        Settings(
            env="test",
            corpus_db_path=str(tmp_path / "corpus.db"),
            embedding_model="fake/embed",
        ),
        search_backend=MockSearchBackend([]),
    )
    with TestClient(app) as c:
        assert (
            c.post(
                "/v1/documents", json={"text": "Alpha cells use lithium iron phosphate chemistry."}
            ).status_code
            == 201
        )
