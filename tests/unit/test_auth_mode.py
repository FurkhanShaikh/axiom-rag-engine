"""
Configured API keys are always enforced; /docs follows the auth mode.

Auth used to depend on AXIOM_ENV alone: a deployment that set AXIOM_API_KEYS
but carried AXIOM_ENV=dev or test (a copied .env, a typo) ran with every
endpoint open. And /docs + /redoc were served by default in production.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

import axiom_rag_engine.bootstrap as bootstrap_module
from axiom_rag_engine.config.settings import Settings
from axiom_rag_engine.main import create_app
from axiom_rag_engine.nodes.retriever import MockSearchBackend

KEY = "configured-key-0123456789"


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bootstrap_module, "_list_ollama_models", lambda _base: [])


def _client(**settings: Any) -> TestClient:
    # Explicit models satisfy production's fail-closed provider check.
    base: dict[str, Any] = {
        "allow_mock_search": True,
        "default_synthesizer_model": "server/synth",
        "default_verifier_model": "server/verifier",
    }
    return TestClient(
        create_app(Settings(**{**base, **settings}), search_backend=MockSearchBackend([]))
    )


@pytest.mark.parametrize("env", ["development", "dev", "local", "test"])
def test_configured_keys_are_enforced_in_any_environment(env: str) -> None:
    assert Settings(env=env, api_keys=[KEY]).auth_required()
    assert Settings(env=env, admin_api_keys=[KEY]).auth_required()
    with _client(env=env, api_keys=[KEY]) as client:
        assert client.get("/v1/status").status_code == 401
        assert client.get("/v1/status", headers={"X-API-Key": KEY}).status_code == 200


def test_without_keys_a_dev_environment_stays_open() -> None:
    assert not Settings(env="development").auth_required()
    with _client(env="development") as client:
        assert client.get("/v1/status").status_code == 200


def test_production_requires_auth_without_keys() -> None:
    assert Settings(env="production").auth_required()


def test_keys_in_dev_do_not_bring_production_strictness() -> None:
    # Authentication, yes; the fail-closed startup checks, no.
    settings = Settings(env="development", api_keys=[KEY])
    assert settings.auth_required()
    assert not settings.is_production()


def test_docs_follow_the_auth_mode_by_default() -> None:
    with _client(env="production", api_keys=[KEY]) as client:
        assert client.get("/docs").status_code == 404
        assert client.get("/redoc").status_code == 404
    with _client(env="development") as client:
        assert client.get("/docs").status_code == 200


def test_docs_can_be_forced_either_way() -> None:
    with _client(env="production", api_keys=[KEY], docs_enabled=True) as client:
        assert client.get("/docs").status_code == 200
    with _client(env="development", docs_enabled=False) as client:
        assert client.get("/docs").status_code == 404
