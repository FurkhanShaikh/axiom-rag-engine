"""
Scoring weights must form a valid pair (RET-2).

``source_weight + chunk_weight`` "should sum to 1.0" but nothing checked it, so
``{"source_weight": 0.5}`` silently scored against 0.5 + the default 0.6. A
lone weight now implies its pair, and an inconsistent pair is rejected (422).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from axiom_rag_engine.config.settings import Settings
from axiom_rag_engine.main import create_app
from axiom_rag_engine.models import AppConfig
from axiom_rag_engine.nodes.retriever import MockSearchBackend


def test_defaults_are_a_valid_pair() -> None:
    config = AppConfig()
    assert config.source_weight + config.chunk_weight == pytest.approx(1.0)


def test_one_weight_implies_the_other() -> None:
    assert AppConfig(source_weight=0.5).chunk_weight == pytest.approx(0.5)
    assert AppConfig(chunk_weight=0.9).source_weight == pytest.approx(0.1)


def test_consistent_pair_is_accepted() -> None:
    config = AppConfig(source_weight=0.3, chunk_weight=0.7)
    assert (config.source_weight, config.chunk_weight) == (0.3, 0.7)


def test_inconsistent_pair_is_rejected() -> None:
    with pytest.raises(ValidationError, match=r"must sum to 1\.0"):
        AppConfig(source_weight=0.5, chunk_weight=0.6)


def test_api_rejects_an_inconsistent_pair_with_422() -> None:
    app = create_app(Settings(env="test"), search_backend=MockSearchBackend([]))
    body = {
        "request_id": "w",
        "user_query": "alpha batteries",
        "app_config": {"source_weight": 0.9, "chunk_weight": 0.9},
    }
    with TestClient(app) as client:
        resp = client.post("/v1/synthesize", json=body)
    assert resp.status_code == 422
    assert "must sum to 1.0" in resp.text
