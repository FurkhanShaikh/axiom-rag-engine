"""
Shared test fixtures for the Axiom Engine test suite.

Centralises common helpers and fixtures so individual test modules
don't have to duplicate setup logic.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import axiom_rag_engine.main as _main_module
from axiom_rag_engine.config.settings import Settings, get_settings
from axiom_rag_engine.main import app
from axiom_rag_engine.nodes.retriever import MockSearchBackend, set_search_backend

# ---------------------------------------------------------------------------
# Reusable helpers
# ---------------------------------------------------------------------------


def mock_litellm_response(content: str) -> MagicMock:
    """Build a MagicMock mimicking a litellm.completion() response."""
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


def make_final_sentence_dict(
    sentence_id: str = "s_01",
    tier: int = 1,
    text: str = "A verified sentence.",
) -> dict[str, Any]:
    """Build a FinalSentence-shaped dict for test assertions."""
    label_map = {
        1: "authoritative",
        2: "multi_source",
        3: "model_assisted",
        4: "misrepresented",
        5: "hallucinated",
        6: "conflicted",
    }
    return {
        "sentence_id": sentence_id,
        "text": text,
        "is_cited": False,
        "citations": [],
        "verification": {
            "tier": tier,
            "tier_label": label_map[tier],
            "mechanical_check": "passed" if tier != 5 else "failed",
            "semantic_check": ("passed" if tier not in (4, 5) else "failed"),
            "failure_reason": None,
        },
    }


SAMPLE_CHUNKS = [
    {
        "chunk_id": "doc_1_chunk_A",
        "text": (
            "Solid-state batteries replace liquid electrolytes "
            "with solid ceramics. "
            "This substitution significantly improves thermal "
            "stability and energy density."
        ),
    },
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


_ENV_LEAK_KEYS = (
    "TAVILY_API_KEY",
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "AXIOM_API_KEYS",
    "AXIOM_AUDIT_RETENTION",
    "AXIOM_REDIS_URL",
    "AXIOM_ALLOW_MOCK_SEARCH",
    "AXIOM_DEFAULT_SYNTHESIZER_MODEL",
    "AXIOM_DEFAULT_VERIFIER_MODEL",
)


@pytest.fixture(autouse=True)
def _reset_settings_cache(monkeypatch):
    """Clear the Settings LRU cache and isolate tests from the operator's `.env`.

    Settings reads env vars once per instantiation, so tests that use
    `monkeypatch.setenv` need a fresh Settings object to observe the
    override. Autouse guarantees the cache is always clean.

    pydantic-settings additionally loads ``.env`` from the working directory
    as a separate source from ``os.environ``. Disabling that source for
    tests ensures the operator's local secrets don't influence assertions.
    """
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    for key in _ENV_LEAK_KEYS:
        monkeypatch.delenv(key, raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture()
def client(monkeypatch):
    """TestClient fixture with clean cache and no Tavily key."""
    _main_module._response_cache.clear()
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setenv("AXIOM_ENV", "test")
    monkeypatch.delenv("AXIOM_API_KEYS", raising=False)
    set_search_backend(MockSearchBackend([]))
    with TestClient(app) as c:
        yield c
