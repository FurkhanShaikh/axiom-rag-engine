"""
Provider auto-selection: OpenRouter support and explicit-model detection.

An OpenRouter key alone is enough to run the engine: startup selects the
``openrouter_*`` defaults when no Anthropic or OpenAI key is set. And an
operator who explicitly sets a model to its built-in default value keeps it —
"explicitly set" is read from the settings sources, not inferred by comparing
the value with the default (CFG-3).
"""

from __future__ import annotations

import os

import pytest

from axiom_rag_engine import bootstrap
from axiom_rag_engine.config.observability import safe_model_label
from axiom_rag_engine.config.settings import Settings


@pytest.fixture(autouse=True)
def _no_ollama(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bootstrap, "_list_ollama_models", lambda _base: [])


def _resolve(**overrides: object) -> tuple[str, str]:
    return bootstrap.resolve_llm_defaults(Settings(env="production", **overrides))


def test_openrouter_key_alone_selects_openrouter_models() -> None:
    assert _resolve(OPENROUTER_API_KEY="or-test") == (
        "openrouter/openai/gpt-4o",
        "openrouter/openai/gpt-4o-mini",
    )


def test_openrouter_models_are_configurable() -> None:
    synth, verif = _resolve(
        OPENROUTER_API_KEY="or-test",
        openrouter_synthesizer_model="openrouter/anthropic/claude-sonnet-4.5",
        openrouter_verifier_model="openrouter/google/gemini-2.5-flash",
    )
    assert synth == "openrouter/anthropic/claude-sonnet-4.5"
    assert verif == "openrouter/google/gemini-2.5-flash"


def test_direct_provider_keys_take_precedence_over_openrouter() -> None:
    assert _resolve(ANTHROPIC_API_KEY="a", OPENROUTER_API_KEY="or") == (
        "claude-opus-4-8",
        "claude-haiku-4-5",
    )


def test_no_provider_still_fails_closed_in_production() -> None:
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        _resolve()


def test_explicitly_set_default_value_is_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    # Pinning the built-in default verifier must not be switched to Haiku just
    # because only an Anthropic key is present.
    monkeypatch.setenv("AXIOM_DEFAULT_VERIFIER_MODEL", "gpt-4o-mini")
    assert _resolve(ANTHROPIC_API_KEY="a")[1] == "gpt-4o-mini"


def test_unset_default_is_still_auto_selected() -> None:
    assert _resolve(ANTHROPIC_API_KEY="a")[1] == "claude-haiku-4-5"


def test_openrouter_key_reaches_litellm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "")  # records the original for teardown
    monkeypatch.delenv("OPENROUTER_API_KEY")
    bootstrap.build_services(Settings(env="test", OPENROUTER_API_KEY="or-test"))
    assert os.environ["OPENROUTER_API_KEY"] == "or-test"


def test_openrouter_defaults_keep_their_metric_label() -> None:
    assert safe_model_label("openrouter/openai/gpt-4o-mini") == "openrouter/openai/gpt-4o-mini"


def test_openrouter_key_is_redacted() -> None:
    redacted = Settings(env="test", OPENROUTER_API_KEY="or-secret").redacted_dict()
    assert "or-secret" not in str(redacted)
