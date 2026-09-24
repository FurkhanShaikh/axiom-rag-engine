"""
``check-config`` never prints credentials.

``Settings.redacted_dict`` masked only ``api_keys`` and the Redis URL, so
``check-config`` printed the Tavily, Anthropic and OpenAI keys in full despite
saying they were "not shown". Secret fields are now recognised by name.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from axiom_rag_engine.config.settings import Settings

_SECRETS = {
    "TAVILY_API_KEY": "tvly-LEAK-1",
    "ANTHROPIC_API_KEY": "sk-ant-LEAK-2",
    "OPENAI_API_KEY": "sk-LEAK-3",
    "AXIOM_API_KEYS": "key-LEAK-4,key-LEAK-5",
    "AXIOM_ADMIN_API_KEYS": "admin-LEAK-6",
    "AXIOM_REDIS_URL": "redis://user:pw-LEAK-7@localhost:6379/0",
}


def test_redacted_dict_masks_every_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in _SECRETS.items():
        monkeypatch.setenv(name, value)
    dumped = json.dumps(Settings().redacted_dict(), default=str)
    assert "LEAK" not in dumped
    assert "***" in dumped


def test_check_config_output_masks_every_secret() -> None:
    env = {"AXIOM_ENV": "development", "PATH": "", **_SECRETS}
    out = subprocess.run(
        [sys.executable, "-m", "axiom_rag_engine", "check-config", "--format", "json"],
        capture_output=True,
        text=True,
        env=env,
        check=True,
        timeout=120,
    ).stdout
    assert "LEAK" not in out
