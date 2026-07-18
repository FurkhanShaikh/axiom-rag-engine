"""Load .env into os.environ for standalone eval runs.

The eval scripts import the production LLM call path, which reaches LiteLLM.
LiteLLM reads provider credentials (OPENROUTER_API_KEY, OPENAI_API_KEY,
ANTHROPIC_API_KEY, ...) from ``os.environ`` directly. The FastAPI server pushes
those keys into the environment during lifespan startup, but the eval scripts
never run that lifespan — so without this, ``python tasks.py evals semantic``
silently has no key and every call fails.

This mirrors the server's ``os.environ.setdefault`` behavior: a value already
present in the real environment always wins over the .env file, so CI secrets
and shell exports are never overridden.
"""

from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]


def load_dotenv(path: Path | None = None) -> list[str]:
    """Populate os.environ from a .env file without overriding existing values.

    Returns the names (not values) of the keys that were newly set, so callers
    can log what was loaded without leaking secrets.

    Parsing is deliberately minimal — ``KEY=value`` with optional surrounding
    single or double quotes on the value, ``#`` comments, and blank lines. It is
    not a full dotenv implementation; the .env files in this repo are simple.
    """
    env_path = path or (_REPO_ROOT / ".env")
    if not env_path.exists():
        return []

    loaded: list[str] = []
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key or key in os.environ:
            # Already set in the real environment — that source wins.
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        os.environ[key] = value
        loaded.append(key)
    return loaded
