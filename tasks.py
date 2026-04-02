#!/usr/bin/env python3
"""
Cross-platform task runner for Axiom Engine.
Requires uv: https://docs.astral.sh/uv/getting-started/installation/

Usage:
    python tasks.py <task>

Quick start for new users:
    python tasks.py install                        # scaffold .env, create venv, install deps
    # edit .env — fill in TAVILY_API_KEY
    python tasks.py run                            # start the server (separate terminal)
    python tasks.py probe "your question"          # send a test query
    python tasks.py probe "your question" --debug  # include audit trail
"""

import json
import pathlib
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime


def _run(*cmd: str) -> None:
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

def install() -> None:
    """Scaffold .env, create virtual environment, and install all dependencies."""
    env = pathlib.Path(".env")
    if not env.exists():
        shutil.copy(".env.example", ".env")
        print("\n  .env created from .env.example.")
        print("  Open it and fill in your TAVILY_API_KEY before running.\n")
    _run("uv", "sync")


def run() -> None:
    """Start the FastAPI development server at http://localhost:8000."""
    _run("uv", "run", "uvicorn", "axiom_engine.main:app", "--reload")


def test() -> None:
    """Run the full test suite with pytest."""
    _run("uv", "run", "pytest")


def lint() -> None:
    """Run ruff (style + imports) and mypy (types) across the codebase."""
    _run("uv", "run", "ruff", "check", "src", "tests")
    _run("uv", "run", "ruff", "format", "--check", "src", "tests")
    _run("uv", "run", "mypy", "src")


def format() -> None:  # noqa: A001
    """Auto-format and fix import order with ruff."""
    _run("uv", "run", "ruff", "format", "src", "tests")
    _run("uv", "run", "ruff", "check", "--fix", "src", "tests")


def probe() -> None:
    """Send a test query to the running server and pretty-print the result.

    Usage:
        python tasks.py probe "your question here"
        python tasks.py probe "your question here" --debug
        python tasks.py probe "your question here" --debug --model ollama/qwen2.5:7b

    Flags:
        --debug          Include full audit trail and pipeline stats in output.
        --model <id>     LiteLLM model ID for both synthesizer and verifier.
                         Default: ollama/qwen3.5:9b
    """
    args = sys.argv[2:]
    debug = "--debug" in args
    args = [a for a in args if a != "--debug"]

    model = "ollama/qwen3.5:9b"
    if "--model" in args:
        idx = args.index("--model")
        model = args[idx + 1]
        args = args[:idx] + args[idx + 2:]

    query = " ".join(args) if args else "What is the capital of France?"

    payload = {
        "request_id": f"probe-{datetime.now().strftime('%H%M%S')}",
        "user_query": query,
        "models": {"synthesizer": model, "verifier": model},
        "include_debug": debug,
    }

    print(f"\n  Query : {query}")
    print(f"  Model : {model}")
    print(f"  Debug : {debug}")
    print(f"  Server: http://localhost:8000\n")

    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        "http://localhost:8000/v1/synthesize",
        data=data,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=700) as resp:
            result = json.loads(resp.read())
    except urllib.error.URLError as exc:
        print(f"  Could not reach server: {exc}")
        print("  Is it running? Try: python tasks.py run")
        sys.exit(1)

    # ── Summary ──────────────────────────────────────────────────────────────
    status = result.get("status", "?")
    score = result.get("confidence_summary", {}).get("overall_score", 0)
    tiers = result.get("confidence_summary", {}).get("tier_breakdown", {})
    sentences = result.get("final_response", [])

    print(f"  Status : {status}")
    print(f"  Score  : {score:.2f}")
    print(f"  Tiers  : { {k: v for k, v in tiers.items() if v > 0} or 'none' }")
    if result.get("error_message"):
        print(f"  Error  : {result['error_message']}")
    print()

    for i, s in enumerate(sentences, 1):
        vr = s.get("verification", {})
        print(f"  [{i}] Tier {vr.get('tier','?')} ({vr.get('tier_label','?')}) — {s['text']}")
        for c in s.get("citations", []):
            print(f"       cite: \"{c['exact_source_quote'][:80]}\"")
            print(f"       from: {c['chunk_id']}")

    # ── Debug info ────────────────────────────────────────────────────────────
    if debug and result.get("debug"):
        dbg = result["debug"]
        stats = dbg.get("pipeline_stats", {})
        print(f"\n  Pipeline stats: {stats}")
        print(f"\n  Audit trail ({len(dbg.get('audit_trail', []))} events):")
        for event in dbg.get("audit_trail", []):
            payload_str = json.dumps(event.get("payload", {}))
            if len(payload_str) > 120:
                payload_str = payload_str[:120] + "…"
            print(f"    [{event['node']}] {event['event_type']}: {payload_str}")


def clean() -> None:
    """Remove the virtual environment and all caches."""
    for name in (".venv", ".pytest_cache", ".ruff_cache", ".mypy_cache"):
        shutil.rmtree(name, ignore_errors=True)
    for p in pathlib.Path(".").rglob("__pycache__"):
        shutil.rmtree(p, ignore_errors=True)
    print("Cleaned.")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

_TASKS = {
    name: fn
    for name, fn in sorted(globals().items())
    if callable(fn) and not name.startswith("_")
}


def _help() -> None:
    print(__doc__)
    print("Available tasks:")
    for name, fn in _TASKS.items():
        print(f"  {name:<10} {fn.__doc__}")


if __name__ == "__main__":
    task = sys.argv[1] if len(sys.argv) > 1 else None
    if task in _TASKS:
        _TASKS[task]()
    else:
        _help()
        if task is not None:
            print(f"\nUnknown task: {task!r}")
            sys.exit(1)
