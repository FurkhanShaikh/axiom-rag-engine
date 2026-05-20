# Contributing to Axiom Engine

Thanks for your interest! Here's how to get started.

## Development setup

```bash
python tasks.py install          # scaffold .env + install deps via uv
# Edit .env — fill in TAVILY_API_KEY for live search
python tasks.py run              # start dev server at localhost:8000
```

Requires **Python 3.11+** and **[uv](https://docs.astral.sh/uv/)**.

## Workflow

1. Fork the repo and create a feature branch from `main`.
2. Make your changes.
3. Run the quality checks:

```bash
python tasks.py lint             # ruff + mypy
python tasks.py test             # pytest with >=70% coverage
python tasks.py security         # dependency vulnerability scan
python tasks.py ci               # CI-style pre-push check
```

4. Open a pull request against `main`.

## Code style

- Formatted with **ruff** (`line-length = 100`).
- Type-checked with **mypy** (non-strict; strictness grows per module).
- Tests live in `tests/unit/` (no live services) and `tests/integration/` (requires Ollama).

## Configuration changes

All `AXIOM_*` env vars should be declared in `src/axiom_rag_engine/config/settings.py`. Do not read env vars directly with `os.getenv` — use `get_settings().<field>`.

## Reporting issues

Open an issue at https://github.com/FurkhanShaikh/axiom-rag-engine/issues with:
- Steps to reproduce
- Expected vs. actual behavior
- Python version + OS
