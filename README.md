# Axiom Engine

**Configuration-driven Agentic RAG with 6-tier verification.**

Axiom Engine is a retrieval-augmented generation (RAG) system that
verifies every cited claim before presenting answers. It assigns each claim a confidence
tier (1–6), but Tier 6 is currently reserved and not produced until explicit
contradiction detection ships.

## Quick Start

```bash
# Prerequisites: Python 3.11+, uv (https://docs.astral.sh/uv/)
python tasks.py install          # create .env, install deps
# Edit .env — fill in TAVILY_API_KEY for live web search
python tasks.py run              # start FastAPI server at http://localhost:8000
python tasks.py probe "your question"   # send a test query
```

## Architecture

```
retriever → scorer → ranker → synthesizer → verifier ─┐
   ▲                    ▲                              │
   │                    └── (rewrite loop) ◄───────────┘  (Tier 4/5 & loop < max)
   └── (re-retrieve) ◄────────────────────────────────┘  (loop exhausted & retries left)
```

| Module | Responsibility |
|---|---|
| **Retriever** | Web search via Tavily, dedup, HTML strip, paragraph chunking |
| **Scorer** | Domain authority + content quality scoring (deterministic) |
| **Ranker** | BM25-based relevance ranking with quality blend |
| **Synthesizer** | LLM-powered answer generation with strict citation format |
| **Verifier** | Two-stage verification: mechanical (exact match) + semantic (LLM) |

## Verification Tiers

| Tier | Label | Meaning |
|---|---|---|
| 1 | Authoritative | Verified against official/primary source |
| 2 | Multi-Source | Verified against multiple independent domains; agreement is not yet proven |
| 3 | Model Assisted | Mechanically verified; semantic relied on model knowledge |
| 4 | Misrepresented | Quote exists but claim distorts context |
| 5 | Hallucinated | Quote not found in source chunk |
| 6 | Conflicted | Reserved for future contradiction detection; not currently assigned |

## Development

```bash
python tasks.py test             # run unit tests
python tasks.py lint             # ruff + mypy
python tasks.py format           # auto-format
python tasks.py clean            # remove caches + venv
```

## API

- `POST /v1/synthesize` — Run the verification pipeline
- `GET /health` — Liveness probe
- `GET /metrics` — Runtime metrics

See the interactive docs at `http://localhost:8000/docs` when the server is running.

## Docker

```bash
docker compose up --build
```

## License

Proprietary. All rights reserved.
