"""
Axiom Engine — per-app runtime services.

Everything a running app needs, built once at startup (``bootstrap.build_services``)
and stored on ``app.state.services``. Keeping it per-app — rather than in module
globals — lets several apps with different settings coexist in one process and
gives tests a clean slate per app.

Deliberately still process-wide: the LLM concurrency semaphore (it bounds calls
across *all* requests to protect provider rate limits), Prometheus metrics (one
registry per process), and the tracer provider. Pipeline code reads the serving
app's settings through ``current_settings()``, bound per request by the API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from axiom_rag_engine.audit_store import AuditStore
from axiom_rag_engine.cache import CacheBackend
from axiom_rag_engine.config.settings import Settings
from axiom_rag_engine.corpus.store import CorpusStore


@dataclass
class AppServices:
    settings: Settings
    engine: Any
    cache: CacheBackend
    audit_store: AuditStore
    corpus_store: CorpusStore | None
    # The retriever's backend for this app. None means "use the retriever's
    # module default" (MockSearchBackend in development, or whatever tests and
    # evals installed with set_search_backend).
    search_backend: Any
    search_backend_mode: str
    default_synthesizer_model: str
    default_verifier_model: str
    started_at: float
    version: str

    def run_config(self) -> dict[str, Any] | None:
        """LangGraph run config carrying this app's search backend, if any."""
        if self.search_backend is None:
            return None
        return {"configurable": {"search_backend": self.search_backend}}
