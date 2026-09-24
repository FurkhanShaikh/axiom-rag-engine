"""
Audit events never carry provider or search-backend error text.

Audit trails reach callers through ``include_debug`` and ``GET /v1/audits``, and
provider messages can embed keys, internal URLs, or account details. Error
events record the exception type only; the full message goes to the server log.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import pytest

from axiom_rag_engine.graph import build_axiom_graph
from axiom_rag_engine.nodes import retriever as retriever_mod
from axiom_rag_engine.nodes.retriever import retriever_node
from axiom_rag_engine.state import make_initial_state

_PROVIDER_MESSAGE = "key=LEAKED-0123456789 at https://internal.example/billing"
_TEXT = (
    "Solid-state batteries replace liquid electrolytes with solid ceramics. "
    "This substitution significantly improves thermal stability and energy density."
)
_SYNTH = json.dumps(
    {
        "is_answerable": True,
        "sentences": [
            {
                "sentence_id": "s_01",
                "text": "Solid-state batteries use solid ceramics.",
                "is_cited": True,
                "citations": [
                    {
                        "citation_id": "cite_1",
                        "chunk_id": "doc_1_chunk_A",
                        "exact_source_quote": (
                            "Solid-state batteries replace liquid electrolytes with solid ceramics"
                        ),
                    }
                ],
            }
        ],
    }
)


class _Backend:
    def __init__(self, fail_on: str | None = None) -> None:
        self.fail_on = fail_on

    def search(self, query: str) -> list[dict[str, Any]]:
        if self.fail_on and self.fail_on in query:
            raise ConnectionError(_PROVIDER_MESSAGE)
        return [{"url": "https://example.com/a", "title": "A", "content": _TEXT}]


def _state(**stages: Any) -> dict[str, Any]:
    return dict(
        make_initial_state(
            request_id="req_redact",
            user_query="What are solid-state batteries?",
            app_config={},
            models_config={"synthesizer": "mock/s", "verifier": "mock/v"},
            pipeline_config={"stages": {"semantic_verification_enabled": True, **stages}},
        )
    )


async def test_search_error_records_type_only(monkeypatch: pytest.MonkeyPatch) -> None:
    # Skip tenacity's real backoff sleeps between search attempts.
    monkeypatch.setattr(retriever_mod._search_with_retry.retry, "sleep", lambda _s: None)
    state = _state()
    # Retry pass: the reformulated "details" query fails, the others succeed.
    state["rewrite_requests"] = ["Sentence s_01 failed"]
    result = await retriever_node(
        state, {"configurable": {"search_backend": _Backend(fail_on="details")}}
    )
    errors = [e for e in result["audit_trail"] if e["event_type"] == "retriever_search_error"]
    assert errors and errors[0]["payload"]["error_type"] == "ConnectionError"
    assert "LEAKED" not in json.dumps(result["audit_trail"])


async def test_semantic_provider_error_records_type_only() -> None:
    async def _fake(node: str, model: str, messages: Any, **_: Any) -> str:
        if node == "synthesizer":
            return _SYNTH
        raise RuntimeError(_PROVIDER_MESSAGE)

    with (
        patch("axiom_rag_engine.nodes.synthesizer.call_llm", _fake),
        patch("axiom_rag_engine.nodes.semantic.call_llm", _fake),
    ):
        result = await build_axiom_graph().ainvoke(
            _state(max_rewrite_loops=0, max_retrieval_retries=0),
            config={"configurable": {"search_backend": _Backend()}},
        )
    errors = [e for e in result["audit_trail"] if e["event_type"] == "semantic_check_error"]
    assert errors and errors[0]["payload"]["error_type"] == "RuntimeError"
    assert "LEAKED" not in json.dumps(result["audit_trail"])
