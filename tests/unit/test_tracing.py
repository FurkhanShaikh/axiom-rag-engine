"""
HTTP request tracing — every request gets a server span, and pipeline spans nest
under it in one trace.

FastAPI instrumentation used to be applied from the lifespan hook, after
Starlette had already built its middleware stack, so it silently produced no
request spans: pipeline spans had no parent and one request's work was scattered
across several traces with no request id on them.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from axiom_rag_engine.config.settings import Settings
from axiom_rag_engine.main import create_app
from axiom_rag_engine.nodes.retriever import MockSearchBackend

_TEXT = "Alpha batteries use lithium iron phosphate chemistry for a long cycle life."
_SYNTH = json.dumps(
    {
        "is_answerable": True,
        "sentences": [
            {
                "sentence_id": "s_01",
                "text": "Alpha batteries use LFP chemistry.",
                "is_cited": True,
                "citations": [
                    {
                        "citation_id": "cite_1",
                        "chunk_id": "doc_1_chunk_A",
                        "exact_source_quote": "Alpha batteries use lithium iron phosphate chemistry",
                    }
                ],
            }
        ],
    }
)


def _reply(content: str) -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content=content))]
    return response


async def _fake_llm(**kwargs: Any) -> MagicMock:
    if "Cognitive Synthesizer" in kwargs["messages"][0]["content"]:
        return _reply(_SYNTH)
    return _reply('{"semantic_check": "passed", "failure_reason": null}')


@pytest.fixture
def spans(monkeypatch: pytest.MonkeyPatch) -> InMemorySpanExporter:
    """Enable tracing with the OTLP exporter swapped for an in-memory one."""
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:4317")
    monkeypatch.setattr(
        "opentelemetry.exporter.otlp.proto.grpc.trace_exporter.OTLPSpanExporter",
        InMemorySpanExporter,
    )
    return InMemorySpanExporter()


def test_request_gets_a_server_span_with_pipeline_spans_nested(
    spans: InMemorySpanExporter,
) -> None:
    app = create_app(
        Settings(env="test", semantic_verification_enabled=True),
        search_backend=MockSearchBackend(
            [{"url": "https://example.com/a", "title": "A", "content": _TEXT}]
        ),
    )
    with TestClient(app) as client, patch("litellm.acompletion", side_effect=_fake_llm):
        # The provider is process-global: capture into it whichever test set it.
        provider = trace.get_tracer_provider()
        assert isinstance(provider, TracerProvider)
        provider.add_span_processor(SimpleSpanProcessor(spans))
        resp = client.post(
            "/v1/synthesize",
            json={
                "request_id": "trace-me",
                "user_query": "alpha batteries chemistry",
                "models": {"synthesizer": "mock/s", "verifier": "mock/v"},
            },
        )
    assert resp.status_code == 200, resp.text

    finished = spans.get_finished_spans()
    server = [s for s in finished if s.kind == trace.SpanKind.SERVER and "/v1/synthesize" in s.name]
    assert server, [s.name for s in finished]
    assert server[0].attributes["axiom.request_id"] == "trace-me"

    trace_id = server[0].context.trace_id
    pipeline = [s for s in finished if s.name in ("verification", "retriever.search")]
    assert pipeline, [s.name for s in finished]
    assert all(s.context.trace_id == trace_id for s in pipeline)


def test_no_instrumentation_without_an_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    app = create_app(Settings(env="test"))
    assert not getattr(app, "_is_instrumented_by_opentelemetry", False)
