"""
Both endpoints emit the same metrics for the same outcome.

The stream endpoint recorded no pipeline duration and swallowed failures of its
completion hook silently; the tier metric counted uncited sentences as tier 3
(the response's tier_breakdown excludes them) and could not tell a verified
tier 3 from an unverified one.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

from axiom_rag_engine.api.sse import stream_pipeline
from axiom_rag_engine.config.settings import Settings
from axiom_rag_engine.main import create_app
from axiom_rag_engine.models import AxiomRequest
from axiom_rag_engine.nodes.retriever import MockSearchBackend
from axiom_rag_engine.state import make_initial_state

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
            },
            {
                "sentence_id": "s_02",
                "text": "In short, chemistry matters.",
                "is_cited": False,
                "citations": [],
            },
        ],
    }
)


def _reply(content: str) -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content=content))]
    return response


async def _llm(**kwargs: Any) -> MagicMock:
    if "Cognitive Synthesizer" in kwargs["messages"][0]["content"]:
        return _reply(_SYNTH)
    return _reply('{"semantic_check": "passed", "failure_reason": null}')


def _sample(name: str, **labels: str) -> float:
    return REGISTRY.get_sample_value(name, labels) or 0.0


def _body(request_id: str) -> dict[str, Any]:
    return {
        "request_id": request_id,
        "user_query": "alpha batteries chemistry",
        "models": {"synthesizer": "mock/s", "verifier": "mock/v"},
    }


@pytest.fixture
def client() -> Any:
    app = create_app(
        Settings(env="test"),
        search_backend=MockSearchBackend(
            [{"url": "https://example.com/a", "title": "A", "content": _TEXT}]
        ),
    )
    with TestClient(app) as c, patch("litellm.acompletion", side_effect=_llm):
        yield c


@pytest.mark.parametrize("path", ["/v1/synthesize", "/v1/synthesize/stream"])
def test_both_endpoints_record_duration_and_claim_tiers(client: TestClient, path: str) -> None:
    durations = _sample("axiom_pipeline_duration_seconds_count")
    verified_t3 = _sample("axiom_tier_assignments_total", tier="3", label="model_assisted")
    unverified_t3 = _sample("axiom_tier_assignments_total", tier="3", label="unverified")

    assert client.post(path, json=_body(f"m-{path}")).status_code == 200

    assert _sample("axiom_pipeline_duration_seconds_count") == durations + 1
    # The cited sentence counts once; the uncited one is not a claim.
    assert (
        _sample("axiom_tier_assignments_total", tier="3", label="model_assisted") == verified_t3 + 1
    )
    assert _sample("axiom_tier_assignments_total", tier="3", label="unverified") == unverified_t3


async def test_stream_logs_on_complete_failures(caplog: pytest.LogCaptureFixture) -> None:
    async def _broken(*_: Any) -> None:
        raise RuntimeError("cache down")

    class _Engine:
        async def astream_events(self, state: Any, **_: Any) -> Any:
            return
            yield

    state = make_initial_state("r", "q", {}, {}, {})
    frames = [
        f
        async for f in stream_pipeline(
            payload=AxiomRequest(request_id="r", user_query="q"),
            engine=_Engine(),
            initial_state=state,
            on_complete=_broken,
        )
    ]
    assert frames[-1].startswith("event: complete")  # the stream still completes
    assert "on_complete hook failed" in caplog.text
