"""
Settings passed to ``create_app`` reach the pipeline.

Pipeline nodes and the LLM helpers read the process-wide ``get_settings()``, so
an app built with explicit settings silently ran its pipeline — budget caps,
reranker, corroboration, timeouts — on the process configuration instead. Each
request now binds its app's settings (``use_settings``) and pipeline code reads
``current_settings()``.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from axiom_rag_engine.config.settings import Settings, current_settings, get_settings
from axiom_rag_engine.main import create_app
from axiom_rag_engine.nodes.retriever import MockSearchBackend

_TEXT_A = "Alpha batteries use lithium iron phosphate chemistry for a long cycle life."
_TEXT_B = "Beta batteries favor energy density over longevity in portable electronics."


def _synth(text: str) -> str:
    return json.dumps(
        {
            "is_answerable": True,
            "sentences": [
                {
                    "sentence_id": "s_01",
                    "text": "A claim.",
                    "is_cited": True,
                    "citations": [
                        {
                            "citation_id": "cite_1",
                            "chunk_id": "doc_1_chunk_A",
                            "exact_source_quote": " ".join(text.split()[:6]),
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


def _backend() -> MockSearchBackend:
    return MockSearchBackend(
        [
            {"url": "https://a.example.com/x", "title": "A", "content": _TEXT_A},
            {"url": "https://b.example.org/y", "title": "B", "content": _TEXT_B},
        ]
    )


class _Recorder:
    def __init__(self) -> None:
        self.rerank_calls = 0

    async def __call__(self, **kwargs: Any) -> MagicMock:
        system = kwargs["messages"][0]["content"]
        if "Cognitive Synthesizer" in system:
            return _reply(_synth(_TEXT_A))
        if "grade how relevant" in system:
            self.rerank_calls += 1
            return _reply("2")
        return _reply('{"semantic_check": "passed", "failure_reason": null}')


def _post(client: TestClient) -> Any:
    return client.post(
        "/v1/synthesize",
        json={
            "request_id": "r1",
            "user_query": "alpha batteries chemistry",
            "models": {"synthesizer": "mock/s", "verifier": "mock/v"},
        },
    )


def test_two_apps_in_one_process_use_their_own_pipeline_settings() -> None:
    with_reranker = create_app(
        Settings(env="test", reranker_model="mock/rerank"), search_backend=_backend()
    )
    without = create_app(Settings(env="test"), search_backend=_backend())
    rec = _Recorder()
    with (
        TestClient(with_reranker) as a,
        TestClient(without) as b,
        patch("litellm.acompletion", side_effect=rec.__call__),
    ):
        assert _post(b).status_code == 200
        assert rec.rerank_calls == 0
        assert _post(a).status_code == 200
        assert rec.rerank_calls > 0


def test_budget_cap_comes_from_the_app() -> None:
    tight = create_app(Settings(env="test", max_llm_calls_per_request=1), search_backend=_backend())
    rec = _Recorder()
    # Process settings allow 64 calls; the app allows 1 (synthesis only), so the
    # semantic check cannot run and the sentence is labelled unverified.
    assert get_settings().max_llm_calls_per_request == 64
    with TestClient(tight) as client, patch("litellm.acompletion", side_effect=rec.__call__):
        body = _post(client).json()
    assert body["final_response"][0]["verification"]["tier_label"] == "unverified"


def test_outside_a_request_the_process_settings_apply() -> None:
    assert current_settings() is get_settings()
