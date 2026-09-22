"""
The cross-source gates (contradiction, corroboration) run concurrently across
sentences. They used to be awaited one sentence at a time, so an answer with N
multi-domain sentences paid N sequential LLM round-trips.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from axiom_rag_engine.config.settings import get_settings
from axiom_rag_engine.nodes import semantic as semantic_module
from axiom_rag_engine.nodes.semantic import semantic_verifier_node
from axiom_rag_engine.state import make_initial_state

_CHUNKS = [
    {
        "chunk_id": "doc_1_chunk_A",
        "text": "The committee approved the measure by a wide margin on Tuesday.",
        "source_url": "https://alpha.example.com/a",
        "domain": "alpha.example.com",
    },
    {
        "chunk_id": "doc_2_chunk_B",
        "text": "The committee approved the measure after a long debate on Tuesday.",
        "source_url": "https://beta.example.org/b",
        "domain": "beta.example.org",
    },
]


def _draft(n: int) -> list[dict[str, Any]]:
    return [
        {
            "sentence_id": f"s_{i:02d}",
            "text": "The committee approved the measure.",
            "is_cited": True,
            "citations": [
                {
                    "citation_id": f"cite_{i}_a",
                    "chunk_id": "doc_1_chunk_A",
                    "exact_source_quote": "The committee approved the measure by a wide margin",
                },
                {
                    "citation_id": f"cite_{i}_b",
                    "chunk_id": "doc_2_chunk_B",
                    "exact_source_quote": "The committee approved the measure after a long debate",
                },
            ],
        }
        for i in range(n)
    ]


def _state(n: int) -> dict[str, Any]:
    state = make_initial_state(
        request_id="req",
        user_query="what did the committee decide?",
        app_config={},
        models_config={"synthesizer": "mock/s", "verifier": "mock/v"},
        pipeline_config={"stages": {"semantic_verification_enabled": True}},
    )
    state["indexed_chunks"] = _CHUNKS
    state["draft_sentences"] = _draft(n)
    mech = {
        "tier": 3,
        "tier_label": "model_assisted",
        "mechanical_check": "passed",
        "semantic_check": "skipped",
        "failure_reason": None,
    }
    state["mechanical_results"] = {
        c["citation_id"]: dict(mech) for s in state["draft_sentences"] for c in s["citations"]
    }
    return dict(state)


def _semantic_pass(**_: Any) -> MagicMock:
    response = MagicMock()
    response.choices = [
        MagicMock(message=MagicMock(content='{"semantic_check": "passed", "failure_reason": null}'))
    ]
    return response


class _ConcurrencyProbe:
    def __init__(self, verdict: tuple[bool, str]) -> None:
        self.in_flight = 0
        self.max_in_flight = 0
        self._verdict = verdict

    async def __call__(self, *_: Any, **__: Any) -> tuple[bool, str]:
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        await asyncio.sleep(0.02)
        self.in_flight -= 1
        return self._verdict


class TestGatesRunConcurrently:
    async def test_contradiction_checks_overlap_across_sentences(self, monkeypatch) -> None:
        monkeypatch.setenv("AXIOM_CONTRADICTION_DETECTION_ENABLED", "true")
        get_settings.cache_clear()
        probe = _ConcurrencyProbe((False, "agree"))
        with (
            patch.object(semantic_module, "_check_contradiction", probe),
            patch(
                "axiom_rag_engine.nodes.semantic.litellm.acompletion",
                new_callable=AsyncMock,
                side_effect=_semantic_pass,
            ),
        ):
            result = await semantic_verifier_node(_state(3))
        assert probe.max_in_flight == 3
        assert [s["verification"]["tier"] for s in result["final_sentences"]] == [2, 2, 2]

    async def test_corroboration_checks_overlap_across_sentences(self, monkeypatch) -> None:
        monkeypatch.setenv("AXIOM_CORROBORATION_ENABLED", "true")
        get_settings.cache_clear()
        probe = _ConcurrencyProbe((True, "both state it"))
        with (
            patch.object(semantic_module, "_check_corroboration", probe),
            patch(
                "axiom_rag_engine.nodes.semantic.litellm.acompletion",
                new_callable=AsyncMock,
                side_effect=_semantic_pass,
            ),
        ):
            result = await semantic_verifier_node(_state(3))
        assert probe.max_in_flight == 3
        assert [s["verification"]["tier"] for s in result["final_sentences"]] == [2, 2, 2]

    async def test_sentence_order_is_preserved(self, monkeypatch) -> None:
        monkeypatch.setenv("AXIOM_CONTRADICTION_DETECTION_ENABLED", "true")
        get_settings.cache_clear()
        with (
            patch.object(semantic_module, "_check_contradiction", _ConcurrencyProbe((True, "x"))),
            patch(
                "axiom_rag_engine.nodes.semantic.litellm.acompletion",
                new_callable=AsyncMock,
                side_effect=_semantic_pass,
            ),
        ):
            result = await semantic_verifier_node(_state(4))
        assert [s["sentence_id"] for s in result["final_sentences"]] == [
            "s_00",
            "s_01",
            "s_02",
            "s_03",
        ]
