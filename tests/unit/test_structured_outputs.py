"""
Structured outputs — ask the provider for schema-conforming JSON.

Every JSON-producing call used to request plain JSON mode and then repair the
reply with regex / balanced-brace salvage. Where the provider supports JSON
Schema, the call now carries the schema so malformed replies (and the parse
retries they trigger) become rare; the salvage parser stays as a safety net.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from axiom_rag_engine.models import Citation, DraftSentence, SynthesizerOutput
from axiom_rag_engine.schemas import (
    CONTRADICTION_SCHEMA,
    CORROBORATION_SCHEMA,
    SEMANTIC_VERDICT_SCHEMA,
    SYNTHESIZER_SCHEMA,
)
from axiom_rag_engine.utils.llm import build_completion_kwargs, call_llm, reset_llm_budget


def _reply(content: str) -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content=content))]
    return response


class TestCompletionKwargs:
    def test_schema_capable_provider_gets_json_schema_response_format(self) -> None:
        kwargs = build_completion_kwargs(
            model="gpt-4o-mini", messages=[], json_schema=("verdict", SEMANTIC_VERDICT_SCHEMA)
        )
        fmt = kwargs["response_format"]
        assert fmt["type"] == "json_schema"
        assert fmt["json_schema"]["name"] == "verdict"
        assert fmt["json_schema"]["schema"] == SEMANTIC_VERDICT_SCHEMA

    def test_provider_without_schema_support_falls_back_to_json_mode(self) -> None:
        kwargs = build_completion_kwargs(
            model="mock/unknown-model",
            messages=[],
            json_schema=("verdict", SEMANTIC_VERDICT_SCHEMA),
        )
        assert kwargs["response_format"] == {"type": "json_object"}

    def test_ollama_gets_the_schema_as_its_format(self) -> None:
        kwargs = build_completion_kwargs(
            model="ollama/qwen3:8b", messages=[], json_schema=("verdict", SEMANTIC_VERDICT_SCHEMA)
        )
        assert kwargs["extra_body"]["format"] == SEMANTIC_VERDICT_SCHEMA
        assert "response_format" not in kwargs

    def test_no_schema_keeps_plain_json_mode(self) -> None:
        kwargs = build_completion_kwargs(model="gpt-4o-mini", messages=[])
        assert kwargs["response_format"] == {"type": "json_object"}
        ollama = build_completion_kwargs(model="ollama/qwen3:8b", messages=[])
        assert ollama["extra_body"]["format"] == "json"

    async def test_call_llm_forwards_the_schema(self) -> None:
        reset_llm_budget(max_calls=5)
        with patch(
            "litellm.acompletion", new_callable=AsyncMock, return_value=_reply("{}")
        ) as mock_llm:
            await call_llm(
                "semantic", "gpt-4o-mini", [], json_schema=("v", SEMANTIC_VERDICT_SCHEMA)
            )
        assert mock_llm.call_args.kwargs["response_format"]["type"] == "json_schema"


def _props(schema: dict[str, Any]) -> set[str]:
    return set(schema["properties"])


class TestSchemasMatchTheModels:
    """Drift guards: the hand-written schemas must describe what the parsers accept."""

    def test_synthesizer_schema_mirrors_pydantic_models(self) -> None:
        assert _props(SYNTHESIZER_SCHEMA) == set(SynthesizerOutput.model_fields)
        sentence = SYNTHESIZER_SCHEMA["properties"]["sentences"]["items"]
        assert _props(sentence) == set(DraftSentence.model_fields)
        citation = sentence["properties"]["citations"]["items"]
        assert _props(citation) == set(Citation.model_fields)

    def test_schemas_are_self_contained(self) -> None:
        # Inline only: several providers reject $ref / $defs in response schemas.
        for schema in (
            SYNTHESIZER_SCHEMA,
            SEMANTIC_VERDICT_SCHEMA,
            CORROBORATION_SCHEMA,
            CONTRADICTION_SCHEMA,
        ):
            text = json.dumps(schema)
            assert "$ref" not in text and "$defs" not in text

    def test_verdict_schemas_require_their_decision_field(self) -> None:
        assert "semantic_check" in SEMANTIC_VERDICT_SCHEMA["required"]
        assert "corroborated" in CORROBORATION_SCHEMA["required"]
        assert "contradicted" in CONTRADICTION_SCHEMA["required"]


@pytest.mark.parametrize(
    ("node_call", "expected_name"),
    [("synthesizer", "synthesizer_output"), ("semantic", "semantic_verdict")],
)
async def test_nodes_request_their_schema(node_call: str, expected_name: str) -> None:
    from axiom_rag_engine.nodes.semantic import semantic_verifier_node
    from axiom_rag_engine.nodes.synthesizer import synthesizer_node
    from axiom_rag_engine.state import make_initial_state

    chunk = {
        "chunk_id": "doc_1_chunk_A",
        "text": "Solid-state batteries replace liquid electrolytes with solid ceramics.",
        "source_url": "https://example.com/a",
        "domain": "example.com",
    }
    state = dict(
        make_initial_state(
            request_id="r",
            user_query="solid-state batteries",
            app_config={},
            models_config={"synthesizer": "gpt-4o", "verifier": "gpt-4o-mini"},
            pipeline_config={"stages": {"semantic_verification_enabled": True}},
        )
    )
    state["ranked_chunks"] = [chunk]
    state["indexed_chunks"] = [chunk]
    state["draft_sentences"] = [
        {
            "sentence_id": "s_01",
            "text": "Solid-state batteries use ceramics.",
            "is_cited": True,
            "citations": [
                {
                    "citation_id": "cite_1",
                    "chunk_id": "doc_1_chunk_A",
                    "exact_source_quote": "Solid-state batteries replace liquid electrolytes",
                }
            ],
        }
    ]
    state["mechanical_results"] = {
        "cite_1": {
            "tier": 3,
            "tier_label": "model_assisted",
            "mechanical_check": "passed",
            "semantic_check": "skipped",
            "failure_reason": None,
        }
    }
    replies = {
        "synthesizer": json.dumps({"is_answerable": False, "sentences": []}),
        "semantic": json.dumps({"semantic_check": "passed", "failure_reason": None}),
    }
    with patch(
        "litellm.acompletion", new_callable=AsyncMock, return_value=_reply(replies[node_call])
    ) as mock_llm:
        if node_call == "synthesizer":
            await synthesizer_node(state)
        else:
            await semantic_verifier_node(state)
    fmt = mock_llm.call_args.kwargs["response_format"]
    assert fmt["type"] == "json_schema"
    assert fmt["json_schema"]["name"] == expected_name


class TestIsCitedIsDerivedFromCitations:
    """``is_cited`` is redundant with ``citations``; a model that gets the flag
    wrong (seen live on llama3.2:1b) should not cost a parse retry. The
    citations decide — verification still judges every one of them."""

    def _parse(self, is_cited: bool, citations: list[dict]) -> Any:
        from axiom_rag_engine.nodes.synthesizer import _parse_llm_response

        return _parse_llm_response(
            json.dumps(
                {
                    "is_answerable": True,
                    "sentences": [
                        {
                            "sentence_id": "s_01",
                            "text": "A claim.",
                            "is_cited": is_cited,
                            "citations": citations,
                        }
                    ],
                }
            )
        )

    def test_flag_false_with_citations_becomes_cited(self) -> None:
        citation = {
            "citation_id": "cite_1",
            "chunk_id": "doc_1_chunk_A",
            "exact_source_quote": "some quoted source text here",
        }
        sentence = self._parse(False, [citation]).sentences[0]
        assert sentence.is_cited is True
        assert len(sentence.citations) == 1

    def test_flag_true_without_citations_becomes_uncited(self) -> None:
        sentence = self._parse(True, []).sentences[0]
        assert sentence.is_cited is False
