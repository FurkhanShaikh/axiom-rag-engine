"""Cross-source corroboration for Tier 2 (opt-in).

When AXIOM_CORROBORATION_ENABLED is on, a provisional Tier 2 (multi-domain
coverage) is confirmed only if >=2 distinct sources independently corroborate the
claim; otherwise it drops to Tier 3. These tests mock the verifier LLM so they
need no keys, routing the two call types (per-citation semantic check vs.
corroboration check) by their system prompt.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from axiom_rag_engine.config.settings import get_settings
from axiom_rag_engine.nodes.semantic import (
    _parse_corroboration_response,
    semantic_verifier_node,
)
from axiom_rag_engine.state import make_initial_state


class TestParseCorroborationResponse:
    def test_parses_true(self) -> None:
        ok, reason = _parse_corroboration_response(
            '{"corroborated": true, "reasoning": "both agree"}'
        )
        assert ok is True
        assert reason == "both agree"

    def test_parses_false(self) -> None:
        ok, _ = _parse_corroboration_response(
            '{"corroborated": false, "reasoning": "different facts"}'
        )
        assert ok is False

    def test_strips_fences_and_think(self) -> None:
        raw = '<think>hmm</think>```json\n{"corroborated": true, "reasoning": "x"}\n```'
        ok, _ = _parse_corroboration_response(raw)
        assert ok is True

    def test_rejects_non_bool(self) -> None:
        with pytest.raises(ValueError, match="corroborated must be a bool"):
            _parse_corroboration_response('{"corroborated": "yes"}')

    def test_rejects_malformed_json(self) -> None:
        with pytest.raises(ValueError, match="not valid JSON"):
            _parse_corroboration_response("not json at all")


# ---------------------------------------------------------------------------
# Integration: the Tier 2 gate
# ---------------------------------------------------------------------------

_SEMANTIC_PASS = json.dumps(
    {"semantic_check": "passed", "failure_reason": None, "reasoning": "faithful"}
)


def _resp(content: str) -> MagicMock:
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    response.usage = None
    return response


def _router(corroboration_json: str) -> Any:
    """Route by system prompt: corroboration vs. per-citation semantic check."""

    async def _route(**kwargs: Any) -> MagicMock:
        system = kwargs["messages"][0]["content"].lower()
        if "corroborate" in system:
            return _resp(corroboration_json)
        return _resp(_SEMANTIC_PASS)

    return _route


def _mech_pass() -> dict[str, Any]:
    return {
        "tier": 3,
        "tier_label": "model_assisted",
        "mechanical_check": "passed",
        "semantic_check": "skipped",
        "failure_reason": None,
    }


# Two chunks from two distinct domains — the multi-domain Tier 2 setup.
_TWO_DOMAIN_CHUNKS = [
    {
        "chunk_id": "doc_1_chunk_A",
        "text": "The measure passed with sixty votes in favor.",
        "source_url": "https://alpha.example.com/a",
        "domain": "alpha.example.com",
    },
    {
        "chunk_id": "doc_2_chunk_B",
        "text": "Reporting confirms the measure received sixty votes to pass.",
        "source_url": "https://beta.example.org/b",
        "domain": "beta.example.org",
    },
]

_TWO_DOMAIN_DRAFT = [
    {
        "sentence_id": "s_01",
        "text": "The measure passed with sixty votes.",
        "is_cited": True,
        "citations": [
            {
                "citation_id": "cite_1",
                "chunk_id": "doc_1_chunk_A",
                "exact_source_quote": "The measure passed with sixty votes in favor.",
            },
            {
                "citation_id": "cite_2",
                "chunk_id": "doc_2_chunk_B",
                "exact_source_quote": "the measure received sixty votes to pass",
            },
        ],
    }
]


def _state() -> dict[str, Any]:
    state = make_initial_state(
        request_id="req",
        user_query="did the measure pass?",
        app_config={"expertise_level": "intermediate", "banned_domains": []},
        models_config={"synthesizer": "mock/s", "verifier": "mock/v"},
        pipeline_config={"stages": {"semantic_verification_enabled": True}},
    )
    state["indexed_chunks"] = _TWO_DOMAIN_CHUNKS
    state["draft_sentences"] = _TWO_DOMAIN_DRAFT
    state["mechanical_results"] = {"cite_1": _mech_pass(), "cite_2": _mech_pass()}
    return state


async def _run(corroboration_json: str) -> dict[str, Any]:
    with patch(
        "axiom_rag_engine.nodes.semantic.litellm.acompletion", new_callable=AsyncMock
    ) as mock:
        mock.side_effect = _router(corroboration_json)
        return await semantic_verifier_node(_state())


class TestCorroborationGate:
    async def test_disabled_by_default_keeps_tier_2(self, monkeypatch) -> None:
        """Without the flag, multi-domain reaches Tier 2 and no corroboration
        call is made."""
        get_settings.cache_clear()  # AXIOM_CORROBORATION_ENABLED unset -> False
        with patch(
            "axiom_rag_engine.nodes.semantic.litellm.acompletion", new_callable=AsyncMock
        ) as mock:
            mock.side_effect = _router('{"corroborated": false}')  # would downgrade IF called
            result = await semantic_verifier_node(_state())
        vr = result["final_sentences"][0]["verification"]
        assert vr["tier"] == 2
        # No corroboration call: every call was a semantic check.
        systems = [c.kwargs["messages"][0]["content"].lower() for c in mock.call_args_list]
        assert not any("corroborate" in s for s in systems)

    async def test_enabled_and_corroborated_keeps_tier_2(self, monkeypatch) -> None:
        monkeypatch.setenv("AXIOM_CORROBORATION_ENABLED", "true")
        get_settings.cache_clear()
        result = await _run('{"corroborated": true, "reasoning": "both state 60 votes"}')
        vr = result["final_sentences"][0]["verification"]
        assert vr["tier"] == 2
        assert any(e["event_type"] == "corroboration_result" for e in result["audit_trail"])

    async def test_enabled_and_not_corroborated_downgrades_to_tier_3(self, monkeypatch) -> None:
        monkeypatch.setenv("AXIOM_CORROBORATION_ENABLED", "true")
        get_settings.cache_clear()
        result = await _run('{"corroborated": false, "reasoning": "different facts"}')
        vr = result["final_sentences"][0]["verification"]
        assert vr["tier"] == 3
        assert vr["failure_reason"] and "corroborate" in vr["failure_reason"].lower()

    async def test_check_error_fails_safe_to_tier_3(self, monkeypatch) -> None:
        """If the corroboration call errors, don't claim corroboration."""
        monkeypatch.setenv("AXIOM_CORROBORATION_ENABLED", "true")
        get_settings.cache_clear()

        async def _route(**kwargs: Any) -> MagicMock:
            system = kwargs["messages"][0]["content"].lower()
            if "corroborate" in system:
                raise RuntimeError("verifier down")
            return _resp(_SEMANTIC_PASS)

        with patch(
            "axiom_rag_engine.nodes.semantic.litellm.acompletion", new_callable=AsyncMock
        ) as mock:
            mock.side_effect = _route
            result = await semantic_verifier_node(_state())
        vr = result["final_sentences"][0]["verification"]
        assert vr["tier"] == 3
        assert any(e["event_type"] == "corroboration_error" for e in result["audit_trail"])
