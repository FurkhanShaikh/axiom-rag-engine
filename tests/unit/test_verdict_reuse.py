"""
Semantic verdicts are reused across passes of the same request.

A rewrite keeps the sentences that verified, yet every pass re-sent a verifier
call for every citation: wasted calls and budget (a common path into running
out of budget on a rewrite), and verdicts that could flip on identical input.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

from axiom_rag_engine.graph import build_axiom_graph
from axiom_rag_engine.state import make_initial_state

_TEXT = (
    "Solid-state batteries replace liquid electrolytes with solid ceramics. "
    "This substitution significantly improves thermal stability and energy density of cells."
)
_KEEP = {
    "sentence_id": "s_01",
    "text": "Solid-state batteries use solid ceramics.",
    "is_cited": True,
    "citations": [
        {
            "citation_id": "cite_1",
            "chunk_id": "doc_1_chunk_A",
            "exact_source_quote": "Solid-state batteries replace liquid electrolytes with solid ceramics",
        }
    ],
}


def _second(quote: str, text: str = "It improves thermal stability.") -> dict[str, Any]:
    return {
        "sentence_id": "s_02",
        "text": text,
        "is_cited": True,
        "citations": [
            {"citation_id": "cite_2", "chunk_id": "doc_1_chunk_A", "exact_source_quote": quote}
        ],
    }


_FIRST = json.dumps(
    {"is_answerable": True, "sentences": [_KEEP, _second("nuclear reactors inside every cell")]}
)
_REWRITE = json.dumps(
    {
        "is_answerable": True,
        "sentences": [
            _KEEP,
            _second("This substitution significantly improves thermal stability"),
        ],
    }
)


class _Backend:
    def search(self, query: str) -> list[dict[str, Any]]:
        return [{"url": "https://example.com/a", "title": "A", "content": _TEXT}]


async def _run(fail_first_semantic: bool = False) -> tuple[dict[str, Any], list[str]]:
    synth = [_FIRST, _REWRITE]
    judged: list[str] = []

    async def _fake(node: str, model: str, messages: Any, **_: Any) -> str:
        if node == "synthesizer":
            return synth.pop(0)
        claim = messages[1]["content"].split("\n")[1]
        judged.append(claim)
        if fail_first_semantic and len(judged) == 1:
            raise RuntimeError("verifier hiccup")
        return json.dumps({"semantic_check": "passed", "failure_reason": None})

    state = make_initial_state(
        request_id="req_reuse",
        user_query="What are solid-state batteries?",
        app_config={},
        models_config={"synthesizer": "mock/s", "verifier": "mock/v"},
        pipeline_config={"stages": {"semantic_verification_enabled": True}},
    )
    with (
        patch("axiom_rag_engine.nodes.synthesizer.call_llm", _fake),
        patch("axiom_rag_engine.nodes.semantic.call_llm", _fake),
    ):
        result = await build_axiom_graph().ainvoke(
            state, config={"configurable": {"search_backend": _Backend()}}
        )
    return result, judged


async def test_unchanged_sentence_is_not_rejudged_on_rewrite() -> None:
    result, judged = await _run()

    # Pass 1 judges s_01 (s_02's quote fails mechanically). Pass 2 judges only
    # the rewritten s_02; s_01 is unchanged and its verdict is reused.
    assert judged == [_KEEP["text"], "It improves thermal stability."]
    assert [s["verification"]["tier"] for s in result["final_sentences"]] == [3, 3]
    reused = [e for e in result["audit_trail"] if e["event_type"] == "semantic_verdicts_reused"]
    assert reused and reused[0]["payload"]["count"] == 1


async def test_errored_check_is_retried_next_pass() -> None:
    # s_01's first check errors (labelled unverified). That is not a verdict,
    # so the rewrite pass judges s_01 again instead of reusing the error.
    _result, judged = await _run(fail_first_semantic=True)
    assert judged.count(_KEEP["text"]) == 2
