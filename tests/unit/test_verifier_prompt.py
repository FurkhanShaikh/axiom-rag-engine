"""
The faithfulness verifier sees only the claim, quote and chunk text.

Its prompt used to include every chunk field — domain, URL, title and the
scorer's authority and quality scores — while telling the model not to infer
authority: an invitation to trust "authoritative" sources more. Authority is the
tiers' job, computed deterministically from the domain.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

from axiom_rag_engine.models import Citation
from axiom_rag_engine.nodes.semantic import _verify_citation


async def test_prompt_carries_no_source_metadata() -> None:
    seen: list[Any] = []

    async def _fake(node: str, model: str, messages: Any, **_: Any) -> str:
        seen.append(messages)
        return json.dumps({"semantic_check": "passed", "failure_reason": None})

    chunk = {
        "chunk_id": "doc_1_chunk_A",
        "text": "Solid-state batteries replace liquid electrolytes with solid ceramics.",
        "domain": "energy.gov",
        "source_url": "https://energy.gov/very-official",
        "title": "Official Government Battery Report",
        "source_quality_score": 0.9,
        "quality_score": 0.87,
        "ranking_score": 0.91,
    }
    citation = Citation(
        citation_id="c1",
        chunk_id="doc_1_chunk_A",
        exact_source_quote="Solid-state batteries replace liquid electrolytes with solid ceramics",
    )
    with patch("axiom_rag_engine.nodes.semantic.call_llm", _fake):
        await _verify_citation(
            "Solid-state batteries use ceramics.", citation, {"doc_1_chunk_A": chunk}, "m", set()
        )

    prompt = json.dumps(seen[0])
    assert "solid ceramics" in prompt  # the text is there
    for leaked in ("energy.gov", "Official Government", "quality_score", "0.87", "META"):
        assert leaked not in prompt
