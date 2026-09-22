"""
Tier 1 ("Authoritative") is about the *document*, not just the domain.

Several primary-source domains also host user-generated content — forums,
mailing-list archives, Q&A, public comments. Subdomain matching let those inherit
Tier 1 (users.rust-lang.org, lists.w3.org, regulations.gov public comments, ...).
They remain citable (Tier 2/3) but can no longer be "Authoritative".
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from axiom_rag_engine.nodes.scorer import build_primary_domain_set, is_primary_source
from axiom_rag_engine.nodes.semantic import semantic_verifier_node
from axiom_rag_engine.state import make_initial_state

PRIMARY = build_primary_domain_set({})


class TestUserGeneratedContentIsNotPrimary:
    @pytest.mark.parametrize(
        "url",
        [
            "https://users.rust-lang.org/t/some-question/123",
            "https://internals.rust-lang.org/t/pre-rfc/456",
            "https://lists.w3.org/Archives/Public/www-style/2020Jan/0001.html",
            "https://mailarchive.ietf.org/arch/msg/quic/abc",
            "https://learn.microsoft.com/en-us/answers/questions/12345/how-to",
            "https://developer.apple.com/forums/thread/700000",
            "https://www.regulations.gov/comment/EPA-HQ-OAR-2021-0317-0001",
            "https://www.postgresql.org/message-id/CAB1234@mail.gmail.com",
        ],
    )
    def test_ugc_page_on_primary_domain_is_not_primary(self, url: str) -> None:
        from axiom_rag_engine.nodes.retriever import extract_domain

        assert is_primary_source(extract_domain(url), url, PRIMARY) is False

    @pytest.mark.parametrize(
        "url",
        [
            "https://doc.rust-lang.org/book/ch01-00-getting-started.html",
            "https://www.w3.org/TR/css-grid-1/",
            "https://learn.microsoft.com/en-us/azure/storage/blobs/overview",
            "https://www.regulations.gov/document/EPA-HQ-OAR-2021-0317-0002",
            "https://www.postgresql.org/docs/current/sql-select.html",
            "https://www.cdc.gov/flu/vaccines-work/index.html",
        ],
    )
    def test_official_pages_stay_primary(self, url: str) -> None:
        from axiom_rag_engine.nodes.retriever import extract_domain

        assert is_primary_source(extract_domain(url), url, PRIMARY) is True

    def test_missing_url_falls_back_to_domain_rule(self) -> None:
        assert is_primary_source("docs.python.org", "", PRIMARY) is True
        assert is_primary_source("users.rust-lang.org", "", PRIMARY) is False


def _semantic_pass(**_: Any) -> MagicMock:
    response = MagicMock()
    response.choices = [
        MagicMock(message=MagicMock(content='{"semantic_check": "passed", "failure_reason": null}'))
    ]
    return response


class TestTierAssignment:
    async def test_forum_citation_does_not_reach_tier_1(self) -> None:
        state = make_initial_state(
            request_id="req",
            user_query="How do Rust lifetimes work?",
            app_config={},
            models_config={"synthesizer": "mock/s", "verifier": "mock/v"},
            pipeline_config={"stages": {"semantic_verification_enabled": True}},
        )
        state["indexed_chunks"] = [
            {
                "chunk_id": "doc_1_chunk_A",
                "text": "Lifetimes ensure that references are valid as long as we need them.",
                "source_url": "https://users.rust-lang.org/t/lifetimes/99",
                "domain": "users.rust-lang.org",
            }
        ]
        state["draft_sentences"] = [
            {
                "sentence_id": "s_01",
                "text": "Lifetimes keep references valid.",
                "is_cited": True,
                "citations": [
                    {
                        "citation_id": "cite_1",
                        "chunk_id": "doc_1_chunk_A",
                        "exact_source_quote": "Lifetimes ensure that references are valid",
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
        with patch(
            "axiom_rag_engine.nodes.semantic.litellm.acompletion",
            new_callable=AsyncMock,
            side_effect=_semantic_pass,
        ):
            result = await semantic_verifier_node(dict(state))
        sentence = result["final_sentences"][0]
        assert sentence["verification"]["tier"] == 3
        assert sentence["citations"][0]["verification"]["tier"] == 3
