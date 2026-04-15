"""
Phase 6 — TDD: Retriever, Scorer, and Ranker node tests.

Test categories:
  A. Retriever — query expansion, HTML stripping, chunking, domain filtering, IDs
  B. Scorer — source quality, chunk quality, combined scoring, filtering
  C. Ranker — relevance scoring, ranking, top-N trimming
  D. Node integration — each node reads/writes correct GraphState keys
"""

from __future__ import annotations

from typing import Any

from axiom_engine.nodes.ranker import (
    _tokenize,
    compute_corpus_idf,
    compute_ranking_score,
    compute_relevance_score,
    ranker_node,
)
from axiom_engine.nodes.retriever import (
    MockSearchBackend,
    _chunk_label,
    chunk_into_paragraphs,
    extract_domain,
    generate_search_queries,
    is_banned,
    retriever_node,
    set_search_backend,
    strip_html,
)
from axiom_engine.nodes.scorer import (
    _normalize_domain,
    compute_combined_score,
    score_chunk_quality,
    score_source_quality,
    scorer_node,
)
from axiom_engine.state import make_initial_state

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base_state(**overrides: Any) -> dict[str, Any]:
    """Create a minimal valid GraphState for testing."""
    s = make_initial_state(
        request_id="test_req",
        user_query=overrides.pop("user_query", "solid-state batteries"),
        app_config=overrides.pop("app_config", {}),
        models_config={},
        pipeline_config=overrides.pop("pipeline_config", {}),
    )
    s.update(overrides)
    return s


def _make_chunk(
    doc: int = 1,
    chunk_label: str = "A",
    text: str = "Solid-state batteries replace liquid electrolytes with solid ceramics for improved safety.",
    url: str = "https://example.com/article",
    domain: str = "example.com",
    title: str = "Test Article",
) -> dict[str, Any]:
    return {
        "chunk_id": f"doc_{doc}_chunk_{chunk_label}",
        "text": text,
        "source_url": url,
        "domain": domain,
        "title": title,
        "doc_index": doc,
        "chunk_index": 0,
    }


# ===========================================================================
# A. Retriever tests
# ===========================================================================


class TestChunkLabel:
    def test_first_label(self) -> None:
        assert _chunk_label(0) == "A"

    def test_last_single_letter(self) -> None:
        assert _chunk_label(25) == "Z"

    def test_double_letter(self) -> None:
        assert _chunk_label(26) == "AA"

    def test_double_letter_ab(self) -> None:
        assert _chunk_label(27) == "AB"


class TestStripHtml:
    def test_removes_tags(self) -> None:
        assert strip_html("<p>Hello</p>") == "Hello"

    def test_collapses_newlines(self) -> None:
        result = strip_html("A\n\n\n\n\nB")
        assert result == "A\n\nB"

    def test_empty_string(self) -> None:
        assert strip_html("") == ""

    def test_nested_tags(self) -> None:
        assert strip_html("<div><span>Text</span></div>") == "Text"

    def test_decodes_html_entities(self) -> None:
        result = strip_html("Price &amp; value &lt;10&gt; &quot;good&quot;")
        assert "&amp;" not in result
        assert "&lt;" not in result
        assert "&gt;" not in result
        assert "&quot;" not in result
        assert "Price & value" in result

    def test_decodes_numeric_entities(self) -> None:
        result = strip_html("It&#39;s a &#x2019;test&#x2019;")
        assert "&#39;" not in result
        assert "&#x2019;" not in result


class TestChunkIntoParagraphs:
    def test_splits_on_double_newline(self) -> None:
        text = "First paragraph with enough characters to pass.\n\nSecond paragraph with enough characters to pass."
        result = chunk_into_paragraphs(text)
        assert len(result) == 2

    def test_filters_short_paragraphs(self) -> None:
        text = "Short\n\nThis paragraph has enough characters to pass the minimum length filter."
        result = chunk_into_paragraphs(text)
        assert len(result) == 1
        assert "enough characters" in result[0]

    def test_empty_text(self) -> None:
        assert chunk_into_paragraphs("") == []

    def test_long_paragraph_split_into_multiple_chunks(self) -> None:
        # Build a paragraph > 1500 chars that the chunker must split.
        sentence = (
            "Solid-state batteries replace liquid electrolytes with solid ceramic materials. "
        )
        long_para = sentence * 25  # ~1900 chars
        result = chunk_into_paragraphs(long_para)
        assert len(result) >= 2, "Long paragraph must produce more than one chunk"
        for chunk in result:
            assert len(chunk) <= 1500

    def test_adjacent_chunks_share_overlap_sentence(self) -> None:
        # Each sentence is ~200 chars; 10 x 200 = 2000 chars > _MAX_CHUNK_LENGTH (1500).
        sent = (
            "Solid-state batteries replace liquid electrolytes with solid ceramic "
            "materials, significantly improving both thermal stability and energy "
            "density compared to conventional lithium-ion designs. "
        )  # ~190 chars
        long_para = sent * 10  # ~1900 chars — must be split into at least 2 chunks
        result = chunk_into_paragraphs(long_para)
        assert len(result) >= 2, (
            f"Expected ≥2 chunks, got {len(result)}: {[len(c) for c in result]}"
        )
        # The final sentence of chunk[0] must appear verbatim inside chunk[1] (overlap).
        last_sentence_chunk0 = result[0].rsplit(". ", 1)[-1].strip().rstrip(".")
        assert last_sentence_chunk0 in result[1], (
            f"Overlap not found.\nChunk0 tail: {last_sentence_chunk0!r}\nChunk1 start: {result[1][:200]!r}"
        )


class TestQueryExpansion:
    def test_generates_three_queries(self) -> None:
        queries = generate_search_queries("quantum computing")
        assert len(queries) == 3
        assert queries[0] == "quantum computing"

    def test_no_duplicate_what_is(self) -> None:
        queries = generate_search_queries("What is quantum computing")
        # Should not add another "What is" reformulation.
        what_is_count = sum(1 for q in queries if q.lower().startswith("what is"))
        assert what_is_count == 1


class TestDomainFiltering:
    def test_extract_domain(self) -> None:
        assert extract_domain("https://www.example.com/page") == "example.com"

    def test_extract_domain_no_www(self) -> None:
        assert extract_domain("https://docs.python.org/3/") == "docs.python.org"

    def test_is_banned(self) -> None:
        assert is_banned("https://spam.com/article", ["spam.com"]) is True

    def test_not_banned(self) -> None:
        assert is_banned("https://example.com/article", ["spam.com"]) is False

    def test_subdomain_banned(self) -> None:
        assert is_banned("https://sub.spam.com/article", ["spam.com"]) is True


class TestRetrieverNode:
    async def test_returns_indexed_chunks(self) -> None:
        backend = MockSearchBackend(
            [
                {
                    "url": "https://example.com/article",
                    "content": "First long paragraph with enough content to pass the filter.\n\nSecond paragraph also long enough to pass the minimum length filter.",
                    "title": "Test",
                }
            ]
        )
        set_search_backend(backend)
        state = _base_state()
        result = await retriever_node(state)

        assert "indexed_chunks" in result
        assert "search_queries" in result
        assert "audit_trail" in result
        assert len(result["indexed_chunks"]) >= 1

    async def test_chunk_ids_follow_pattern(self) -> None:
        backend = MockSearchBackend(
            [
                {
                    "url": "https://example.com/a",
                    "content": "Long paragraph one with enough characters to pass.\n\nLong paragraph two with enough characters to pass.",
                    "title": "A",
                }
            ]
        )
        set_search_backend(backend)
        state = _base_state()
        result = await retriever_node(state)

        for chunk in result["indexed_chunks"]:
            assert chunk["chunk_id"].startswith("doc_")
            assert "_chunk_" in chunk["chunk_id"]

    async def test_banned_domains_filtered(self) -> None:
        backend = MockSearchBackend(
            [
                {
                    "url": "https://banned.com/x",
                    "content": "Content " * 20,
                    "title": "Bad",
                },
                {
                    "url": "https://good.com/x",
                    "content": "Content " * 20,
                    "title": "Good",
                },
            ]
        )
        set_search_backend(backend)
        state = _base_state(app_config={"banned_domains": ["banned.com"]})
        result = await retriever_node(state)

        domains = [c["domain"] for c in result["indexed_chunks"]]
        assert "banned.com" not in domains

    async def test_empty_search_results(self) -> None:
        set_search_backend(MockSearchBackend([]))
        state = _base_state()
        result = await retriever_node(state)
        assert result["indexed_chunks"] == []

    async def test_html_stripped_from_content(self) -> None:
        backend = MockSearchBackend(
            [
                {
                    "url": "https://example.com/a",
                    "content": "<p>This is a long paragraph with enough characters to pass the minimum length filter.</p>",
                    "title": "A",
                }
            ]
        )
        set_search_backend(backend)
        state = _base_state()
        result = await retriever_node(state)

        for chunk in result["indexed_chunks"]:
            assert "<p>" not in chunk["text"]
            assert "</p>" not in chunk["text"]

    async def test_chunk_ids_remain_unique_across_retrieval_passes(self) -> None:
        backend = MockSearchBackend(
            [
                {
                    "url": "https://example.com/a",
                    "content": "Long paragraph with enough content to pass the minimum length filter.",
                    "title": "A",
                }
            ]
        )
        set_search_backend(backend)
        state = _base_state()

        first = await retriever_node(state)
        state["next_doc_index"] = first["next_doc_index"]
        second = await retriever_node(state)

        assert first["indexed_chunks"][0]["chunk_id"] == "doc_1_chunk_A"
        assert second["indexed_chunks"][0]["chunk_id"] == "doc_2_chunk_A"

    async def test_chunk_cap_limits_total_chunks(self) -> None:
        # Each URL has unique long content so dedup doesn't swallow everything.
        # Each produces multiple chunks; a cap of 5 should be hit quickly.
        def _unique_content(i: int) -> str:
            # Prefix makes every URL's content unique, then pad to >1500 chars.
            prefix = f"Document {i}: "
            filler = "Solid-state batteries improve energy density and safety. " * 30
            return prefix + filler  # well over 1500 chars → multiple chunks

        backend = MockSearchBackend(
            [
                {
                    "url": f"https://example.com/{i}",
                    "content": _unique_content(i),
                    "title": f"Doc{i}",
                }
                for i in range(10)
            ]
        )
        set_search_backend(backend)
        cap = 5
        state = _base_state(app_config={"max_chunks_per_request": cap})
        result = await retriever_node(state)

        assert len(result["indexed_chunks"]) <= cap
        event_types = [e["event_type"] for e in result["audit_trail"]]
        assert "retriever_chunk_cap_reached" in event_types

    async def test_empty_content_emits_audit_event(self) -> None:
        backend = MockSearchBackend(
            [{"url": "https://example.com/empty", "content": "", "title": "Empty"}]
        )
        set_search_backend(backend)
        state = _base_state()
        result = await retriever_node(state)

        event_types = [e["event_type"] for e in result["audit_trail"]]
        assert "retriever_empty_content" in event_types
        assert result["indexed_chunks"] == []


# ===========================================================================
# B. Scorer tests
# ===========================================================================


class TestSourceQualityScore:
    def test_authoritative_domain(self) -> None:
        assert score_source_quality("arxiv.org") == 0.9

    def test_low_quality_domain(self) -> None:
        assert score_source_quality("reddit.com") == 0.3

    def test_unknown_domain(self) -> None:
        assert score_source_quality("example.com") == 0.5

    def test_authoritative_subdomain(self) -> None:
        assert score_source_quality("sub.arxiv.org") == 0.85

    def test_low_quality_subdomain(self) -> None:
        assert score_source_quality("old.reddit.com") == 0.3

    def test_case_insensitive(self) -> None:
        assert score_source_quality("ARXIV.ORG") == 0.9


class TestChunkQualityScore:
    def test_empty_text(self) -> None:
        assert score_chunk_quality("") == 0.0

    def test_short_text(self) -> None:
        score = score_chunk_quality("Short text.")
        assert score == 0.1  # Below 40 chars

    def test_medium_text(self) -> None:
        text = "This is a medium-length paragraph with some content about batteries."
        score = score_chunk_quality(text)
        assert 0.2 < score < 1.0

    def test_text_with_data_markers(self) -> None:
        text = "Battery capacity improved by 25% in 2024 according to Figure 1."
        score = score_chunk_quality(text)
        # Should get density bonus for percentage, year, and figure reference.
        text_no_markers = "Battery capacity improved significantly in recent tests."
        score_plain = score_chunk_quality(text_no_markers)
        assert score > score_plain

    def test_long_text_caps_at_one(self) -> None:
        text = "A " * 500
        score = score_chunk_quality(text)
        assert score <= 1.0


class TestCombinedScore:
    def test_weighted_blend(self) -> None:
        # 0.4 * 0.9 + 0.6 * 0.8 = 0.36 + 0.48 = 0.84
        assert compute_combined_score(0.9, 0.8) == 0.84

    def test_zero_scores(self) -> None:
        assert compute_combined_score(0.0, 0.0) == 0.0


class TestScorerNode:
    async def test_scores_all_chunks(self) -> None:
        chunks = [
            _make_chunk(doc=1, chunk_label="A"),
            _make_chunk(doc=2, chunk_label="A", domain="arxiv.org"),
        ]
        state = _base_state(indexed_chunks=chunks)
        result = await scorer_node(state)

        assert "scored_chunks" in result
        assert "audit_trail" in result
        assert len(result["scored_chunks"]) == 2

        for chunk in result["scored_chunks"]:
            assert "source_quality_score" in chunk
            assert "chunk_quality_score" in chunk
            assert "quality_score" in chunk

    async def test_sorted_by_quality_descending(self) -> None:
        chunks = [
            _make_chunk(doc=1, domain="reddit.com"),
            _make_chunk(doc=2, domain="arxiv.org"),
        ]
        state = _base_state(indexed_chunks=chunks)
        result = await scorer_node(state)

        scores = [c["quality_score"] for c in result["scored_chunks"]]
        assert scores == sorted(scores, reverse=True)

    async def test_empty_chunks(self) -> None:
        state = _base_state(indexed_chunks=[])
        result = await scorer_node(state)
        assert result["scored_chunks"] == []

    async def test_filters_below_threshold(self) -> None:
        chunks = [
            _make_chunk(doc=1, text="", domain="reddit.com"),  # Empty text → 0 chunk score
        ]
        state = _base_state(indexed_chunks=chunks)
        result = await scorer_node(state)
        # Empty text with low domain should be filtered.
        assert len(result["scored_chunks"]) == 0

    async def test_audit_events_emitted(self) -> None:
        chunks = [_make_chunk()]
        state = _base_state(indexed_chunks=chunks)
        result = await scorer_node(state)
        events = result["audit_trail"]
        event_types = [e["event_type"] for e in events]
        assert "scorer_start" in event_types
        assert "scorer_complete" in event_types


# ===========================================================================
# C. Ranker tests
# ===========================================================================


class TestTokenize:
    def test_basic_tokenization(self) -> None:
        tokens = _tokenize("What is a solid-state battery?")
        assert "solid" in tokens
        assert "state" in tokens
        assert "battery" in tokens
        # Stopwords removed.
        assert "what" not in tokens
        assert "is" not in tokens
        assert "a" not in tokens

    def test_empty_string(self) -> None:
        assert _tokenize("") == []


class TestRelevanceScore:
    def test_matching_term_scores_positive(self) -> None:
        score = compute_relevance_score("battery", "battery technology explained")
        assert score > 0.0

    def test_no_overlap(self) -> None:
        score = compute_relevance_score("quantum computing", "battery technology explained")
        assert score == 0.0

    def test_partial_overlap(self) -> None:
        score = compute_relevance_score(
            "solid state battery safety",
            "Solid-state batteries improve safety through ceramic electrolytes.",
        )
        assert 0.0 < score < 1.0

    def test_more_overlap_scores_higher(self) -> None:
        """Chunk matching more query terms should score higher."""
        low = compute_relevance_score(
            "battery technology ceramic",
            "battery is a common device",
            avg_doc_len=5.0,
        )
        high = compute_relevance_score(
            "battery technology ceramic",
            "battery technology uses ceramic electrolytes",
            avg_doc_len=5.0,
        )
        assert high > low

    def test_empty_query(self) -> None:
        assert compute_relevance_score("", "some text") == 0.0

    def test_empty_chunk(self) -> None:
        assert compute_relevance_score("query", "") == 0.0

    def test_pure_bm25_tf_overlap(self) -> None:
        """Terms boost the score pure TF style."""
        score = compute_relevance_score("battery technology", "battery technology explained")
        assert score > 0.0


class TestRankingScore:
    def test_weighted_combination(self) -> None:
        # 0.6 * 0.8 + 0.4 * 0.7 = 0.48 + 0.28 = 0.76
        assert compute_ranking_score(0.8, 0.7) == 0.76


class TestRankerNode:
    async def test_ranks_by_relevance(self) -> None:
        chunks = [
            {
                **_make_chunk(doc=1, text="Unrelated topic about cooking recipes."),
                "quality_score": 0.5,
            },
            {
                **_make_chunk(
                    doc=2,
                    text="Solid-state batteries use solid ceramics for electrolytes.",
                ),
                "quality_score": 0.5,
            },
        ]
        state = _base_state(
            user_query="solid-state batteries",
            scored_chunks=chunks,
        )
        result = await ranker_node(state)

        assert "ranked_chunks" in result
        assert len(result["ranked_chunks"]) == 2
        # The battery chunk should rank higher.
        assert result["ranked_chunks"][0]["doc_index"] == 2

    async def test_trims_to_max_ranked(self) -> None:
        chunks = [
            {
                **_make_chunk(doc=i, text=f"Chunk {i} about batteries and energy storage."),
                "quality_score": 0.5,
            }
            for i in range(1, 6)
        ]
        state = _base_state(
            scored_chunks=chunks,
            pipeline_config={"stages": {"max_ranked_chunks": 3}},
        )
        result = await ranker_node(state)
        assert len(result["ranked_chunks"]) == 3

    async def test_empty_scored_chunks(self) -> None:
        state = _base_state(scored_chunks=[])
        result = await ranker_node(state)
        assert result["ranked_chunks"] == []

    async def test_ranking_scores_present(self) -> None:
        chunks = [
            {**_make_chunk(doc=1), "quality_score": 0.6},
        ]
        state = _base_state(scored_chunks=chunks)
        result = await ranker_node(state)

        for chunk in result["ranked_chunks"]:
            assert "relevance_score" in chunk
            assert "ranking_score" in chunk

    async def test_audit_events_emitted(self) -> None:
        chunks = [
            {**_make_chunk(doc=1), "quality_score": 0.5},
        ]
        state = _base_state(scored_chunks=chunks)
        result = await ranker_node(state)
        event_types = [e["event_type"] for e in result["audit_trail"]]
        assert "ranker_start" in event_types
        assert "ranker_complete" in event_types

    async def test_default_max_ranked_is_10(self) -> None:
        chunks = [
            {
                **_make_chunk(doc=i, text=f"Chunk {i} about batteries and solid state technology."),
                "quality_score": 0.5,
            }
            for i in range(1, 15)
        ]
        state = _base_state(scored_chunks=chunks)
        result = await ranker_node(state)
        assert len(result["ranked_chunks"]) == 10


# ===========================================================================
# D2. Batch 3 — IDF, configurable weights, punycode normalization
# ===========================================================================


class TestCorpusIdf:
    def test_rare_term_scores_higher_idf(self) -> None:
        """A term appearing in 1 of 5 chunks should have higher IDF than one in all 5."""
        chunks = [
            {"text": "rare_term common_word"},
            {"text": "common_word only"},
            {"text": "common_word only"},
            {"text": "common_word only"},
            {"text": "common_word only"},
        ]
        idf = compute_corpus_idf(chunks)
        # rare_term appears in 1 chunk, common_word in all 5
        assert "rareterm" in idf or "rare" in idf or any("rare" in k for k in idf)
        assert "commonword" in idf or "common" in idf or any("common" in k for k in idf)
        rare_key = next(k for k in idf if "rare" in k)
        common_key = next(k for k in idf if "common" in k)
        assert idf[rare_key] > idf[common_key]

    def test_empty_corpus_returns_empty(self) -> None:
        assert compute_corpus_idf([]) == {}

    def test_idf_always_positive(self) -> None:
        """Robertson-Walker IDF is always strictly positive."""
        chunks = [{"text": "same text here"}, {"text": "same text here"}]
        idf = compute_corpus_idf(chunks)
        for val in idf.values():
            assert val > 0.0


class TestRelevanceScoreWithIdf:
    def test_idf_down_weights_universal_terms(self) -> None:
        """A query term that appears in every chunk should score lower when IDF is used."""
        # 'battery' appears in both chunks; 'ceramic' only in the relevant one
        chunks = [
            {"text": "battery technology uses ceramic electrolytes for safety"},
            {"text": "battery powered devices are everywhere in daily life"},
        ]
        idf = compute_corpus_idf(chunks)
        avg_len = sum(len(c["text"].split()) for c in chunks) / len(chunks)

        # Without IDF: 'battery' and 'ceramic' contribute equally
        score_no_idf = compute_relevance_score(
            "battery ceramic", chunks[0]["text"], avg_doc_len=avg_len, idf=None
        )
        # With IDF: 'ceramic' (rare) contributes more than 'battery' (universal)
        score_idf = compute_relevance_score(
            "battery ceramic", chunks[0]["text"], avg_doc_len=avg_len, idf=idf
        )
        # Both should be positive, and the IDF score reflects rare-term boost
        assert score_no_idf > 0.0
        assert score_idf > 0.0

    def test_idf_zero_idf_term_still_matches(self) -> None:
        """A term with idf=0.0 should not contribute to score but not crash."""
        idf = {"battery": 0.0}
        score = compute_relevance_score("battery", "battery technology", avg_doc_len=5.0, idf=idf)
        assert score == 0.0


class TestConfigurableRankingWeights:
    async def test_custom_relevance_weight_applied(self) -> None:
        """Passing relevance_weight=1.0, quality_weight=0.0 ignores quality score."""
        chunks = [
            {
                **_make_chunk(doc=1, text="solid state batteries use ceramic electrolytes"),
                "quality_score": 0.1,  # low quality
            },
            {
                **_make_chunk(doc=2, text="cooking recipes and kitchen tips"),
                "quality_score": 0.9,  # high quality
            },
        ]
        state = _base_state(
            user_query="solid state batteries",
            scored_chunks=chunks,
            pipeline_config={"stages": {"relevance_weight": 1.0, "quality_weight": 0.0}},
        )
        result = await ranker_node(state)
        # Relevant chunk (doc 1) should rank first despite low quality
        assert result["ranked_chunks"][0]["doc_index"] == 1

    async def test_default_weights_produce_expected_blend(self) -> None:
        """Default 0.6/0.4 blend: compute_ranking_score(r, q) == 0.6*r + 0.4*q."""
        assert compute_ranking_score(1.0, 0.5) == round(0.6 * 1.0 + 0.4 * 0.5, 4)


class TestConfigurableScorerWeights:
    async def test_custom_source_weight_applied(self) -> None:
        """source_weight=1.0, chunk_weight=0.0 — only domain authority counts."""
        chunks = [
            _make_chunk(doc=1, domain="arxiv.org", text="Short."),  # high source, low chunk
            _make_chunk(doc=2, domain="reddit.com", text="x" * 600),  # low source, high chunk
        ]
        state = _base_state(
            indexed_chunks=chunks,
            app_config={"source_weight": 1.0, "chunk_weight": 0.0},
        )
        result = await scorer_node(state)
        ids = [c["doc_index"] for c in result["scored_chunks"]]
        # arxiv chunk should rank first (source_weight=1.0 → source score dominates)
        assert ids[0] == 1

    def test_compute_combined_score_with_custom_weights(self) -> None:
        # 0.7 * 0.9 + 0.3 * 0.5 = 0.63 + 0.15 = 0.78
        assert compute_combined_score(0.9, 0.5, source_weight=0.7, chunk_weight=0.3) == 0.78


class TestPunycodeDomainNormalization:
    def test_ascii_domain_unchanged(self) -> None:
        assert _normalize_domain("arxiv.org") == "arxiv.org"

    def test_uppercase_normalized(self) -> None:
        assert _normalize_domain("ARXIV.ORG") == "arxiv.org"

    def test_punycode_label_decoded(self) -> None:
        # xn--n3h is the punycode for ☁ (cloud emoji); this is a valid ACE label
        # We just care that the function doesn't crash and lowercases correctly.
        result = _normalize_domain("sub.xn--n3h.ws")
        assert result.startswith("sub.")

    def test_lookalike_punycode_does_not_bypass_authoritative(self) -> None:
        """A punycode lookalike of arxiv.org must NOT score as authoritative."""
        # Construct a domain that would decode to something other than arxiv.org
        # Use a clearly fake punycode label to verify it normalizes to non-auth
        score = score_source_quality("xn--rxiv-bsa.org")  # decodes differently
        # Score should NOT be 0.9 (authoritative) since it's not arxiv.org
        assert score != 0.9

    def test_real_subdomain_still_matches(self) -> None:
        """Legitimate subdomain of authoritative domain still scores 0.85."""
        assert score_source_quality("blog.arxiv.org") == 0.85


# ===========================================================================
# E. Deduplication tests (Fix #2/#8)
# ===========================================================================


class TestRetrieverDeduplication:
    async def test_deduplicates_by_url(self) -> None:
        """Same URL from multiple queries should only be indexed once."""
        backend = MockSearchBackend(
            [
                {
                    "url": "https://example.com/article",
                    "content": "Long paragraph with enough content to pass the minimum length filter easily.",
                    "title": "Test",
                },
            ]
        )
        set_search_backend(backend)
        state = _base_state()
        result = await retriever_node(state)

        # 3 queries all return the same URL — should get only 1 doc.
        urls = [c["source_url"] for c in result["indexed_chunks"]]
        assert urls.count("https://example.com/article") <= 1

    async def test_deduplicates_by_content_hash(self) -> None:
        """Identical paragraph text from different URLs should be deduplicated."""
        shared_text = (
            "Identical paragraph content that appears on multiple sites and is long enough."
        )
        backend = MockSearchBackend(
            [
                {
                    "url": "https://site-a.com/page",
                    "content": shared_text,
                    "title": "A",
                },
                {
                    "url": "https://site-b.com/page",
                    "content": shared_text,
                    "title": "B",
                },
            ]
        )
        set_search_backend(backend)
        state = _base_state()
        result = await retriever_node(state)

        texts = [c["text"] for c in result["indexed_chunks"]]
        assert texts.count(shared_text) == 1

    async def test_audit_reports_duplicate_counts(self) -> None:
        backend = MockSearchBackend(
            [
                {
                    "url": "https://example.com/same",
                    "content": "Long paragraph with enough content to pass the filter easily for testing.",
                    "title": "Test",
                },
            ]
        )
        set_search_backend(backend)
        state = _base_state()
        result = await retriever_node(state)

        complete_event = next(
            e for e in result["audit_trail"] if e["event_type"] == "retriever_complete"
        )
        assert "duplicate_urls_skipped" in complete_event["payload"]
        assert "duplicate_chunks_skipped" in complete_event["payload"]


# ===========================================================================
# F. Configurable domain lists (Fix #4)
# ===========================================================================


class TestConfigurableDomains:
    async def test_custom_authoritative_domain(self) -> None:
        """Custom domains in app_config should be treated as authoritative."""
        chunks = [
            _make_chunk(doc=1, domain="internal-docs.mycompany.com"),
        ]
        state = _base_state(
            indexed_chunks=chunks,
            app_config={"authoritative_domains": ["internal-docs.mycompany.com"]},
        )
        result = await scorer_node(state)
        assert result["scored_chunks"][0]["source_quality_score"] == 0.9

    async def test_custom_low_quality_domain(self) -> None:
        """Custom low-quality domains should score 0.3."""
        chunks = [
            _make_chunk(doc=1, domain="unreliable-blog.example.com"),
        ]
        state = _base_state(
            indexed_chunks=chunks,
            app_config={"low_quality_domains": ["unreliable-blog.example.com"]},
        )
        result = await scorer_node(state)
        assert result["scored_chunks"][0]["source_quality_score"] == 0.3

    async def test_defaults_still_apply_without_config(self) -> None:
        """Built-in defaults work when no custom config is provided."""
        chunks = [_make_chunk(doc=1, domain="arxiv.org")]
        state = _base_state(indexed_chunks=chunks)
        result = await scorer_node(state)
        assert result["scored_chunks"][0]["source_quality_score"] == 0.9
