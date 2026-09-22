"""
Pre-LLM answerability gate and the multilingual BM25 tokenizer it depends on.

The gate used to compare the best ``ranking_score`` against a 0.15 floor, but
chunk *quality* alone clears that floor: a 100-character chunk from an unknown
domain with zero query relevance scores 0.153. So the gate only ever fired on an
empty retrieval. It now also requires lexical overlap with the query (unless the
dense arm ran, which covers vocabulary mismatch).

That rule is only safe if tokenization works for every script. The old
tokenizer (``[a-z0-9]+``) dropped Arabic and CJK entirely, giving every
non-Latin query zero relevance — so the tokenizer is fixed first.
"""

from __future__ import annotations

from axiom_rag_engine.nodes.ranker import _tokenize, compute_relevance_score
from axiom_rag_engine.nodes.synthesizer import _pre_llm_unanswerable_reason


class TestMultilingualTokenizer:
    def test_english_tokens_and_stopwords_unchanged(self) -> None:
        assert _tokenize("What is the boiling point of water?") == ["boiling", "point", "water"]

    def test_accented_latin_word_is_one_token(self) -> None:
        assert _tokenize("naïve café") == ["naïve", "café"]

    def test_arabic_words_are_tokens(self) -> None:
        assert _tokenize("عاصمة فرنسا") == ["عاصمة", "فرنسا"]

    def test_cjk_run_becomes_character_bigrams(self) -> None:
        assert _tokenize("沸点是多少") == ["沸点", "点是", "是多", "多少"]

    def test_single_cjk_character_is_kept(self) -> None:
        assert _tokenize("水") == ["水"]

    def test_mixed_script_token_is_split_by_script(self) -> None:
        assert _tokenize("5g网络") == ["5g", "网络"]


class TestNonLatinRelevance:
    def test_arabic_query_matches_arabic_chunk(self) -> None:
        score = compute_relevance_score("ما هي عاصمة فرنسا", "باريس هي عاصمة فرنسا وأكبر مدنها")
        assert score > 0

    def test_cjk_query_matches_cjk_chunk(self) -> None:
        score = compute_relevance_score(
            "水在标准大气压下的沸点是多少", "在标准大气压下，水的沸点是摄氏100度。"
        )
        assert score > 0


def _chunk(relevance: float, ranking: float = 0.5, **extra: float) -> dict:
    return {
        "chunk_id": "doc_1_chunk_A",
        "text": "x",
        "relevance_score": relevance,
        "ranking_score": ranking,
        **extra,
    }


class TestAnswerabilityGate:
    def test_fires_when_no_chunk_shares_a_query_term(self) -> None:
        reason = _pre_llm_unanswerable_reason([_chunk(0.0), _chunk(0.0)])
        assert reason is not None
        assert "query term" in reason

    def test_clears_when_a_chunk_shares_a_query_term(self) -> None:
        assert _pre_llm_unanswerable_reason([_chunk(0.0), _chunk(0.2)]) is None

    def test_dense_scores_waive_the_lexical_floor(self) -> None:
        # Hybrid retrieval ran: a paraphrase-only match is legitimate.
        assert _pre_llm_unanswerable_reason([_chunk(0.0, dense_score=0.71)]) is None

    def test_ranking_floor_still_applies(self) -> None:
        assert _pre_llm_unanswerable_reason([_chunk(0.3, ranking=0.05)]) is not None

    def test_unranked_chunks_defer_to_caller(self) -> None:
        assert _pre_llm_unanswerable_reason([{"chunk_id": "doc_1_chunk_A", "text": "x"}]) is None

    def test_irrelevant_golden_case_is_caught_deterministically(self) -> None:
        # Mirrors evals/golden seed gold_unanswerable_irrelevant end to end
        # through the real ranker scoring.
        query = "What is the error-correction threshold for topological quantum codes?"
        chunk_text = (
            "Tomatoes grow best in full sun with well-drained soil. Water them deeply "
            "once or twice a week and add mulch to keep the roots cool in summer."
        )
        relevance = compute_relevance_score(query, chunk_text)
        assert relevance == 0.0
        assert _pre_llm_unanswerable_reason([_chunk(relevance, ranking=0.4)]) is not None
