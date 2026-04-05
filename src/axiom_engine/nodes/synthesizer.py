"""
Axiom Engine v2.3 — Cognitive Synthesizer Node (Module 6)

Responsibilities:
  - Calls the configured heavy LLM via LiteLLM.
  - Forces structured JSON output conforming to SynthesizerOutput schema.
  - Implements the is_answerable escape hatch: if retrieved chunks cannot
    answer the query the LLM sets is_answerable=false and returns no
    sentences, aborting generation before a hallucination loop begins.
  - On rewrite pass: injects rewrite_requests as correction context so the
    LLM understands exactly which citations failed and why.
  - Updates GraphState keys: draft_sentences, is_answerable, audit_trail.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from functools import partial
from typing import Any

import litellm
from pydantic import ValidationError

from axiom_engine.models import SynthesizerOutput
from axiom_engine.state import GraphState
from axiom_engine.utils.audit import make_audit_event
from axiom_engine.utils.llm import build_completion_kwargs

_audit = partial(make_audit_event, "synthesizer")
logger = logging.getLogger("axiom_engine.synthesizer")

# Concurrency limit for LLM calls — prevents overloading providers / local Ollama.
_MAX_CONCURRENT = int(os.environ.get("AXIOM_MAX_CONCURRENT_LLM", "5"))
_llm_semaphore = threading.Semaphore(_MAX_CONCURRENT)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are the Cognitive Synthesizer for the Axiom Engine, a hallucination-free \
research assistant. Your single responsibility is to answer the user's query \
using ONLY the source chunks provided. You must never invent, infer, or \
paraphrase beyond what the chunks explicitly state.

OUTPUT FORMAT — You must respond with a single valid JSON object matching \
this exact schema (no markdown fences, no extra keys):

{
  "is_answerable": <true | false>,
  "sentences": [
    {
      "sentence_id": "s_01",
      "text": "<one complete sentence of your answer>",
      "is_cited": <true | false>,
      "citations": [
        {
          "citation_id": "cite_1",
          "chunk_id": "<exact chunk id from context, e.g. doc_1_chunk_A>",
          "exact_source_quote": "<verbatim substring copied character-for-character from the chunk text>"
        }
      ]
    }
  ]
}

STRICT RULES:
1. is_answerable ESCAPE HATCH: If the provided chunks do not contain \
sufficient information to answer the query, you MUST set is_answerable=false \
and return an empty sentences array. Do NOT fabricate an answer.
2. Every sentence where is_cited=true MUST have at least one citation.
3. exact_source_quote MUST be a verbatim substring copied directly from the \
chunk text. No paraphrasing, no summarising, no smart quotes. Copy the \
characters exactly.
4. chunk_id MUST be the exact ID from the provided context (format: \
doc_<N>_chunk_<X>).
5. sentence_id values must be sequential: s_01, s_02, s_03, ...
6. citation_id values must be sequential: cite_1, cite_2, cite_3, ...
7. Sentences that contain no factual claim (e.g. transition sentences) may \
have is_cited=false and an empty citations array.
8. Do NOT wrap your response in markdown code fences.
"""

_USER_PROMPT_TEMPLATE = """\
EXPERTISE LEVEL: {expertise_level}

USER QUERY:
{user_query}

SOURCE CHUNKS:
{chunks_block}

{rewrite_section}
Answer the query now using only the source chunks above. Output valid JSON only.
"""

_REWRITE_SECTION_TEMPLATE = """\
CORRECTION INSTRUCTIONS (Rewrite Pass {loop_count}):
The following citations from your previous response failed verification. \
You MUST fix every listed failure. Do not repeat the same mistakes.

{rewrite_requests}

"""

_CHUNK_ITEM_TEMPLATE = "--- chunk_id: {chunk_id} ---\n{text}\n"


def _build_chunks_block(ranked_chunks: list[dict[str, Any]]) -> str:
    """Render the ranked chunks into the prompt context block."""
    parts: list[str] = []
    for chunk in ranked_chunks:
        parts.append(
            _CHUNK_ITEM_TEMPLATE.format(
                chunk_id=chunk["chunk_id"],
                text=chunk.get("text", ""),
            )
        )
    return "\n".join(parts)


def _build_rewrite_section(state: GraphState) -> str:
    """Build the correction block injected on rewrite passes."""
    requests: list[str] = list(state.get("rewrite_requests") or [])
    if not requests:
        return ""
    # Deduplicate while preserving order — prevents bloating the context
    # window when the same citation fails across multiple loop iterations.
    seen: set[str] = set()
    unique: list[str] = []
    for r in requests:
        if r not in seen:
            seen.add(r)
            unique.append(r)
    numbered = "\n".join(f"  {i + 1}. {r}" for i, r in enumerate(unique))
    return _REWRITE_SECTION_TEMPLATE.format(
        loop_count=state.get("loop_count", 1),
        rewrite_requests=numbered,
    )


def _parse_llm_response(raw: str) -> SynthesizerOutput:
    """
    Extract and validate JSON from the LLM response string.
    Strips accidental markdown fences before parsing.
    Raises ValueError if the JSON is invalid or fails Pydantic validation.
    """
    # Strip <think>...</think> blocks (common in Qwen-family models).
    clean = re.sub(r"<think>.*?</think>", "", raw.strip(), flags=re.DOTALL)
    # Strip ```json ... ``` or ``` ... ``` fences if the LLM ignored rule 8.
    clean = re.sub(r"^```(?:json)?\s*", "", clean.strip(), flags=re.IGNORECASE)
    clean = re.sub(r"\s*```$", "", clean.strip())

    try:
        data = json.loads(clean)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM response is not valid JSON: {exc}") from exc

    try:
        return SynthesizerOutput.model_validate(data)
    except ValidationError as exc:
        raise ValueError(f"LLM JSON does not match SynthesizerOutput schema: {exc}") from exc


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

MAX_PARSE_RETRIES = 2  # Category 3 error handling: max 2 attempts on malformed output


def synthesizer_node(state: GraphState) -> dict[str, Any]:
    """
    LangGraph node — Cognitive Synthesizer.

    Reads ranked_chunks (or indexed_chunks fallback) from state, calls the
    heavy LLM, validates the structured output, and returns a partial state
    update dict.

    Returns keys: draft_sentences, is_answerable, audit_trail
    """
    audit: list[dict[str, Any]] = []

    models_cfg: dict = state.get("models_config") or {}
    app_cfg: dict = state.get("app_config") or {}

    model: str = models_cfg.get("synthesizer", "claude-3-5-sonnet-20241022")
    expertise_level: str = app_cfg.get("expertise_level", "intermediate")

    # Prefer pre-ranked chunks; fall back to all indexed chunks.
    chunks: list[dict] = list(state.get("ranked_chunks") or state.get("indexed_chunks") or [])

    audit.append(
        _audit(
            "synthesizer_start",
            {
                "model": model,
                "loop_count": state.get("loop_count", 0),
                "chunk_count": len(chunks),
                "is_rewrite": bool(state.get("rewrite_requests")),
            },
        )
    )

    chunks_block = _build_chunks_block(chunks)
    rewrite_section = _build_rewrite_section(state)

    user_prompt = _USER_PROMPT_TEMPLATE.format(
        expertise_level=expertise_level,
        user_query=state["user_query"],
        chunks_block=chunks_block,
        rewrite_section=rewrite_section,
    )

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    last_error: Exception | None = None
    output: SynthesizerOutput | None = None
    raw_content: str = ""  # Initialized so the retry correction message is always safe.

    for attempt in range(1, MAX_PARSE_RETRIES + 1):
        try:
            completion_kwargs = build_completion_kwargs(
                model=model,
                messages=messages,
                temperature=0.0,  # deterministic output for citation integrity
            )
            with _llm_semaphore:
                response = litellm.completion(**completion_kwargs)
            raw_content = response.choices[0].message.content or ""
            output = _parse_llm_response(raw_content)
            break

        except ValueError as exc:
            # Category 3: malformed LLM response — inject correction and retry.
            last_error = exc
            audit.append(
                _audit(
                    "synthesizer_malformed_response",
                    {"attempt": attempt, "error": str(exc)},
                )
            )
            # Inject targeted correction into messages for next attempt.
            messages.append({"role": "assistant", "content": raw_content})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Your previous response was invalid: {exc}\n"
                        "Please respond again with ONLY a valid JSON object matching "
                        "the SynthesizerOutput schema. No markdown fences."
                    ),
                }
            )

        except Exception as exc:
            # Category 2: LLM API failure — record and surface immediately.
            last_error = exc
            audit.append(
                _audit(
                    "synthesizer_api_error",
                    {"attempt": attempt, "error": str(exc)},
                )
            )
            break

    if output is None:
        # All attempts exhausted — degrade gracefully.
        audit.append(
            _audit(
                "synthesizer_failed",
                {"error": str(last_error), "action": "marking unanswerable"},
            )
        )
        return {
            "is_answerable": False,
            "draft_sentences": [],
            "audit_trail": audit,
        }

    # is_answerable escape hatch triggered by the LLM itself.
    if not output.is_answerable:
        audit.append(
            _audit(
                "synthesizer_unanswerable",
                {"reason": "LLM set is_answerable=false — chunks lack sufficient data."},
            )
        )
        return {
            "is_answerable": False,
            "draft_sentences": [],
            "audit_trail": audit,
        }

    draft_dicts = [s.model_dump() for s in output.sentences]

    audit.append(
        _audit(
            "synthesizer_complete",
            {
                "sentence_count": len(draft_dicts),
                "total_citations": sum(len(s["citations"]) for s in draft_dicts),
            },
        )
    )

    return {
        "is_answerable": True,
        "draft_sentences": draft_dicts,
        "audit_trail": audit,
    }
