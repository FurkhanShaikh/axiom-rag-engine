"""
Axiom Engine — JSON Schemas for structured LLM output.

Sent with each JSON-producing call so providers that support structured output
(OpenAI, Anthropic, Gemini via LiteLLM; Ollama natively) return conforming JSON
instead of relying on post-hoc repair. The Pydantic models and the ``_parse_*``
functions remain the source of truth: every reply is still validated, and the
lenient parser still handles providers that ignore the schema.

Deliberately portable: inline (no ``$ref`` / ``$defs``), and no ``pattern`` or
``additionalProperties`` — several providers' schema subsets reject them.
Constraints they would express are enforced by Pydantic after parsing.
Drift against the Pydantic models is guarded by tests/unit/test_structured_outputs.py.
"""

from __future__ import annotations

from typing import Any

_CITATION: dict[str, Any] = {
    "type": "object",
    "properties": {
        "citation_id": {
            "type": "string",
            "description": "Globally unique across the response: cite_1, cite_2, ...",
        },
        "chunk_id": {
            "type": "string",
            "description": "Exact chunk id from the context, e.g. doc_1_chunk_A.",
        },
        "exact_source_quote": {
            "type": "string",
            "description": "Verbatim substring copied character-for-character from the chunk.",
        },
    },
    "required": ["citation_id", "chunk_id", "exact_source_quote"],
}

_SENTENCE: dict[str, Any] = {
    "type": "object",
    "properties": {
        "sentence_id": {"type": "string", "description": "Sequential: s_01, s_02, ..."},
        "text": {"type": "string", "description": "One complete sentence of the answer."},
        "is_cited": {"type": "boolean"},
        "citations": {"type": "array", "items": _CITATION},
    },
    "required": ["sentence_id", "text", "is_cited", "citations"],
}

SYNTHESIZER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "is_answerable": {
            "type": "boolean",
            "description": "False when the chunks cannot answer the query (sentences then empty).",
        },
        "sentences": {"type": "array", "items": _SENTENCE},
    },
    "required": ["is_answerable", "sentences"],
}

SEMANTIC_VERDICT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "semantic_check": {"type": "string", "enum": ["passed", "failed"]},
        "failure_reason": {
            "type": "string",
            "description": "The specific semantic mismatch when failed; empty string when passed.",
        },
        "reasoning": {"type": "string", "description": "One sentence explaining the decision."},
    },
    "required": ["semantic_check", "failure_reason", "reasoning"],
}

CORROBORATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "corroborated": {"type": "boolean"},
        "reasoning": {"type": "string", "description": "One sentence."},
    },
    "required": ["corroborated", "reasoning"],
}

CONTRADICTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "contradicted": {"type": "boolean"},
        "reasoning": {"type": "string", "description": "One sentence."},
    },
    "required": ["contradicted", "reasoning"],
}
