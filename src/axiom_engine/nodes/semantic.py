"""
Axiom Engine v2.3 — Semantic Verifier Node (Module 7, Stage 2)

Responsibilities:
  - Runs AFTER Mechanical Verification has already passed for a citation.
  - Takes the claim text, the exact_source_quote, AND the full chunk_text
    (v2.3 context-deprivation patch — prevents context-stripping detection gaps).
  - Calls a lightweight LLM via LiteLLM to assess faithful representation.
  - Assigns Verification Tiers 1, 2, 3, 4, or 6 (Tier 5 is Mechanical's domain).
  - Is configurable — can be disabled in pipeline_config. When disabled,
    all mechanically-passed citations are upgraded/held at Tier 3 with a
    warning (Category 2 degradation, architecture §7).
  - Updates GraphState keys: final_sentences, rewrite_requests, loop_count,
    audit_trail.

Tier assignment logic (architecture §4):
  Tier 1 — Authoritative: mechanically + semantically verified vs. an
            authoritative/official source.
  Tier 2 — Consensus: mechanically + semantically verified vs. multiple
            independent agreeing sources.
  Tier 3 — Model Assisted: mechanically verified; semantic verifier had to
            rely on model training knowledge (no external source available),
            OR semantic verification is disabled.
  Tier 4 — Misrepresented: mechanically verified (quote exists) but semantic
            check found the claim distorts or strips context from the quote.
            Triggers a Synthesizer rewrite request.
  Tier 6 — Conflicted: mechanically + semantically verified but multiple
            sources contradict each other without explanation.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from functools import partial
from typing import Any, Literal, cast

import litellm

from axiom_engine.models import (
    Citation,
    FinalSentence,
    VerificationResult,
)
from axiom_engine.state import GraphState
from axiom_engine.utils.audit import make_audit_event
from axiom_engine.utils.llm import build_completion_kwargs

logger = logging.getLogger("axiom_engine.semantic")
_audit = partial(make_audit_event, "semantic_verifier")

_MAX_CONCURRENT = int(os.environ.get("AXIOM_MAX_CONCURRENT_LLM", "5"))
_llm_semaphore = threading.Semaphore(_MAX_CONCURRENT)

# ---------------------------------------------------------------------------
# Tier label lookup
# ---------------------------------------------------------------------------

# Higher value = more severe degradation. Used to track the worst citation per sentence.
# Tier 4 (Misrepresented) is the most severe semantic outcome; 6 (Conflicted) next.
_DEGRADATION_ORDER: dict[int, int] = {0: 0, 1: 1, 2: 2, 3: 3, 6: 4, 4: 5}

_TIER_LABELS: dict[int, str] = {
    1: "authoritative",
    2: "consensus",
    3: "model_assisted",
    4: "misrepresented",
    5: "hallucinated",  # assigned by Mechanical, never by Semantic
    6: "conflicted",
}

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are the Semantic Verifier for the Axiom Engine. Your job is to assess \
whether a cited claim faithfully represents its source chunk.

You will be given:
  - CLAIM: one sentence the Synthesizer produced
  - QUOTE: the exact substring the Synthesizer cited
  - CHUNK_TEXT: the full source paragraph the quote was taken from

You must respond with a single valid JSON object (no markdown fences):

{
  "tier": <integer: 1, 2, 3, 4, or 6>,
  "semantic_check": "passed" | "failed",
  "failure_reason": "<string if failed, else null>",
  "reasoning": "<one sentence explaining your decision>"
}

TIER ASSIGNMENT RULES:
  1 (Authoritative): The claim faithfully represents the quote, and the chunk \
originates from an official, authoritative, or primary source.
  2 (Consensus): The claim faithfully represents the quote, and the content \
is consistent with multiple independent sources (cross-source agreement).
  3 (Model Assisted): The claim faithfully represents the quote but you cannot \
confirm the source authority or cross-source agreement from the chunk alone. \
Use this as the default when the claim is accurate but source authority is unclear.
  4 (Misrepresented): The quote exists in the chunk (mechanical check already \
confirmed this) BUT the claim distorts, overstates, cherry-picks, or strips \
critical context from the quote. Set semantic_check="failed".
  6 (Conflicted): The claim and quote are faithful to this chunk, but the \
chunk itself signals an unresolved contradiction with other sources.

IMPORTANT:
- Never assign Tier 5 — that is the Mechanical Verifier's domain.
- If the claim accurately reflects the quote in context, default to Tier 3 \
rather than guessing authority (Tier 1) or consensus (Tier 2).
- Tier 4 requires a specific failure_reason explaining what context was stripped \
or distorted.
- Do NOT wrap your JSON in markdown code fences.
"""

_USER_PROMPT_TEMPLATE = """\
CLAIM:
{claim}

QUOTE:
{quote}

CHUNK_TEXT (full source paragraph):
{chunk_text}

SOURCE METADATA:
{source_metadata}

Assess the claim and output valid JSON only.
"""


def _parse_semantic_response(raw: str) -> dict[str, Any]:
    """
    Parse and validate the semantic verifier's JSON response.
    Strips accidental markdown fences.
    Raises ValueError on parse or schema errors.
    """
    # Strip <think>...</think> blocks (common in Qwen-family models).
    clean = re.sub(r"<think>.*?</think>", "", raw.strip(), flags=re.DOTALL)
    # Strip markdown fences.
    clean = re.sub(r"^```(?:json)?\s*", "", clean.strip(), flags=re.IGNORECASE)
    clean = re.sub(r"\s*```$", "", clean.strip())

    try:
        data: dict[str, Any] = json.loads(clean)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Semantic verifier response is not valid JSON: {exc}") from exc

    tier = data.get("tier")
    if tier not in (1, 2, 3, 4, 6):
        raise ValueError(
            f"Semantic verifier returned invalid tier={tier!r}. Must be 1, 2, 3, 4, or 6."
        )
    if data.get("semantic_check") not in ("passed", "failed"):
        raise ValueError(
            f"semantic_check must be 'passed' or 'failed', got {data.get('semantic_check')!r}"
        )
    return data


def _build_rewrite_request(
    sentence_id: str,
    citation_id: str,
    chunk_id: str,
    tier: int,
    failure_reason: str,
) -> str:
    label = _TIER_LABELS[tier]
    return (
        f"Sentence {sentence_id}, citation {citation_id} (chunk {chunk_id}): "
        f"Tier {tier} ({label}) failure — {failure_reason}"
    )


def _degraded_verification(citation_id: str, chunk_id: str) -> VerificationResult:
    """Tier 3 fallback used when semantic verification is disabled or errors."""
    return VerificationResult(
        tier=3,
        tier_label="model_assisted",
        mechanical_check="passed",
        semantic_check="skipped",
        failure_reason="Semantic verification disabled or unavailable; degraded to Tier 3.",
    )


# ---------------------------------------------------------------------------
# Per-citation semantic check
# ---------------------------------------------------------------------------


def _verify_citation(
    claim_text: str,
    citation: dict[str, Any],
    chunk_lookup: dict[str, dict[str, Any]],
    model: str,
) -> tuple[VerificationResult, dict[str, Any] | None]:
    """
    Run semantic verification on one citation.

    Returns:
        (VerificationResult, rewrite_dict | None)
        rewrite_dict is non-None only for Tier 4 failures.
    """
    chunk_id: str = citation["chunk_id"]
    exact_quote: str = citation["exact_source_quote"]

    chunk_data = chunk_lookup.get(chunk_id, {})
    chunk_text: str = chunk_data.get("text", "")
    source_metadata: str = json.dumps(
        {k: v for k, v in chunk_data.items() if k not in ("text", "chunk_id")},
        indent=2,
    )

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": _USER_PROMPT_TEMPLATE.format(
                claim=claim_text,
                quote=exact_quote,
                chunk_text=chunk_text or "(chunk text unavailable)",
                source_metadata=source_metadata or "{}",
            ),
        },
    ]

    try:
        completion_kwargs = build_completion_kwargs(
            model=model,
            messages=messages,
            temperature=0.0,
        )
        with _llm_semaphore:
            response = litellm.completion(**completion_kwargs)
        raw: str = response.choices[0].message.content or ""
        data = _parse_semantic_response(raw)
    except Exception as exc:
        # Category 2 degradation: semantic verifier failure → Tier 3 + warning.
        logger.warning(
            "Semantic verification failed for chunk %s, degrading to Tier 3: %s",
            chunk_id,
            exc,
        )
        return (
            VerificationResult(
                tier=3,
                tier_label="model_assisted",
                mechanical_check="passed",
                semantic_check="skipped",
                failure_reason=f"Semantic verifier error (degraded to Tier 3): {exc}",
            ),
            None,
        )

    tier = cast(Literal[1, 2, 3, 4, 6], data["tier"])
    semantic_check = cast(Literal["passed", "failed"], data["semantic_check"])
    failure_reason: str | None = data.get("failure_reason")

    vr = VerificationResult(
        tier=tier,
        tier_label=cast(
            Literal[
                "authoritative",
                "consensus",
                "model_assisted",
                "misrepresented",
                "hallucinated",
                "conflicted",
            ],
            _TIER_LABELS[tier],
        ),
        mechanical_check="passed",  # Semantic only runs after mechanical pass
        semantic_check=semantic_check,
        failure_reason=failure_reason,
    )

    rewrite: dict[str, Any] | None = None
    if tier == 4:
        rewrite = {
            "citation_id": citation["citation_id"],
            "chunk_id": chunk_id,
            "tier": 4,
            "failure_reason": failure_reason or "Context misrepresented.",
        }

    return vr, rewrite


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


def semantic_verifier_node(state: GraphState) -> dict[str, Any]:
    """
    LangGraph node — Semantic Verifier (Stage 2).

    Iterates over draft_sentences from state. For each citation that already
    has a mechanical 'passed' status (determined by the caller via
    mechanical_results in state), runs the lightweight LLM semantic check.

    Returns keys: final_sentences, rewrite_requests, loop_count, audit_trail
    """
    audit: list[dict[str, Any]] = []

    pipeline_cfg: dict = state.get("pipeline_config") or {}
    stages_cfg: dict = pipeline_cfg.get("stages") or {}
    semantic_enabled: bool = stages_cfg.get("semantic_verification_enabled", True)

    models_cfg: dict = state.get("models_config") or {}
    model: str = models_cfg.get("verifier", "gpt-4o-mini")

    draft_sentences: list[dict] = list(state.get("draft_sentences") or [])

    # Build chunk lookup: chunk_id → chunk dict (text + metadata)
    indexed_chunks: list[dict] = list(state.get("indexed_chunks") or [])
    chunk_lookup: dict[str, dict] = {c["chunk_id"]: c for c in indexed_chunks}

    # mechanical_results: dict[citation_id, "passed"|"failed"] — set by verifier node
    mechanical_results: dict[str, str] = cast(dict[str, str], state.get("mechanical_results") or {})

    audit.append(
        _audit(
            "semantic_verifier_start",
            {
                "semantic_enabled": semantic_enabled,
                "model": model,
                "sentence_count": len(draft_sentences),
                "loop_count": state.get("loop_count", 0),
            },
        )
    )

    final_sentences: list[dict] = []
    rewrite_requests: list[str] = []

    for sentence_dict in draft_sentences:
        sentence_id: str = sentence_dict["sentence_id"]
        claim_text: str = sentence_dict["text"]
        citations: list[dict] = sentence_dict.get("citations") or []

        # Sentence-level verification result: use the worst citation tier.
        sentence_verification: VerificationResult | None = None

        if not sentence_dict.get("is_cited") or not citations:
            # Uncited sentence — no verification needed.
            sentence_verification = VerificationResult(
                tier=3,
                tier_label="model_assisted",
                mechanical_check="skipped",
                semantic_check="skipped",
                failure_reason=None,
            )
        else:
            worst_tier: int = 0  # 0 = sentinel (no citation processed yet)
            worst_vr: VerificationResult | None = None

            for citation in citations:
                cit_id: str = citation["citation_id"]
                chunk_id: str = citation["chunk_id"]

                # Skip citations that failed mechanical verification —
                # the verification_node handles those as Tier 5 rewrite requests.
                if mechanical_results.get(cit_id) == "failed":
                    audit.append(
                        _audit(
                            "semantic_skip_mechanical_fail",
                            {"citation_id": cit_id, "chunk_id": chunk_id},
                        )
                    )
                    continue

                if not semantic_enabled:
                    vr = _degraded_verification(cit_id, chunk_id)
                    audit.append(
                        _audit(
                            "semantic_skipped_disabled",
                            {"citation_id": cit_id, "chunk_id": chunk_id, "tier": 3},
                        )
                    )
                else:
                    vr, rewrite = _verify_citation(
                        claim_text=claim_text,
                        citation=citation,
                        chunk_lookup=chunk_lookup,
                        model=model,
                    )
                    if rewrite is not None:
                        rewrite_requests.append(
                            _build_rewrite_request(
                                sentence_id=sentence_id,
                                citation_id=cit_id,
                                chunk_id=chunk_id,
                                tier=4,
                                failure_reason=rewrite["failure_reason"],
                            )
                        )
                    audit.append(
                        _audit(
                            "semantic_citation_result",
                            {
                                "citation_id": cit_id,
                                "chunk_id": chunk_id,
                                "tier": vr.tier,
                                "semantic_check": vr.semantic_check,
                                "failure_reason": vr.failure_reason,
                            },
                        )
                    )

                # Track worst (highest degradation priority) citation tier.
                if _DEGRADATION_ORDER.get(vr.tier, 0) > _DEGRADATION_ORDER.get(worst_tier, 0):
                    worst_tier = vr.tier
                    worst_vr = vr

            sentence_verification = worst_vr or VerificationResult(
                tier=3,
                tier_label="model_assisted",
                mechanical_check="passed",
                semantic_check="skipped",
                failure_reason="No mechanically-passed citations to semantically verify.",
            )

        # Build FinalSentence dict
        final_sentence = FinalSentence(
            sentence_id=sentence_dict["sentence_id"],
            text=sentence_dict["text"],
            is_cited=sentence_dict.get("is_cited", False),
            citations=[Citation(**c) for c in citations],
            verification=sentence_verification,
        )
        final_sentences.append(final_sentence.model_dump())

    audit.append(
        _audit(
            "semantic_verifier_complete",
            {
                "final_sentence_count": len(final_sentences),
                "rewrite_request_count": len(rewrite_requests),
            },
        )
    )

    return {
        "final_sentences": final_sentences,
        "rewrite_requests": rewrite_requests,  # operator.add appends
        "loop_count": state.get("loop_count", 0) + 1,
        "audit_trail": audit,
    }
