"""
Axiom Engine v2.3 — Pydantic V2 Data Models
All I/O contracts for the API Gateway and the LangGraph DAG.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, StrictBool, model_validator

from axiom_rag_engine.config.settings import get_settings

# ---------------------------------------------------------------------------
# INPUT MODELS
# ---------------------------------------------------------------------------


class AppConfig(BaseModel):
    """Caller-supplied behavioural constraints for the pipeline."""

    expertise_level: Literal["beginner", "intermediate", "expert"] = "intermediate"
    banned_domains: list[str] = Field(default_factory=list, max_length=100)
    authoritative_domains: list[str] = Field(default_factory=list, max_length=100)
    low_quality_domains: list[str] = Field(default_factory=list, max_length=100)
    exclude_default_domains: list[str] = Field(
        default_factory=list,
        max_length=100,
        description=(
            "Domains to remove from the built-in authoritative/low-quality default sets. "
            "Use this to demote sources such as 'en.wikipedia.org' from the defaults."
        ),
    )
    source_weight: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description=(
            "Weight applied to the domain-authority score when computing the combined "
            "chunk score. Paired with ``chunk_weight``; the two should sum to 1.0."
        ),
    )
    chunk_weight: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description=(
            "Weight applied to the chunk content-quality score when computing the "
            "combined chunk score. Paired with ``source_weight``."
        ),
    )


class ModelConfig(BaseModel):
    """LiteLLM model identifiers for each pipeline stage."""

    synthesizer: str = Field(
        default_factory=lambda: get_settings().default_synthesizer_model,
        description="Heavy model used for answer generation.",
    )
    verifier: str = Field(
        default_factory=lambda: get_settings().default_verifier_model,
        description="Lightweight model used for semantic verification.",
    )


class MechanicalVerificationStageConfig(BaseModel):
    """Stage-level config for Mechanical Verification (non-negotiable floor)."""

    enabled: bool = Field(default=True, frozen=True)
    configurable: bool = Field(default=False, frozen=True)


class PipelineStagesConfig(BaseModel):
    mechanical_verification: MechanicalVerificationStageConfig = Field(
        default_factory=MechanicalVerificationStageConfig
    )
    semantic_verification_enabled: bool = True
    max_ranked_chunks: int = Field(default=10, ge=1, le=50)
    max_rewrite_loops: int = Field(default=3, ge=1, le=5)
    max_retrieval_retries: int = Field(
        default=1,
        ge=0,
        le=3,
        description="Max re-retrieve attempts after rewrite loops are exhausted.",
    )


class PipelineConfig(BaseModel):
    stages: PipelineStagesConfig = Field(default_factory=PipelineStagesConfig)


class AxiomRequest(BaseModel):
    """Top-level request payload from a calling application."""

    request_id: str = Field(..., min_length=1, max_length=256)
    user_query: str = Field(..., min_length=1, max_length=10_000)
    app_config: AppConfig = Field(default_factory=AppConfig)
    models: ModelConfig = Field(default_factory=ModelConfig)
    pipeline_config: PipelineConfig = Field(default_factory=PipelineConfig)
    include_debug: bool = Field(
        default=False,
        description="When true, the response includes the full audit trail and pipeline stats.",
    )


# ---------------------------------------------------------------------------
# SYNTHESIZER OUTPUT SCHEMA  (structured output contract for the heavy LLM)
# ---------------------------------------------------------------------------


class Citation(BaseModel):
    """A single sentence-level citation produced by the Synthesizer."""

    citation_id: str = Field(..., min_length=1)
    chunk_id: str = Field(
        ...,
        pattern=r"^doc_\d+_chunk_[A-Z0-9]+$",
        description="Strict alphanumeric chunk reference, e.g. doc_1_chunk_A.",
    )
    exact_source_quote: str = Field(
        ...,
        min_length=1,
        max_length=2_000,
        description="Verbatim substring of the source chunk. No paraphrasing allowed.",
    )


class DraftSentence(BaseModel):
    """One sentence of the Synthesizer's draft response."""

    sentence_id: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)
    is_cited: bool
    citations: list[Citation] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_citation_shape(self) -> DraftSentence:
        if self.is_cited and not self.citations:
            raise ValueError("Cited draft sentences must include at least one citation.")
        if not self.is_cited and self.citations:
            raise ValueError("Uncited draft sentences must not include citations.")
        return self


class SynthesizerOutput(BaseModel):
    """Full structured output from the Cognitive Synthesizer node."""

    is_answerable: StrictBool = Field(
        ...,
        description=(
            "Escape hatch. False if the retrieved chunks cannot answer the query. "
            "Setting False aborts generation and prevents forced hallucination loops."
        ),
    )
    sentences: list[DraftSentence] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# VERIFICATION & FINAL OUTPUT MODELS
# ---------------------------------------------------------------------------

# Verification tier semantics (doc §4):
#   1 Authoritative  — mechanical + semantic pass; source is a primary official document
#                      (government body, official spec, platform docs).
#                      Tertiary sources (Wikipedia, arXiv) cannot reach Tier 1.
#   2 Consensus      — mechanical + semantic pass; citations span ≥2 distinct domains.
#                      NOTE: this is multi-domain coverage, NOT an agreement check.
#                      Pairwise NLI / contradiction detection is not yet implemented;
#                      Tier 2 should be read as "multi-source" until that ships.
#   3 Model Assisted — mechanical pass; semantic check passed or disabled.
#   4 Misrepresented — mechanical pass; semantic fail (context stripped/inverted).
#   5 Hallucinated   — mechanical fail (quote does not exist in any source sentence).
#   6 Conflicted     — NOT YET IMPLEMENTED. Requires pairwise NLI contradiction
#                      detection across citation pairs. Reserved for future use;
#                      the verifier will never currently assign this tier.

VerificationTier = Literal[1, 2, 3, 4, 5, 6]


class VerificationResult(BaseModel):
    """Outcome of the two-stage verification pipeline for one citation."""

    tier: VerificationTier = Field(
        ...,
        description="Confidence tier 1-6 per the Axiom verification taxonomy.",
    )
    tier_label: Literal[
        "authoritative",
        "consensus",  # Deprecated — use "multi_source" for new code.
        "multi_source",
        "model_assisted",
        "misrepresented",
        "hallucinated",
        "conflicted",
    ] = Field(..., description="Human-readable tier label.")
    mechanical_check: Literal["passed", "failed", "skipped"]
    semantic_check: Literal["passed", "failed", "skipped"]
    failure_reason: str | None = None

    model_config = {"frozen": True}

    @model_validator(mode="after")
    def validate_tier_contract(self) -> VerificationResult:
        if self.tier == 5 and self.mechanical_check != "failed":
            raise ValueError("Tier 5 requires mechanical_check='failed'.")
        if self.tier == 4 and (
            self.mechanical_check != "passed" or self.semantic_check != "failed"
        ):
            raise ValueError("Tier 4 requires mechanical pass and semantic failure.")
        if self.tier in (1, 2) and (
            self.mechanical_check != "passed" or self.semantic_check != "passed"
        ):
            raise ValueError("Tier 1 and Tier 2 require both checks to pass.")
        if self.tier == 6 and self.semantic_check == "skipped":
            raise ValueError("Tier 6 requires an explicit contradiction verdict.")
        return self


class VerifiedCitation(Citation):
    """A citation annotated with verification outcome."""

    verification: VerificationResult


class FinalSentence(BaseModel):
    """A verified sentence ready for the response payload."""

    sentence_id: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)
    is_cited: bool
    citations: list[VerifiedCitation] = Field(default_factory=list)
    verification: VerificationResult

    @model_validator(mode="after")
    def validate_final_citation_shape(self) -> FinalSentence:
        if self.is_cited and not self.citations:
            raise ValueError("Cited final sentences must include verified citations.")
        if not self.is_cited and self.citations:
            raise ValueError("Uncited final sentences must not include citations.")
        return self


# ---------------------------------------------------------------------------
# RESPONSE MODELS
# ---------------------------------------------------------------------------


class UsageByModel(BaseModel):
    """Per-model LLM consumption for one request."""

    calls: int = Field(default=0, ge=0)
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    cost_usd: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Best-effort cost via litellm.completion_cost. "
            "Local backends (Ollama) and untracked providers report 0.0."
        ),
    )


class UsageSummary(BaseModel):
    """Aggregate LLM consumption for one request."""

    calls: int = Field(default=0, ge=0)
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0.0)
    by_model: dict[str, UsageByModel] = Field(default_factory=dict)


class TierBreakdown(BaseModel):
    tier_1_claims: int = 0
    tier_2_claims: int = 0
    tier_3_claims: int = 0
    tier_4_claims: int = 0
    tier_5_claims: int = 0
    tier_6_claims: int = 0


class ConfidenceSummary(BaseModel):
    overall_score: float = Field(..., ge=0.0, le=1.0)
    tier_breakdown: TierBreakdown


# ---------------------------------------------------------------------------
# AUDIT TRAIL MODELS
# ---------------------------------------------------------------------------


class AuditEvent(BaseModel):
    """A single immutable record emitted by any node into the audit trail."""

    event_id: str
    node: str = Field(..., description="Name of the emitting LangGraph node.")
    event_type: str
    payload: dict = Field(default_factory=dict)
    timestamp_utc: str = Field(..., description="ISO-8601 UTC timestamp at emission time.")


class DebugInfo(BaseModel):
    """Optional debug payload included when include_debug=true in the request."""

    audit_trail: list[AuditEvent] = Field(default_factory=list)
    pipeline_stats: dict = Field(default_factory=dict)


class AxiomResponse(BaseModel):
    """Top-level response payload returned to the calling application."""

    request_id: str
    status: Literal["success", "partial", "unanswerable", "error"]
    is_answerable: bool
    confidence_summary: ConfidenceSummary
    final_response: list[FinalSentence] = Field(default_factory=list)
    error_message: str | None = Field(
        default=None,
        description="Human-readable error detail; populated only when status='error'.",
    )
    debug: DebugInfo | None = Field(
        default=None,
        description="Full audit trail and pipeline stats. Populated only when include_debug=true.",
    )
    usage: UsageSummary | None = Field(
        default=None,
        description="LLM token counts and best-effort USD cost for this request.",
    )
