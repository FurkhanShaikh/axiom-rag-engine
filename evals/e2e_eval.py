"""Layer 2 eval — end-to-end pipeline behavior on the golden seed set.

Each case in evals/golden/seed.jsonl injects a pinned corpus through
MockSearchBackend and runs the full compiled LangGraph (retriever ->
scorer -> ranker -> synthesizer -> verifier, with rewrite loops), then
checks the marshalled response against the case's expectations.

Cases run sequentially because the search backend is a module-level global.

Usage:
    python tasks.py evals e2e -- --model gpt-4o-mini
    uv run python evals/e2e_eval.py --model ollama/qwen3:8b
    uv run python evals/e2e_eval.py --validate-only   # no LLM: schema +
        deterministic retrieval/scoring/ranking stages only
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import gate

EVALS_DIR = Path(__file__).resolve().parent
SEED_PATH = EVALS_DIR / "golden" / "seed.jsonl"
RESULTS_DIR = EVALS_DIR / "results"
BASELINE_PATH = EVALS_DIR / "baselines" / "e2e-golden.json"


def _echo(message: str = "") -> None:
    sys.stdout.write(f"{message}\n")


@dataclass
class GoldenCase:
    case_id: str
    query: str
    search_results: list[dict[str, Any]]
    expect: dict[str, Any]
    comment: str = ""
    # Optional per-case overrides merged onto the defaults. Lets a case exercise
    # trust policy (banned_domains, low_quality_domains), expertise level, or
    # pipeline stage toggles without a bespoke runner per class.
    app_config: dict[str, Any] = field(default_factory=dict)
    pipeline_config: dict[str, Any] = field(default_factory=dict)


@dataclass
class CaseResult:
    case_id: str
    passed: bool
    failures: list[str] = field(default_factory=list)
    status: str = ""
    is_answerable: bool | None = None
    overall_score: float | None = None
    tier_counts: dict[str, int] = field(default_factory=dict)
    usage: dict[str, Any] = field(default_factory=dict)
    latency_s: float = 0.0


def load_cases() -> list[GoldenCase]:
    cases: list[GoldenCase] = []
    with SEED_PATH.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            for key in ("id", "query", "search_results", "expect"):
                if key not in raw:
                    raise ValueError(f"seed.jsonl line {line_no}: missing key {key!r}")
            cases.append(
                GoldenCase(
                    case_id=raw["id"],
                    query=raw["query"],
                    search_results=raw["search_results"],
                    expect=raw["expect"],
                    comment=raw.get("comment", ""),
                    app_config=raw.get("app_config", {}),
                    pipeline_config=raw.get("pipeline_config", {}),
                )
            )
    return cases


def _merged_configs(case: GoldenCase) -> tuple[dict[str, Any], dict[str, Any]]:
    """Case overrides merged onto the model defaults (shallow, one level deep)."""
    from axiom_rag_engine.models import AppConfig, PipelineConfig

    app_config = AppConfig().model_dump()
    app_config.update(case.app_config)

    pipeline_config = PipelineConfig().model_dump()
    for key, value in case.pipeline_config.items():
        if key == "stages" and isinstance(value, dict):
            pipeline_config["stages"].update(value)
        else:
            pipeline_config[key] = value
    return app_config, pipeline_config


def _check_expectations(case: GoldenCase, response: Any) -> list[str]:
    """Return a list of human-readable expectation failures (empty = pass)."""
    failures: list[str] = []
    expect = case.expect

    if "answerable" in expect and response.is_answerable != expect["answerable"]:
        failures.append(
            f"answerable: expected {expect['answerable']}, got {response.is_answerable}"
        )
    if "status_in" in expect and response.status not in expect["status_in"]:
        failures.append(f"status: expected one of {expect['status_in']}, got {response.status!r}")
    if "max_tier5" in expect:
        tier5 = response.confidence_summary.tier_breakdown.tier_5_claims
        if tier5 > expect["max_tier5"]:
            failures.append(f"tier5 sentences: expected <= {expect['max_tier5']}, got {tier5}")
    if "max_tier1" in expect:
        # Guards the primary-vs-reference distinction: a reference source
        # (arXiv, Wikipedia) must not be promoted to Tier 1.
        tier1 = response.confidence_summary.tier_breakdown.tier_1_claims
        if tier1 > expect["max_tier1"]:
            failures.append(f"tier1 sentences: expected <= {expect['max_tier1']}, got {tier1}")
    if "min_overall_score" in expect:
        score = response.confidence_summary.overall_score
        if score < expect["min_overall_score"]:
            failures.append(
                f"overall_score: expected >= {expect['min_overall_score']}, got {score}"
            )
    return failures


async def run_case(case: GoldenCase, engine: Any, model: str) -> CaseResult:
    from axiom_rag_engine.marshalling import marshal_response
    from axiom_rag_engine.nodes.retriever import MockSearchBackend, set_search_backend
    from axiom_rag_engine.state import make_initial_state
    from axiom_rag_engine.utils.llm import get_llm_usage_snapshot, reset_llm_budget

    app_config, pipeline_config = _merged_configs(case)
    set_search_backend(MockSearchBackend(case.search_results))
    initial_state = make_initial_state(
        request_id=f"eval-{case.case_id}",
        user_query=case.query,
        app_config=app_config,
        models_config={"synthesizer": model, "verifier": model},
        pipeline_config=pipeline_config,
    )
    reset_llm_budget()
    start = time.monotonic()
    try:
        graph_result = await engine.ainvoke(initial_state)
    except Exception as exc:
        return CaseResult(
            case_id=case.case_id,
            passed=False,
            failures=[f"pipeline raised {type(exc).__name__}: {exc}"],
            latency_s=round(time.monotonic() - start, 2),
        )
    latency = round(time.monotonic() - start, 2)

    response = marshal_response(
        f"eval-{case.case_id}", graph_result, False, get_llm_usage_snapshot()
    )
    failures = _check_expectations(case, response)
    breakdown = response.confidence_summary.tier_breakdown
    tier_counts = {f"tier_{n}": getattr(breakdown, f"tier_{n}_claims") for n in range(1, 7)}
    usage = response.usage.model_dump() if response.usage else {}
    usage.pop("by_model", None)
    return CaseResult(
        case_id=case.case_id,
        passed=not failures,
        failures=failures,
        status=response.status,
        is_answerable=response.is_answerable,
        overall_score=response.confidence_summary.overall_score,
        tier_counts={k: v for k, v in tier_counts.items() if v},
        usage=usage,
        latency_s=latency,
    )


async def validate_case(case: GoldenCase) -> CaseResult:
    """LLM-free pass: run the deterministic stages and report the pre-LLM gate."""
    from axiom_rag_engine.config.settings import get_settings
    from axiom_rag_engine.nodes.ranker import ranker_node
    from axiom_rag_engine.nodes.retriever import (
        MockSearchBackend,
        retriever_node,
        set_search_backend,
    )
    from axiom_rag_engine.nodes.scorer import scorer_node
    from axiom_rag_engine.state import make_initial_state

    app_config, pipeline_config = _merged_configs(case)
    set_search_backend(MockSearchBackend(case.search_results))
    state: dict[str, Any] = dict(
        make_initial_state(
            request_id=f"eval-{case.case_id}",
            user_query=case.query,
            app_config=app_config,
            models_config={},
            pipeline_config=pipeline_config,
        )
    )
    state.update(await retriever_node(state))  # type: ignore[arg-type]
    state.update(await scorer_node(state))  # type: ignore[arg-type]
    state.update(await ranker_node(state))  # type: ignore[arg-type]

    ranked = state.get("ranked_chunks") or []
    best = max((c.get("ranking_score", 0.0) for c in ranked), default=0.0)
    gate_fires = not ranked or best < get_settings().min_usable_ranking_score

    failures: list[str] = []
    # Only the deterministic gate can be asserted without an LLM: a case that
    # expects answerable=True must at least survive the pre-LLM gate.
    if case.expect.get("answerable") is True and gate_fires:
        failures.append(
            f"pre-LLM gate would mark this unanswerable (best ranking_score={best:.3f})"
        )
    return CaseResult(
        case_id=case.case_id,
        passed=not failures,
        failures=failures,
        status=f"validate-only: chunks={len(ranked)} best_ranking={best:.3f} "
        f"pre_llm_gate={'fires' if gate_fires else 'clears'}",
    )


def _observed_metrics(results: list[CaseResult], validate_only: bool) -> dict[str, float]:
    """Flatten case results into the metric names the baseline gates on."""
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    pass_rate = passed / total if total else 0.0
    if validate_only:
        return {"validate_pass_rate": round(pass_rate, 4)}
    return {
        "pass_rate": round(pass_rate, 4),
        "cost_usd": round(sum(float(r.usage.get("cost_usd") or 0.0) for r in results), 6),
    }


async def run(model: str, validate_only: bool, gate_baseline: Path | None) -> int:
    cases = load_cases()
    _echo(f"Loaded {len(cases)} golden cases from {SEED_PATH}")

    results: list[CaseResult] = []
    if validate_only:
        for case in cases:
            results.append(await validate_case(case))
    else:
        from axiom_rag_engine.graph import build_axiom_graph

        engine = build_axiom_graph()
        _echo(f"Running full pipeline with model={model} (cases run sequentially) ...")
        for case in cases:
            result = await run_case(case, engine, model)
            marker = "PASS" if result.passed else "FAIL"
            _echo(f"  [{marker}] {result.case_id} ({result.latency_s}s) {result.status}")
            for failure in result.failures:
                _echo(f"         - {failure}")
            results.append(result)

    passed = sum(1 for r in results if r.passed)
    total_cost = sum(float(r.usage.get("cost_usd") or 0.0) for r in results)
    observed = _observed_metrics(results, validate_only)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    mode = "validate" if validate_only else "full"
    out_path = RESULTS_DIR / f"e2e-{mode}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out_path.write_text(
        json.dumps(
            {
                "eval": "e2e_golden",
                "mode": mode,
                "model": None if validate_only else model,
                "summary": {
                    "passed": passed,
                    "total": len(results),
                    "cost_usd": total_cost,
                    "metrics": observed,
                },
                "cases": [r.__dict__ for r in results],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    _echo()
    _echo(f"  passed : {passed}/{len(results)}")
    if not validate_only:
        _echo(f"  cost   : ${total_cost:.4f}")
    _echo(f"Full records: {out_path}")
    if validate_only:
        for r in results:
            _echo(f"  {r.case_id}: {r.status}")

    # Per-case expectations are the first gate: any failing case is a
    # regression regardless of the baseline.
    cases_ok = passed == len(results)

    if gate_baseline is not None:
        baseline = gate.load_baseline(gate_baseline)
        report = gate.evaluate_gate(observed, baseline)
        _echo()
        _echo(report.render())
        if report.gating_failed or not cases_ok:
            return 1
        return 0

    return 0 if cases_ok else 1


def main() -> None:
    from _env import load_dotenv

    load_dotenv()  # so litellm sees OPENROUTER_API_KEY / OPENAI_API_KEY / etc.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt-4o-mini", help="LiteLLM model id for both stages")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="No LLM: validate seed schema and run deterministic stages only",
    )
    parser.add_argument(
        "--gate",
        nargs="?",
        const=str(BASELINE_PATH),
        default=None,
        metavar="BASELINE",
        help=(
            "Fail (exit 1) if metrics regress against a baseline JSON. "
            f"Defaults to {BASELINE_PATH.name} when given no path."
        ),
    )
    args = parser.parse_args()
    baseline = Path(args.gate) if args.gate else None
    sys.exit(asyncio.run(run(args.model, args.validate_only, baseline)))


if __name__ == "__main__":
    main()
