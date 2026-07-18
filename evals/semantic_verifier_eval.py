"""Layer 1 eval — semantic verifier accuracy on SciFact.

Feeds expert-labeled (claim, evidence, abstract) triples from SciFact
through the *production* semantic-verification code path and scores the
verdicts against the dataset labels:

    SUPPORT    -> expected semantic_check = "passed"
    CONTRADICT -> expected semantic_check = "failed"

Positive class for precision/recall is "unfaithful detected" (CONTRADICT).
Low recall  = misrepresentations slip through (missed Tier 4).
Low precision = faithful claims bounced into rewrite loops (wasted budget).

Usage:
    python tasks.py evals semantic -- --model gpt-4o-mini --limit 50
    uv run python evals/semantic_verifier_eval.py --model ollama/qwen3:8b --limit 20
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import gate

EVALS_DIR = Path(__file__).resolve().parent
SCIFACT_DIR = EVALS_DIR / "data" / "scifact"
RESULTS_DIR = EVALS_DIR / "results"
BASELINE_PATH = EVALS_DIR / "baselines" / "semantic-verifier.json"


def _echo(message: str = "") -> None:
    sys.stdout.write(f"{message}\n")


@dataclass
class Example:
    """One (claim, evidence, chunk) triple with a gold label."""

    example_id: str
    claim: str
    quote: str  # labeled evidence sentences — plays exact_source_quote
    chunk: str  # full abstract — plays chunk_text
    label: str  # "SUPPORT" | "CONTRADICT"


def load_examples(split: str = "dev") -> list[Example]:
    claims_path = SCIFACT_DIR / f"claims_{split}.jsonl"
    corpus_path = SCIFACT_DIR / "corpus.jsonl"
    if not claims_path.exists() or not corpus_path.exists():
        _echo(f"SciFact not found in {SCIFACT_DIR}.")
        _echo("Run: python tasks.py evals download")
        sys.exit(1)

    corpus: dict[str, dict[str, Any]] = {}
    with corpus_path.open(encoding="utf-8") as fh:
        for line in fh:
            doc = json.loads(line)
            corpus[str(doc["doc_id"])] = doc

    examples: list[Example] = []
    with claims_path.open(encoding="utf-8") as fh:
        for line in fh:
            claim_obj = json.loads(line)
            evidence: dict[str, list[dict[str, Any]]] = claim_obj.get("evidence") or {}
            for doc_id, evidence_sets in evidence.items():
                doc = corpus.get(str(doc_id))
                if doc is None or not evidence_sets:
                    continue
                abstract: list[str] = [s.strip() for s in doc.get("abstract") or []]
                if not abstract:
                    continue
                # One example per (claim, doc): first evidence set carries the label.
                ev = evidence_sets[0]
                label = ev.get("label")
                if label not in ("SUPPORT", "CONTRADICT"):
                    continue
                sentence_idxs = [i for i in ev.get("sentences") or [] if 0 <= i < len(abstract)]
                if not sentence_idxs:
                    continue
                quote = " ".join(abstract[i] for i in sentence_idxs)
                chunk = " ".join(abstract)
                examples.append(
                    Example(
                        example_id=f"scifact-{claim_obj['id']}-{doc_id}",
                        claim=claim_obj["claim"],
                        quote=quote,
                        chunk=chunk,
                        label=label,
                    )
                )
    return examples


@dataclass
class Record:
    """Verdict for one example."""

    example_id: str
    label: str
    expected: str  # "passed" | "failed"
    got: str  # "passed" | "failed" | "error"
    correct: bool
    failure_reason: str | None
    latency_s: float


# Transient provider throttling (HTTP 429) is not a verifier error — it says
# nothing about whether the verifier judges correctly, so retrying it keeps the
# measurement honest. Free/shared model endpoints throttle aggressively with
# short Retry-After windows, so an eval that counted every 429 as an "error"
# would report a meaningless error_rate.
_MAX_RATE_LIMIT_RETRIES = 4
_MAX_BACKOFF_SECONDS = 40.0


def _is_rate_limited(exc: Exception) -> bool:
    name = type(exc).__name__.lower()
    return "ratelimit" in name or "429" in str(exc)


def _retry_after_seconds(exc: Exception, attempt: int) -> float:
    """Honor the provider's Retry-After if present, else exponential backoff."""
    match = re.search(r'"retry_after_seconds"\s*:\s*([0-9.]+)', str(exc))
    if match:
        return min(float(match.group(1)) + 1.0, _MAX_BACKOFF_SECONDS)
    return min(2.0**attempt, _MAX_BACKOFF_SECONDS)


async def _verify_with_retry(**kwargs: Any) -> tuple[Any, str | None]:
    """Call the production verifier, retrying only on transient rate limits."""
    from axiom_rag_engine.nodes.semantic import _verify_citation

    last_exc: Exception | None = None
    for attempt in range(_MAX_RATE_LIMIT_RETRIES + 1):
        try:
            return await _verify_citation(**kwargs)
        except Exception as exc:
            last_exc = exc
            if not _is_rate_limited(exc) or attempt == _MAX_RATE_LIMIT_RETRIES:
                raise
            await asyncio.sleep(_retry_after_seconds(exc, attempt))
    raise last_exc  # unreachable; for type-checkers


async def _judge(example: Example, model: str) -> Record:
    """Run one example through the production semantic-verification path."""
    # Private import is deliberate: the eval must exercise the exact code the
    # verifier node runs, not a reimplementation of its prompt.
    from axiom_rag_engine.models import Citation

    citation = Citation(
        citation_id="eval_cite_1",
        chunk_id="doc_1_chunk_A",
        exact_source_quote=example.quote[:2000],
    )
    chunk_lookup = {
        "doc_1_chunk_A": {
            "chunk_id": "doc_1_chunk_A",
            "text": example.chunk,
            "domain": "eval.invalid",
            "title": "",
        }
    }
    expected = "passed" if example.label == "SUPPORT" else "failed"
    start = time.monotonic()
    try:
        verification, _reason = await _verify_with_retry(
            claim_text=example.claim,
            citation=citation,
            chunk_lookup=chunk_lookup,
            model=model,
            primary=set(),
        )
        got = verification.semantic_check
        failure_reason = verification.failure_reason
    except Exception as exc:  # LLM/parse failure — count, don't abort the run
        got = "error"
        failure_reason = f"{type(exc).__name__}: {exc}"
    return Record(
        example_id=example.example_id,
        label=example.label,
        expected=expected,
        got=got,
        correct=(got == expected),
        failure_reason=failure_reason,
        latency_s=round(time.monotonic() - start, 2),
    )


def summarize(records: list[Record]) -> dict[str, Any]:
    scored = [r for r in records if r.got != "error"]
    errors = len(records) - len(scored)
    # Positive class: unfaithful detected (label CONTRADICT / verdict failed).
    tp = sum(1 for r in scored if r.label == "CONTRADICT" and r.got == "failed")
    fn = sum(1 for r in scored if r.label == "CONTRADICT" and r.got == "passed")
    fp = sum(1 for r in scored if r.label == "SUPPORT" and r.got == "failed")
    tn = sum(1 for r in scored if r.label == "SUPPORT" and r.got == "passed")
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    error_rate = errors / len(records) if records else 0.0
    return {
        "total": len(records),
        "scored": len(scored),
        "errors": errors,
        "error_rate": round(error_rate, 4),
        "accuracy": round((tp + tn) / len(scored), 4) if scored else 0.0,
        "unfaithful_precision": round(precision, 4),
        "unfaithful_recall": round(recall, 4),
        "unfaithful_f1": round(f1, 4),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


def _gate_metrics(summary: dict[str, Any]) -> dict[str, float]:
    """Flatten the summary into the metric names the baseline gates on."""
    return {
        "unfaithful_recall": summary["unfaithful_recall"],
        "unfaithful_precision": summary["unfaithful_precision"],
        "unfaithful_f1": summary["unfaithful_f1"],
        "accuracy": summary["accuracy"],
        "error_rate": summary["error_rate"],
    }


async def run(model: str, limit: int, seed: int, split: str, gate_baseline: Path | None) -> int:
    examples = load_examples(split)
    by_label = {
        "SUPPORT": sum(1 for e in examples if e.label == "SUPPORT"),
        "CONTRADICT": sum(1 for e in examples if e.label == "CONTRADICT"),
    }
    _echo(f"Loaded {len(examples)} labeled examples from SciFact {split}: {by_label}")

    rng = random.Random(seed)  # noqa: S311 - reproducible sampling, not crypto
    rng.shuffle(examples)
    sample = examples[:limit] if limit else examples
    _echo(f"Evaluating {len(sample)} examples with model={model} ...")

    records = list(await asyncio.gather(*[_judge(e, model) for e in sample]))
    summary = summarize(records)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"semantic-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out_path.write_text(
        json.dumps(
            {
                "eval": "semantic_verifier",
                "dataset": f"scifact/{split}",
                "model": model,
                "seed": seed,
                "summary": summary,
                "records": [asdict(r) for r in records],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    _echo()
    _echo(f"  accuracy             : {summary['accuracy']}")
    _echo(f"  unfaithful precision : {summary['unfaithful_precision']}")
    _echo(f"  unfaithful recall    : {summary['unfaithful_recall']}")
    _echo(f"  unfaithful f1        : {summary['unfaithful_f1']}")
    _echo(f"  confusion            : {summary['confusion']}")
    _echo(f"  errors               : {summary['errors']}/{summary['total']}")
    _echo()
    _echo(f"Full records: {out_path}")

    if gate_baseline is not None:
        baseline = gate.load_baseline(gate_baseline)
        report = gate.evaluate_gate(_gate_metrics(summary), baseline)
        _echo()
        _echo(report.render())
        return 1 if report.gating_failed else 0
    return 0


def main() -> None:
    from _env import load_dotenv

    load_dotenv()  # so litellm sees OPENROUTER_API_KEY / OPENAI_API_KEY / etc.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt-4o-mini", help="LiteLLM model id for the verifier")
    parser.add_argument("--limit", type=int, default=50, help="Max examples (0 = all)")
    parser.add_argument("--seed", type=int, default=13, help="Sampling seed")
    parser.add_argument("--split", default="dev", choices=("dev", "train"))
    parser.add_argument(
        "--gate",
        nargs="?",
        const=str(BASELINE_PATH),
        default=None,
        metavar="BASELINE",
        help=(
            "Compare metrics against a baseline JSON and exit 1 on regression "
            "(unless the baseline is report_only). "
            f"Defaults to {BASELINE_PATH.name} when given no path."
        ),
    )
    args = parser.parse_args()
    baseline = Path(args.gate) if args.gate else None
    sys.exit(asyncio.run(run(args.model, args.limit, args.seed, args.split, baseline)))


if __name__ == "__main__":
    main()
