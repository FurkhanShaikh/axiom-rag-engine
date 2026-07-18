"""Regression gate for the quality evals.

The eval scripts measure whether the pipeline still does what it claims. This
module turns a measurement into a **pass/fail decision** against a committed
baseline, so a quality regression can fail CI instead of silently shipping.

Design
------
A baseline is a small JSON document listing the metrics that matter and the
bound each must respect::

    {
      "eval": "e2e_golden",
      "enforcement": "enforce",          # "enforce" | "report_only"
      "recorded_at": "2026-07-17",
      "model": null,
      "metrics": {
        "validate_pass_rate": {"floor": 1.0,  "tolerance": 0.0},
        "cost_usd":           {"ceiling": 0.5, "tolerance": 0.1}
      }
    }

Each metric declares exactly one of ``floor`` (observed must not drop below) or
``ceiling`` (observed must not rise above), plus a ``tolerance`` slack band that
absorbs the run-to-run noise inherent in LLM-scored metrics. A floor metric
passes when ``observed >= floor - tolerance``; a ceiling metric passes when
``observed <= ceiling + tolerance``.

``enforcement`` lets a baseline be committed before real numbers exist: a
``report_only`` baseline prints the comparison but never fails, so a nightly job
wired up ahead of the first keyed run stays green until someone records
defensible floors and flips it to ``enforce``.

The gate is deliberately dependency-free and pure so it can be unit-tested
without keys, datasets, or a pipeline run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Direction = Literal["floor", "ceiling"]
Enforcement = Literal["enforce", "report_only"]


@dataclass(frozen=True)
class MetricCheck:
    """The outcome of comparing one observed metric against its bound."""

    name: str
    direction: Direction
    threshold: float
    tolerance: float
    observed: float | None  # None when the run did not produce this metric
    passed: bool
    detail: str

    @property
    def effective_bound(self) -> float:
        """The bound after the tolerance band is applied."""
        return (
            self.threshold - self.tolerance
            if self.direction == "floor"
            else self.threshold + self.tolerance
        )


@dataclass(frozen=True)
class GateReport:
    """The overall gate decision plus a per-metric breakdown."""

    eval_name: str
    enforcement: Enforcement
    checks: list[MetricCheck]
    # True when every metric passed. Independent of enforcement  -  a report_only
    # baseline still records whether it *would* have failed.
    metrics_ok: bool

    @property
    def gating_failed(self) -> bool:
        """True only when an enforcing baseline observed a regression.

        This is the value CI should turn into an exit code: report_only
        baselines never gate, even when a metric is below its bound.
        """
        return self.enforcement == "enforce" and not self.metrics_ok

    def render(self) -> str:
        """Human-readable table for CI logs and local runs.

        ASCII only  -  the eval scripts write to the platform-encoded stdout, and
        this output must survive a Windows cp1252 terminal without crashing.
        """
        lines = [
            f"Regression gate: {self.eval_name}  (enforcement={self.enforcement})",
            f"{'metric':<26} {'observed':>10} {'bound':>18} {'result':>8}",
            f"{'-' * 26} {'-' * 10} {'-' * 18} {'-' * 8}",
        ]
        for c in self.checks:
            obs = "n/a" if c.observed is None else f"{c.observed:.4f}"
            sign = ">=" if c.direction == "floor" else "<="
            bound = f"{sign} {c.effective_bound:.4f}"
            if c.tolerance:
                bound += f" (+/-{c.tolerance:g})"
            lines.append(f"{c.name:<26} {obs:>10} {bound:>18} {'PASS' if c.passed else 'FAIL':>8}")
            if not c.passed:
                lines.append(f"    -> {c.detail}")
        verdict = (
            "PASS"
            if self.metrics_ok
            else ("FAIL (blocking)" if self.enforcement == "enforce" else "FAIL (report-only)")
        )
        lines.append(f"{'=' * 26}")
        lines.append(f"overall: {verdict}")
        return "\n".join(lines)


def _check_metric(name: str, spec: dict, observed: float | None) -> MetricCheck:
    has_floor = "floor" in spec
    has_ceiling = "ceiling" in spec
    if has_floor == has_ceiling:
        raise ValueError(
            f"metric {name!r} must declare exactly one of 'floor' or 'ceiling' (got {sorted(spec)})"
        )
    direction: Direction = "floor" if has_floor else "ceiling"
    threshold = float(spec["floor" if has_floor else "ceiling"])
    tolerance = float(spec.get("tolerance", 0.0))
    if tolerance < 0:
        raise ValueError(f"metric {name!r} tolerance must be non-negative, got {tolerance}")

    if observed is None:
        # A metric named in the baseline but absent from the run is always a
        # failure: you cannot certify a bound you did not measure. This catches
        # a renamed/removed summary field before it silently stops being gated.
        return MetricCheck(
            name=name,
            direction=direction,
            threshold=threshold,
            tolerance=tolerance,
            observed=None,
            passed=False,
            detail=f"metric not produced by the run  -  cannot gate {name!r}",
        )

    if direction == "floor":
        passed = observed >= threshold - tolerance
        detail = (
            f"{observed:.4f} < floor {threshold:.4f} - tol {tolerance:g} = {threshold - tolerance:.4f}"
            if not passed
            else "ok"
        )
    else:
        passed = observed <= threshold + tolerance
        detail = (
            f"{observed:.4f} > ceiling {threshold:.4f} + tol {tolerance:g} = {threshold + tolerance:.4f}"
            if not passed
            else "ok"
        )
    return MetricCheck(
        name=name,
        direction=direction,
        threshold=threshold,
        tolerance=tolerance,
        observed=observed,
        passed=passed,
        detail=detail,
    )


def evaluate_gate(observed: dict[str, float], baseline: dict) -> GateReport:
    """Compare a run's flattened metrics against a baseline document.

    Args:
        observed: metric name -> value produced by the eval run.
        baseline: parsed baseline document (see module docstring).

    Returns:
        A GateReport. Read ``.gating_failed`` for the CI exit decision and
        ``.render()`` for the log output.
    """
    enforcement: Enforcement = baseline.get("enforcement", "enforce")
    if enforcement not in ("enforce", "report_only"):
        raise ValueError(f"enforcement must be 'enforce' or 'report_only', got {enforcement!r}")

    metrics: dict = baseline.get("metrics") or {}
    if not metrics:
        raise ValueError("baseline declares no metrics  -  nothing to gate")

    checks = [_check_metric(name, spec, observed.get(name)) for name, spec in metrics.items()]
    return GateReport(
        eval_name=baseline.get("eval", "unknown"),
        enforcement=enforcement,
        checks=checks,
        metrics_ok=all(c.passed for c in checks),
    )


def load_baseline(path: Path) -> dict:
    """Load and lightly validate a baseline JSON document."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if "metrics" not in data:
        raise ValueError(f"{path}: baseline is missing a 'metrics' object")
    return data
