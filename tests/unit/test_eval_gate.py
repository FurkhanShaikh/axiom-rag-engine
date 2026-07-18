"""Regression-gate logic (evals/gate.py).

The gate decides CI pass/fail from measured eval metrics, so its boundary
behavior — tolerance bands, missing metrics, report-only mode — must be exact.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# evals/ is not an installed package; load gate.py by path so the eval harness
# stays runnable as plain scripts without packaging it.
_GATE_PATH = Path(__file__).resolve().parents[2] / "evals" / "gate.py"
_spec = importlib.util.spec_from_file_location("axiom_eval_gate", _GATE_PATH)
assert _spec and _spec.loader
gate = importlib.util.module_from_spec(_spec)
sys.modules["axiom_eval_gate"] = gate
_spec.loader.exec_module(gate)


def _baseline(metrics: dict, enforcement: str = "enforce") -> dict:
    return {"eval": "unit", "enforcement": enforcement, "metrics": metrics}


class TestFloorMetrics:
    def test_above_floor_passes(self) -> None:
        report = gate.evaluate_gate({"recall": 0.82}, _baseline({"recall": {"floor": 0.8}}))
        assert report.metrics_ok
        assert not report.gating_failed

    def test_below_floor_fails(self) -> None:
        report = gate.evaluate_gate({"recall": 0.70}, _baseline({"recall": {"floor": 0.8}}))
        assert not report.metrics_ok
        assert report.gating_failed

    def test_exactly_on_floor_passes(self) -> None:
        report = gate.evaluate_gate({"recall": 0.80}, _baseline({"recall": {"floor": 0.8}}))
        assert report.metrics_ok

    def test_within_tolerance_band_passes(self) -> None:
        # 0.77 is below the 0.80 floor but inside the 0.05 slack band.
        report = gate.evaluate_gate(
            {"recall": 0.77}, _baseline({"recall": {"floor": 0.8, "tolerance": 0.05}})
        )
        assert report.metrics_ok

    def test_just_past_tolerance_band_fails(self) -> None:
        report = gate.evaluate_gate(
            {"recall": 0.74}, _baseline({"recall": {"floor": 0.8, "tolerance": 0.05}})
        )
        assert not report.metrics_ok


class TestCeilingMetrics:
    def test_below_ceiling_passes(self) -> None:
        report = gate.evaluate_gate(
            {"error_rate": 0.02}, _baseline({"error_rate": {"ceiling": 0.05}})
        )
        assert report.metrics_ok

    def test_above_ceiling_fails(self) -> None:
        report = gate.evaluate_gate(
            {"error_rate": 0.09}, _baseline({"error_rate": {"ceiling": 0.05}})
        )
        assert not report.metrics_ok

    def test_within_tolerance_band_passes(self) -> None:
        report = gate.evaluate_gate(
            {"cost_usd": 0.55}, _baseline({"cost_usd": {"ceiling": 0.5, "tolerance": 0.1}})
        )
        assert report.metrics_ok


class TestMissingMetric:
    def test_baseline_metric_absent_from_run_fails(self) -> None:
        """A bound you cannot measure must not silently pass."""
        report = gate.evaluate_gate({}, _baseline({"recall": {"floor": 0.8}}))
        assert not report.metrics_ok
        check = report.checks[0]
        assert check.observed is None
        assert "cannot gate" in check.detail

    def test_extra_observed_metrics_are_ignored(self) -> None:
        report = gate.evaluate_gate(
            {"recall": 0.9, "unrelated": 0.0}, _baseline({"recall": {"floor": 0.8}})
        )
        assert report.metrics_ok


class TestReportOnly:
    def test_report_only_never_gates_even_on_regression(self) -> None:
        report = gate.evaluate_gate(
            {"recall": 0.1},
            _baseline({"recall": {"floor": 0.8}}, enforcement="report_only"),
        )
        # The metric genuinely failed...
        assert not report.metrics_ok
        # ...but a report-only baseline must not fail CI.
        assert not report.gating_failed

    def test_enforce_gates_on_regression(self) -> None:
        report = gate.evaluate_gate(
            {"recall": 0.1}, _baseline({"recall": {"floor": 0.8}}, enforcement="enforce")
        )
        assert report.gating_failed


class TestValidation:
    def test_metric_with_both_bounds_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="exactly one"):
            gate.evaluate_gate({"x": 1.0}, _baseline({"x": {"floor": 0.0, "ceiling": 1.0}}))

    def test_metric_with_neither_bound_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="exactly one"):
            gate.evaluate_gate({"x": 1.0}, _baseline({"x": {"tolerance": 0.1}}))

    def test_negative_tolerance_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            gate.evaluate_gate({"x": 1.0}, _baseline({"x": {"floor": 0.5, "tolerance": -0.1}}))

    def test_unknown_enforcement_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="enforce"):
            gate.evaluate_gate({"x": 1.0}, _baseline({"x": {"floor": 0.5}}, enforcement="maybe"))

    def test_empty_metrics_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="no metrics"):
            gate.evaluate_gate({"x": 1.0}, _baseline({}))


class TestRender:
    def test_render_marks_failing_metric(self) -> None:
        report = gate.evaluate_gate({"recall": 0.5}, _baseline({"recall": {"floor": 0.8}}))
        out = report.render()
        assert "recall" in out
        assert "FAIL" in out
        assert "blocking" in out

    def test_render_shows_report_only_verdict(self) -> None:
        report = gate.evaluate_gate(
            {"recall": 0.5}, _baseline({"recall": {"floor": 0.8}}, enforcement="report_only")
        )
        assert "report-only" in report.render()


class TestCommittedBaselines:
    """The baselines shipped in evals/baselines/ must be well-formed."""

    def test_all_committed_baselines_parse_and_are_valid(self) -> None:
        baselines_dir = _GATE_PATH.parent / "baselines"
        files = sorted(baselines_dir.glob("*.json"))
        assert files, "no committed baselines found"
        for path in files:
            baseline = gate.load_baseline(path)
            # A dummy run where every metric sits comfortably inside its bound
            # proves the baseline is structurally valid (bounds parse, no metric
            # declares both/neither bound).
            observed = {}
            for name, spec in baseline["metrics"].items():
                observed[name] = spec["floor"] if "floor" in spec else spec["ceiling"]
            report = gate.evaluate_gate(observed, baseline)
            assert report.metrics_ok, f"{path.name}: valid observed values did not pass"

    def test_e2e_baseline_is_enforcing(self) -> None:
        """The deterministic e2e gate must actually block — it needs no keys."""
        baseline = gate.load_baseline(_GATE_PATH.parent / "baselines" / "e2e-golden.json")
        assert baseline["enforcement"] == "enforce"
