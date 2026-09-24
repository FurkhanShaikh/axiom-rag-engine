"""
Deterministic eval floors track the observed values; comparisons carry a CI.

The BM25 gate is fully deterministic, yet its floors sat ~2 points below the
observed values, so recall@10 could fall by four claims without failing. Floors
are now pinned, a ``--ratchet`` raises them when a change improves retrieval
(never lowers them), and ``--compare`` gives a paired-bootstrap interval for the
difference between two methods instead of a bare point difference.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_EVALS = Path(__file__).resolve().parents[2] / "evals"
sys.path.insert(0, str(_EVALS))
_spec = importlib.util.spec_from_file_location("axiom_retrieval_eval", _EVALS / "retrieval_eval.py")
assert _spec and _spec.loader
reval = importlib.util.module_from_spec(_spec)
sys.modules["axiom_retrieval_eval"] = reval
_spec.loader.exec_module(reval)
gate = reval.gate


def _write(path: Path, data: dict) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _baseline(tmp_path: Path) -> Path:
    return _write(
        tmp_path / "baseline.json",
        {
            "eval": "unit",
            "enforcement": "enforce",
            "recorded_at": "2026-01-01",
            "notes": "kept",
            "metrics": {
                "recall": {"floor": 0.80, "tolerance": 0.001},
                "latency": {"ceiling": 2.0, "tolerance": 0.001},
            },
        },
    )


class TestRatchet:
    def test_raises_floors_and_lowers_ceilings(self, tmp_path: Path) -> None:
        path = _baseline(tmp_path)
        moved = gate.ratchet_baseline(path, {"recall": 0.85123, "latency": 1.5}, "2026-09-24")
        data = json.loads(path.read_text(encoding="utf-8"))
        assert moved == ["recall", "latency"]
        assert data["metrics"]["recall"]["floor"] == 0.8512
        assert data["metrics"]["latency"]["ceiling"] == 1.5
        assert data["recorded_at"] == "2026-09-24"
        assert data["notes"] == "kept"

    def test_never_loosens(self, tmp_path: Path) -> None:
        path = _baseline(tmp_path)
        before = path.read_text(encoding="utf-8")
        assert gate.ratchet_baseline(path, {"recall": 0.79, "latency": 2.5}, "2026-09-24") == []
        assert path.read_text(encoding="utf-8") == before  # untouched, date included

    def test_ignores_metrics_the_run_did_not_report(self, tmp_path: Path) -> None:
        path = _baseline(tmp_path)
        assert gate.ratchet_baseline(path, {"recall": 0.9}, "2026-09-24") == ["recall"]
        assert json.loads(path.read_text(encoding="utf-8"))["metrics"]["latency"]["ceiling"] == 2.0


class TestImproved:
    def _check(self, observed: float, direction: str = "floor") -> object:
        spec = {"floor": 0.8} if direction == "floor" else {"ceiling": 0.8}
        report = gate.evaluate_gate(
            {"m": observed}, {"metrics": {"m": {**spec, "tolerance": 0.01}}}
        )
        return report.checks[0]

    def test_floor_beaten_beyond_tolerance_is_improved(self) -> None:
        assert self._check(0.82).improved
        assert not self._check(0.805).improved  # inside the rounding band

    def test_ceiling_beaten_beyond_tolerance_is_improved(self) -> None:
        assert self._check(0.78, "ceiling").improved
        assert not self._check(0.795, "ceiling").improved

    def test_render_suggests_ratcheting(self) -> None:
        report = gate.evaluate_gate(
            {"m": 0.9}, {"eval": "unit", "metrics": {"m": {"floor": 0.8, "tolerance": 0.01}}}
        )
        assert "ratchet the baseline" in report.render()


class TestPairedBootstrap:
    def test_consistent_gain_excludes_zero(self) -> None:
        base = [0.5, 0.6, 0.4, 0.7, 0.5] * 20
        mean, low, _high = gate.paired_bootstrap(base, [b + 0.1 for b in base])
        assert mean == pytest.approx(0.1)
        assert low > 0

    def test_noise_straddles_zero(self) -> None:
        base = [0.0, 1.0] * 50
        cand = [1.0, 0.0] * 50  # same mean, every query flips
        _, low, high = gate.paired_bootstrap(base, cand)
        assert low < 0 < high

    def test_is_reproducible(self) -> None:
        base, cand = [0.1, 0.5, 0.9, 0.3], [0.2, 0.4, 1.0, 0.6]
        assert gate.paired_bootstrap(base, cand) == gate.paired_bootstrap(base, cand)

    def test_rejects_unpaired_samples(self) -> None:
        with pytest.raises(ValueError, match="equal-length"):
            gate.paired_bootstrap([1.0], [1.0, 2.0])


def _results(path: Path, method: str, scores: dict[str, float]) -> Path:
    records = [
        {"claim_id": qid, "recall_at_10": s, "ndcg_at_10": s, "rr": s} for qid, s in scores.items()
    ]
    return _write(path, {"method": method, "records": records})


def test_compare_reports_a_significant_gain(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ids = [f"q{i}" for i in range(60)]
    base = _results(tmp_path / "a.json", "bm25", dict.fromkeys(ids, 0.5))
    cand = _results(
        tmp_path / "b.json", "hybrid", {q: 0.5 + 0.2 * (i % 2) for i, q in enumerate(ids)}
    )
    assert reval.compare(base, cand) == 0
    out = capsys.readouterr().out
    assert "hybrid vs bm25 on 60 shared queries" in out
    assert out.count("  significant") == 3


def test_compare_needs_shared_queries(tmp_path: Path) -> None:
    base = _results(tmp_path / "a.json", "bm25", {"q1": 0.5})
    cand = _results(tmp_path / "b.json", "hybrid", {"q2": 0.5})
    assert reval.compare(base, cand) == 1
