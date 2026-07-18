"""Golden seed-set integrity.

The e2e eval loads evals/golden/seed.jsonl. A malformed line or a duplicate
case id silently corrupts a benchmark run, so validate the committed set in the
unit suite where it fails fast and without keys.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

SEED_PATH = Path(__file__).resolve().parents[2] / "evals" / "golden" / "seed.jsonl"

_REQUIRED_KEYS = ("id", "query", "search_results", "expect")
_KNOWN_EXPECT_KEYS = frozenset(
    {"answerable", "status_in", "max_tier5", "max_tier1", "min_overall_score"}
)


def _load_lines() -> list[dict]:
    cases = []
    for line in SEED_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            cases.append(json.loads(line))
    return cases


def test_seed_file_exists_and_is_nonempty() -> None:
    cases = _load_lines()
    assert len(cases) >= 16, f"expected the expanded golden set, got {len(cases)} cases"


def test_every_case_has_required_keys() -> None:
    for case in _load_lines():
        missing = [k for k in _REQUIRED_KEYS if k not in case]
        assert not missing, f"case {case.get('id')!r} missing keys: {missing}"


def test_case_ids_are_unique() -> None:
    ids = [c["id"] for c in _load_lines()]
    dupes = {i for i in ids if ids.count(i) > 1}
    assert not dupes, f"duplicate case ids: {dupes}"


def test_search_results_are_well_formed() -> None:
    for case in _load_lines():
        results = case["search_results"]
        assert isinstance(results, list), f"{case['id']}: search_results must be a list"
        for r in results:
            assert "url" in r and "content" in r, f"{case['id']}: result missing url/content"


def test_expectations_use_known_keys() -> None:
    """A typo'd expectation key (e.g. 'max_teir5') silently never fires."""
    for case in _load_lines():
        unknown = set(case["expect"]) - _KNOWN_EXPECT_KEYS
        assert not unknown, f"{case['id']}: unknown expectation keys {unknown}"


def test_answerable_cases_declare_a_status() -> None:
    for case in _load_lines():
        if case["expect"].get("answerable") is True:
            assert "status_in" in case["expect"], (
                f"{case['id']}: answerable case should pin status_in"
            )


@pytest.mark.parametrize("case", _load_lines(), ids=lambda c: c["id"])
def test_each_case_carries_a_regression_comment(case: dict) -> None:
    """Each case must document the specific regression class it guards."""
    assert case.get("comment", "").strip(), f"{case['id']}: missing diagnostic comment"
