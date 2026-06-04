"""Adversarial failure suites (PRD §6.3) — fabrication, PII leakage, missing element.

Each is a hard gate that must score 100% before release. ``all_cases`` exposes
every case's (input, ideal_target) so the dummy ``GoldenModel`` can answer them
correctly; ``run_all`` executes the three suites against a model + judge.
"""

from __future__ import annotations

from typing import Any

from eval.failure_suites import fabrication, missing_element, pii_leakage
from eval.judge import Judge
from eval.models import SuiteResult

SUITES = {
    "fabrication": fabrication,
    "pii_leakage": pii_leakage,
    "missing_element": missing_element,
}


def all_cases() -> list[Any]:
    """Every SuiteCase across all suites."""
    cases: list[Any] = []
    for module in SUITES.values():
        cases.extend(module.CASES)
    return cases


def ideal_examples() -> list[dict[str, Any]]:
    """(input -> ideal_target) records, so GoldenModel can answer suite inputs."""
    return [{"input": c.input, "target": c.ideal_target} for c in all_cases()]


def run_all(model: object, judge: Judge) -> list[SuiteResult]:
    return [
        fabrication.run(model, judge),
        pii_leakage.run(model),
        missing_element.run(model),
    ]
