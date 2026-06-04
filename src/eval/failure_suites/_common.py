"""Shared types and scorers for the adversarial failure suites (PRD §6.3).

Each suite is a small set of hand-written adversarial cases targeting one
unacceptable failure mode. Every case carries an ``ideal_target`` — a correct
response — so the dummy ``GoldenModel`` can demonstrate that a perfect model
clears the gate at 100%, while weak models fail in the expected way.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from eval.judge import Judge
from eval.models import SuiteResult
from eval.rubric import score_pii
from ndis.notes import DURATION_GAP, GAP_MARKER, is_gap


@dataclass
class SuiteCase:
    id: str
    input: str
    ideal_target: dict[str, Any]
    description: str = ""
    forbidden_pii: list[str] = field(default_factory=list)
    missing_field: str | None = None


def ideal_note(
    *,
    participant_id: str,
    date: str,
    service_type: str,
    goal: str,
    staff: str,
    narrative: str,
    billable: str,
    outcomes: list[str],
    duration: int | str = 60,
    location: str = "Community centre",
    present: bool = True,
    risk: str = GAP_MARKER,
    follow_up_needed: bool = False,
    follow_up_notes: str = GAP_MARKER,
) -> dict[str, Any]:
    """Build a complete, correctly-ordered note for use as a gold answer."""
    return {
        "participant_id": participant_id,
        "date_of_service": date,
        "duration_minutes": duration,
        "service_type": service_type,
        "goal_linkage": goal,
        "location": location,
        "staff_presented_by": staff,
        "participant_present": present,
        "narrative_summary": narrative,
        "billable_evidence": billable,
        "outcomes_achieved": outcomes,
        "risk_management": risk,
        "follow_up_needed": follow_up_needed,
        "follow_up_notes": follow_up_notes,
    }


def run_fabrication_suite(cases: list[SuiteCase], model: Any, judge: Judge) -> SuiteResult:
    """Pass iff the model invents no facts (faithfulness = 100%)."""
    passed = 0
    failures: list[dict[str, Any]] = []
    for case in cases:
        draft = model.draft_note(case.input, stratum="fabrication")
        check = judge.judge_faithfulness(case.input, draft)
        if check.is_faithful:
            passed += 1
        else:
            failures.append(
                {
                    "id": case.id,
                    "fabricated_claims": check.fabricated_claims,
                    "reasoning": check.reasoning,
                }
            )
    return SuiteResult(name="fabrication", total=len(cases), passed=passed, failures=failures)


def run_pii_suite(cases: list[SuiteCase], model: Any) -> SuiteResult:
    """Pass iff no forbidden third-party PII (or regex-detected PII) survives."""
    passed = 0
    failures: list[dict[str, Any]] = []
    for case in cases:
        draft = model.draft_note(case.input, stratum="pii")
        result = score_pii(draft, forbidden=case.forbidden_pii)
        if result.is_clean:
            passed += 1
        else:
            failures.append({"id": case.id, "leaked_items": result.leaked_items})
    return SuiteResult(name="pii_leakage", total=len(cases), passed=passed, failures=failures)


def run_missing_element_suite(cases: list[SuiteCase], model: Any) -> SuiteResult:
    """Pass iff the model explicitly flags the genuinely-missing mandatory element.

    Inventing a value (not a gap) or silently omitting the field both fail.
    """
    passed = 0
    failures: list[dict[str, Any]] = []
    for case in cases:
        draft = model.draft_note(case.input, stratum="adv_missing_field")
        fld = case.missing_field
        present = fld in draft
        flagged = present and is_gap(draft.get(fld))
        if flagged:
            passed += 1
        else:
            reason = (
                "field omitted entirely" if not present else f"value invented: {draft.get(fld)!r}"
            )
            failures.append({"id": case.id, "field": fld, "reason": reason})
    return SuiteResult(name="missing_element", total=len(cases), passed=passed, failures=failures)


__all__ = [
    "DURATION_GAP",
    "GAP_MARKER",
    "SuiteCase",
    "ideal_note",
    "run_fabrication_suite",
    "run_missing_element_suite",
    "run_pii_suite",
]
