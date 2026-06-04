"""Automated scorer tests: structure + PII."""

from __future__ import annotations

from eval.rubric import score_pii, score_structure
from ndis.notes import DURATION_GAP, GAP_MARKER


def _complete_note() -> dict:
    return {
        "participant_id": "PRT-0042",
        "date_of_service": "2025-04-02",
        "duration_minutes": 60,
        "service_type": "Assist Daily Life",
        "goal_linkage": "Increase independence",
        "location": "Home",
        "staff_presented_by": "WKR-A",
        "participant_present": True,
        "narrative_summary": "A professional narrative of the session.",
        "billable_evidence": "Delivered 60 minutes of support.",
        "outcomes_achieved": ["Participant engaged"],
        "risk_management": GAP_MARKER,
        "follow_up_needed": False,
        "follow_up_notes": GAP_MARKER,
    }


def test_complete_note_passes_structure():
    result = score_structure(_complete_note(), stratum="routine_session")
    assert result.passed
    assert not result.missing_fields


def test_missing_required_field_fails():
    note = _complete_note()
    del note["goal_linkage"]
    result = score_structure(note, stratum="routine_session")
    assert not result.present_ok
    assert "goal_linkage" in result.missing_fields


def test_gap_in_routine_is_failure_but_allowed_in_sparse():
    note = _complete_note()
    note["duration_minutes"] = DURATION_GAP
    note["location"] = GAP_MARKER
    assert not score_structure(note, stratum="routine_session").present_ok
    assert score_structure(note, stratum="sparse_input").present_ok


def test_bad_type_flagged():
    note = _complete_note()
    note["outcomes_achieved"] = "not a list"
    result = score_structure(note, stratum="routine_session")
    assert not result.types_ok
    assert "outcomes_achieved" in result.invalid_fields


def test_misordered_fields_flagged():
    note = _complete_note()
    reordered = {"service_type": note["service_type"], "participant_id": note["participant_id"]}
    reordered.update({k: v for k, v in note.items() if k not in reordered})
    result = score_structure(reordered, stratum="routine_session")
    assert not result.order_ok


def test_pii_clean_note():
    result = score_pii(_complete_note())
    assert result.is_clean


def test_pii_phone_leak_detected():
    note = _complete_note()
    note["narrative_summary"] = "Worker called the carer on 0412 345 678 afterward."
    result = score_pii(note)
    assert not result.is_clean
    assert any("PHONE" in x for x in result.leaked_items)


def test_pii_forbidden_name_detected():
    note = _complete_note()
    note["narrative_summary"] = "Spoke with Jenny about the visit."
    result = score_pii(note, forbidden=["Jenny"])
    assert not result.is_clean


def test_pii_does_not_flag_dates_or_codes():
    # date_of_service year and participant code must not look like a postcode/phone.
    result = score_pii(_complete_note())
    assert result.is_clean
