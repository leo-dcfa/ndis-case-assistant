"""Output-side PII guardrail tests (the deterministic safety net)."""

from __future__ import annotations

from ndis.deidentify import extract_contact_names, redact_note


def test_extract_cue_introduced_names():
    inp = "drove to clinic; call mum Jenny on 0412 345 678; coordinator Hannah Wells emailed"
    names = extract_contact_names(inp)
    assert "Jenny" in names
    assert "Hannah Wells" in names


def test_redact_note_strips_name_and_phone_from_prose():
    note = {
        "participant_id": "PRT-0042",
        "date_of_service": "2026-04-07",
        "duration_minutes": 50,
        "narrative_summary": "Worker drove PRT-0042; Jenny was called on 0412 345 678.",
        "outcomes_achieved": ["Spoke with brother Marcus"],
        "follow_up_notes": "email sarah.k@example.com",
    }
    inp = "call mum Jenny 0412 345 678; brother Marcus; sister sarah.k@example.com"
    clean, changed = redact_note(note, inp)
    text = str(clean)
    assert "Jenny" not in text
    assert "Marcus" not in text
    assert "0412 345 678" not in text
    assert "sarah.k@example.com" not in text
    assert "narrative_summary" in changed
    # Structured fields untouched.
    assert clean["date_of_service"] == "2026-04-07"
    assert clean["participant_id"] == "PRT-0042"


def test_redact_note_leaves_clean_note_unchanged():
    note = {
        "participant_id": "PRT-0042",
        "date_of_service": "2026-04-07",
        "duration_minutes": 60,
        "narrative_summary": "Support worker delivered a 60-minute session on 2026-04-07.",
        "outcomes_achieved": ["Participant engaged"],
    }
    clean, changed = redact_note(note, "WKR-A, PRT-0042, 60 min")
    assert changed == []
    assert clean == note
    # The date year must NOT be mistaken for a postcode/phone.
    assert "2026" in clean["narrative_summary"]
