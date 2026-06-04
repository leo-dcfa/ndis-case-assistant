"""CaseNote schema + config tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ndis.config_loader import get_config
from ndis.models import CaseNote


def _valid_payload() -> dict:
    return {
        "participant_id": "PRT-0042",
        "date_of_service": "2025-04-02",
        "duration_minutes": 60,
        "service_type": "Assistance with Daily Life",
        "goal_linkage": "Increase independence",
        "location": "Participant's home",
        "staff_presented_by": "WKR-A (Alex)",
        "participant_present": True,
        "narrative_summary": "Support worker delivered a session.",
        "billable_evidence": "Delivered 60 minutes of support.",
        "outcomes_achieved": ["Participant engaged"],
        "follow_up_needed": False,
    }


def test_valid_note_parses():
    note = CaseNote.from_dict(_valid_payload())
    assert note.participant_id == "PRT-0042"
    assert note.duration_minutes == 60


def test_invalid_service_type_rejected():
    payload = _valid_payload()
    payload["service_type"] = "Not A Real Category"
    with pytest.raises(ValidationError):
        CaseNote.from_dict(payload)


def test_out_of_range_duration_rejected():
    payload = _valid_payload()
    payload["duration_minutes"] = 99999
    with pytest.raises(ValidationError):
        CaseNote.from_dict(payload)


def test_empty_required_field_rejected():
    payload = _valid_payload()
    payload["goal_linkage"] = "   "
    with pytest.raises(ValidationError):
        CaseNote.from_dict(payload)


def test_config_loads_required_fields():
    cfg = get_config().required_fields
    names = {f.name for f in cfg.fields}
    assert "participant_id" in names
    assert "narrative_summary" in names
    assert cfg.structure_order[0] == "participant_id"
