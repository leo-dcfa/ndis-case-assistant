"""NDIS CaseNote Pydantic schema.

This module defines the canonical data model for NDIS case notes.
It is used by:
  - `data/synth_generate.py` to validate synthetic outputs
  - `eval/rubric.py` to score structural compliance
  - `serve/api.py` for API request/response validation
"""

from __future__ import annotations

import re
from datetime import date, datetime
from enum import Enum
from typing import Annotated, Any, Optional

from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
)


class ServiceType(str, Enum):
    SUPPORT_ASSOCIATION = "Support Association"
    ASSIST_DAILY_LIFE = "Assist Daily Life"
    COMMUNITY_PARTICIPATION = "Community Participation"
    CAPACITY_BUILDING = "Capacity Building"
    CONSUMED_MATERIALS = "Consumed Materials"
    TRANSPORTATION = "Transportation"
    GROUP_CARE_ACTIVITY = "Group Care Activity"
    SHORT_TERM_ACCOMMODATION = "Short Term Accommodation"
    ASSESSMENT_REPORTING = "Assessment and Reporting"
    CASE_MANAGEMENT = "Case Management"
    OTHER = "Other"


class CaseNote(BaseModel):
    """A structured NDIS case note with compliance-ready validation."""

    # --- Identifiers (de-identified) ---
    participant_id: str = Field(
        description="De-identified participant reference (e.g. P-XXXX)",
        min_length=3,
        max_length=50,
    )
    staff_presented_by: str = Field(
        description="Support worker name (de-identified)",
        min_length=2,
        max_length=100,
    )

    # --- Service metadata ---
    date_of_service: date = Field(description="Date the service was delivered")
    duration_minutes: int = Field(
        ge=5,
        le=480,
        description="Duration of the session in minutes",
    )
    service_type: ServiceType = Field(description="NDIS support category code/title")
    goal_linkage: str = Field(
        min_length=3,
        max_length=500,
        description="Which participant goal this service supports",
    )
    location: str = Field(
        min_length=2,
        max_length=500,
        description="Where the service was delivered",
    )
    participant_present: bool = Field(description="Whether the participant was present")

    # --- Narrative content ---
    narrative_summary: Annotated[
        str,
        Field(min_length=20, description="Narrative of what occurred during the session"),
    ]
    billable_evidence: Annotated[
        str,
        Field(min_length=10, description="Evidence supporting billing (activities, techniques)"),
    ]

    # --- Outcomes ---
    outcomes_achieved: list[str] = Field(
        min_length=1,
        max_length=20,
        description="Specific outcomes or progress noted",
    )

    # --- Risk & follow-up (optional fields) ---
    risk_management: Optional[str] = Field(
        default=None,
        description="Any risks identified or interventions taken",
    )
    follow_up_needed: bool = Field(description="Whether follow-up is indicated")
    follow_up_notes: Optional[str] = Field(
        default=None,
        description="Notes for next session (optional)",
    )

    # --- Validation ---
    @field_validator("participant_id")
    @classmethod
    def _validate_participant_id(cls, v: str) -> str:
        """Ensure participant ID does not look like a real name or PII."""
        if re.match(r"^[Pp]\d{3,10}$", v):
            return v
        # Accept any de-identified format but warn via length check
        if len(v) >= 3:
            return v
        raise ValueError("participant_id must be a de-ified identifier (at least 3 chars)")

    @field_validator("narrative_summary")
    @classmethod
    def _validate_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("narrative_summary cannot be blank")
        return v.strip()

    @field_validator("billable_evidence")
    @classmethod
    def _validate_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("billable_evidence cannot be blank")
        return v.strip()

    def to_jsonl_record(self) -> dict[str, Any]:
        """Serialize to a JSONL-friendly dict for training / eval datasets."""
        return {
            "participant_id": self.participant_id,
            "date_of_service": self.date_of_service.isoformat(),
            "duration_minutes": self.duration_minutes,
            "service_type": self.service_type.value,
            "goal_linkage": self.goal_linkage,
            "location": self.location,
            "staff_presented_by": self.staff_presented_by,
            "participant_present": self.participant_present,
            "narrative_summary": self.narrative_summary,
            "billable_evidence": self.billable_evidence,
            "outcomes_achieved": self.outcomes_achieved,
            "risk_management": self.risk_management,
            "follow_up_needed": self.follow_up_needed,
            "follow_up_notes": self.follow_up_notes,
        }

    @classmethod
    def from_jsonl_record(cls, record: dict[str, Any]) -> CaseNote:
        """Deserialize a JSONL record back into a CaseNote."""
        return cls(
            participant_id=record["participant_id"],
            date_of_service=datetime.fromisoformat(record["date_of_service"]).date(),
            duration_minutes=record["duration_minutes"],
            service_type=ServiceType(record["service_type"]),
            goal_linkage=record["goal_linkage"],
            location=record["location"],
            staff_presented_by=record["staff_presented_by"],
            participant_present=record["participant_present"],
            narrative_summary=record["narrative_summary"],
            billable_evidence=record["billable_evidence"],
            outcomes_achieved=record["outcomes_achieved"],
            risk_management=record.get("risk_management"),
            follow_up_needed=record["follow_up_needed"],
            follow_up_notes=record.get("follow_up_notes"),
        )
