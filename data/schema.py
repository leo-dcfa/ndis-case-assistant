"""NDIS CaseNote Pydantic model + compliance validation.

Loads required fields from config/required_fields.yaml so the compliance
rule set is editable without code changes (PRD §5.1).
"""

from __future__ import annotations

import json
import pathlib
from datetime import date, datetime
from typing import Any, List, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

CONFIG_PATH = pathlib.Path(__file__).resolve().parents[1] / "config" / "required_fields.yaml"


def _load_required_fields() -> dict[str, dict[str, Any]]:
    """Return {field_name: field_spec} from required_fields.yaml."""
    text = CONFIG_PATH.read_text()
    cfg = yaml.safe_load(text)
    fields: dict[str, dict[str, Any]] = {}
    for spec in cfg.get("required_fields", []):
        fields[spec["field"]] = spec
    return fields


_REQUIRED_FIELDS = _load_required_fields()

_STRUCTURE_ORDER: list[str] = [
    f["field"] for f in _REQUIRED_FIELDS.values() if f.get("required")
]


# ---------------------------------------------------------------------------
# CaseNote schema
# ---------------------------------------------------------------------------


class CaseNote(BaseModel):
    """Structured NDIS case note. Fields driven by config/required_fields.yaml."""

    model_config = {"extra": "forbid"}

    participant_id: str = Field(description="De-identified participant reference")
    date_of_service: date = Field(description="Date the service was delivered")
    duration_minutes: int = Field(description="Duration of the session in minutes")
    service_type: str = Field(description="NDIS support category/title")
    goal_linkage: str = Field(description="Which participant goal this service supports")
    location: str = Field(description="Where the service was delivered")
    staff_presented_by: str = Field(description="Support worker name (de-identified)")
    participant_present: bool = Field(description="Whether the participant was present")
    narrative_summary: str = Field(description="Narrative of what occurred during the session")
    billable_evidence: str = Field(description="Evidence supporting billing")
    outcomes_achieved: List[str] = Field(description="Specific outcomes or progress noted")
    risk_management: Optional[str] = Field(default=None, description="Risks identified or interventions")
    follow_up_needed: bool = Field(description="Whether follow-up is indicated")
    follow_up_notes: Optional[str] = Field(default=None, description="Notes for next session")

    # ---- computed / validation helpers ------------------------------------

    @property
    def required_fields_list(self) -> list[str]:
        """Fields that are required, in config order."""
        return list(_STRUCTURE_ORDER)

    @field_validator("service_type")
    @classmethod
    def _check_service_type(cls, v: str) -> str:
        allowed = _REQUIRED_FIELDS.get("service_type", {}).get("options", [])
        if allowed and v not in allowed:
            raise ValueError(f"service_type must be one of {allowed}, got '{v}'")
        return v

    @field_validator("date_of_service")
    @classmethod
    def _check_date_format(cls, v: date) -> date:
        if not isinstance(v, date):
            raise ValueError("date_of_service must be a date")
        return v

    @field_validator("duration_minutes")
    @classmethod
    def _check_duration(cls, v: int) -> int:
        if v <= 0 or v > 1440:
            raise ValueError("duration_minutes must be between 1 and 1439")
        return v

    @model_validator(mode="after")
    def check_required_present(self) -> "CaseNote":
        """Ensure all required fields are non-empty (PRD §6.1 automated checks)."""
        for field_name in _STRUCTURE_ORDER:
            val = getattr(self, field_name)
            if isinstance(val, str) and not val.strip():
                raise ValueError(f"required field '{field_name}' is empty")
            if isinstance(val, list) and len(val) == 0:
                raise ValueError(f"required field '{field_name}' is empty list")
        return self

    def to_dict(self, exclude_none: bool = True) -> dict[str, Any]:
        """Serialize for JSONL storage."""
        d = self.model_dump(exclude_none=exclude_none)
        # dates serialise as ISO strings automatically in model_dump
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CaseNote":
        """Deserialize from a dict (e.g. loaded from JSONL)."""
        if isinstance(data.get("date_of_service"), str):
            data = dict(data)  # copy so we don't mutate the input
            data["date_of_service"] = datetime.strptime(
                data["date_of_service"], "%Y-%m-%d"
            ).date()
        return cls(**data)

    @classmethod
    def from_jsonl_line(cls, line: str) -> "CaseNote":
        """Parse a single JSONL record."""
        return cls.from_dict(json.loads(line))

    @property
    def structure_order(self) -> list[str]:
        """Field order as defined in config."""
        return list(_STRUCTURE_ORDER)
