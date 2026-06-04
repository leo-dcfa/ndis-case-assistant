"""Pydantic + dataclass schemas for evaluation outputs.

The Pydantic models are the *contract* for the local LLM judge: it must return
JSON matching these shapes, which keeps fuzzy judgements structured and lets us
swap a heuristic judge for an LLM judge without touching the harness. The
dataclasses hold deterministic scorer results and the aggregated scorecard.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from pydantic import BaseModel, Field

# --------------------------------------------------------------------------- #
# Judge output contracts (LLM must return JSON matching these).
# --------------------------------------------------------------------------- #


class FaithfulnessCheck(BaseModel):
    """Every fact in the note must trace to the worker input — no fabrication."""

    is_faithful: bool = Field(description="True iff the note invents no facts.")
    fabricated_claims: list[str] = Field(
        default_factory=list,
        description="Specific claims in the note not supported by the input.",
    )
    reasoning: str = Field(default="", description="Short justification.")


class RegisterCheck(BaseModel):
    """Professional register / tone per the style rubric."""

    meets_register: bool = Field(description="True iff tone/format is professional.")
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    issues: list[str] = Field(default_factory=list)
    reasoning: str = Field(default="")


class BillableEvidenceCheck(BaseModel):
    present: bool = Field(description="True iff billable evidence is clearly stated.")
    reasoning: str = Field(default="")


class PIIComplianceCheck(BaseModel):
    is_clean: bool = Field(description="True iff no PII / wrong identifiers leak.")
    leaked_items: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Deterministic scorer results.
# --------------------------------------------------------------------------- #


@dataclass
class StructureResult:
    present_ok: bool
    order_ok: bool
    types_ok: bool
    missing_fields: list[str] = field(default_factory=list)
    misordered: bool = False
    invalid_fields: list[str] = field(default_factory=list)
    gap_fields: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.present_ok and self.order_ok and self.types_ok

    @property
    def score(self) -> float:
        return float(sum([self.present_ok, self.order_ok, self.types_ok])) / 3.0


@dataclass
class PIIResult:
    is_clean: bool
    leaked_items: list[str] = field(default_factory=list)


@dataclass
class ExampleScore:
    id: str
    stratum: str
    structure: StructureResult
    pii: PIIResult
    faithfulness: FaithfulnessCheck
    register: RegisterCheck
    billable: BillableEvidenceCheck

    @property
    def passed(self) -> bool:
        return (
            self.structure.passed
            and self.pii.is_clean
            and self.faithfulness.is_faithful
            and self.register.meets_register
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "stratum": self.stratum,
            "structure": asdict(self.structure),
            "pii": asdict(self.pii),
            "faithfulness": self.faithfulness.model_dump(),
            "register": self.register.model_dump(),
            "billable": self.billable.model_dump(),
            "passed": self.passed,
        }


@dataclass
class SuiteResult:
    name: str
    total: int
    passed: int
    failures: list[dict[str, Any]] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "total": self.total,
            "passed": self.passed,
            "pass_rate": self.pass_rate,
            "failures": self.failures,
        }
