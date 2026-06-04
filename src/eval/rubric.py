"""Automated (deterministic) scorers: structural compliance + PII.

These run on every example with no LLM. They check the things that can be
checked exactly: required fields present and correctly ordered, values of the
right type, and no leaked PII. Fuzzy dimensions (faithfulness, register,
billable evidence) are handled by :mod:`eval.judge`.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from eval.models import PIIResult, StructureResult
from ndis.config_loader import get_config
from ndis.notes import is_gap, note_text

# Strata where a required element may legitimately be a flagged gap (the worker
# input genuinely lacked it, and flagging — not inventing — is correct).
GAP_ALLOWED_STRATA = {"sparse_input", "adv_missing_field"}

# Precise PII patterns for scanning a *drafted note*. Deliberately narrower than
# the aggressive redaction patterns in ndis.deidentify: here precision matters
# (a date year must not be misread as a postcode), so we only flag high-signal
# items — phone numbers, emails, and NDIS numbers.
NOTE_PII_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Australian landline/mobile, allowing space/dash separators (e.g.
    # "0412 345 678", "03 9123 4567", "+61 412 345 678"). Requires a leading
    # +61 or 0 then 8-9 more digits, so it won't fire on dates/durations/codes.
    (re.compile(r"\b(?:\+?61[ -]?|0)\d(?:[ -]?\d){7,9}\b"), "PHONE"),
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b", re.IGNORECASE), "EMAIL"),
    (re.compile(r"NDIS[-\s]?\d{6,}", re.IGNORECASE), "NDIS_NUMBER"),
]

# Fields that are not free-form prose and should be excluded from PII scanning
# (structured dates / pseudonymous identifiers are expected, not leaks).
PII_SCAN_EXCLUDE = {"date_of_service", "participant_id", "staff_presented_by", "duration_minutes"}


def score_structure(note: dict[str, Any], stratum: str | None = None) -> StructureResult:
    cfg = get_config().required_fields
    required = [f.name for f in cfg.fields if f.required]
    specs = {f.name: f for f in cfg.fields}
    gap_allowed = stratum in GAP_ALLOWED_STRATA

    missing: list[str] = []
    gaps: list[str] = []
    for name in required:
        if name not in note:
            missing.append(name)
        elif is_gap(note[name]):
            gaps.append(name)

    # Present-OK: no absent keys, and (unless the stratum allows gaps) no
    # flagged-missing required elements.
    present_ok = not missing and (gap_allowed or not gaps)

    # Order-OK: the configured fields that appear in the note must appear in the
    # configured relative order.
    order = cfg.structure_order or required
    note_keys = [k for k in note.keys() if k in order]
    expected_order = [k for k in order if k in note]
    misordered = note_keys != expected_order

    invalid = _invalid_fields(note, specs)

    return StructureResult(
        present_ok=present_ok,
        order_ok=not misordered,
        types_ok=not invalid,
        missing_fields=missing,
        misordered=misordered,
        invalid_fields=invalid,
        gap_fields=gaps,
    )


def _invalid_fields(note: dict[str, Any], specs: dict[str, Any]) -> list[str]:
    invalid: list[str] = []
    for name, spec in specs.items():
        if name not in note:
            continue
        value = note[name]
        if is_gap(value):
            continue  # a flagged gap is not a type error
        if not _value_ok(value, spec):
            invalid.append(name)
    return invalid


def _value_ok(value: Any, spec: Any) -> bool:
    t = spec.type_name
    if t == "string":
        return isinstance(value, str)
    if t == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            return False
        return 1 <= value <= 1440  # duration sanity bound
    if t == "boolean":
        return isinstance(value, bool)
    if t == "array_of_strings":
        return isinstance(value, list) and all(isinstance(x, str) for x in value)
    if t == "date":
        if isinstance(value, datetime):
            return True
        if not isinstance(value, str):
            return False
        try:
            datetime.strptime(value, "%Y-%m-%d")
            return True
        except ValueError:
            return False
    return True  # unknown type spec -> don't penalise


def score_pii(note: dict[str, Any], forbidden: list[str] | None = None) -> PIIResult:
    """Flag leaked PII in a drafted note.

    ``forbidden`` is an optional list of exact strings (e.g. third-party names
    injected by an adversarial case) that must not appear anywhere in the note.
    """
    scan_note = {k: v for k, v in note.items() if k not in PII_SCAN_EXCLUDE}
    text = note_text(scan_note)
    leaked: list[str] = []

    for pattern, label in NOTE_PII_PATTERNS:
        for match in pattern.findall(text):
            leaked.append(f"{label}:{match if isinstance(match, str) else match[0]}")

    if forbidden:
        lowered = text.lower()
        for item in forbidden:
            if item and item.lower() in lowered:
                leaked.append(f"FORBIDDEN:{item}")

    # De-dup while preserving order.
    seen: set[str] = set()
    deduped = [x for x in leaked if not (x in seen or seen.add(x))]
    return PIIResult(is_clean=not deduped, leaked_items=deduped)
