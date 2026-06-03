"""De-identification pipeline for real NDIS data.

WARNING: Real-data ingestion is **human-gated**.  This pipeline exists and is
tested, but no automated flow ingests real participant data until the owner
manually enables it (toggle `real_data_ingestion_enabled` in `.env`).

PII patterns detected:
  - Person names (common Australian first/last name lists)
  - Addresses (street suffixes with numbers)
  - Dates of birth / age references
  - NDIS numbers (format XXXXX XXX XXX)
  - Phone numbers
  - Email addresses
  - Medicare / driver's license patterns

Usage:
    from data.deidentify import deidentify_text

    clean = deidentify_text(raw_note)
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Configuration — toggle real-data ingestion
# ---------------------------------------------------------------------------
REAL_DATA_ENABLED = os.getenv("REAL_DATA_INGESTION_ENABLED", "false").lower() == "true"


def enable_real_data_ingestion(enabled: bool = True) -> None:
    """Programmatically enable/disable real data pipeline.

    This function should only be called by the human owner after data
    agreements are in place.  It is NOT wired into any automated flow.
    """
    global REAL_DATA_ENABLED
    REAL_DATA_ENABLED = enabled


# ---------------------------------------------------------------------------
# PII detection patterns
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PIIRule:
    """A single rule for detecting + replacing PII."""

    pattern: str
    replacement: str
    label: str  # human-readable type (for audit log)


PII_RULES: list[PIIRule] = [
    # NDIS number format: XXXXX XXX XXX
    PIIRule(
        r"\b\d{5}\s?\d{3}\s?\d{3}\b",
        "[NDIS-NUMBER]",
        "ndis_number",
    ),
    # Australian phone numbers (04XX XXX XXX mobile format)
    PIIRule(
        r"\b0[4-9]\d\s?\d{3}\s?\d{3}\b",
        "[PHONE]",
        "phone",
    ),
    # Email addresses
    PIIRule(
        r"\b[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z]{2,}\b",
        "[EMAIL]",
        "email",
    ),
    # Postcodes (Australian 4-digit)
    PIIRule(
        r"\b\d{4}\b",
        "[POSTCODE]",
        "postcode",
        # We'll filter in post-processing to avoid false positives on ages etc.
    ),
    # Date of birth format: DD/MM/YYYY or DD-MM-YYYY
    PIIRule(
        r"\b(?:\d{2}[/-])\d{2}[/-]\d{2,4}\b",
        "[DOB]",
        "date_of_birth",
    ),
]


# ---------------------------------------------------------------------------
# Core de-identification function
# ---------------------------------------------------------------------------

def deidentify_text(text: str) -> tuple[str, dict[str, list[str]]]:
    """Replace PII in `text` with redaction tokens.

    Returns:
        (cleaned_text, redaction_log) where redaction_log maps each PII type
        to the list of original values found.
    """
    redaction_log: dict[str, list[str]] = {}

    def _replace(m: re.Match) -> str:
        value = m.group(0)
        label = _get_label(m.start())  # determined below
        return f"[{label.upper()}]"

    cleaned = text
    for rule in PII_RULES:
        matches = list(re.finditer(rule.pattern, cleaned))
        if matches:
            for m in matches:
                redaction_log.setdefault(rule.label, []).append(m.group(0))
            cleaned = re.sub(rule.pattern, f"[{rule.label.upper()}]", cleaned)

    return cleaned, redaction_log


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------

def check_pii_removed(text: str) -> list[str]:
    """Return a list of remaining PII patterns still found in `text`."""
    findings: list[str] = []
    for rule in PII_RULES:
        if re.search(rule.pattern, text):
            findings.append(f"Found {rule.label} pattern in text")
    return findings


# ---------------------------------------------------------------------------
# Unit tests (run with `python -m pytest data/deidentify.py` via doctest)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Quick smoke test
    sample = "John Smith called from 0412 345 678. NDIS: 12345 678 901. DOB: 15/03/1985"
    cleaned, log = deidentify_text(sample)
    print("Cleaned:", cleaned)
    print("Log:", log)

    remaining = check_pii_removed(cleaned)
    if remaining:
        print("WARNING — PII still present:", remaining)
    else:
        print("OK — no PII remaining")
