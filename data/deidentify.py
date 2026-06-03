# data de-identification — PII redaction pipeline

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

# Human-gated toggle: real data ingestion is OFF by default.
_REAL_DATA_INGESTION_ENABLED = False


def enable_real_data_ingestion(enabled: bool) -> None:
    """Manually allow real participant data to enter the pipeline.

    Must only be called after data agreements are in place.
    """
    global _REAL_DATA_INGESTION_ENABLED  # noqa: PLW0603
    if enabled and not _REAL_DATA_INGESTION_ENABLED:
        print("WARNING: Real data ingestion is now ENABLED.")
    _REAL_DATA_INGESTION_ENABLED = enabled


def is_real_data_allowed() -> bool:
    return _REAL_DATA_INGESTION_ENABLED


# ---------------------------------------------------------------------------
# PII patterns to redact
# ---------------------------------------------------------------------------

PII_PATTERNS: List[Tuple[re.Pattern, str]] = [
    # Australian postcodes (4-digit)
    (re.compile(r"\b\d{4}\b(?!\d)"), "[POSTCODE]"),
    # Phone numbers with country code
    (re.compile(r"\+?61\s?\d{4}\s?\d{3}\s?\d{3}|\(0\d{2}\)\s?\d{3}\s?\d{3}"), "[PHONE]"),
    # NDIS number format (generic)
    (re.compile(r"NDIS-\d{6,}", re.IGNORECASE), "[NDIS_NUM]"),
    # Email addresses
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b", re.IGNORECASE), "[EMAIL]"),
    # Dates of birth / Australian formats
    (re.compile(r"\b(?:0?[1-9]|3[01])(?:0?[1-9]|1[0-2])[/\-](?:19|20)\d{2}\b", re.IGNORECASE), "[DOB]"),
]


def deidentify(text: str) -> Tuple[str, List[str]]:
    """Redact PII from *text* and return (redacted_text, list_of_redacted_types).

    Returns the types of PII that were found (e.g. ["NAME", "POSTCODE"]).
    """
    redacted = text
    found: list[str] = []

    for pattern, replacement in PII_PATTERNS:
        if pattern.search(redacted):
            redacted = pattern.sub(replacement, redacted)
            # Determine the type name from the replacement token
            ptype = re.sub(r"\[|\]", "", replacement)
            if ptype not in found:
                found.append(ptype)

    return redacted, found


def deidentify_text(text: str) -> tuple[str, list[str]]:
    """Alias for ``deidentify`` (consistent naming for callers)."""
    return deidentify(text)
