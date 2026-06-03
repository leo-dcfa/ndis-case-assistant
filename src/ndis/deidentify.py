"""De-identification pipeline for NDIS data."""

from __future__ import annotations

import re
from typing import List, Tuple

PII_PATTERNS: List[tuple[re.Pattern, str]] = [
    (re.compile(r"\b\d{4}\b(?!\d)"), "[POSTCODE]"),
    (re.compile(r"\+?61\s?\d{4}\s?\d{3}\s?\d{3}|\(0\d{2}\)\s?\d{3}\s?\d{3}"), "[PHONE]"),
    (re.compile(r"NDIS-\d{6,}", re.IGNORECASE), "[NDIS_NUM]"),
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b", re.IGNORECASE), "[EMAIL]"),
    (re.compile(r"\b(?:0?[1-9]|3[01])(?:0?[1-9]|1[0-2])[/\-](?:19|20)\d{2}\b", re.IGNORECASE), "[DOB]"),
]


def deidentify(text: str) -> tuple[str, List[str]]:
    redacted = text
    found: list[str] = []

    for pattern, replacement in PII_PATTERNS:
        if pattern.search(redacted):
            redacted = pattern.sub(replacement, redacted)
            ptype = re.sub(r"\[|\]", "", replacement)
            if ptype not in found:
                found.append(ptype)

    return redacted, found


def deidentify_text(text: str) -> tuple[str, list[str]]:
    return deidentify(text)
