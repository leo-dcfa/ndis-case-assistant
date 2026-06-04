"""De-identification pipeline for NDIS data."""

from __future__ import annotations

import re

# Order matters: longer / more specific patterns first so a phone number is not
# partially eaten by the bare 4-digit postcode rule.
PII_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b", re.IGNORECASE), "[EMAIL]"),
    (re.compile(r"NDIS[-\s]?\d{6,}", re.IGNORECASE), "[NDIS_NUM]"),
    # Australian landline/mobile with optional +61 and space/dash separators.
    (re.compile(r"\b(?:\+?61[ -]?|\(0\d\)\s?|0)\d(?:[ -]?\d){7,9}\b"), "[PHONE]"),
    (
        re.compile(r"\b(?:0?[1-9]|3[01])(?:0?[1-9]|1[0-2])[/\-](?:19|20)\d{2}\b", re.IGNORECASE),
        "[DOB]",
    ),
    (re.compile(r"\b\d{4}\b(?!\d)"), "[POSTCODE]"),
]


def deidentify(text: str) -> tuple[str, list[str]]:
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
