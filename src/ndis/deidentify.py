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


# --------------------------------------------------------------------------- #
# Output-side guardrail (defense-in-depth)
#
# The fine-tuned model does the smart reshaping/redaction; this deterministic
# scrubber runs on its OUTPUT so a hard 100% PII gate does not rest on a
# probabilistic model. It is intentionally PRECISE (no bare-postcode / DOB rules
# that would false-positive on a note's date/duration), and it removes
# third-party people who were introduced in the worker INPUT by a relationship
# or title cue (mum, brother, carer, Dr, Mrs, coordinator, …) — the way third
# parties almost always appear in case notes.
# (Production hardening: swap the cue heuristic for a local NER model to also
# catch cue-less names. Documented in docs/PLAN.md.)
# --------------------------------------------------------------------------- #

REDACTED = "[redacted]"

_NOTE_PII_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b", re.IGNORECASE),  # email
    re.compile(r"NDIS[-\s]?\d{6,}", re.IGNORECASE),  # NDIS number
    re.compile(r"\b(?:\+?61[ -]?|\(0\d\)\s?|0)\d(?:[ -]?\d){7,9}\b"),  # AU phone
]

# Relationship / role / title cues that introduce a third party in worker notes.
_NAME_CUES = (
    r"mum|mother|dad|father|parent|brother|sister|sibling|son|daughter|aunt|uncle|"
    r"cousin|grandma|grandpa|nan|pop|partner|wife|husband|friend|flatmate|housemate|"
    r"neighbour|neighbor|carer|caregiver|guardian|advocate|coordinator|worker|staff|"
    r"teacher|nurse|gp|doctor|dr|mr|mrs|ms|miss|sir"
)
# A cue followed by 1–3 capitalised name words (e.g. "brother Marcus",
# "coordinator Hannah Wells", "Dr Aisha Khan").
_CUE_NAME_RE = re.compile(rf"\b(?:{_NAME_CUES})\b[\s,'’.\-]*([A-Z][a-z]+(?:\s[A-Z][a-z]+){{0,2}})")
# Standalone honorific + name even without a relationship cue (e.g. "Dr Nguyen").
_TITLE_NAME_RE = re.compile(
    r"\b(?:Dr|Mr|Mrs|Ms|Miss|Prof)\.?\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+){0,2})"
)


def extract_contact_names(input_text: str) -> list[str]:
    """Third-party person names introduced in the worker input by a cue/title."""
    names: list[str] = []
    for rx in (_CUE_NAME_RE, _TITLE_NAME_RE):
        for m in rx.finditer(input_text):
            name = m.group(1).strip()
            if name and name not in names:
                names.append(name)
    # Longest first so multi-word names are removed before their first-name substring.
    return sorted(names, key=len, reverse=True)


def _scrub_text(text: str, names: list[str]) -> tuple[str, bool]:
    changed = False
    for name in names:
        new = re.sub(rf"\b{re.escape(name)}\b", REDACTED, text)
        if new != text:
            text, changed = new, True
    for rx in _NOTE_PII_PATTERNS:
        new = rx.sub(REDACTED, text)
        if new != text:
            text, changed = new, True
    return text, changed


# Structured / pseudonymous fields that are not free prose — left untouched so a
# date or participant code is never mangled.
_NON_PROSE_FIELDS = {
    "date_of_service",
    "duration_minutes",
    "participant_present",
    "follow_up_needed",
}


def redact_note(note: dict, input_text: str = "") -> tuple[dict, list[str]]:
    """Scrub residual PII from a drafted note. Returns (clean_note, fields_changed)."""
    names = extract_contact_names(input_text)
    out: dict = {}
    changed_fields: list[str] = []
    for key, value in note.items():
        if key in _NON_PROSE_FIELDS or not isinstance(value, (str, list)):
            out[key] = value
            continue
        if isinstance(value, list):
            scrubbed = []
            hit = False
            for item in value:
                if isinstance(item, str):
                    s, c = _scrub_text(item, names)
                    scrubbed.append(s)
                    hit = hit or c
                else:
                    scrubbed.append(item)
            out[key] = scrubbed
            if hit:
                changed_fields.append(key)
        else:
            s, c = _scrub_text(value, names)
            out[key] = s
            if c:
                changed_fields.append(key)
    return out, changed_fields
