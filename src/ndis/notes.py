"""Shared helpers for working with drafted case notes.

A *drafted* note is a plain ``dict`` (what a model emits / what synthetic
targets store), distinct from the strict :class:`ndis.models.CaseNote` which
only accepts a fully-complete, valid note. Drafts may legitimately contain
explicit gap markers (sparse / adversarial-missing strata) where the correct
behaviour is to *flag* a missing element rather than invent one.

These helpers give the synth generator and the eval harness a single,
consistent notion of "this field is a flagged gap" and "this is the note's
prose", so faithfulness/structure scoring stays honest about what counts as
present vs. fabricated.
"""

from __future__ import annotations

from typing import Any

# Canonical marker a model should emit when an element is genuinely absent
# from the worker's input. Treated as "present-but-flagged-missing".
GAP_MARKER = "[not recorded]"

# Sentinel for the numeric duration field when not recorded.
DURATION_GAP = -1

# String fields whose free text we treat as the note's "prose" for faithfulness
# and register judging.
PROSE_FIELDS = (
    "narrative_summary",
    "billable_evidence",
    "goal_linkage",
    "risk_management",
    "follow_up_notes",
)


def is_gap(value: Any) -> bool:
    """True if ``value`` represents a flagged-missing element."""
    if value is None:
        return True
    if isinstance(value, str):
        s = value.strip().lower()
        return s == "" or s == GAP_MARKER.lower() or s in {"[missing]", "n/a", "none", "unknown"}
    if isinstance(value, int) and not isinstance(value, bool):
        return value == DURATION_GAP
    if isinstance(value, (list, tuple)):
        return len(value) == 0
    return False


def note_text(note: dict[str, Any]) -> str:
    """Concatenate the human-readable prose of a note for text-level checks."""
    chunks: list[str] = []
    for key, value in note.items():
        if is_gap(value):
            continue
        if isinstance(value, str):
            chunks.append(value)
        elif isinstance(value, list):
            chunks.extend(str(v) for v in value)
    return "\n".join(chunks)


def coerce_draft(raw: Any) -> dict[str, Any]:
    """Best-effort normalise a model's raw output into a note dict.

    Accepts an already-parsed dict, or a JSON string (optionally wrapped in
    markdown fences / surrounded by chatter). Returns ``{}`` if nothing
    parseable is found — callers treat an empty draft as a total structural
    failure rather than crashing.
    """
    import json

    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return {}

    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(line for line in lines[1:] if not line.strip().startswith("```"))

    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        return {}
    # NB: kept as two clauses rather than `except (ValueError, TypeError)` — the
    # pinned ruff-format (v0.15.15) miscompiles a bare parenthesised except tuple
    # into invalid Python, so we avoid that exact construct.
    try:
        parsed = json.loads(cleaned[start : end + 1])
    except ValueError:
        return {}
    except TypeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}
