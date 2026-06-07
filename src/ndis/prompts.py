"""The model's input/output contract — shared by training and inference.

Fine-tuning only transfers if the model sees the SAME prompt format at train time
and at serve time. So the drafting system prompt, the user-message template, and
the target serialisation all live here, and both the trainer (``train/``) and the
served model (``eval.model_under_test.OpenAIModel``) import them.
"""

from __future__ import annotations

import json
from typing import Any

from ndis.notes import GAP_MARKER

DRAFT_SYSTEM_PROMPT = (
    "You convert a support worker's rough input (bullet points or dictation) into a "
    "structured, compliance-ready NDIS case note. You FAITHFULLY reshape the input — "
    "you NEVER invent clinical or factual content. Where a required element is absent "
    f'from the input, write "{GAP_MARKER}" rather than guessing. Redact any third-party '
    "personal details (names, phone numbers, emails, addresses). Write in calm, "
    "objective, professional third-person past tense. Output STRICT JSON only — no "
    "markdown fences, no commentary."
)

DRAFT_FIELDS = [
    "participant_id",
    "date_of_service",
    "duration_minutes",
    "service_type",
    "goal_linkage",
    "location",
    "staff_presented_by",
    "participant_present",
    "narrative_summary",
    "billable_evidence",
    "outcomes_achieved",
    "risk_management",
    "follow_up_needed",
    "follow_up_notes",
]


def build_user_message(input_text: str) -> str:
    """The user turn: the worker's raw input + the required output keys."""
    return (
        "Draft the NDIS case note for this worker input. Return STRICT JSON with "
        f"exactly these keys: {', '.join(DRAFT_FIELDS)}.\n\n"
        "--- WORKER INPUT ---\n"
        f"{input_text}\n"
        "--------------------"
    )


def target_to_assistant(target: dict[str, Any]) -> str:
    """The assistant turn: the gold note serialised as compact strict JSON."""
    return json.dumps(target, ensure_ascii=False)


def build_chat(input_text: str, target: dict[str, Any] | None = None) -> list[dict[str, str]]:
    """Chat messages for one example. With ``target`` -> a full training example;
    without -> the prompt only (for inference)."""
    messages = [
        {"role": "system", "content": DRAFT_SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(input_text)},
    ]
    if target is not None:
        messages.append({"role": "assistant", "content": target_to_assistant(target)})
    return messages
