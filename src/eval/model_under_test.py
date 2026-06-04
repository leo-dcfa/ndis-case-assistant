"""Model-under-test abstraction for the evaluation harness.

The harness scores any object that can turn a worker's raw input into a drafted
note dict. Three implementations are provided:

* :class:`GoldenModel` — the "dummy" model the Phase 1 DoD calls for. It returns
  the dataset's gold target for each input, so the harness can be exercised
  end-to-end and we can confirm a *perfect* model clears every gate. It is a
  harness self-test, not a real model.
* :class:`NaiveModel` — a deliberately weak, no-LLM baseline (copies the raw
  input into the narrative, guesses a few fields, invents a duration when one is
  missing). It exercises every failure mode so the gates demonstrably *fail*
  when they should.
* :class:`OpenAIModel` — drafts via an OpenAI-compatible endpoint (local Ollama
  or vLLM). This is what Phase 2's prompted-baseline and later the fine-tuned
  model plug into. No frontier APIs — license-clean.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from ndis.notes import GAP_MARKER, coerce_draft

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


@runtime_checkable
class ModelUnderTest(Protocol):
    name: str

    def draft_note(self, input_text: str, stratum: str | None = None) -> dict[str, Any]: ...


class GoldenModel:
    """Returns the dataset's gold target for each input (harness self-test)."""

    name = "golden-dummy"

    def __init__(self, examples: list[dict[str, Any]]) -> None:
        self._by_input: dict[str, dict[str, Any]] = {ex["input"]: ex["target"] for ex in examples}

    def draft_note(self, input_text: str, stratum: str | None = None) -> dict[str, Any]:
        target = self._by_input.get(input_text)
        if target is None:
            return {}
        return dict(target)


class NaiveModel:
    """A weak, no-LLM baseline that fails on purpose in instructive ways."""

    name = "naive-baseline"

    def draft_note(self, input_text: str, stratum: str | None = None) -> dict[str, Any]:
        fields = _crude_parse(input_text)
        return {
            "participant_id": fields.get("participant", ""),
            "date_of_service": fields.get("date", ""),
            # Fabrication: always asserts 60 min even when none was recorded.
            "duration_minutes": 60,
            "service_type": fields.get("type", "Other"),
            "goal_linkage": "",  # structural gap
            "location": fields.get("location", ""),
            "staff_presented_by": fields.get("worker", ""),
            "participant_present": True,
            # PII/register failure: dumps the raw worker input verbatim.
            "narrative_summary": input_text,
            "billable_evidence": "",
            "outcomes_achieved": [],
            "risk_management": GAP_MARKER,
            "follow_up_needed": False,
            "follow_up_notes": GAP_MARKER,
        }


class OpenAIModel:
    """Drafts via an OpenAI-compatible endpoint (local Ollama / vLLM)."""

    def __init__(
        self,
        model: str = "qwen3:8b",
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "ollama",
        temperature: float = 0.2,
    ) -> None:
        from openai import OpenAI

        self.name = f"openai:{model}"
        self.model = model
        self.temperature = temperature
        self._client = OpenAI(base_url=base_url, api_key=api_key)

    def draft_note(self, input_text: str, stratum: str | None = None) -> dict[str, Any]:
        user = (
            "Draft the NDIS case note for this worker input. Return STRICT JSON with "
            f"exactly these keys: {', '.join(DRAFT_FIELDS)}.\n\n"
            "--- WORKER INPUT ---\n"
            f"{input_text}\n"
            "--------------------"
        )
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": DRAFT_SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            temperature=self.temperature,
            max_tokens=1600,
            # Ollama: disable Qwen3 "thinking" for clean, fast JSON output.
            extra_body={"think": False},
        )
        return coerce_draft(resp.choices[0].message.content)


def _crude_parse(text: str) -> dict[str, str]:
    """Naive 'key: value' line scraper used by NaiveModel."""
    out: dict[str, str] = {}
    normalised = text.replace(";", "\n")
    for line in normalised.splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            out[key.strip().lower()] = value.strip()
    return out


def build_model(mode: str, examples: list[dict[str, Any]]) -> ModelUnderTest:
    if mode == "dummy":
        return GoldenModel(examples)
    if mode == "naive":
        return NaiveModel()
    if mode == "openai":
        return OpenAIModel()
    raise ValueError(f"unknown model mode: {mode!r} (expected dummy|naive|openai)")
