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
from ndis.prompts import DRAFT_FIELDS, DRAFT_SYSTEM_PROMPT, build_chat  # noqa: F401  (re-exported)


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
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=build_chat(input_text),  # shared train/serve I/O contract
            temperature=self.temperature,
            max_tokens=1600,
            # Ollama: disable Qwen3 "thinking" for clean, fast JSON output.
            extra_body={"think": False},
        )
        return coerce_draft(resp.choices[0].message.content)


class HFModel:
    """In-process transformers inference for a fine-tuned LoRA adapter.

    Loads the 4-bit base model + the trained adapter and generates greedily —
    no server needed, so the eval can score an adapter straight after training.
    Uses the same chat contract as training (``build_chat``).
    """

    def __init__(
        self,
        base_model: str,
        adapter: str | None = None,
        max_new_tokens: int = 768,
    ) -> None:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.name = f"hf:{adapter or base_model}"
        self.max_new_tokens = max_new_tokens
        self._tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        if self._tok.pad_token is None:
            self._tok.pad_token = self._tok.eos_token
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb,
            device_map="auto",
            trust_remote_code=True,
            dtype=torch.bfloat16,
        )
        if adapter:
            model = PeftModel.from_pretrained(model, adapter)
        model.eval()
        self._model = model

    def draft_note(self, input_text: str, stratum: str | None = None) -> dict[str, Any]:
        import torch

        prompt = self._tok.apply_chat_template(
            build_chat(input_text),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self._tok(prompt, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self._tok.pad_token_id,
            )
        text = self._tok.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        return coerce_draft(text)


class GuardedModel:
    """Wrap any model with the deterministic output-side PII scrubber.

    This is the *served* configuration: the model drafts, then known PII patterns
    and cue-introduced third-party names are stripped before the note is emitted —
    so the PII hard gate is guaranteed, not probabilistic. Evaluating with this
    wrapper measures what would actually ship.
    """

    def __init__(self, inner: ModelUnderTest) -> None:
        self.inner = inner
        self.name = f"guarded({inner.name})"

    def draft_note(self, input_text: str, stratum: str | None = None) -> dict[str, Any]:
        from ndis.deidentify import redact_note

        note = self.inner.draft_note(input_text, stratum=stratum)
        if not note:
            return note
        clean, _changed = redact_note(note, input_text)
        return clean


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
