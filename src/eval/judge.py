"""Judges for the fuzzy evaluation dimensions.

Two interchangeable implementations:

* :class:`HeuristicJudge` — deterministic, no LLM. Grounds every number/date/
  identifier in the note against the worker input (faithfulness), checks tone
  markers (register), and checks billable evidence is stated. Lets the whole
  harness run offline and gives stable CI signal.
* :class:`LLMJudge` — a *local, open-weight* model (Ollama / vLLM) returning
  JSON matching the contracts in :mod:`eval.models`. Local-only so real notes
  never leave on-prem; never a frontier API (license-clean).

Both satisfy the :class:`Judge` protocol so the harness is judge-agnostic.
Calibrate the LLM judge against human labels (``eval.calibrate_judge``) before
trusting it.
"""

from __future__ import annotations

import re
from typing import Any, Protocol, runtime_checkable

from eval.models import BillableEvidenceCheck, FaithfulnessCheck, RegisterCheck
from ndis.notes import is_gap, note_text

FIRST_PERSON = re.compile(r"\b(i|we|my|our|me|us)\b", re.IGNORECASE)
FILLERS = re.compile(
    r"\b(um+|uh+|yeah|gonna|wanna|kinda|like,|so,|okay so|right so)\b", re.IGNORECASE
)


@runtime_checkable
class Judge(Protocol):
    name: str

    def judge_faithfulness(self, input_text: str, note: dict[str, Any]) -> FaithfulnessCheck: ...
    def judge_register(self, note: dict[str, Any]) -> RegisterCheck: ...
    def judge_billable(self, input_text: str, note: dict[str, Any]) -> BillableEvidenceCheck: ...


# --------------------------------------------------------------------------- #
# Heuristic judge
# --------------------------------------------------------------------------- #


class HeuristicJudge:
    name = "heuristic"

    # Structured fields whose factual content must be grounded in the input.
    _GROUNDED_FIELDS = ("duration_minutes", "date_of_service", "participant_id", "service_type")

    def judge_faithfulness(self, input_text: str, note: dict[str, Any]) -> FaithfulnessCheck:
        src = input_text.lower()
        fabricated: list[str] = []

        # 1) Structured facts.
        for fld in self._GROUNDED_FIELDS:
            val = note.get(fld)
            if val is None or is_gap(val):
                continue
            if str(val).lower() not in src:
                fabricated.append(f"{fld}={val!r} not in input")

        # 2) Numbers and identifiers appearing in the note prose.
        prose = note_text({k: v for k, v in note.items() if k not in self._GROUNDED_FIELDS})
        for num in set(re.findall(r"\d{2,}", prose)):
            if num not in input_text:
                fabricated.append(f"number {num} not in input")
        for code in set(re.findall(r"(?:PRT|WKR)[-\w]*", prose, re.IGNORECASE)):
            if code.lower() not in src:
                fabricated.append(f"identifier {code} not in input")

        ok = not fabricated
        return FaithfulnessCheck(
            is_faithful=ok,
            fabricated_claims=fabricated,
            reasoning="All facts grounded in input." if ok else "Ungrounded facts present.",
        )

    def judge_register(self, note: dict[str, Any]) -> RegisterCheck:
        narrative = str(note.get("narrative_summary", ""))
        issues: list[str] = []
        checks: list[bool] = []

        no_first_person = not FIRST_PERSON.search(narrative)
        checks.append(no_first_person)
        if not no_first_person:
            issues.append("uses first-person voice")

        no_fillers = not FILLERS.search(narrative)
        checks.append(no_fillers)
        if not no_fillers:
            issues.append("contains dictation fillers / informal markers")

        well_formed = (
            bool(narrative) and narrative[:1].isupper() and narrative.rstrip().endswith(".")
        )
        checks.append(well_formed)
        if not well_formed:
            issues.append("not a well-formed sentence (capitalisation/terminal period)")

        substantive = len(narrative.split()) >= 8
        checks.append(substantive)
        if not substantive:
            issues.append("narrative too short")

        score = sum(checks) / len(checks)
        return RegisterCheck(
            meets_register=score >= 0.75,
            score=round(score, 3),
            issues=issues,
            reasoning="Heuristic tone/format check.",
        )

    def judge_billable(self, input_text: str, note: dict[str, Any]) -> BillableEvidenceCheck:
        evidence = note.get("billable_evidence")
        present = not is_gap(evidence) and len(str(evidence).split()) >= 4
        return BillableEvidenceCheck(
            present=present,
            reasoning="Billable evidence stated."
            if present
            else "Billable evidence missing/thin.",
        )


# --------------------------------------------------------------------------- #
# Local LLM judge
# --------------------------------------------------------------------------- #

_JUDGE_SYSTEM = (
    "You are a strict, local evaluation judge for NDIS case notes. You return ONLY "
    "JSON matching the requested schema — no markdown, no commentary. Be conservative: "
    "if uncertain about fabrication or PII, fail the note."
)


class LLMJudge:
    def __init__(
        self,
        model: str = "qwen3.6:27b",
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "ollama",
    ) -> None:
        from openai import OpenAI

        self.name = f"llm:{model}"
        self.model = model
        self._client = OpenAI(base_url=base_url, api_key=api_key)

    def _ask(self, prompt: str) -> dict[str, Any]:
        from ndis.notes import coerce_draft

        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _JUDGE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=600,
        )
        return coerce_draft(resp.choices[0].message.content)

    def judge_faithfulness(self, input_text: str, note: dict[str, Any]) -> FaithfulnessCheck:
        prompt = (
            "Decide whether EVERY fact in the NOTE is supported by the WORKER INPUT. "
            "Any invented fact (number, date, name, event) is a fabrication.\n\n"
            f"WORKER INPUT:\n{input_text}\n\nNOTE:\n{note}\n\n"
            'Return JSON: {"is_faithful": bool, "fabricated_claims": [str], "reasoning": str}'
        )
        try:
            return FaithfulnessCheck.model_validate(self._ask(prompt))
        except Exception as exc:  # fail closed on any judge/parse error
            return FaithfulnessCheck(
                is_faithful=False, fabricated_claims=[], reasoning=f"judge error: {exc}"
            )

    def judge_register(self, note: dict[str, Any]) -> RegisterCheck:
        prompt = (
            "Rate the professional register of this NDIS case note narrative "
            "(objective, third-person, past tense, no fillers).\n\n"
            f"NOTE:\n{note}\n\n"
            'Return JSON: {"meets_register": bool, "score": 0..1, "issues": [str], "reasoning": str}'
        )
        try:
            return RegisterCheck.model_validate(self._ask(prompt))
        except Exception as exc:
            return RegisterCheck(meets_register=False, score=0.0, reasoning=f"judge error: {exc}")

    def judge_billable(self, input_text: str, note: dict[str, Any]) -> BillableEvidenceCheck:
        prompt = (
            "Does this note clearly state billable evidence (what was delivered to justify "
            "the claim)?\n\n"
            f"WORKER INPUT:\n{input_text}\n\nNOTE:\n{note}\n\n"
            'Return JSON: {"present": bool, "reasoning": str}'
        )
        try:
            return BillableEvidenceCheck.model_validate(self._ask(prompt))
        except Exception as exc:
            return BillableEvidenceCheck(present=False, reasoning=f"judge error: {exc}")


def make_judge(kind: str = "heuristic", **kwargs: Any) -> Judge:
    if kind == "heuristic":
        return HeuristicJudge()
    if kind in {"llm", "openai"}:
        return LLMJudge(**kwargs)
    raise ValueError(f"unknown judge kind: {kind!r} (expected heuristic|llm)")
