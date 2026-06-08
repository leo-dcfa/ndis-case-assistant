"""Compliance verifier for synthetic training data (rejection-sampling gate).

A teacher-generated pair is kept only if its target note clears **all four**
gated behaviours the eval grades:

1. **Structurally correct** — required fields present, ordered, valid types.
2. **Schema-valid** — covered by the structure check (types/enums/dates).
3. **PII-clean** — no leaked phone/email/NDIS numbers (or forbidden strings).
4. **Faithful** — every fact traces to the input (local LLM judge).

Cheap deterministic checks run first; the expensive LLM faithfulness check runs
last, so obviously-broken pairs never reach the judge. This is what turns naive
distillation into *filtered* distillation: the teacher proposes, this verifies,
and only fully-compliant exemplars become training data.

Everything is LOCAL (rubric is pure Python; the judge is a local Ollama model).
"""

from __future__ import annotations

from collections.abc import Callable

from eval.judge import make_judge
from eval.rubric import score_pii, score_structure


def build_compliance_verifier(
    judge_kind: str = "heuristic", *, verbose: bool = True
) -> Callable[..., bool]:
    """Return a ``verify(input, target, stratum, forbidden=None) -> bool`` gate.

    ``judge_kind`` selects the faithfulness judge: 'heuristic' (instant, catches
    number/ID fabrication only) or 'llm' (local 27B, catches embellished prose —
    recommended for real training data). ``forbidden`` is the exact list of
    third-party PII strings that must not survive (names regex can't catch).
    """
    judge = make_judge(judge_kind)

    def verify(
        input_text: str, target: dict, stratum: str, forbidden: list[str] | None = None
    ) -> bool:
        # 1+2. Structure / schema (deterministic, instant).
        st = score_structure(target, stratum=stratum)
        if not st.passed:
            if verbose:
                reasons = []
                if st.missing_fields:
                    reasons.append(f"missing={st.missing_fields}")
                if st.gap_fields:
                    reasons.append(f"unflagged-gaps={st.gap_fields}")
                if st.invalid_fields:
                    reasons.append(f"invalid-types={st.invalid_fields}")
                if not st.order_ok:
                    reasons.append("wrong-field-order")
                print(f"    ✗ structure: {', '.join(reasons) or 'present_ok failed'}")
            return False

        # 3. PII (deterministic, instant) — regex + exact forbidden third-party strings.
        pii = score_pii(target, forbidden=forbidden)
        if not pii.is_clean:
            if verbose:
                print(f"    ✗ pii leak: {pii.leaked_items}")
            return False

        # 4. Faithfulness (LLM — most expensive, so checked last).
        fc = judge.judge_faithfulness(input_text, target)
        if not fc.is_faithful:
            if verbose:
                print(f"    ✗ fabrication: {fc.fabricated_claims[:2]}")
            return False

        return True

    verify.judge_name = judge.name  # type: ignore[attr-defined]
    return verify
