# Baseline — prompted Qwen3-8B (no fine-tune)

**Phase 2 deliverable.** This is the bar fine-tuning must beat. Numbers are as
*measured* — deliberately not softened. Reproduce with `make baseline` (heuristic
judge) or the command below (LLM judge).

## Configuration

| | |
|---|---|
| Model under test | `qwen3:8b`, prompted (no fine-tune), temp 0.2, thinking disabled, via Ollama OpenAI endpoint |
| Judge (fuzzy dims) | `qwen3.6:27b`, local, thinking on — **calibrated** 100%/100% (faithfulness/register), κ=1.0 on the 12-sample fixture |
| Automated dims | deterministic structural + PII scorers |
| Test split | 13 stratified synthetic examples (offline generator, seed 42) |
| Failure suites | 10 cases each (fabrication / PII leakage / missing element) |
| Hardware | local (single RTX 5090, per PRD) |

```
uv run python -m eval.eval_suite --mode openai --judge llm --split test \
  --out baselines/qwen3-8b.json --html baselines/qwen3-8b.html
```

## Result: **BLOCKED** (fails hard gates)

| Metric | Value | Target | Type | Result |
|---|---|---|---|---|
| Structural compliance | **0.0%** | ≥98% | target | ❌ |
| Professional register | 100% | ≥95% | target | ✅ |
| Strata faithfulness | 100% | ≥98% | target | ✅ |
| Strata PII clean | 100% | 100% | target | ✅ |
| Strata billable evidence | 92.3% | ≥95% | target | ❌ |
| **Suite: fabrication** | **80%** | 100% | hard | ❌ |
| **Suite: PII leakage** | **60%** | 100% | hard | ❌ |
| Suite: missing element | 100% | 100% | hard | ✅ |

**Draft latency** (local, n=13): mean 3.36s, p50 2.79s, p95 4.67s, max 6.35s.

## What the failures actually are

- **Structural 0% — wrong value types.** Order is correct and no fields are
  missing, but the model emits `participant_present: "yes"` (string, not bool) and
  `outcomes_achieved` as a string (not a list). The strict schema (PRD §6.1
  "valid schema") counts these as failures. This is the single biggest and most
  tractable gap — fine-tuning, or constrained/JSON-schema decoding, fixes it.
- **PII leakage 60% — the cardinal-adjacent risk.** On 4/10 adversarial cases the
  8B copied a third-party name/phone into the note despite the redaction
  instruction.
- **Fabrication 80%.** 2/10 adversarial cases invented a fact (e.g. a duration
  the input didn't contain).
- **Billable 92.3%** — one sparse case lacked clear billable evidence.

## The bar for fine-tuning (Phase 3)

Must move **structural 0% → ≥98%**, **fabrication 80% → 100%**, **PII 60% → 100%**,
while holding register/faithfulness. Latency is already well within any reasonable
target (~3s/draft locally).

## Caveats (read before trusting the green cells)

- **n=13 is a smoke-sized split.** The 100% strata cells have ±~18% error — they
  mean "no failures seen in a tiny easy set", not "solved". Grow the stratified
  set to a few hundred before treating these as real (see README sizing note).
- **Synthetic data measures fit to the generator's distribution**, not real-world
  faithfulness. Real (de-identified) notes are the eventual ground truth.
- Strata faithfulness/PII here use easy routine inputs; the **adversarial suites**
  are the honest signal, and they fail.
