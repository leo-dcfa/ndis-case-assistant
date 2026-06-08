# v1 fine-tune — Qwen3-8B QLoRA (heuristic judge)

First fine-tune on the 800-example v1 dataset (3 epochs). Scored on the frozen
120-example test set + adversarial suites with the **heuristic** judge.

| Metric | Baseline (prompted) | v1 (fine-tuned) |
|---|---|---|
| Structural compliance | 0% | **100%** |
| Professional register | 100% | 100% |
| Strata faithfulness | 100% (n=13) | 100% (n=120) |
| Fabrication suite (hard) | 80% | **100%** |
| PII leakage suite (hard) | 60% | **50%** |
| Missing-element suite (hard) | 100% | **60%** |
| **Verdict** | BLOCKED | **BLOCKED** |

Fine-tuning fixed structure + fabrication + all in-distribution dimensions, but
regressed on the two safety behaviours on *out-of-distribution* adversarial
cases. Root cause: training data had only 3 PII patterns and always omitted the
same 2 fields, so the model learned surface forms, not the behaviour. Fixed in
the v2 generator (wide PII variety + missing-field rotation across all 8 fields).
Note: PII/missing are deterministic gates, so this verdict is judge-independent;
faithfulness/fabrication here are heuristic (optimistic) pending an LLM-judged pass.
