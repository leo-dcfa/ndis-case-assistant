# v2 fine-tune — Qwen3-8B QLoRA (heuristic judge)

Trained on the v2 dataset (widened adv PII + missing-field rotation).

| Hard gate | v1 | v2 |
|---|---|---|
| Structural compliance | 100% | 100% |
| Fabrication | 100% | 100% |
| Missing-element | 60% | **90%** ✅ |
| PII leakage | 50% | **40%** ❌ |
| Verdict | BLOCKED | BLOCKED |

The missing-field rotation fix worked (60→90%). PII still fails: the model leaks
third-party **names and emails** woven into the input — because v2 training always
injected PII as a single droppable `aside:` line, so it learned to delete that
line, not to redact PII anywhere. v3 fix: inject PII in varied positions
(inline/note/aside/trailing) and increase the adv_pii share.
