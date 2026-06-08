# v3 fine-tune — Qwen3-8B QLoRA (heuristic judge)

Trained on v3 data (PII woven into varied positions; adv_pii share increased).

| Hard gate | v1 | v2 | v3 |
|---|---|---|---|
| Structural | 100% | 100% | 100% |
| Fabrication | 100% | 100% | 100% |
| Missing-element | 60% | 90% | 90% |
| PII leakage | 50% | 40% | **70%** |
| Verdict | BLOCKED | BLOCKED | BLOCKED |

Varied PII placement + more examples moved PII 40→70%. Remaining leaks are bare
first names (Marcus, Hannah Wells, Amy) — phones/emails are now reliably redacted,
but standalone names embedded in content still survive sometimes. SFT alone is
unlikely to guarantee 100% on names; the recommended path is defence-in-depth:
keep improving the data AND add a deterministic output-side PII scrubber
(reuse ndis/deidentify.py) so the hard gate is guaranteed, not probabilistic.
