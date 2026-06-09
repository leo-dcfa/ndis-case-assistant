# Results summary — Qwen3-8B QLoRA, eval-driven iterations

Frozen 120-example test set + adversarial hard-gate suites (heuristic-judged;
PII/missing/structure gates are deterministic, so judge-independent).
"+G" = with the deterministic output-side PII guardrail (the served config).

| Hard gate | Baseline | v1 | v2 | v3 | v3+G | v4+G |
|---|---|---|---|---|---|---|
| Structural compliance | 0% | 100 | 100 | 100 | 100 | 100 |
| Fabrication | 80% | 100 | 100 | 100 | 100 | 100 |
| PII leakage | 60% | 50 | 40 | 70 | **100** | **100** |
| Missing-element | — | 60 | 90 | 90 | 90 | 80 |
| Release (hard gates) | BLOCKED | … | … | … | BLOCKED | BLOCKED |

**Solved:** structural, fabrication, and PII (the last via a deterministic
guardrail — defense-in-depth, 100% and judge-independent).

**Open: missing-element (~80–90%).** It bounces because the suite is only 10
cases (±10%/case) and per-field SFT signal is thin (e.g. 6 billable-missing
examples). Blind data tweaks whack-a-mole between fields (v4's billable fix
traded for a service_type regression). The right next steps are NOT another blind
iteration but:
1. grow the failure suites to ~50 cases each for a stable signal;
2. raise per-missing-field training counts once the suite can measure it;
3. rely on the PRD's mandated **human-in-the-loop** (no autonomous submission)
   as the backstop for the residual — the system flags low-confidence/edge
   cases for review rather than auto-submitting.

**Recommended current adapter:** v3 + guardrail (missing-element 90%, all other
hard gates 100%).
