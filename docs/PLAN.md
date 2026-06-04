# Build Plan & Methodology

Companion to [`prd.md`](prd.md). This captures *how* we're building the model and
the decisions made along the way. Status as of the current branch.

## Status

| Phase | What | State |
|---|---|---|
| 0 | Scaffold, schema, config, synthetic generator (offline + teacher), frozen splits | ✅ |
| 1 | Eval harness: scorers, strata, 3 failure suites, **calibrated** local judge, scorecard | ✅ |
| 2 | Baseline: prompted Qwen3-8B measured → **BLOCKED** ([`../baselines/qwen3-8b.md`](../baselines/qwen3-8b.md)) | ✅ |
| 2.5 | Accurate NDIS taxonomy + **compliance-verified** data pipeline | ✅ (this work) |
| 3 | QLoRA fine-tune v1 on verified data; iterate until it clears the §2 bar | ⬜ next |
| 4 | Shrink protocol (4B, 1.5B) — smallest model that still clears the bar | ⬜ |
| 5 | vLLM serving + cascade fallback | ⬜ |
| 6 | FastAPI `POST /draft-note` | ⬜ |
| 7 | Monitoring + retrain runbook | ⬜ |

## Method: filtered (rejection-sampling) distillation

We are **not** doing logit/soft-label distillation. We do **sequence-level
distillation with verification**: a local teacher *generates* candidate
input→note pairs, a local verifier *checks* each one, and only fully-compliant
pairs become training data. The student (small model) is then fine-tuned (QLoRA)
on the survivors.

Why this and not naive distillation: the teacher is a *general* model, and we
measured that it **embellishes** (invents professional-sounding detail). Training
on raw teacher output would distil fabrication into the student — the cardinal
failure. Because correctness here is *verifiable* (faithfulness, schema, PII),
rejection-sampling is strictly better: **noisy generator → strict verifier → keep
only the good.**

### Roles (don't conflate them)
- **Teacher / generator** — proposes draft pairs. Can be *weak*; the verifier
  polices it. We use **qwen3:8b** (fast) by default; 27B optional for richer prose.
- **Verifier** — the quality gate. Must be *accurate*. We use a deterministic
  rubric (structure + PII) plus the **qwen3.6:27b** judge (faithfulness).
- **Judge** — the LLM half of the verifier; also scores the eval. **Calibrated at
  100%/100% agreement with human labels (κ=1.0)** — see `make calibrate`.

The valuable asymmetry is **weak generator + strong verifier**.

### The compliance verifier (`eval/verify.py`)
A pair is kept only if its target note is **all four** of:
1. correctly **structured** (required fields present + ordered),
2. **schema-valid** (right types / enums / date format),
3. **PII-clean** (no leaked phone/email/NDIS numbers),
4. **faithful** (every fact traces to the input — local LLM judge).

Cheap deterministic checks run first; the expensive LLM faithfulness check runs
last, so broken pairs never reach the judge. Each kept pair is therefore a *fully
compliant exemplar of every behaviour the eval grades*.

## Do we need a bigger model than 27B?

Not now. The 27B **verifier** is already calibrated to human judgement (κ=1.0), so
it isn't the bottleneck — and verification ("is this fact in the input?") is far
easier than generation. Reasons to stay:
- **On-prem premise:** single RTX 5090 (32GB). 27B fits quantized; 70B+ is slow
  and risks breaking the "runs locally" guarantee.
- **Recency > size:** a current 27B beats an older 70B.
- A bigger model would mainly help generator *diversity*, which prompt/temperature
  variation gives for free.

Revisit only if you grow the human-labelled calibration set with harder cases and
the 27B's agreement drops below ~90%.

## Data-generation recipe (Phase 3 input)

```bash
# Quality run — 8B generator + 27B faithfulness verifier (overnight batch).
uv run python -m ndis.synth_generate --count 1200 \
    --teacher-model qwen3:8b --filter --filter-judge llm \
    --out-dir data/splits
```

- **Count:** ~1,000–1,500 verified pairs for v1 (then add where strata fail).
- **Composition:** weight toward the hard behaviours — bump `STRATUM_PLAN` so
  `sparse_input` / `adv_pii_check` / `adv_missing_field` are a larger share
  (they teach gap-flagging and redaction, the gated safety behaviours).
- **Throughput:** structure/PII drops are instant; only structurally-valid pairs
  reach the 27B judge. Expect a meaningful drop rate (the teacher isn't perfect —
  that's the point). Budget it as an overnight batch.
- **Safety:** generation is hard-capped (max attempts + consecutive failures per
  stratum) so it can never run away; shortfalls are reported, not hidden.

## Test-set sizing

See the "How much test data?" section in [`../README.md`](../README.md). Short
version: ~150–300 stratified for iteration (±~10%/stratum); grow the failure
suites toward 100–300 each for a low *provable* fabrication rate (rule of three);
diversity beats raw count.

## Guarantees

- **Local / license-clean:** teacher, verifier, judge, and student are all local
  open-weight Qwen via Ollama. No frontier-API outputs in the training lineage.
- **No real data:** synthetic-first; real notes only via the human-gated de-id
  step once data agreements exist.
- **Human-in-the-loop:** the model drafts; a person reviews and submits.
