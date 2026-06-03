# PRD — NDIS Case Note Assistant (owned small-model fine-tune)

**Owner:** Leo / Azul Labs
**Status:** Draft v1 — build plan

---

## 1. Overview

Build a small, fine-tuned, locally-served language model that turns a support worker's
rough input (bullet points or a dictation transcript) into a properly structured,
compliance-ready NDIS case note. The model **faithfully reshapes the worker's input** into
the required note structure — it must never invent clinical or factual content.

This is an **owned-model** deliverable: the model and its full training lineage are
license-clean (no frontier-API outputs in the training set), and it runs on-premises so
participant data never leaves controlled infrastructure.

### Why a small model fits
The task is *constrained generation*, not open prose: the required note structure acts as
scaffolding, so a small fine-tuned model can match a large general model on this narrow task
at a fraction of the serving cost — and small enough to run on-prem.

---

## 2. Success criteria (the bar)

The project is successful when, on the held-out test set:

- **Structural compliance ≥ 98%** — every required NDIS note element present and correctly placed.
- **Faithfulness = 100% on the adversarial fabrication suite** — zero fabricated facts. This is a hard gate; any failure blocks release.
- **PII handling = 100% on the privacy suite** — no leakage, no incorrect identifiers.
- **Professional register ≥ 95%** — appropriate tone/format per the style rubric.
- **Smallest model that clears all of the above** — see shrink protocol (§7, Phase 4).
- Serving latency and cost meet targets defined once baseline is measured.

**Eval-driven development:** the evaluation harness (§6) is built and validated *before* any
fine-tuning. We do not train until we can measure.

---

## 3. Scope & non-goals

**In scope:** data schema + de-identification pipeline, synthetic seed data, evaluation harness,
baseline measurement, QLoRA fine-tuning, model-size shrink protocol, local serving with cascade
fallback, FastAPI integration surface, monitoring hooks.

**Non-goals (do NOT build):**
- No free-form article/clinical-content generation — transformation only.
- No frontier-API outputs used as training data (license-clean requirement).
- No real participant data in the repo or in any cloud call — synthetic-first; real data only
  enters via the de-id pipeline once data agreements exist (human-gated, see §5).
- No multi-GPU/distributed training — single RTX 5090, QLoRA.
- No autonomous note submission — human review remains in the loop.

---

## 4. Tech stack & constraints

- **Language/runtime:** Python 3.14+, `uv` for env management.
- **Base model:** Qwen3 family (Apache-2.0). Start at **Qwen3-8B**, shrink toward 4B / 1.5B.
- **Fine-tuning:** Unsloth (QLoRA, 4-bit) on a single RTX 5090 (32GB). Axolotl as fallback.
- **Teacher (if synthetic distillation is used):** a **permissive open-weight** model only
  (e.g. a large Qwen3). **Never** Claude/GPT/Gemini outputs — license-clean is a hard rule.
- **Serving:** vLLM, OpenAI-compatible endpoint.
- **Eval:** Pydantic Evals + a custom rubric scorer; **local open-weight judge** for fuzzy
  dimensions (judge runs locally so real notes never leave on-prem).
- **API surface:** FastAPI.
- **Experiment tracking:** local JSON run logs + optional Weights & Biases (off by default).
- **Data format:** chat-style JSONL (`system` / `user` / `assistant`).

Hardware note: 8B QLoRA fine-tune and serving both fit comfortably on the 5090; no larger GPU
is required for this project.

---

## 5. Data

### 5.1 Schema
Define a `CaseNote` Pydantic model capturing required NDIS elements (goal linkage, service
type, date/duration, participant identifier reference, narrative, billable evidence, outcomes).
Treat the exact required-fields list as **configurable** — load it from `config/required_fields.yaml`
so the compliance rule set can be edited without code changes.

### 5.2 Synthetic-first
Generate a synthetic seed dataset of realistic-but-fake NDIS case notes (input → target note
pairs) using the open-weight teacher, covering all strata in §6.2. This lets the entire
pipeline — training, eval, serving — be built and tested with **zero real participant data**.

### 5.3 De-identification pipeline (for real data, later)
Build `data/deidentify.py`: detects and redacts/pseudonymises PII (names, addresses, dates of
birth, NDIS numbers, etc.) before any note enters a training or eval set. Real data ingestion
is **human-gated**: the pipeline exists, but ingesting real notes is a manual step the owner runs
only once data agreements are in place. Document this clearly; do not wire real-data ingestion
into any automated flow.

### 5.4 Splits
Train / validation / **held-out test**. The test set is frozen and never seen during training or
hyperparameter selection.

---

## 6. Evaluation harness (build this FIRST)

### 6.1 Structure
`eval/` runs every candidate model over the test set and the failure suites, producing a scorecard
against the §2 bar. Each example is scored on multiple dimensions:

**Automated (run on every example):**
- All required fields present + valid schema.
- Structure/ordering correct.
- No PII errors (cross-check against the de-id allowlist).

**Judged (local open-weight LLM judge, calibrated to human):**
- Faithfulness — every fact in the note traceable to the input; **no fabrication**.
- Professional register / tone.
- Billable evidence present where required.

### 6.2 Stratified test set
Target **150–300** real examples (synthetic for now), stratified to cover every case type with
enough examples per stratum to read a within-stratum failure rate:
- Routine session notes
- Incident notes
- Each distinct service type
- Sparse-input cases (minimal worker input)
- Edge/unusual cases

### 6.3 Adversarial failure suites (hard gates)
Separate, strictly-scored suites of 10–20 cases each for the unacceptable failure modes:
- **Fabrication** — inputs that tempt the model to invent content; must score 100% faithful.
- **PII leakage** — must score 100% clean.
- **Missing mandatory element** — must be caught.

### 6.4 Judge calibration
Provide a script for the owner to validate a sample of judge "passes" against human judgement,
and report judge–human agreement. Do not trust the automated judge until agreement is confirmed.

---

## 7. Build phases

Each phase has a definition of done. Do not advance until acceptance criteria pass.

**Phase 0 — Scaffold**
Repo structure (§8), env, config files, `CaseNote` schema, synthetic data generator.
*DoD:* `make synth` produces a stratified synthetic dataset; schema validates.

**Phase 1 — Evaluation harness**
Build the full scorer (automated + judged), stratified test set, failure suites, judge
calibration script.
*DoD:* harness runs end-to-end on a dummy model and emits a scorecard; judge calibration script works.

**Phase 2 — Baseline**
Measure **prompted Qwen3-8B (no fine-tune)** on the eval. This sets the bar fine-tuning must beat.
*DoD:* baseline scorecard committed; targets in §2 finalised with real latency/cost numbers.

**Phase 3 — Fine-tune v1**
QLoRA fine-tune Qwen3-8B on the training set. Iterate data quality → filtering → epochs.
*DoD:* v1 beats baseline and clears the §2 bar on the test set, including 100% on failure suites.

**Phase 4 — Shrink protocol**
With v1 passing, retrain at 4B, then 1.5B. Pick the **smallest model that still clears the full
bar**; stop one size above where the eval breaks.
*DoD:* chosen model documented with the scorecard at each size.

**Phase 5 — Serving + cascade**
vLLM endpoint for the chosen model. Implement **cascade fallback**: when automated structural
checks fail or a confidence/uncertainty heuristic trips, escalate the case to a larger model (or
flag for human review). Never auto-submit.
*DoD:* endpoint serves; cascade routes failing cases correctly; latency target met.

**Phase 6 — Integration surface**
FastAPI wrapper exposing a clean `POST /draft-note` (input → drafted note + confidence + flags),
plus a review payload for human-in-the-loop.
*DoD:* API documented (OpenAPI), integration test passes.

**Phase 7 — Monitoring + retrain loop**
Log inputs/outputs/flags (de-identified) for drift monitoring; document the retraining trigger
and procedure.
*DoD:* monitoring writes structured logs; retrain runbook committed.

---

## 8. Repo structure

```
ndis-case-note-model/
├── config/
│   ├── required_fields.yaml      # editable NDIS compliance rule set
│   └── model.yaml                # base model, LoRA, training hyperparams
├── data/
│   ├── synth_generate.py         # synthetic seed data (open-weight teacher)
│   ├── deidentify.py             # PII redaction (human-gated for real data)
│   ├── schema.py                 # CaseNote Pydantic model
│   └── splits/                   # train / val / test (test frozen)
├── eval/
│   ├── rubric.py                 # automated + judged scorers
│   ├── strata.py                 # stratified test-set assembly
│   ├── failure_suites/           # fabrication, PII, missing-element
│   ├── judge.py                  # local open-weight judge
│   └── calibrate_judge.py        # judge–human agreement
├── train/
│   └── finetune_qlora.py         # Unsloth QLoRA
├── serve/
│   ├── vllm_server.py
│   ├── cascade.py                # fallback / escalation logic
│   └── api.py                    # FastAPI: POST /draft-note
├── runs/                         # scorecards + run logs
├── Makefile                      # synth, eval, train, shrink, serve
├── pyproject.toml
└── README.md
```

---

## 9. Risks & guardrails

- **Fabrication is the cardinal risk** — case notes feed billing and audits. The faithfulness
  gate (100% on the adversarial suite) is non-negotiable and blocks release on any failure.
- **Privacy** — no real participant data in repo, in cloud calls, or seen by any frontier API.
  De-id pipeline + local judge + on-prem serving enforce this.
- **License cleanliness** — training data lineage must contain no frontier-model outputs. Document
  the teacher used for every synthetic batch.
- **Over-shrinking** — do not chase the smallest model past the eval bar; keep one size of headroom.
- **Human-in-the-loop** — the system drafts; a person reviews and submits. No autonomous submission.

---

## 10. First instruction to Claude Code

Start with **Phase 0 and Phase 1 only.** Build the scaffold, the `CaseNote` schema, the synthetic
data generator, and the full evaluation harness with failure suites. Do **not** fine-tune yet —
produce a working eval that scores a dummy model end-to-end, then stop and report the scorecard
format for review.
