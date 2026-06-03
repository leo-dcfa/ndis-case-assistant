# NDIS Case Note Assistant

> Owned small-model fine-tune for structured NDIS case note generation.  
> Builds → evaluates → fine-tunes → serves a locally-runnable model that turns rough worker input into compliance-ready NDIS case notes — **without inventing facts**.

---

## Status: Phase 0 + Phase 1 complete (scaffold + evaluation harness)

| Phase | Description | Status |
|-------|-------------|--------|
| 0 — Scaffold | Repo structure, schema, config, synthetic data generator | ✅ Done |
| 1 — Evaluation harness | Rubric scorer, Pydantic Evals metrics, failure suites | ✅ Done |
| 2 — Baseline | Prompted Qwen3-8B measurement | ⏳ Next |
| 3 — Fine-tune v1 | QLoRA fine-tune on training set | ⏳ Awaiting Phase 2 |
| 4 — Shrink protocol | Retrain at 4B, then 1.5B | ⏳ |
| 5 — Serving + cascade | vLLM endpoint with fallback | ⏳ |
| 6 — Integration surface | FastAPI `POST /draft-note` | ⏳ |
| 7 — Monitoring + retrain loop | Drift monitoring, runbook | ⏳ |

---

## Quick start

```bash
# 1. Set up the environment (requires uv)
make dev-init

# 2. Generate synthetic training/eval data (Phase 0 DoD)
make synth

# 3. Run the evaluation harness on a dummy model (Phase 1 DoD)
make eval

# 4. Generate HTML report
make eval-html
```

---

## Project structure

```
ndis-case-assistant/
├── config/
│   ├── required_fields.yaml    ← Editable NDIS compliance rule set (§5.1)
│   └── model.yaml              ← Base model, LoRA, training hyperparams
├── data/
│   ├── __init__.py
│   ├── schema.py               ← CaseNote Pydantic model (§5.1)
│   ├── deidentify.py           ← PII redaction pipeline (§5.3)
│   ├── synth_generate.py       ← Synthetic seed data generator (§5.2)
│   └── splits/                 ← train / val / test (test frozen)
├── eval/
│   ├── __init__.py
│   ├── eval_suite.py          ← Pydantic Evals evaluation harness (§6.1)
│   ├── models.py               ← Pydantic output schemas (CaseNoteOutputSchema, FaithfulnessCheck, PIIComplianceCheck)
│   ├── strata.py               ← Stratified test-set loader (§6.2)
│   └── failure_suites/         ← Adversarial failure suites (§6.3)
│       ├── fabrication.py      ← Fabrication suite (hard gate)
│       ├── pii_leakage.py      ← PII leakage suite (hard gate)
│       └── missing_element.py  ← Missing element suite (hard gate)
├── train/
│   ├── __init__.py
│   └── finetune_qlora.py       ← Unsloth QLoRA fine-tuning (§7 Phase 3)
├── serve/
│   ├── __init__.py
│   ├── vllm_server.py          ← vLLM endpoint (§5)
│   ├── cascade.py              ← Fallback / escalation logic (§7 Phase 5)
│   └── api.py                  ← FastAPI: POST /draft-note (§7 Phase 6)
├── runs/                       ← Scorecards + run logs (JSON)
├── Makefile                    ← synth, eval, train, serve commands
├── pyproject.toml              ← Dependencies & tool config
└── README.md                   ← This file
```

---

## Success criteria (PRD §2)

| Metric | Target | Hard gate? |
|--------|--------|-----------|
| Structural compliance | ≥ 98% | No |
| Faithfulness (adversarial suite) | 100% | **Yes** |
| PII handling (privacy suite) | 100% | **Yes** |
| Professional register | ≥ 95% | No |

---

## Key design decisions

1. **Synthetic-first pipeline** — all data is fake until the human owner enables real-data ingestion via `enable_real_data_ingestion()` or `.env` toggle.
2. **Pydantic Evals harness** — deterministic metrics (faithfulness, PII) use Pydantic-validated models; replaceable with LLM judge later.
3. **Hard gates on faithfulness & PII** — zero tolerance for fabrication or PII leakage blocks release, per PRD requirements.
4. **Config-driven compliance** — edit `config/required_fields.yaml` to update rules without code changes.

---

## Environment setup

Requires:
- Python ≥ 3.12 (or uv-managed environment)
- `uv` for dependency management
- RTX 5090 (32GB VRAM) when starting training phase

Install dependencies:
```bash
make dev-init
```

---

## Adding real data later

Real data ingestion is **disabled by default**. To enable:

1. Ensure data agreements are in place
2. Set `REAL_DATA_INGESTION_ENABLED=true` in `.env`, or call:
   ```python
   from data.deidentify import enable_real_data_ingestion
   enable_real_data_ingestion(True)
   ```
3. Run the de-identification pipeline before any data enters training/eval

---

## License

MIT — see LICENSE file. Training lineage is license-clean (no frontier-API outputs).
