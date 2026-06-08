# NDIS Case Note Assistant

> Owned small-model fine-tune for structured NDIS case note generation.
> Builds → evaluates → fine-tunes → serves a locally-runnable model that turns rough worker
> input into compliance-ready NDIS case notes — **without inventing facts**.

See [`docs/prd.md`](docs/prd.md) for the full product spec.

---

## Status

Eval-driven build: the harness and baseline come *before* any fine-tuning. See
[`docs/PLAN.md`](docs/PLAN.md) for the full methodology (filtered distillation, model
choices, the data-generation recipe).

| Phase | Description | Status |
|-------|-------------|--------|
| 0 — Scaffold | Repo, schema, config, synthetic generator (LLM + offline), frozen splits | ✅ Done |
| 1 — Evaluation harness | Automated + judged scorers, strata, 3 failure suites, judge calibration, scorecard | ✅ Done |
| 2 — Baseline | Prompted Qwen3-8B measured + 27B judge calibrated → [`baselines/qwen3-8b.md`](baselines/qwen3-8b.md) (**BLOCKED**: sets the bar) | ✅ Done |
| 2.5 — Data pipeline | Accurate NDIS taxonomy + **compliance-verified** synthetic generation | ✅ Done |
| 3 — Fine-tune | QLoRA on verified data; iterated v1→v3 (see Results) — **structure/fabrication solved; PII gate not yet 100%** | 🔄 In progress |
| 4 — Shrink protocol | Retrain at 4B, then 1.5B (phone-viable) | ⏳ |
| 5 — Serving + cascade | vLLM endpoint + **deterministic PII guardrail** on output | ⏳ |
| 6 — Integration surface | FastAPI `POST /draft-note` | ⏳ |
| 7 — Monitoring + retrain loop | Drift monitoring, runbook | ⏳ |

### Results — the eval-driven loop (Qwen3-8B QLoRA, heuristic-judged, frozen test set)

| Hard gate | Baseline | v1 | v2 | v3 | Target |
|---|---|---|---|---|---|
| Structural compliance | 0% | 100% | 100% | 100% | ≥98% |
| Fabrication suite | 80% | 100% | 100% | 100% | 100% |
| Missing-element suite | — | 60% | 90% | 90% | 100% |
| PII-leakage suite | 60% | 50% | 40% | 70% | 100% |
| **Release** | BLOCKED | BLOCKED | BLOCKED | BLOCKED | — |

Each iteration the eval pinpointed the exact failure and we fixed the **data**, not the model:
v2 rotated *which* mandatory field is omitted (missing-element 60→90%); v3 wove PII into varied
input positions (PII 40→70%). Residual PII leaks are bare names — the planned route to a
guaranteed 100% is a **deterministic output-side PII scrubber** (`ndis/deidentify.py`) layered on
the model (defense-in-depth), plus larger failure suites for statistical confidence. Per-version
scorecards: [`baselines/`](baselines/).

---

## Quick start

```bash
make dev-init          # uv sync

# Phase 0 — synthetic data
make synth-offline     # deterministic, no LLM needed → data/splits/{seed,train,val,test}.jsonl
make synth             # via the local open-weight teacher (Ollama qwen3.6:27b)

# Phase 1 — evaluation harness (the DoD)
make eval              # score the dummy (golden) model end-to-end → scorecard, RELEASE-OK
make eval-naive        # score a deliberately weak baseline → gates FAIL (proves they bite)
make eval-html         # also write runs/scorecard.{json,html}
make calibrate         # judge-vs-human agreement on the calibration fixture
make test              # pytest

# Phase 2 — baseline (prompted, un-tuned)
make baseline          # measure prompted Qwen3-8B → baselines/

# Phase 3 — fine-tune + evaluate (single GPU)
make train-smoke       # validate the pipeline fast on a local 3B (30 steps, no download)
make train             # QLoRA fine-tune Qwen3-8B on train.jsonl → runs/adapters/...
make eval-finetune     # score the adapter on the frozen test set vs the baseline

# Try the model on your own note (loads base + adapter locally)
uv run ndis draft "saw PRT-0042 today, 60 min, cooking at home, went ok" \
    --base-model Qwen/Qwen3-8B --adapter runs/adapters/qwen3-8b-v3

# Presentable demo (curated examples + optional live drafting)
uvx marimo run notebooks/demo.py        # or: uv run --with marimo marimo run notebooks/demo.py
```

`make eval` self-bootstraps an offline dataset if none exists, so it always runs standalone.

---

## How the harness works

A **model-under-test** turns worker input → a drafted note dict
(`src/eval/model_under_test.py`):

- `dummy` (`GoldenModel`) — returns the gold target; a perfect-model self-test that should
  clear every gate. Used for the Phase 1 DoD.
- `naive` (`NaiveModel`) — a weak no-LLM baseline that fails on purpose (leaks PII, fabricates
  a duration, drops fields) so the gates are demonstrably effective.
- `openai` (`OpenAIModel`) — drafts via a local OpenAI-compatible endpoint (Ollama / vLLM).
  This is the Phase 2 prompted baseline and, later, the fine-tuned model. **No frontier APIs.**

Each example is scored on:

| Dimension | How | Source |
|-----------|-----|--------|
| Structural compliance | deterministic | `eval/rubric.py` — required fields present, ordered, typed; gap-flag aware |
| PII clean | deterministic | `eval/rubric.py` — phone/email/NDIS regex + per-case forbidden strings |
| Faithfulness | judged | `eval/judge.py` — grounds every number/date/ID in the input |
| Professional register | judged | `eval/judge.py` — third-person, no fillers, well-formed |
| Billable evidence | judged | `eval/judge.py` |

The judge is pluggable: `--judge heuristic` (deterministic, offline, used in CI) or
`--judge llm` (a **local open-weight** model — never a frontier API; point it at a remote
endpoint such as a MacBook with `--judge-url`). Calibrate the LLM judge against human labels
(`make calibrate`) before trusting it for gating.

The model-under-test is also pluggable: `--mode dummy|naive|openai|hf`. `--mode hf` loads a
fine-tuned base + LoRA adapter **in-process** (no server) so an adapter can be scored straight
after training; the same adapter is also what `ndis draft` uses.

**Adversarial failure suites** (`src/eval/failure_suites/`) are committed fixtures, each a hard
gate that must score **100%**: `fabrication`, `pii_leakage`, `missing_element`.

The **scorecard** (`eval/scorecard.py`) separates *targets* (structural ≥ 98%, register ≥ 95%)
from *hard gates* (the three suites = 100%). `release_ok` is true only when every hard gate passes.

---

## Project structure

```
ndis-case-assistant/
├── config/
│   ├── required_fields.yaml    ← editable NDIS compliance rule set (drives validation)
│   └── model.yaml              ← base model, LoRA, training hyperparams
├── src/
│   ├── cli.py                  ← typer CLI (synth, deid, draft)
│   ├── ndis/
│   │   ├── models.py           ← CaseNote schema + config dataclasses + load_config
│   │   ├── config_loader.py    ← cached config accessor
│   │   ├── notes.py            ← gap markers + lenient draft parsing (shared)
│   │   ├── prompts.py          ← shared system/user/assistant chat contract (train == serve)
│   │   ├── synth_generate.py   ← synthetic generator: LLM teacher + offline template + CLI
│   │   ├── splits.py           ← stratified train/val/test (test frozen)
│   │   └── deidentify.py       ← PII redaction (real-data pipeline + planned output guardrail)
│   ├── train/
│   │   ├── format_dataset.py   ← {input,target} → chat 'messages' dataset
│   │   └── finetune_qlora.py   ← TRL SFTTrainer + peft 4-bit QLoRA (reads config/model.yaml)
│   └── eval/
│       ├── eval_suite.py       ← harness entry point (python -m eval.eval_suite)
│       ├── model_under_test.py ← dummy / naive / openai / hf (base+LoRA adapter) models
│       ├── rubric.py           ← automated structural + PII scorers
│       ├── judge.py            ← heuristic + local-LLM judges
│       ├── verify.py           ← 4-check compliance verifier for training data
│       ├── strata.py           ← stratified test-set loader
│       ├── scorecard.py        ← aggregate vs §2 bar; console/JSON/HTML
│       ├── calibrate_judge.py  ← judge–human agreement
│       ├── calibration/        ← human_labels.jsonl fixture
│       ├── models.py           ← pydantic judge contracts + result dataclasses
│       └── failure_suites/     ← fabrication, pii_leakage, missing_element (hard gates)
├── notebooks/demo.py           ← marimo demo (curated examples + live drafting)
├── baselines/                  ← committed per-version scorecards (baseline, v1, v2, v3)
├── data/splits/                ← generated datasets (gitignored)
├── runs/                       ← scorecards, adapters, run logs (gitignored)
├── tests/                      ← pytest
├── Makefile · pyproject.toml · README.md
```

`serve/` (vLLM endpoint + cascade, Phase 5) is the main piece not built yet.

---

## Success criteria (PRD §2)

| Metric | Target | Hard gate? |
|--------|--------|-----------|
| Structural compliance | ≥ 98% | No |
| Faithfulness (adversarial suite) | 100% | **Yes** |
| PII handling (privacy suite) | 100% | **Yes** |
| Missing mandatory element (suite) | 100% | **Yes** |
| Professional register | ≥ 95% | No |

---

## Key design decisions

1. **License-clean teacher** — synthetic data comes from a local open-weight model (Ollama
   Qwen3.6) or the deterministic offline generator. **No frontier-API outputs** ever enter the
   training/eval lineage. Each batch records its teacher in `meta`.
2. **Filtered (rejection-sampling) distillation** — the teacher *proposes* pairs; a local
   **4-check verifier** (`eval/verify.py`) keeps only those that are faithful + schema-valid +
   PII-clean + correctly structured, so every training example is a fully-compliant exemplar and
   the teacher's embellishments never become training signal. Enable with
   `--filter` (see `make synth` / `make synth-fast`). Methodology: [`docs/PLAN.md`](docs/PLAN.md).
2. **Offline-by-default eval** — the harness runs with zero external dependencies (deterministic
   generator + heuristic judge), so CI and the Phase 1 DoD are reproducible. The LLM teacher/judge
   are opt-in for higher fidelity.
3. **Hard gates block release** — fabrication and PII leakage are non-negotiable; `release_ok`
   gates on them.
4. **Config-driven compliance** — edit `config/required_fields.yaml` to change the rule set
   without code changes.
5. **Gap markers, not crashes** — drafts may flag a genuinely-missing element (`[not recorded]`)
   rather than invent one; the scorers treat that as correct in sparse/adversarial strata and as
   a structural failure elsewhere.

---

## How much test data?

Test-set sizing is about confidence in a pass-rate, not training volume — two regimes:

- **Stratified quality set** (structural ≥98%, register ≥95%): margin of error ≈ 1/√n.
  ~150–300 total (PRD §6.2) gives ±~10% per stratum — fine for iteration. A few hundred
  *per stratum* (low thousands total) to publish tight numbers.
- **Hard-gate failure suites** (faithfulness/PII = 100%): with 0 failures, the true rate
  could still be ≈ 3/n (rule of three). 20 cases → ≤~15%; 100 → ≤~3%; 300 → ≤~1%;
  3,000 → ≤~0.1%. This is where volume buys assurance — but **diversity beats count**
  (50 distinct fabrication temptations >> 1,000 near-duplicates).

Two caveats: synthetic data measures fit to the *generator's* distribution (a few hundred
synthetic + real de-identified notes beats thousands of synthetic); and every LLM-judged
example costs ~10–15s on the 27B, so use the heuristic judge for big sweeps and the LLM
judge on a calibrated subsample. The committed baseline uses n=13 (illustrative) — grow it
before trusting the per-stratum numbers.

## Real data (later, human-gated)

Real participant data never enters the repo or any cloud call. The de-id pipeline
(`src/ndis/deidentify.py`) exists, but ingesting real notes is a manual step the owner runs only
once data agreements are in place (PRD §5.3). Use the CLI to sanity-check redaction:

```bash
uv run ndis deid "call mum Jenny on 0412 345 678"
```

---

## License

MIT. Training lineage is license-clean (no frontier-API outputs).
