.PHONY: synth eval train serve dev-init clean help

# ---------------------------------------------------------------------------
# Help / project info
# ---------------------------------------------------------------------------

help: ## Show this help message
	@echo "NDIS Case Note Assistant — Makefile targets"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# Development setup
# ---------------------------------------------------------------------------

dev-init: ## Install dependencies with uv
	uv sync --all-extras

# ---------------------------------------------------------------------------
# Phase 0: Synthetic data generation (make synth)
# ---------------------------------------------------------------------------

synth-offline: ## Fastest: deterministic no-LLM generator (no Ollama needed)
	cd $(CURDIR) && uv run python -m ndis.synth_generate --count 60 --offline

synth-fast: ## Quick teacher run: 8B teacher + heuristic filter (~5s/record, catches number/ID fab)
	cd $(CURDIR) && uv run python -m ndis.synth_generate --count 20 \
		--teacher-model qwen3:8b --filter --filter-judge heuristic

synth: ## Quality teacher run: 27B teacher + LLM faithfulness filter (SLOW ~1 min/record)
	cd $(CURDIR) && uv run python -m ndis.synth_generate --count 20 --filter --filter-judge llm

synth-large: ## Large quality dataset (27B teacher + LLM filter) — run as an overnight batch
	cd $(CURDIR) && uv run python -m ndis.synth_generate --count 1000 --filter --filter-judge llm

# ---------------------------------------------------------------------------
# Phase 1: Evaluation harness
# ---------------------------------------------------------------------------

eval: synth-offline ## Run eval harness end-to-end on the dummy (golden) model
	cd $(CURDIR) && uv run python -m eval.eval_suite --mode dummy --judge heuristic

eval-naive: synth-offline ## Run eval on a deliberately weak baseline (exercises the gates)
	cd $(CURDIR) && uv run python -m eval.eval_suite --mode naive --judge heuristic

eval-html: synth-offline ## Same as eval but also write an HTML scorecard
	cd $(CURDIR) && uv run python -m eval.eval_suite --mode dummy --judge heuristic \
		--out runs/scorecard.json --html runs/scorecard.html

calibrate: ## Report judge-vs-human agreement on the calibration fixture
	cd $(CURDIR) && uv run python -m eval.calibrate_judge

test: ## Run the pytest suite
	cd $(CURDIR) && uv run pytest -q

# ---------------------------------------------------------------------------
# Phase 2: Baseline — prompted Qwen3-8B (no fine-tune), committed scorecard
# ---------------------------------------------------------------------------

baseline: ## Measure prompted Qwen3-8B on the test split (needs Ollama; writes baselines/)
	cd $(CURDIR) && uv run python -m eval.eval_suite --mode openai --judge heuristic \
		--split test --out baselines/qwen3-8b.json --html baselines/qwen3-8b.html

# ---------------------------------------------------------------------------
# Phase 3: QLoRA fine-tune (TRL + peft, 4-bit, single GPU)
# ---------------------------------------------------------------------------

BASE_MODEL ?= Qwen/Qwen3-8B
ADAPTER    ?= runs/adapters/qwen3-8b-v1

train-smoke: ## Validate the pipeline fast on a local model (30 steps, no download)
	cd $(CURDIR) && uv run python -m train.finetune_qlora \
		--base-model Qwen/Qwen2.5-3B-Instruct --max-steps 30 --out runs/adapters/smoke

train: ## QLoRA fine-tune the base model on train.jsonl -> $(ADAPTER)
	cd $(CURDIR) && uv run python -m train.finetune_qlora \
		--base-model $(BASE_MODEL) --out $(ADAPTER)

eval-finetune: ## Eval the fine-tuned adapter on the frozen test set (LLM judge)
	cd $(CURDIR) && uv run python -m eval.eval_suite --mode hf \
		--base-model $(BASE_MODEL) --adapter $(ADAPTER) --judge llm \
		--out runs/scorecard-v1.json --html runs/scorecard-v1.html

# ---------------------------------------------------------------------------
# Serving (Phase 5+) — placeholder until model is trained
# ---------------------------------------------------------------------------

serve: ## Start the vLLM serving endpoint
	@echo "Serving not yet available — no trained model."
	@echo "Run 'serve' after Phase 4 completion."

api-test: ## Test the POST /draft-note API endpoint (requires server running)
	curl -s http://localhost:8000/draft-note \
		-H "Content-Type: application/json" \
		-d '{"input_text": "Worker-A saw P-1001. Daily living support. 90min."}'

# ---------------------------------------------------------------------------
# Utility targets
# ---------------------------------------------------------------------------

clean: ## Remove generated files (keep source)
	rm -rf data/splits/seed.jsonl data/splits/train.jsonl data/splits/val.jsonl data/splits/test.jsonl runs/*.json
	@echo "Cleaned generated data and run artifacts."

check: ## Run linting
	@echo "Running ruff linter..."
	uv run ruff check src/
	@echo "Done."
