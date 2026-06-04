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

synth: ## Generate stratified synthetic dataset via the Ollama teacher (default ~20 records)
	cd $(CURDIR) && uv run python -m ndis.synth_generate --count 20

synth-offline: ## Generate dataset with the deterministic no-LLM generator (no Ollama needed)
	cd $(CURDIR) && uv run python -m ndis.synth_generate --count 60 --offline

synth-large: ## Generate a larger dataset via the teacher (~340 records)
	cd $(CURDIR) && uv run python -m ndis.synth_generate --count 340

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
# Training (Phase 3+) — placeholder until fine-tuning is needed
# ---------------------------------------------------------------------------

train: ## Start QLoRA fine-tune training
	@echo "Training not yet implemented. Edit config/model.yaml first."
	@echo "Run 'make train-real' when ready."

train-real: ## Actually run the fine-tune (UnSloth QLoRA on RTX 5090)
	cd $(CURDIR) && python -m train.finetune_qlora

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
