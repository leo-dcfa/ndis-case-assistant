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

synth: ## Generate stratified synthetic dataset (default 20 per stratum = 100 total)
	cd $(CURDIR) && python -m data.synth_generate --count 20

synth-large: ## Generate larger synthetic dataset (500 per stratum = 2500 total)
	cd $(CURDIR) && python -m data.synth_generate --count 500

# ---------------------------------------------------------------------------
# Phase 1: Evaluation harness (Pydantic Evals)
# ---------------------------------------------------------------------------

eval: synth ## Run eval harness on dummy model (golden reference)
	cd $(CURDIR) && python -m eval.eval_suite --mode dummy

eval-html: synth ## Same but output HTML report
	cd $(CURDIR) && python -m eval.eval_suite --mode dummy --output runs/eval.html --format html

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

check: ## Run linting and type checking
	@echo "Running ruff linter..."
	uv run ruff check data/ eval/ train/ serve/
	@echo "Done."
