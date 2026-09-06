# Convenience targets. Run `make help` for the list.
.DEFAULT_GOAL := help
SHELL := /bin/bash

PYTHON      ?= .venv/bin/python
PIP         ?= .venv/bin/pip
IMAGE       ?= llmops-rag-api
REGION      ?= us-central1
SERVICE     ?= llmops-rag-api

.PHONY: help venv install test lint fmt run docker-build docker-run clean deploy destroy cost-check eval sweep prompts-lock drift

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

venv: ## Create the local virtualenv (Python 3.12, matching the container)
	uv venv --python 3.12 .venv || python3 -m venv .venv

install: ## Install dev + runtime dependencies
	$(PIP) install -r requirements-dev.txt

test: ## Run the test suite (no network, no spend)
	$(PYTHON) -m pytest -v

eval: ## Run the RAG eval against real OpenAI (SPENDS ~$0.002)
	# The free, offline version of this runs in `make test` as the CI gate.
	# This one measures what the real model actually does.
	$(PYTHON) scripts/run_eval.py

sweep: ## Sweep retrieval parameters and compare in MLflow (~$0.002)
	# Retrieval only -- no generation -- so a whole grid costs a fraction of
	# a cent. Confirm any winner with `make eval` before adopting it.
	$(PYTHON) scripts/run_sweep.py

prompts-lock: ## Re-pin the prompt lock after editing a template (VERSION=v2)
	# Free. Bump VERSION whenever the text changes -- that string is how an
	# MLflow run months from now says which words produced it.
	$(PYTHON) scripts/lock_prompts.py --version $(VERSION)

drift: ## Compare recent production traffic against the eval baseline (free)
	# Reads MLflow only -- no model calls, no spend. Needs a baseline from
	# `make eval` and some recorded /query traffic to compare against.
	$(PYTHON) scripts/check_drift.py

lint: ## Lint and check formatting
	.venv/bin/ruff check src tests
	.venv/bin/ruff format --check src tests

fmt: ## Auto-format
	.venv/bin/ruff check --fix src tests
	.venv/bin/ruff format src tests

run: ## Run the API locally on :8080
	$(PYTHON) -m uvicorn app.main:app --reload --port 8080 --app-dir src

docker-build: ## Build the container image (linux/amd64, as Cloud Run requires)
	# Cloud Run runs x86_64 only. On an Apple Silicon Mac a plain `docker build`
	# produces an arm64 image that fails on Cloud Run with an exec format error,
	# so the platform is pinned explicitly here.
	docker build --platform linux/amd64 -t $(IMAGE):local .

docker-build-native: ## Build for the local architecture (fast, for local testing only)
	docker build -t $(IMAGE):native .

docker-run: ## Run the container locally (reads .env)
	docker run --rm -p 8080:8080 --env-file .env -e CHROMA_DIR=/tmp/chroma $(IMAGE):local

deploy: ## Provision infra with Terraform (see terraform/README)
	cd terraform && terraform apply

destroy: ## TEAR DOWN every billable resource
	./scripts/teardown.sh

cost-check: ## Show what is currently running and could be costing money
	./scripts/cost_check.sh

clean: ## Remove local state and caches
	rm -rf data/ mlruns/ .pytest_cache/ .ruff_cache/ **/__pycache__/
