# Development tasks. Each target mirrors a CI step, so a green `make check`
# means a green pipeline.
.DEFAULT_GOAL := help
PYTHON ?= python

.PHONY: help install lint format format-check typecheck test test-cov check build clean hooks

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Install the project with dev dependencies and git hooks
	$(PYTHON) -m pip install -e ".[dev]"
	$(PYTHON) -m pre_commit install

hooks: ## Run every pre-commit hook against all files
	$(PYTHON) -m pre_commit run --all-files

lint: ## Lint with ruff
	$(PYTHON) -m ruff check .

format: ## Auto-format and auto-fix
	$(PYTHON) -m ruff check --fix .
	$(PYTHON) -m black .

format-check: ## Verify formatting without writing
	$(PYTHON) -m black --check --diff .

typecheck: ## Type-check with mypy
	$(PYTHON) -m mypy

test: ## Run the test suite
	$(PYTHON) -m pytest

test-network: ## Run the dataset-downloading tests (exp9, exp10)
	$(PYTHON) -m pytest -m network

test-cov: ## Run tests with a coverage report
	$(PYTHON) -m pytest --cov=drift_or_shift --cov=caliblab \
		--cov-report=term-missing --cov-report=html

check: lint format-check typecheck test ## Run everything CI runs

build: ## Build sdist + wheel and validate the metadata
	$(PYTHON) -m build
	$(PYTHON) -m twine check dist/*

clean: ## Remove build and tooling artifacts
	rm -rf build dist htmlcov .coverage coverage.xml
	rm -rf .pytest_cache .mypy_cache .ruff_cache .benchmarks
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	find . -name '*.egg-info' -type d -prune -exec rm -rf {} +
