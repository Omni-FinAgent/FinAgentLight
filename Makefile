.PHONY: install pre-commit-init lint lint-fix type-check test clean all

# Define paths
CONFIG_DIR=dev_config/python
SRC_DIR=finagent

# Install dependencies
install:
	poetry install

# Initialize pre-commit hooks
pre-commit-init:
	poetry run pre-commit install --config $(CONFIG_DIR)/.pre-commit-config.yaml

# Run pre-commit checks
lint:
	poetry run pre-commit run --all-files --config $(CONFIG_DIR)/.pre-commit-config.yaml

# Auto-fix issues found by pre-commit
lint-fix:
	poetry run pre-commit run --all-files --hook-stage manual --config $(CONFIG_DIR)/.pre-commit-config.yaml

# Run static type checks using mypy
type-check:
	poetry run mypy --config-file $(CONFIG_DIR)/mypy.ini $(SRC_DIR)

# Run tests (assumes pytest is being used)
test:
	poetry run pytest

# Clean temporary files and caches
clean:
	rm -rf __pycache__ .mypy_cache .pytest_cache .ruff_cache

# Run all steps in sequence
all: install pre-commit-init lint type-check test
