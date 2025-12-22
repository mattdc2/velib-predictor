.PHONY: help setup install install-dev test lint format clean

help:
	@echo "Available commands:"
	@echo "  make setup          - Setup UV and install dependencies"
	@echo "  make install        - Install production dependencies"
	@echo "  make install-dev    - Install with dev dependencies"
	@echo "  make test           - Run tests"
	@echo "  make lint           - Run linters"
	@echo "  make format         - Format code"
	@echo "  make clean          - Clean generated files"
	@echo "  make db-setup       - Setup database"
	@echo "  make train-baseline - Train baseline model"
	@echo "  make train-lstm     - Train LSTM model"
	@echo "  make train-gnn      - Train GNN model"

setup:
	@echo "Installing UV..."
	@curl -LsSf https://astral.sh/uv/install.sh | sh
	@echo "Setting up project..."
	$(MAKE) install-dev
	$(MAKE) db-setup

install:
	uv sync

install-dev:
	uv sync --extra dev

test:
	uv run pytest tests/ -v --cov=src

lint:
	uv run ruff check src/ tests/
	uv run mypy src/

format:
	uv run black src/ tests/
	uv run isort src/ tests/
	uv run ruff check --fix src/ tests/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .coverage htmlcov .mypy_cache .ruff_cache

db-setup:
	bash scripts/setup_database.sh

train-baseline:
	uv run python -m src.models.baseline.train

train-lstm:
	uv run python -m src.models.lstm.trainer

train-gnn:
	uv run python -m src.models.gnn.trainer

collect-data:
	uv run python -m src.data.collector

run-inference:
	uv run python -m src.inference.predictor