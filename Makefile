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

	
# ============================================================
# Makefile additions for Docker
# ============================================================

# Add these to your existing Makefile:

.PHONY: docker-build docker-up docker-down docker-logs docker-shell

# Docker commands
docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

docker-shell:
	docker-compose exec timescaledb psql -U $(DB_USER) -d $(DB_NAME)

docker-collector-logs:
	docker-compose logs -f collector

docker-restart-collector:
	docker-compose restart collector

# Build specific images
docker-build-collector:
	docker build --target collector -t velib-collector .

docker-build-training:
	docker build --target training -t velib-training .

docker-build-dev:
	docker build --target development -t velib-dev .

# Development workflow
docker-dev:
	docker-compose up -d timescaledb mlflow jupyter
	@echo "Services started:"
	@echo "  - Database: postgresql://localhost:5432/velib"
	@echo "  - MLflow: http://localhost:5000"
	@echo "  - Jupyter: http://localhost:8888"

# Backup database
docker-backup:
	@mkdir -p backups
	docker-compose exec -T timescaledb pg_dump -U $(DB_USER) $(DB_NAME) > backups/velib_$(shell date +%Y%m%d_%H%M%S).sql
	@echo "Backup saved to backups/"

# Restore database
docker-restore:
	@echo "Restoring from $(BACKUP_FILE)..."
	cat $(BACKUP_FILE) | docker-compose exec -T timescaledb psql -U $(DB_USER) $(DB_NAME)

