# Velib Predictor - Repository Architecture

```
velib-predictor/
│
├── README.md                          # Project overview, setup instructions
├── LICENSE                            # MIT or Apache 2.0
├── .gitignore                         # Python, data, models, secrets
├── .env.example                       # Environment variables template
├── requirements.txt                   # Core dependencies
├── requirements-dev.txt               # Development dependencies (pytest, black, etc.)
├── setup.py                           # Make package installable
├── pyproject.toml                     # Modern Python project config (black, isort, etc.)
├── Makefile                           # Common commands (setup, test, train, etc.)
│
├── docker-compose.yml                 # PostgreSQL + TimescaleDB
├── Dockerfile                         # For deployment (optional)
│
├── config/                            # Configuration files
│   ├── config.yaml                    # Main configuration
│   ├── model_config.yaml              # Model hyperparameters
│   └── logging_config.yaml            # Logging configuration
│
├── data/                              # Data directory (gitignored except .gitkeep)
│   ├── .gitkeep
│   ├── raw/                           # Raw data from API (if stored locally)
│   ├── processed/                     # Processed features
│   ├── external/                      # External data (holidays, weather, etc.)
│   └── predictions/                   # Model predictions output
│
├── models/                            # Saved models (gitignored)
│   ├── .gitkeep
│   ├── baseline/                      # Baseline models
│   ├── lstm/                          # LSTM models
│   └── gnn/                           # GNN models
│
├── mlruns/                            # MLflow tracking (gitignored)
│   └── .gitkeep
│
├── notebooks/                         # Jupyter notebooks for exploration
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_baseline_models.ipynb
│   ├── 04_lstm_experiments.ipynb
│   └── 05_gnn_experiments.ipynb
│
├── scripts/                           # Standalone scripts
│   ├── setup_database.sh              # Initialize PostgreSQL/TimescaleDB
│   ├── collect_data.sh                # Wrapper for cron job
│   ├── train_baseline.sh              # Train baseline models
│   ├── train_lstm.sh                  # Train LSTM model
│   ├── train_gnn.sh                   # Train GNN model
│   ├── run_inference.sh               # Run inference pipeline
│   └── evaluate_model.sh              # Evaluate model performance
│
├── src/                               # Main source code
│   ├── __init__.py
│   │
│   ├── config/                        # Configuration management
│   │   ├── __init__.py
│   │   └── config.py                  # Load and validate config
│   │
│   ├── data/                          # Data collection and processing
│   │   ├── __init__.py
│   │   ├── collector.py               # Fetch data from Velib API
│   │   ├── database.py                # Database connection and operations
│   │   ├── schema.py                  # SQLAlchemy models/schemas
│   │   ├── preprocessor.py            # Data preprocessing
│   │   └── feature_engineering.py     # Feature creation
│   │
│   ├── models/                        # ML models
│   │   ├── __init__.py
│   │   ├── base.py                    # Base model interface
│   │   ├── baseline/
│   │   │   ├── __init__.py
│   │   │   ├── persistence.py         # Simple persistence model
│   │   │   └── linear.py              # Linear/XGBoost baseline
│   │   ├── lstm/
│   │   │   ├── __init__.py
│   │   │   ├── model.py               # LSTM architecture
│   │   │   ├── dataset.py             # PyTorch Dataset
│   │   │   └── trainer.py             # Training logic
│   │   └── gnn/
│   │       ├── __init__.py
│   │       ├── model.py               # GNN architecture
│   │       ├── graph_builder.py       # Build graph structure
│   │       ├── dataset.py             # PyTorch Geometric Dataset
│   │       └── trainer.py             # Training logic
│   │
│   ├── training/                      # Training utilities
│   │   ├── __init__.py
│   │   ├── trainer.py                 # Generic trainer class
│   │   ├── callbacks.py               # Training callbacks
│   │   └── metrics.py                 # Evaluation metrics
│   │
│   ├── inference/                     # Inference pipeline
│   │   ├── __init__.py
│   │   ├── predictor.py               # Generate predictions
│   │   └── postprocessor.py           # Post-process predictions
│   │
│   ├── evaluation/                    # Model evaluation
│   │   ├── __init__.py
│   │   ├── evaluator.py               # Evaluate model performance
│   │   └── visualizer.py              # Visualization utilities
│   │
│   └── utils/                         # Utility functions
│       ├── __init__.py
│       ├── logger.py                  # Logging setup
│       ├── spatial.py                 # Spatial calculations (distance, k-NN)
│       ├── temporal.py                # Temporal feature utilities
│       └── mlflow_utils.py            # MLflow helper functions
│
├── tests/                             # Unit and integration tests
│   ├── __init__.py
│   ├── conftest.py                    # Pytest fixtures
│   ├── test_data/
│   │   ├── test_collector.py
│   │   ├── test_database.py
│   │   └── test_feature_engineering.py
│   ├── test_models/
│   │   ├── test_baseline.py
│   │   ├── test_lstm.py
│   │   └── test_gnn.py
│   └── test_utils/
│       ├── test_spatial.py
│       └── test_temporal.py
│
├── docs/                              # Documentation
│   ├── architecture.md                # System architecture
│   ├── data_schema.md                 # Database schema documentation
│   ├── model_details.md               # Model architecture details
│   ├── api.md                         # API documentation (if applicable)
│   └── deployment.md                  # Deployment guide
│
└── .github/                           # GitHub specific files
    ├── workflows/
    │   ├── ci.yml                     # CI pipeline (tests, linting)
    │   └── deploy.yml                 # Deployment workflow
    └── ISSUE_TEMPLATE/
        └── bug_report.md
```

## Key Design Decisions

### 1. **Separation of Concerns**
- `data/`: Everything related to data collection and processing
- `models/`: Model architectures only
- `training/`: Training logic separate from model definitions
- `inference/`: Prediction pipeline
- `evaluation/`: Model evaluation and metrics

### 2. **Configuration Management**
- YAML configs for easy experimentation
- Environment variables for secrets (API keys, DB credentials)
- `config.py` centralizes all configuration loading

### 3. **Model Organization**
- Each model type (baseline/lstm/gnn) in its own module
- Consistent interface via `base.py`
- Makes it easy to swap models

### 4. **Testability**
- Clear test structure mirroring `src/`
- Fixtures in `conftest.py` for reusability
- Integration tests for end-to-end workflows

### 5. **Reproducibility**
- MLflow for experiment tracking
- Version control for configs and code
- Docker for consistent environments

### 6. **Scripts for Common Tasks**
- Shell scripts for cron jobs and orchestration
- `Makefile` for developer convenience
- Clear separation between library code and executables



# ============================================================
# README section for Docker setup
# ============================================================

## Docker Setup

### Quick Start

```bash
# 1. Create .env file
cp .env.example .env
# Edit .env with your passwords

# 2. Start all services
docker-compose up -d

# 3. Check services are running
docker-compose ps

# 4. View logs
docker-compose logs -f
```

### Services

The docker-compose setup includes:

1. **TimescaleDB** (localhost:5432)
   - PostgreSQL with TimescaleDB extension
   - Persistent data storage
   - Automatic initialization

2. **MLflow** (localhost:5000)
   - Experiment tracking
   - Model registry
   - Artifact storage

3. **Collector** (optional)
   - Automatic data collection every 15 min
   - Can be disabled to run on host

4. **Jupyter** (localhost:8888)
   - Development environment
   - Pre-configured with project dependencies

### Development Workflow

```bash
# Start development environment
make docker-dev

# Access Jupyter
# Open http://localhost:8888 in browser

# Access database
make docker-shell

# View collector logs
docker-compose logs -f collector

# Restart collector
docker-compose restart collector
```

### Production Deployment

```bash
# Build optimized images
docker build --target collector -t velib-collector:v1.0 .
docker build --target inference -t velib-api:v1.0 .

# Run in production
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### Backup & Restore

```bash
# Backup database
make docker-backup

# Restore database
make docker-restore BACKUP_FILE=backups/velib_20241222_120000.sql
```

### Troubleshooting

```bash
# Check service health
docker-compose ps

# View service logs
docker-compose logs timescaledb
docker-compose logs collector

# Restart services
docker-compose restart

# Clean restart (WARNING: deletes data)
docker-compose down -v
docker-compose up -d
```

### Resource Configuration

Edit `docker-compose.yml` to adjust resource limits:

```yaml
services:
  timescaledb:
    deploy:
      resources:
        limits:
          cpus: '2'      # Adjust based on your system
          memory: 2G
```

### Environment Variables

Required in `.env`:

```bash
# Database
DB_NAME=velib
DB_USER=velib_user
DB_PASSWORD=your_secure_password  # REQUIRED

# Optional
DB_PORT=5432
LOG_LEVEL=INFO
```