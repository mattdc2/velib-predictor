

# ============================================================
# scripts/docker-entrypoint.sh
# Entrypoint script for Docker containers
# ============================================================

#!/bin/bash
set -e

# Wait for database to be ready
wait_for_db() {
    echo "Waiting for database at $DB_HOST:$DB_PORT..."
    
    until pg_isready -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" > /dev/null 2>&1; do
        echo "Database is unavailable - sleeping"
        sleep 2
    done
    
    echo "Database is ready!"
}

# Initialize database if needed
init_db() {
    echo "Checking database schema..."
    
    python -c "
from src.data.database import DatabaseManager
db = DatabaseManager()
if not db.table_exists('station_information'):
    print('Database not initialized! Please run init_db.sql first.')
    exit(1)
db.close()
print('Database schema OK')
"
}

# Run database migrations (placeholder for future)
run_migrations() {
    echo "Running database migrations..."
    # Add migration tool here (e.g., Alembic)
}

# Main entrypoint logic
case "${1:-}" in
    collector)
        wait_for_db
        init_db
        echo "Starting data collector..."
        exec python -m src.data.collector
        ;;
    
    train)
        wait_for_db
        init_db
        echo "Starting model training..."
        exec python -m src.models."${2:-baseline}".trainer
        ;;
    
    api)
        wait_for_db
        echo "Starting API server..."
        exec uvicorn src.api.main:app --host 0.0.0.0 --port 8000
        ;;
    
    notebook)
        wait_for_db
        echo "Starting Jupyter Lab..."
        exec jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
        ;;
    
    bash)
        exec /bin/bash
        ;;
    
    *)
        # If no recognized command, execute whatever was passed
        exec "$@"
        ;;
esac
