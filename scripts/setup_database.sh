# ----------------------------------------------------------
# 1. setup_database.sh
#    Initialize PostgreSQL database with TimescaleDB
# ----------------------------------------------------------

setup_database() {
    echo "========================================="
    echo "Setting up Velib Predictor Database"
    echo "========================================="
    
    # Load environment variables
    if [ -f .env ]; then
        echo "Found .env"
        export $(cat ../.env | grep -v '^#' | xargs)
        sleep 2
    else
        echo "Error: .env file not found!"
        exit 1
    fi
    
    # Check if Docker is running (for docker-compose setup)
    if command -v docker-compose &> /dev/null; then
        echo "Starting PostgreSQL + TimescaleDB with Docker..."
        docker-compose up -d timescaledb
        
        # Wait for database to be ready
        echo "Waiting for database to be ready..."
        sleep 10
        
        # Check connection
        until docker-compose exec -T timescaledb pg_isready -U $DB_USER -d $DB_NAME; do
            echo "Waiting for database..."
            sleep 2
        done
    fi
    
    echo "Database is ready!"
    
    # Run schema initialization
    echo "Creating database schema..."
    PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME \
        -f scripts/init_db.sql
    
    if [ $? -eq 0 ]; then
        echo "✓ Database schema created successfully!"
    else
        echo "✗ Failed to create database schema"
        exit 1
    fi
    
    # Verify installation
    echo "Verifying database setup..."
    python -c "
from src.data.database import DatabaseManager
db = DatabaseManager()
tables = ['station_information', 'station_status', 'predictions', 'model_runs']
for table in tables:
    exists = db.table_exists(table)
    print(f'  Table {table}: {'✓' if exists else '✗'}')
db.close()
"
    
    echo "========================================="
    echo "Database setup complete!"
    echo "========================================="
}