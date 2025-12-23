# Velib Predictor - Data Collection Setup Guide

## Overview

This guide will help you set up the data collection pipeline for the Velib Predictor project. The pipeline collects station status data every 15 minutes and stores it in a PostgreSQL/TimescaleDB database.

## Prerequisites

- Python 3.11+
- UV package manager installed
- Docker and Docker Compose (recommended) OR local PostgreSQL installation
- Unix-like system with cron (for scheduled data collection)

## Architecture

```
┌─────────────────┐
│  Velib API      │  Every 15 minutes
│  (Public GBFS)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Data Collector │  Python script
│  (collector.py) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  PostgreSQL +   │  Time-series storage
│  TimescaleDB    │  Automatic compression
└─────────────────┘
```

## Step-by-Step Setup

### 1. Environment Configuration

Create a `.env` file in the project root:

```bash
# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=velib
DB_USER=velib_user
DB_PASSWORD=your_secure_password_here

# MLflow (for later)
MLFLOW_TRACKING_URI=http://localhost:5000

# Logging
LOG_LEVEL=INFO
```

### 2. Database Setup

#### Option A: Using Docker (Recommended)

```bash
# Start PostgreSQL + TimescaleDB
docker-compose up -d timescaledb

# Wait for database to be ready
sleep 10

# Initialize schema
bash scripts/setup_database.sh database
```

#### Option B: Local PostgreSQL Installation

If you have PostgreSQL installed locally:

```bash
# Install TimescaleDB extension
# On Ubuntu/Debian:
sudo apt-get install timescaledb-postgresql-14

# On macOS with Homebrew:
brew install timescaledb

# Create database
createdb -U postgres velib
psql -U postgres -d velib -c "CREATE USER velib_user WITH PASSWORD 'your_password';"
psql -U postgres -d velib -c "GRANT ALL PRIVILEGES ON DATABASE velib TO velib_user;"

# Enable TimescaleDB
psql -U postgres -d velib -c "CREATE EXTENSION IF NOT EXISTS timescaledb;"

# Run schema initialization
psql -U velib_user -d velib -f scripts/init_db.sql
```

### 3. Verify Database Setup

```bash
# Check tables were created
uv run python -c "
from src.data.database import DatabaseManager
db = DatabaseManager()
tables = ['station_information', 'station_status', 'predictions', 'model_runs']
for table in tables:
    print(f'{table}: {\"✓\" if db.table_exists(table) else \"✗\"}')
db.close()
"
```

Expected output:
```
station_information: ✓
station_status: ✓
predictions: ✓
model_runs: ✓
```

### 4. Initial Data Load

Load station information and first status snapshot:

```bash
bash scripts/setup_database.sh initial-load
```

This will:
1. Fetch information for ~1504 Velib stations
2. Store station metadata (location, capacity)
3. Collect the first status snapshot

### 5. Setup Automated Data Collection

Configure cron job to run collector every 15 minutes:

```bash
bash scripts/setup_database.sh cron
```

This creates a cron entry:
```cron
*/15 * * * * cd /path/to/velib-predictor && uv run python -m src.data.collector >> logs/collector.log 2>&1
```

### 6. Verify Data Collection

After waiting 15-30 minutes, verify data is being collected:

```bash
bash scripts/setup_database.sh verify
```

Expected output:
```
Total stations: 1504
Records in last hour: 6016
Stations with data: 1504
✓ Data is fresh (12.3 minutes old)
```

## Database Schema Overview

### Tables

1. **station_information** (static)
   - Station metadata: ID, name, location, capacity
   - Updated daily or when stations change

2. **station_status** (time-series)
   - Real-time availability data
   - Collected every 15 minutes
   - Automatically compressed after 7 days
   - Retained for 6 months

3. **predictions** (model outputs)
   - Forecasts from ML models
   - Stored with metadata for evaluation

4. **model_runs** (training logs)
   - Training metadata and metrics

### Views & Aggregates

- **latest_station_status**: Most recent status per station
- **station_status_hourly**: Pre-aggregated hourly statistics
- **stale_stations**: Stations with missing data

## Data Collection Monitoring

### Check Recent Activity

```sql
-- View collection stats
SELECT * FROM collection_stats LIMIT 24;

-- Check for stale stations
SELECT * FROM stale_stations;

-- Latest data per station
SELECT 
    si.name,
    lss.time,
    lss.num_bikes_available,
    lss.num_mechanical,
    lss.num_ebike
FROM latest_station_status lss
JOIN station_information si ON lss.station_id = si.station_id
LIMIT 10;
```

### View Logs

```bash
# Collector logs
tail -f logs/collector.log

# Cron logs (if using system cron)
grep velib /var/log/syslog  # Ubuntu/Debian
grep velib /var/log/cron    # CentOS/RHEL
```

## Troubleshooting

### Issue: No data being collected

```bash
# Test manual collection
uv run python -m src.data.collector

# Check cron is running
crontab -l
ps aux | grep cron

# Check database connection
uv run python -c "
from src.data.database import DatabaseManager
db = DatabaseManager()
print('Connection successful!')
db.close()
"
```

### Issue: Database connection failed

- Verify credentials in `.env`
- Check PostgreSQL is running: `pg_isready -h localhost`
- Check firewall rules allow port 5432

### Issue: TimescaleDB extension not found

```sql
-- Install extension manually
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Verify installation
SELECT * FROM pg_extension WHERE extname = 'timescaledb';
```

## Data Storage Estimates

With 1504 stations collected every 15 minutes:

- **Records per day**: ~144,000 (1504 stations × 96 intervals)
- **Storage per day**: ~15 MB (uncompressed), ~5 MB (compressed)
- **1 month of data**: ~150 MB (compressed)
- **6 months retention**: ~900 MB

TimescaleDB compression reduces storage by ~70% after 7 days.

## Next Steps

Once you have 3-4 weeks of data:

1. **Exploratory Analysis**: Run notebooks in `notebooks/01_data_exploration.ipynb`
2. **Feature Engineering**: Build temporal and spatial features
3. **Baseline Models**: Train simple forecasting models
4. **Deep Learning**: Train LSTM and GNN models

## Maintenance

### Weekly Tasks

```bash
# Check data quality
uv run python -c "
from src.data.database import DatabaseManager
db = DatabaseManager()

# Check for gaps in data
query = '''
    SELECT 
        DATE(time) as date,
        COUNT(DISTINCT station_id) as stations,
        COUNT(*) / 1504.0 as collections_per_station
    FROM station_status
    WHERE time > NOW() - INTERVAL '7 days'
    GROUP BY DATE(time)
    ORDER BY date DESC
'''
for row in db.fetch_all(query):
    print(row)
db.close()
"
```

### Monthly Tasks

- Review storage usage
- Update station information: `collector.update_station_information()`
- Backup database: `pg_dump velib > backups/velib_$(date +%Y%m%d).sql`

## API Rate Limits

Velib API has no strict rate limits, but:
- Be respectful: 15-minute intervals are sufficient
- API may be slow during peak hours
- Implement retries in case of transient failures

## Security Notes

- Never commit `.env` files to git
- Use strong passwords for database
- Restrict database access to localhost in production
- Consider encrypting backups

## Support

- Check logs: `logs/collector.log`
- Database issues: Check `docker-compose logs timescaledb`
- Open an issue on GitHub for bugs