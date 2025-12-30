#!/bin/bash
# scripts/add_weather.sh
# Migrate database to add weather tables and backfill data

set -e

echo "========================================="
echo "Adding Weather Data to Velib Predictor"
echo "========================================="

# 1. Add weather schema to database
echo ""
echo "Step 1: Adding weather tables to database..."
docker-compose exec -T timescaledb psql -U velib_user -d velib << 'EOF'

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS cube;
CREATE EXTENSION IF NOT EXISTS earthdistance;

-- Weather data table
CREATE TABLE IF NOT EXISTS weather_data (
    time TIMESTAMPTZ NOT NULL PRIMARY KEY,
    temperature FLOAT NOT NULL,
    apparent_temperature FLOAT,
    precipitation FLOAT NOT NULL DEFAULT 0,
    rain FLOAT DEFAULT 0,
    snowfall FLOAT DEFAULT 0,
    wind_speed FLOAT,
    wind_direction INT,
    wind_gusts FLOAT,
    pressure FLOAT,
    humidity INT,
    cloud_cover INT,
    weather_code INT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Convert to hypertable if not exists
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM timescaledb_information.hypertables 
        WHERE hypertable_name = 'weather_data'
    ) THEN
        PERFORM create_hypertable('weather_data', 'time', 
            chunk_time_interval => INTERVAL '7 days');
    END IF;
END $$;

-- Add compression
ALTER TABLE weather_data SET (
    timescaledb.compress,
    timescaledb.compress_orderby = 'time DESC'
);

-- Add policies if not exist
DO $$
BEGIN
    -- Compression policy
    IF NOT EXISTS (
        SELECT 1 FROM timescaledb_information.jobs 
        WHERE proc_name = 'policy_compression' 
        AND hypertable_name = 'weather_data'
    ) THEN
        PERFORM add_compression_policy('weather_data', INTERVAL '7 days');
    END IF;
    
    -- Retention policy
    IF NOT EXISTS (
        SELECT 1 FROM timescaledb_information.jobs 
        WHERE proc_name = 'policy_retention' 
        AND hypertable_name = 'weather_data'
    ) THEN
        PERFORM add_retention_policy('weather_data', INTERVAL '6 months');
    END IF;
END $$;

-- Index
CREATE INDEX IF NOT EXISTS idx_weather_time ON weather_data (time DESC);

-- Views
CREATE OR REPLACE VIEW weather_conditions AS
SELECT 
    time,
    temperature,
    precipitation,
    wind_speed,
    CASE weather_code
        WHEN 0 THEN 'Clear sky'
        WHEN 1 THEN 'Mainly clear'
        WHEN 2 THEN 'Partly cloudy'
        WHEN 3 THEN 'Overcast'
        WHEN 51 THEN 'Light drizzle'
        WHEN 53 THEN 'Moderate drizzle'
        WHEN 55 THEN 'Dense drizzle'
        WHEN 61 THEN 'Slight rain'
        WHEN 63 THEN 'Moderate rain'
        WHEN 65 THEN 'Heavy rain'
        WHEN 71 THEN 'Slight snow'
        WHEN 73 THEN 'Moderate snow'
        WHEN 75 THEN 'Heavy snow'
        WHEN 95 THEN 'Thunderstorm'
        ELSE 'Other'
    END as condition_description,
    CASE
        WHEN weather_code IN (51, 53, 55, 61, 63, 65, 80, 81, 82) THEN TRUE
        ELSE FALSE
    END as is_raining
FROM weather_data;

\echo '✓ Weather schema added successfully'
EOF

if [ $? -ne 0 ]; then
    echo "✗ Failed to add weather schema"
    exit 1
fi

# 2. Rebuild collector with weather support
echo ""
echo "Step 2: Rebuilding collector with weather support..."
docker-compose build collector

if [ $? -ne 0 ]; then
    echo "✗ Failed to rebuild collector"
    exit 1
fi

# 3. Backfill historical weather data
echo ""
echo "Step 3: Backfilling historical weather data..."
echo "(This will fetch weather for all dates with bike data)"

docker-compose run --rm collector python -c "
from src.data.database import DatabaseManager
from src.data.weather_collector import OpenMeteoClient, WeatherCollector

db = DatabaseManager()
api = OpenMeteoClient()
collector = WeatherCollector(db, api)

try:
    print('Backfilling historical weather...')
    count = collector.backfill_historical_weather()
    print(f'✓ Backfilled {count} weather records')
except Exception as e:
    print(f'✗ Backfill failed: {e}')
    exit(1)
finally:
    db.close()
"

if [ $? -ne 0 ]; then
    echo "✗ Failed to backfill weather data"
    exit 1
fi

# 4. Restart collector
echo ""
echo "Step 4: Restarting collector..."
docker-compose restart collector

# 5. Verify everything works
echo ""
echo "Step 5: Verifying weather collection..."
sleep 5

docker-compose exec -T timescaledb psql -U velib_user -d velib << 'EOF'
SELECT 
    COUNT(*) as total_records,
    MIN(time) as oldest,
    MAX(time) as newest,
    ROUND(AVG(temperature)::numeric, 1) as avg_temp,
    ROUND(SUM(precipitation)::numeric, 1) as total_precip
FROM weather_data;
EOF

echo ""
echo "========================================="
echo "✓ Weather integration complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "  - Weather will be collected every 15 minutes"
echo "  - Check logs: docker-compose logs -f collector"
echo "  - View data: docker-compose exec timescaledb psql -U velib_user -d velib"
echo "              SELECT * FROM weather_conditions ORDER BY time DESC LIMIT 10;"
echo ""