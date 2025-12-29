-- ============================================================
-- Velib Predictor Database Schema
-- PostgreSQL + TimescaleDB
-- ============================================================

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;
CREATE EXTENSION IF NOT EXISTS cube;
CREATE EXTENSION IF NOT EXISTS earthdistance;

-- ============================================================
-- 1. STATION INFORMATION (Semi-static data)
-- ============================================================
CREATE TABLE station_information (
    station_id BIGINT PRIMARY KEY,
    station_code VARCHAR(20) NOT NULL,
    name VARCHAR(255) NOT NULL,
    lat DOUBLE PRECISION NOT NULL,
    lon DOUBLE PRECISION NOT NULL,
    capacity INT NOT NULL,
    
    -- Metadata
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    
    -- Indexes
    CONSTRAINT unique_station_code UNIQUE (station_code)
);

-- Spatial index for nearest neighbor queries
-- CREATE INDEX idx_station_location ON station_information USING GIST (
--     ll_to_earth(lat, lon)
-- );

-- Standard index on location for k-NN
CREATE INDEX idx_station_lat_lon ON station_information (lat, lon);

COMMENT ON TABLE station_information IS 'Static information about Velib stations';
COMMENT ON COLUMN station_information.station_id IS 'Unique station identifier from Velib API';
COMMENT ON COLUMN station_information.capacity IS 'Total docks at station (bikes + empty)';


-- ============================================================
-- 2. STATION STATUS (Time series data)
-- ============================================================
CREATE TABLE station_status (
    time TIMESTAMPTZ NOT NULL,
    station_id BIGINT NOT NULL,
    
    -- Availability counts
    num_bikes_available INT NOT NULL,
    num_mechanical INT NOT NULL,
    num_ebike INT NOT NULL,
    num_docks_available INT NOT NULL,
    
    -- Station state flags
    is_installed BOOLEAN NOT NULL,
    is_returning BOOLEAN NOT NULL,
    is_renting BOOLEAN NOT NULL,
    
    -- API metadata
    last_reported BIGINT NOT NULL, -- Unix timestamp from API
    
    -- Constraints
    PRIMARY KEY (time, station_id),
    FOREIGN KEY (station_id) REFERENCES station_information(station_id) ON DELETE CASCADE,
    
    -- Data quality constraints
    CONSTRAINT check_bikes_sum CHECK (num_mechanical + num_ebike = num_bikes_available),
    CONSTRAINT check_positive_counts CHECK (
        num_bikes_available >= 0 AND 
        num_mechanical >= 0 AND 
        num_ebike >= 0 AND 
        num_docks_available >= 0
    )
);

-- Convert to TimescaleDB hypertable (automatic partitioning on time dimension)
SELECT create_hypertable(
    'station_status', 
    'time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- Add compression policy (compress data older than 7 days)
ALTER TABLE station_status SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'station_id',
    timescaledb.compress_orderby = 'time DESC'
);

SELECT add_compression_policy('station_status', INTERVAL '7 days');

-- Add retention policy (keep data for 6 months)
SELECT add_retention_policy('station_status', INTERVAL '6 months');

-- Indexes for efficient queries
CREATE INDEX idx_station_status_station_time ON station_status (station_id, time DESC);
CREATE INDEX idx_station_status_time ON station_status (time DESC);

COMMENT ON TABLE station_status IS 'Time series data of station availability';
COMMENT ON COLUMN station_status.time IS 'Timestamp when data was collected (not from API)';
COMMENT ON COLUMN station_status.last_reported IS 'Unix timestamp from Velib API';


-- ============================================================
-- 3. PREDICTIONS (Model outputs)
-- ============================================================
CREATE TABLE predictions (
    time TIMESTAMPTZ NOT NULL,
    station_id BIGINT NOT NULL,
    prediction_horizon_minutes INT NOT NULL, -- e.g., 15, 30, 60
    
    -- Predictions
    predicted_mechanical FLOAT NOT NULL,
    predicted_ebike FLOAT NOT NULL,
    predicted_total FLOAT NOT NULL,
    predicted_available_docks FLOAT NOT NULL,
    
    -- Model metadata
    model_name VARCHAR(50) NOT NULL,
    model_version VARCHAR(20) NOT NULL,
    
    -- Confidence (optional)
    confidence_score FLOAT,
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    PRIMARY KEY (time, station_id, prediction_horizon_minutes, model_name, model_version),
    FOREIGN KEY (station_id) REFERENCES station_information(station_id) ON DELETE CASCADE,
    
    CONSTRAINT check_prediction_positive CHECK (
        predicted_mechanical >= 0 AND 
        predicted_ebike >= 0 AND 
        predicted_total >= 0 AND
        predicted_available_docks >= 0
    )
);

-- Convert to hypertable
SELECT create_hypertable(
    'predictions', 
    'time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- Indexes
CREATE INDEX idx_predictions_station_time ON predictions (station_id, time DESC);
CREATE INDEX idx_predictions_model ON predictions (model_name, model_version, time DESC);

-- Retention policy (keep predictions for 3 months)
SELECT add_retention_policy('predictions', INTERVAL '3 months');

COMMENT ON TABLE predictions IS 'Model predictions for future station availability';
COMMENT ON COLUMN predictions.prediction_horizon_minutes IS 'How many minutes ahead this prediction is for';


-- ============================================================
-- 4. MODEL RUNS (Training metadata)
-- ============================================================
CREATE TABLE model_runs (
    run_id SERIAL PRIMARY KEY,
    model_name VARCHAR(50) NOT NULL,
    model_version VARCHAR(20) NOT NULL,
    
    -- Training info
    started_at TIMESTAMPTZ NOT NULL,
    completed_at TIMESTAMPTZ,
    status VARCHAR(20) NOT NULL, -- 'running', 'completed', 'failed'
    
    -- Data splits
    train_start_date DATE NOT NULL,
    train_end_date DATE NOT NULL,
    val_start_date DATE,
    val_end_date DATE,
    
    -- Hyperparameters (stored as JSON)
    hyperparameters JSONB,
    
    -- Metrics
    metrics JSONB, -- e.g., {"mae": 2.3, "rmse": 3.1, "r2": 0.85}
    
    -- Model artifact path
    model_path TEXT,
    
    -- Metadata
    notes TEXT,
    git_commit VARCHAR(40),
    
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_model_runs_name_version ON model_runs (model_name, model_version, started_at DESC);
CREATE INDEX idx_model_runs_status ON model_runs (status);

COMMENT ON TABLE model_runs IS 'Metadata about model training runs';


-- ============================================================
-- 5. MATERIALIZED VIEWS for common queries
-- ============================================================

-- Latest status for each station (for quick dashboard queries)
CREATE MATERIALIZED VIEW latest_station_status AS
SELECT DISTINCT ON (station_id)
    station_id,
    time,
    num_bikes_available,
    num_mechanical,
    num_ebike,
    num_docks_available,
    is_renting,
    is_returning
FROM station_status
ORDER BY station_id, time DESC;

CREATE UNIQUE INDEX idx_latest_station_status ON latest_station_status (station_id);

COMMENT ON MATERIALIZED VIEW latest_station_status IS 'Most recent status for each station';

-- Refresh policy (every 15 minutes)
-- Note: In production, this would be triggered by your data collection job


-- ============================================================
-- 6. CONTINUOUS AGGREGATES (TimescaleDB feature)
-- ============================================================

-- Hourly aggregates for each station
CREATE MATERIALIZED VIEW station_status_hourly
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('1 hour', time) AS bucket,
    station_id,
    AVG(num_mechanical) as avg_mechanical,
    AVG(num_ebike) as avg_ebike,
    AVG(num_bikes_available) as avg_bikes_available,
    AVG(num_docks_available) as avg_docks_available,
    MIN(num_bikes_available) as min_bikes_available,
    MAX(num_bikes_available) as max_bikes_available,
    COUNT(*) as num_measurements
FROM station_status
GROUP BY bucket, station_id
WITH NO DATA;

-- Refresh policy (every hour)
SELECT add_continuous_aggregate_policy('station_status_hourly',
    start_offset => INTERVAL '3 hours',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour');

COMMENT ON MATERIALIZED VIEW station_status_hourly IS 'Hourly aggregates for faster historical queries';


-- ============================================================
-- 7. HELPER FUNCTIONS
-- ============================================================

-- Function to get k-nearest stations
CREATE OR REPLACE FUNCTION get_nearest_stations(
    target_station_id BIGINT,
    k INT DEFAULT 5
)
RETURNS TABLE (
    station_id BIGINT,
    distance_km FLOAT
) AS $$
BEGIN
    RETURN QUERY
    WITH target AS (
        SELECT lat, lon 
        FROM station_information 
        WHERE station_information.station_id = target_station_id
    )
    SELECT 
        si.station_id,
        earth_distance(
            ll_to_earth(si.lat, si.lon),
            ll_to_earth(target.lat, target.lon)
        ) / 1000.0 AS distance_km
    FROM station_information si, target
    WHERE si.station_id != target_station_id
    ORDER BY earth_distance(
        ll_to_earth(si.lat, si.lon),
        ll_to_earth(target.lat, target.lon)
    )
    LIMIT k;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION get_nearest_stations IS 'Get k nearest stations to a given station';


-- Function to compute fill rate
CREATE OR REPLACE FUNCTION compute_fill_rate(
    bikes_available INT,
    capacity INT
)
RETURNS FLOAT AS $$
BEGIN
    IF capacity = 0 THEN
        RETURN 0.0;
    END IF;
    RETURN bikes_available::FLOAT / capacity::FLOAT;
END;
$$ LANGUAGE plpgsql IMMUTABLE;


-- ============================================================
-- 8. INITIAL DATA QUALITY VIEWS
-- ============================================================

-- View to identify stations with no recent data
CREATE VIEW stale_stations AS
SELECT 
    si.station_id,
    si.name,
    MAX(ss.time) as last_update,
    NOW() - MAX(ss.time) as time_since_update
FROM station_information si
LEFT JOIN station_status ss ON si.station_id = ss.station_id
GROUP BY si.station_id, si.name
HAVING MAX(ss.time) < NOW() - INTERVAL '1 hour' OR MAX(ss.time) IS NULL;

COMMENT ON VIEW stale_stations IS 'Stations with no data in the last hour';


-- View for data collection monitoring
CREATE VIEW collection_stats AS
SELECT 
    DATE_TRUNC('hour', time) as hour,
    COUNT(DISTINCT station_id) as stations_collected,
    COUNT(*) as total_records,
    (SELECT COUNT(*) FROM station_information) as total_stations
FROM station_status
WHERE time > NOW() - INTERVAL '24 hours'
GROUP BY hour
ORDER BY hour DESC;

COMMENT ON VIEW collection_stats IS 'Data collection statistics for monitoring';