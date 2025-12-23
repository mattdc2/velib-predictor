
# ----------------------------------------------------------
# 5. initial_data_load.sh
#    Load station information and initial status
# ----------------------------------------------------------

initial_data_load() {
    echo "========================================="
    echo "Initial Data Load"
    echo "========================================="
    
    # First, load station information
    echo "Loading station information..."
    python -c "
from src.data.collector import VelibDataCollector, VelibAPIClient
from src.data.database import DatabaseManager

db = DatabaseManager()
api = VelibAPIClient()
collector = VelibDataCollector(db, api)

# Load station information
count = collector.update_station_information()
print(f'Loaded {count} stations')

# Load initial status
count = collector.collect_station_status()
print(f'Loaded {count} status records')

db.close()
print('✓ Initial data load complete!')
"
    
    echo "========================================="
}