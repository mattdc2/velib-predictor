# ----------------------------------------------------------
# 6. verify_data.sh
#    Verify data collection is working
# ----------------------------------------------------------

verify_data() {
    echo "========================================="
    echo "Data Collection Verification"
    echo "========================================="
    
    python -c "
from src.data.database import DatabaseManager
from datetime import datetime, timedelta

db = DatabaseManager()

# Check station information
station_count = db.get_table_row_count('station_information')
print(f'Total stations: {station_count}')

# Check recent data
query = '''
    SELECT 
        COUNT(*) as total_records,
        COUNT(DISTINCT station_id) as stations_with_data,
        MIN(time) as oldest_record,
        MAX(time) as newest_record
    FROM station_status
    WHERE time > NOW() - INTERVAL '1 hour'
'''
result = db.fetch_one(query)

if result:
    print(f\"Records in last hour: {result['total_records']}\")
    print(f\"Stations with data: {result['stations_with_data']}\")
    print(f\"Oldest record: {result['oldest_record']}\")
    print(f\"Newest record: {result['newest_record']}\")
    
    # Check data freshness
    newest = result['newest_record']
    if newest:
        age_minutes = (datetime.now() - newest).total_seconds() / 60
        if age_minutes < 20:
            print(f'✓ Data is fresh ({age_minutes:.1f} minutes old)')
        else:
            print(f'⚠ Data is stale ({age_minutes:.1f} minutes old)')
else:
    print('✗ No recent data found!')

db.close()
"
    
    echo "========================================="
}
