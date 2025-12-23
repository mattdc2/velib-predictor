
# ----------------------------------------------------------
# 4. setup_cron.sh
#    Setup cron job for data collection
# ----------------------------------------------------------

setup_cron() {
    echo "========================================="
    echo "Setting up Cron Job"
    echo "========================================="
    
    # Get absolute path to project
    PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    
    # Create logs directory
    mkdir -p "$PROJECT_DIR/logs"
    
    # Create cron job entry
    CRON_JOB="*/15 * * * * cd $PROJECT_DIR && uv run python -m src.data.collector >> logs/collector.log 2>&1"
    
    # Check if cron job already exists
    if crontab -l 2>/dev/null | grep -q "src.data.collector"; then
        echo "Cron job already exists!"
    else
        # Add cron job
        (crontab -l 2>/dev/null; echo "$CRON_JOB") | crontab -
        echo "✓ Cron job added successfully!"
        echo "Data will be collected every 15 minutes"
    fi
    
    # Show current crontab
    echo ""
    echo "Current crontab:"
    crontab -l | grep velib
    
    echo "========================================="
}
