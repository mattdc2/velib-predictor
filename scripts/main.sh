# ----------------------------------------------------------
# 7. Main execution
# ----------------------------------------------------------

case "${1:-}" in
    database)
        setup_database
        ;;
    cron)
        setup_cron
        ;;
    initial-load)
        initial_data_load
        ;;
    verify)
        verify_data
        ;;
    all)
        setup_database
        initial_data_load
        setup_cron
        verify_data
        ;;
    *)
        echo "Usage: $0 {database|cron|initial-load|verify|all}"
        echo ""
        echo "Commands:"
        echo "  database      - Setup database schema"
        echo "  cron          - Setup cron job for data collection"
        echo "  initial-load  - Load initial data"
        echo "  verify        - Verify data collection is working"
        echo "  all           - Run all setup steps"
        exit 1
        ;;
esac