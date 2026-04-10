#!/bin/bash

# ==============================================================================
# ML Heating Control Add-on Entry Point Script
# ==============================================================================

set -e

# Configure logging
echo "[INFO] Starting ML Heating Control Add-on..."

# Check if we're running in Home Assistant environment
if [[ -f "/etc/services.d" ]] || [[ -n "${SUPERVISOR_TOKEN}" ]]; then
    # We're in HA addon environment, use bashio if available
    if command -v bashio &> /dev/null; then
        echo "[INFO] Running in Home Assistant addon environment with bashio support"
        
        # Initialize configuration using bashio
        echo "[INFO] Initializing configuration..."
        python3 /app/config_adapter.py
        
        # Source the environment file written by config_adapter so that
        # all mapped configuration variables are available to src.main.
        if [[ -f /data/config/env_vars ]]; then
            source /data/config/env_vars
            echo "[INFO] Sourced environment variables from /data/config/env_vars"
        else
            echo "[WARNING] Environment file /data/config/env_vars not found"
        fi
        
        # Setup data directories with proper permissions
        mkdir -p /data/{models,backups,logs,config}
        
        # Check if required environment variables are set
        if [[ -z "${SUPERVISOR_TOKEN}" ]]; then
            echo "[ERROR] Home Assistant Supervisor token not available"
            exit 1
        fi
    else
        echo "[WARNING] Running in HA environment but bashio not available, using fallback mode"
    fi
else
    echo "[INFO] Running in standalone/development mode"
    
    # Setup data directories for standalone mode
    mkdir -p /data/{models,backups,logs,config}
    
    # Set default environment variables for standalone mode
    export SUPERVISOR_TOKEN="${SUPERVISOR_TOKEN:-standalone_mode}"
fi

# Create log file if it doesn't exist
touch /data/logs/ml_heating.log

# Start the ML heating system
echo "[INFO] Starting ML heating system..."

# Change to app directory
cd /app

# Start the health check server in the background (port 3002)
echo "[INFO] Starting health check server on port 3002..."
python3 /app/dashboard/health.py &
HEALTH_PID=$!

# Start the Streamlit dashboard in the background (port 3001 = ingress_port)
echo "[INFO] Starting Streamlit dashboard on port 3001 (ingress)..."
streamlit run /app/dashboard/app.py \
    --server.port=3001 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false \
    --browser.gatherUsageStats=false &
DASHBOARD_PID=$!

# Execute the main ML application in the foreground
echo "[INFO] Starting ML backend..."
python3 -m src.main &
ML_PID=$!

# Wait for any process to exit and propagate the signal
wait -n $HEALTH_PID $DASHBOARD_PID $ML_PID
EXIT_CODE=$?
echo "[ERROR] A process exited unexpectedly with code $EXIT_CODE"

# Clean up remaining processes
kill $HEALTH_PID $DASHBOARD_PID $ML_PID 2>/dev/null
exit $EXIT_CODE
