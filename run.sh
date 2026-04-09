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

# Change to app directory and start the main application
cd /app

# Execute the main application directly
exec python3 -m src.main
