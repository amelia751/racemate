#!/bin/bash
# Stop Cognirace Prediction API

echo "🛑 Stopping Cognirace Prediction API..."

if [ -f /tmp/cognirace_api.pid ]; then
    API_PID=$(cat /tmp/cognirace_api.pid)
    if ps -p $API_PID > /dev/null; then
        kill $API_PID
        echo "✓ API stopped (PID: $API_PID)"
    else
        echo "ℹ️  API not running"
    fi
    rm /tmp/cognirace_api.pid
else
    echo "ℹ️  PID file not found"
    pkill -f "python.*main.py" && echo "✓ Killed running API processes" || echo "ℹ️  No API processes found"
fi

