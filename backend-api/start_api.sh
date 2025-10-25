#!/bin/bash
# Start Cognirace Prediction API on port 8005

cd "$(dirname "$0")"

echo "🏁 Starting Cognirace Prediction API..."

# Activate virtual environment
source venv/bin/activate

# Start API
python main.py > /tmp/cognirace_api.log 2>&1 &
API_PID=$!

echo $API_PID > /tmp/cognirace_api.pid

sleep 3

if ps -p $API_PID > /dev/null; then
    echo "✓ API started successfully (PID: $API_PID)"
    echo "  URL: http://localhost:8005"
    echo "  Docs: http://localhost:8005/docs"
    echo "  Logs: /tmp/cognirace_api.log"
    echo ""
    echo "To stop: kill $API_PID"
else
    echo "✗ Failed to start API"
    cat /tmp/cognirace_api.log
fi

