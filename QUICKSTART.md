# Cognirace - Quick Start Guide

## 🚀 Get Started in 5 Minutes

### Prerequisites
- Python 3.13+
- Google Cloud account with service account credentials
- Port 8005 available

---

## Step 1: Start the API Server

```bash
cd backend-api
source venv/bin/activate
python main.py
```

**Expected Output**:
```
Cognirace API starting up...
Cognirace API startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8005
```

---

## Step 2: Test the API

```bash
curl http://localhost:8005/health
```

**Expected Response**:
```json
{
  "status": "healthy",
  "message": "Cognirace API is running",
  "version": "1.0.0",
  "uptime_seconds": 123.45
}
```

---

## Step 3: Run Comprehensive Test

```bash
python3 tests/test_end_to_end.py
```

**Expected Result**:
```
✅ 5/5 tests passed (100%)
✅ ALL TESTS PASSED! System is fully operational!
```

---

## Step 4: Use the Agent System

### Simple Python Example

```python
from agents.specialized.chief_agent import ChiefAgent
from agents.tools.api_client import CogniraceAPIClient

# Initialize
api_client = CogniraceAPIClient()
chief = ChiefAgent(api_client)

# Prepare context
context = {
    "telemetry": {
        "speed": 180.5,
        "nmot": 7200,
        "gear": 5,
        "aps": 95.2,
        "lap": 15,
        "fuel_level": 28.0,
        "cum_brake_energy": 2500,
        "cum_lateral_load": 3200
    },
    "race_info": {"total_laps": 40},
    "last_pit_lap": 5,
    "weather": {"air_temp": 28.5}
}

# Get comprehensive analysis
response = chief.process("Give me a full analysis", context)
print(response)

# Get strategy recommendation
strategy = chief.get_strategy_recommendation(context)
print(f"Should Pit: {strategy['should_pit']}")
print(f"Urgency: {strategy['urgency']}")
```

---

## Step 5: Test Streaming Data

### Run Telemetry Simulator

```python
from streaming.simulator.telemetry_simulator import TelemetrySimulator, SimulatorConfig

# Configure simulator
config = SimulatorConfig(
    frequency_hz=20.0,     # 20 Hz (realistic)
    base_speed=160.0,      # 160 km/h average
    lap_time_seconds=90.0, # 90 second laps
    total_laps=40          # 40 lap race
)

simulator = TelemetrySimulator(config)

# Stream telemetry for 60 seconds
for sample in simulator.stream(duration_seconds=60):
    print(f"Lap {sample['lap']}: {sample['speed']:.1f} km/h, Gear {sample['gear']}")
```

---

## Available Agents

### 1. ChiefAgent (Orchestrator)
Coordinates all agents and provides comprehensive analysis.

```python
chief.process("Should we pit now?", context)
```

### 2. FuelAgent (Fuel Strategy)
Analyzes fuel consumption and predicts remaining laps.

```python
fuel_agent.process("What's our fuel situation?", context)
```

### 3. TireAgent (Tire Strategy)
Monitors tire degradation and recommends pit timing.

```python
tire_agent.process("How are the tires?", context)
```

### 4. TelemetryAgent (Data Manager)
Buffers and analyzes telemetry streams.

```python
telemetry_agent.process("Show current data", context)
```

---

## API Endpoints

### Health Check
```bash
GET http://localhost:8005/health
```

### Fuel Prediction
```bash
POST http://localhost:8005/predict/fuel
{
  "telemetry": {
    "speed": 180,
    "nmot": 7200,
    "gear": 5,
    "aps": 95,
    "lap": 15
  }
}
```

### Lap Time Prediction
```bash
POST http://localhost:8005/predict/laptime
{
  "telemetry_sequence": [/* array of telemetry samples */],
  "current_lap_time": 88.5
}
```

### Tire Degradation Prediction
```bash
POST http://localhost:8005/predict/tire
{
  "telemetry": {
    "cum_brake_energy": 2500,
    "cum_lateral_load": 3200,
    "lap_number": 15
  }
}
```

### Traffic Analysis
```bash
POST http://localhost:8005/predict/traffic
{
  "traffic_scenario": [/* array of car telemetry */]
}
```

---

## Interactive Swagger UI

Visit: **http://localhost:8005/docs**

Interactive API documentation with built-in testing.

---

## Troubleshooting

### API Won't Start
1. Check if port 8005 is available:
   ```bash
   lsof -i :8005
   ```
2. Verify virtual environment is activated
3. Check `.env.local` configuration

### Models Not Loading
1. Verify GCS credentials in `.env.local`
2. Check GCS bucket exists and contains models
3. Review logs in `/tmp/cognirace_api.log`

### Agent Errors
1. Ensure API is running
2. Check API connectivity: `curl http://localhost:8005/health`
3. Verify agent dependencies installed: `pip install -r agents/requirements.txt`

---

## What's Next?

### Explore the System
- Run the comprehensive test suite
- Try different telemetry scenarios
- Experiment with agent queries

### Customize
- Adjust simulator parameters (speed, frequency, track)
- Modify agent thresholds (fuel margin, tire grip levels)
- Add new API endpoints

### Deploy
- Deploy API to Cloud Run
- Set up Pub/Sub for real-time streaming
- Implement frontend dashboard

---

## Quick Commands Cheat Sheet

```bash
# Start API
cd backend-api && python main.py

# Run tests
python3 tests/test_end_to_end.py

# Check API health
curl http://localhost:8005/health

# View API docs
open http://localhost:8005/docs

# Test simulator
python3 streaming/simulator/telemetry_simulator.py

# Test individual agent
python3 agents/specialized/fuel_agent.py
```

---

## Support

- **Documentation**: See `PHASE_2_COMPLETE.md` for full system documentation
- **Issues**: Check logs in `/tmp/cognirace_api.log`
- **Project Status**: See `PROJECT_STATUS.md`

---

**System Status**: 🟢 FULLY OPERATIONAL
**Last Updated**: October 22, 2025

