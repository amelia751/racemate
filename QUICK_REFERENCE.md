# Cognirace - Quick Reference Card

**Status**: 🟢 FULLY OPERATIONAL  
**Port**: 8005  
**Updated**: October 22, 2025

---

## 🚀 Quick Commands

### Start System
```bash
# Start API
cd backend-api && python main.py

# Check health
curl http://localhost:8005/health

# Open docs
open http://localhost:8005/docs
```

### Run Tests
```bash
# Comprehensive E2E test (5/5 tests)
python3 tests/test_end_to_end.py

# Individual agent tests
python3 agents/specialized/fuel_agent.py
python3 agents/specialized/tire_agent.py
```

### Run Demo
```bash
# Interactive demo (4 scenarios)
python3 demo_system.py

# Telemetry simulator test
python3 streaming/simulator/telemetry_simulator.py
```

---

## 📡 API Endpoints

### Health & Status
```bash
GET  /health              # Health check
GET  /health/ready        # Readiness
GET  /predict/models      # List models
```

### Predictions
```bash
POST /predict/fuel        # Fuel consumption
POST /predict/laptime     # Lap time delta
POST /predict/tire        # Tire degradation
POST /predict/traffic     # Traffic impact
```

---

## 🤖 Agents

### ChiefAgent (Orchestrator)
```python
from agents.specialized.chief_agent import ChiefAgent
from agents.tools.api_client import CogniraceAPIClient

api = CogniraceAPIClient()
chief = ChiefAgent(api)

# Get comprehensive analysis
response = chief.process("Give me a full analysis", context)

# Get strategy recommendation
strategy = chief.get_strategy_recommendation(context)
```

### FuelAgent
```python
from agents.specialized.fuel_agent import FuelAgent

agent = FuelAgent(api)
response = agent.process("What's our fuel situation?", context)
```

### TireAgent
```python
from agents.specialized.tire_agent import TireAgent

agent = TireAgent(api)
response = agent.process("How are the tires?", context)
```

### TelemetryAgent
```python
from agents.specialized.telemetry_agent import TelemetryAgent

agent = TelemetryAgent()
agent.add_telemetry(telemetry_data)
response = agent.process("Show current telemetry")
stats = agent.calculate_statistics(window=10)
```

---

## 📊 Context Structure

```python
context = {
    "telemetry": {
        "speed": 180.5,       # km/h
        "nmot": 7200,         # RPM
        "gear": 5,            # 1-7
        "aps": 95.2,          # Throttle %
        "lap": 15,            # Current lap
        "fuel_level": 28.0,   # Liters
        "cum_brake_energy": 2500,    # Cumulative
        "cum_lateral_load": 3200,    # Cumulative
        "pbrake_f": 280,      # Front brake (bar)
        "pbrake_r": 220       # Rear brake (bar)
    },
    "race_info": {
        "total_laps": 40
    },
    "last_pit_lap": 5,
    "weather": {
        "air_temp": 28.5      # Celsius
    }
}
```

---

## 📡 Streaming Simulator

```python
from streaming.simulator.telemetry_simulator import TelemetrySimulator, SimulatorConfig

config = SimulatorConfig(
    frequency_hz=20.0,        # Samples per second
    base_speed=160.0,         # km/h
    lap_time_seconds=90.0,    # Seconds
    total_laps=40,            # Race length
    track_name="Virtual Circuit"
)

simulator = TelemetrySimulator(config)

# Stream for 60 seconds
for sample in simulator.stream(duration_seconds=60):
    print(sample)

# Stream 1000 samples
for sample in simulator.stream(max_samples=1000):
    process(sample)
```

---

## 📂 File Locations

### Documentation
```
README.md                    # Project overview
QUICKSTART.md                # 5-minute guide
PHASE_2_COMPLETE.md          # Phase 2 detailed report
PROJECT_STATUS.md            # System status
SESSION_SUMMARY.md           # Session summary
QUICK_REFERENCE.md           # This file
```

### Code
```
backend-api/main.py          # FastAPI server
agents/specialized/          # 4 agents
streaming/simulator/         # Telemetry simulator
tests/test_end_to_end.py    # Comprehensive test
demo_system.py               # Interactive demo
```

### ML Pipeline
```
ml-pipeline/models/          # 8 trained models
ml-pipeline/training/        # Training scripts
ml-pipeline/deployment/      # Endpoints
```

---

## 🔧 Configuration

### API (.env.local in backend-api/)
```bash
GCP_PROJECT_ID=cognirace
GCP_SERVICE_ACCOUNT_PATH=../ml-pipeline/config/gcp_credentials.json
GCP_REGION=us-central1
GCS_BUCKET_MODELS=cognirace-model-artifacts
API_PORT=8005
API_HOST=0.0.0.0
MODEL_CACHE_DIR=/tmp/cognirace_models
```

### Agents (.env.local in agents/)
```bash
API_BASE_URL=http://localhost:8005
API_TIMEOUT=30
GCP_PROJECT_ID=cognirace
GCP_REGION=us-central1
VERTEX_AI_MODEL=gemini-1.5-flash
```

---

## 🧪 Test Results

```
✅ Test 1: API Connectivity       PASSED
✅ Test 2: Telemetry Simulator    PASSED
✅ Test 3: Individual Agents      PASSED
✅ Test 4: Chief Agent            PASSED
✅ Test 5: Streaming Pipeline     PASSED

Overall: 5/5 tests (100%)
```

---

## 📈 Performance

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| API Response | < 200ms | < 100ms | ✅ |
| Model Loading | < 5s | < 3s | ✅ |
| Agent Response | < 200ms | < 200ms | ✅ |
| Stream Rate | 20 Hz | 20 Hz | ✅ |
| Test Pass | 100% | 100% | ✅ |

---

## 🚨 Troubleshooting

### API Won't Start
```bash
# Check if port is in use
lsof -i :8005

# Check logs
tail -f /tmp/cognirace_api.log

# Verify credentials
ls -la backend-api/.env.local
```

### Models Not Loading
```bash
# Check GCS connection
gsutil ls gs://cognirace-model-artifacts/

# Verify cache directory
ls -la /tmp/cognirace_models/

# Check credentials
cat ml-pipeline/config/gcp_credentials.json | head -3
```

### Tests Failing
```bash
# Check API is running
curl http://localhost:8005/health

# Run individual tests
python3 -c "from agents.specialized.fuel_agent import FuelAgent; print('OK')"

# Check dependencies
pip list | grep -E "(fastapi|torch|pydantic)"
```

---

## 🔗 Quick Links

- **Swagger UI**: http://localhost:8005/docs
- **ReDoc**: http://localhost:8005/redoc
- **Health**: http://localhost:8005/health
- **GCP Console**: https://console.cloud.google.com

---

## 📊 System Stats

- **8 ML Models**: Trained & deployed
- **4 Agents**: Operational
- **7 Endpoints**: Active
- **5,300+ LOC**: Written
- **38+ Files**: Created
- **100% Tests**: Passing
- **< $5 Cost**: Development

---

## 🎯 Demo Scenarios

1. **Early Race**: All green, STAY OUT
2. **Mid-Race**: Tire warning, MONITOR
3. **Critical**: Low fuel + tires, **PIT NOW**
4. **Live Stream**: 10s real-time processing

---

## 💡 Quick Tips

- Use `chief.process()` for comprehensive analysis
- Use specific agents for focused queries
- Check `telemetry_agent.telemetry_buffer` for history
- Strategy recommendations via `get_strategy_recommendation()`
- Simulator frequency: 20 Hz production, 5-10 Hz testing

---

## 📞 Support

- **Docs**: See README.md and QUICKSTART.md
- **Tests**: Run `python3 tests/test_end_to_end.py`
- **Demo**: Run `python3 demo_system.py`
- **Logs**: Check `/tmp/cognirace_api.log`

---

**Status**: 🟢 FULLY OPERATIONAL  
**Ready for**: Production, Demo, Submission

*Quick reference for Cognirace v2.0.0*

