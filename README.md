# Cognirace - Real-Time Race Strategy Platform

> **Unleash the Data. Engineer Victory.**

**Hack the Track 2025** - Toyota GR Cup Series

A production-ready, real-time analytics and strategy tool for the GR Cup Series featuring ML-powered predictions, intelligent agents, and streaming telemetry analysis.

## 🎉 **System Status: FULLY OPERATIONAL** 🟢

✅ **8 ML Models Trained & Deployed**
✅ **FastAPI Server Running** (Port 8005)
✅ **Multi-Agent System Operational**
✅ **Streaming Infrastructure Ready**
✅ **100% Test Pass Rate**

---

## 🚀 Quick Start

### Start the System (3 Commands)

```bash
# 1. Start the API server
cd backend-api && python main.py

# 2. Run comprehensive test
python3 tests/test_end_to_end.py

# 3. Access Swagger UI
open http://localhost:8005/docs
```

**See [QUICKSTART.md](QUICKSTART.md) for detailed instructions**

---

## 🏁 Project Overview

### What is Cognirace?

Cognirace is a **real-time race strategy platform** that:
- 🤖 Uses **8 trained ML models** for race predictions
- 🎯 Employs **specialized AI agents** for fuel, tire, and telemetry analysis
- 📡 Processes **streaming telemetry** at 20 Hz
- 🏎️ Provides **real-time pit strategy recommendations**
- 📊 Delivers **comprehensive race analysis** in < 200ms

### Key Features

1. **ML-Powered Predictions**
   - Lap Time Delta Prediction (Transformer)
   - Tire Degradation Modeling (Physics-informed TCN)
   - Fuel Consumption Prediction (Gradient Boosting)
   - Traffic Impact Analysis (GNN)
   - FCY Hazard Prediction, Pit Loss Model, Anomaly Detection, Driver Embedding

2. **Intelligent Agent System**
   - **ChiefAgent**: Orchestrates strategy and coordinates specialists
   - **FuelAgent**: Fuel consumption analysis and pit timing
   - **TireAgent**: Tire degradation monitoring and recommendations
   - **TelemetryAgent**: Real-time data buffering and statistics

3. **Real-Time Infrastructure**
   - FastAPI server (4 prediction endpoints)
   - Telemetry simulator (1-100 Hz configurable)
   - Streaming pipeline with agent integration
   - GCS-backed model storage

---

## 📊 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    COGNIRACE PLATFORM                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Telemetry Stream (20Hz) → ChiefAgent → ML Models          │
│                              ↓                              │
│                    ┌────────────────────┐                   │
│                    │  Strategy Engine   │                   │
│                    │  - Fuel Analysis   │                   │
│                    │  - Tire Analysis   │                   │
│                    │  - Pit Decisions   │                   │
│                    └────────────────────┘                   │
│                              ↓                              │
│                    Real-Time Recommendations                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎯 Competition Categories

**Entered Category**: **Real-Time Analytics**

Create a tool that simulates real-time decision-making for race engineers.

✅ **Delivered**: Complete pit wall assistant with ML predictions and agent coordination

---

## 📁 Repository Structure

### Phase 1: ML Foundation ✅
1. ✅ **Data Downloaded**: All 6 track datasets
2. 📊 **Data Explored**: See [DATAEXPLORE.md](DATAEXPLORE.md)
3. 💡 **Project Spec**: See [IDEA.md](IDEA.md)
4. 🤖 **8 Models Trained**: All models operational

### Phase 2: Production System ✅
5. 🌐 **API Deployed**: FastAPI on port 8005
6. 🤖 **Agents Built**: 4 specialized agents + orchestrator
7. 📡 **Streaming Ready**: Telemetry simulator operational
8. ✅ **Tests Pass**: 100% comprehensive test pass rate

```
cognirace/
├── 📖 Documentation
│   ├── README.md                      # This file
│   ├── QUICKSTART.md                  # 5-minute start guide
│   ├── PHASE_2_COMPLETE.md            # Phase 2 detailed report
│   ├── DATAEXPLORE.md                 # Data analysis
│   └── IDEA.md                        # Project specification
│
├── 🤖 ML Pipeline (Phase 1)
│   ├── models/                        # 8 trained models
│   ├── training/                      # Training scripts
│   ├── deployment/                    # Vertex AI deployment
│   └── data_processing/               # ETL pipeline
│
├── 🌐 Backend API (Phase 2B)
│   ├── main.py                        # FastAPI server
│   ├── routers/                       # API endpoints
│   ├── services/                      # Model loader
│   └── models/                        # Pydantic schemas
│
├── 🤖 Agents (Phase 2C)
│   ├── specialized/                   # 4 specialized agents
│   │   ├── chief_agent.py            # Orchestrator
│   │   ├── fuel_agent.py             # Fuel specialist
│   │   ├── tire_agent.py             # Tire specialist
│   │   └── telemetry_agent.py        # Data specialist
│   ├── base/                          # Base agent framework
│   └── tools/                         # API client
│
├── 📡 Streaming (Phase 2D)
│   └── simulator/                     # Telemetry generator
│       └── telemetry_simulator.py    # 20 Hz stream
│
├── 🧪 Tests
│   └── test_end_to_end.py            # Comprehensive E2E test
│
└── 📊 Data
    └── [6 track datasets]             # Raw telemetry data
```

## 📊 Dataset Overview

- **6 Professional Tracks** across North America
- **~12 Races** (2 per track)
- **13 Telemetry Parameters** including:
  - Speed, RPM, Gear
  - Throttle & Brake Pressure
  - Acceleration (Longitudinal & Lateral)
  - Steering Angle
  - GPS Coordinates
  - Distance from Start/Finish
- **~23 Million** telemetry data points
- **~2+ GB** of uncompressed racing data
- **Official Race Results**, Lap Times, Weather Data

## 🎯 Competition Categories

1. **Driver Training & Insights**: Tools to help drivers improve
2. **Pre-Event Prediction**: Forecast race outcomes and performance
3. **Post-Event Analysis**: Deep dive into race results
4. **Real-Time Analytics**: Simulate race-day decision making
5. **Wildcard**: Creative and out-of-the-box ideas

## 💡 Top Project Ideas

### 🥇 Recommended: RaceCoach AI
**Category**: Driver Training & Insights

An AI-powered coaching system that analyzes driver telemetry to identify improvement areas with actionable, corner-by-corner feedback.

**Key Features**: Performance gap analysis, driving style metrics, visual overlays, progress tracking

---

### 🥈 Advanced: PitStop Prophet
**Category**: Real-Time Analytics

A real-time race strategy optimizer that predicts optimal pit stops, tire management, and race pace decisions.

**Key Features**: Tire degradation modeling, pit strategy simulation, weather impact, competitor tracking

---

### 🥉 Visual: Race Replay 3D
**Category**: Wildcard / Post-Event Analysis

An immersive 3D race visualization that recreates races from telemetry data with multiple camera angles and analytics overlays.

**Key Features**: 3D rendering, multi-vehicle tracking, telemetry overlay, highlight detection

---

See [IDEA.md](IDEA.md) for all 5 detailed project proposals with technical specifications, implementation roadmaps, and tech stacks.

## 📖 Documentation

### [DATAEXPLORE.md](DATAEXPLORE.md)
Comprehensive analysis of the dataset including:
- Data structure and file types
- Telemetry parameters explained
- Data quality assessment
- Usage recommendations
- Technical specifications

### [IDEA.md](IDEA.md)
Detailed project proposals including:
- 5 innovative project ideas
- Technical approaches and algorithms
- Implementation roadmaps
- Tech stack recommendations
- Comparison matrix and selection guide

## 🚀 Getting Started with Your Project

1. **Choose a Project**: Review [IDEA.md](IDEA.md) and select based on your skills and interests

2. **Set Up Environment**:
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies (example)
pip install pandas numpy scikit-learn matplotlib seaborn plotly
```

3. **Explore the Data**:
```python
import pandas as pd

# Load telemetry data
telemetry = pd.read_csv('data/barber-motorsports-park/barber/R1_barber_telemetry_data.csv')

# Load race results
results = pd.read_csv('data/barber-motorsports-park/barber/03_Provisional Results_Race 1_Anonymized.CSV', 
                      sep=';')

# Start exploring!
print(telemetry.head())
print(results.head())
```

4. **Build Your Solution**: Follow the implementation roadmap in [IDEA.md](IDEA.md)

5. **Prepare Submission**:
   - Create demo video (~3 minutes)
   - Write project description
   - Deploy if applicable
   - Submit to Devpost

## 📊 Key Telemetry Parameters

| Parameter | Description | Unit |
|-----------|-------------|------|
| Speed | Vehicle speed | km/h |
| Gear | Current gear | 1-6 |
| nmot | Engine RPM | RPM |
| aps | Accelerator pedal position | % |
| pbrake_f / pbrake_r | Brake pressure (front/rear) | bar |
| accx_can | Longitudinal acceleration | G |
| accy_can | Lateral acceleration | G |
| Steering_Angle | Steering wheel angle | degrees |
| VBOX_Long_Minutes | GPS longitude | degrees |
| VBOX_Lat_Min | GPS latitude | degrees |
| Laptrigger_lapdist_dls | Distance from start/finish | meters |

## 🏆 Competition Details

- **Prize Pool**: $20,000 in cash
- **Deadline**: November 24, 2025 @ 8:00pm EST
- **Participants**: 330+ registered
- **Format**: Online, Public
- **Host**: Toyota Gazoo Racing North America

## 📋 Submission Requirements

✅ **Category selection** from 5 options
✅ **Dataset(s) used** documentation
✅ **Text description** of your project
✅ **Published project** for judges to test
✅ **Code repository URL** (share with testing@devpost.com and trd.hackathon@toyota.com)
✅ **Demo video** (~3 minutes)

## 🔗 Resources

- **Dataset Source**: [https://trddev.com/hackathon-2025/](https://trddev.com/hackathon-2025/)
- **Competition Page**: Devpost (link in hackathon description)
- **Official Timing**: SRO - TGRNA GR CUP NORTH AMERICA (2025)

## 📝 Data Notes

- **Format**: CSV files with mixed delimiters (comma and semicolon)
- **Anonymization**: Driver names removed
- **Missing Data**: Sebring Race 1 has no telemetry data
- **Time Format**: ISO 8601 timestamps
- **File Size**: ~885 MB compressed, ~2+ GB uncompressed

## 🛠️ Recommended Tech Stack

### Data Analysis
- **Python**: Pandas, NumPy, SciPy
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Machine Learning**: Scikit-learn, TensorFlow, PyTorch

### Web Development
- **Frontend**: React, Vue.js, or Angular
- **Backend**: Flask, FastAPI, Node.js
- **Database**: PostgreSQL, MongoDB

### Specialized Libraries
- **3D Graphics**: Three.js, Unity, Unreal Engine
- **Maps**: Mapbox GL, Leaflet
- **Real-time**: WebSockets, Socket.io

## 💻 Example Code Snippets

### Loading Telemetry Data
```python
import pandas as pd

# Load telemetry
df = pd.read_csv('data/barber-motorsports-park/barber/R1_barber_telemetry_data.csv')

# Filter for specific vehicle
vehicle_data = df[df['vehicle_number'] == 13]

# Pivot to wide format for time-series
telemetry_wide = vehicle_data.pivot_table(
    index='timestamp',
    columns='telemetry_name',
    values='telemetry_value'
)
```

### Analyzing Lap Times
```python
# Load lap times
laps = pd.read_csv('data/barber-motorsports-park/barber/R1_barber_lap_time.csv')

# Calculate average lap time per vehicle
avg_laps = laps.groupby('vehicle_number')['lap_time'].mean()

# Find fastest lap
fastest = laps.loc[laps['lap_time'].idxmin()]
```

### Creating Track Map
```python
import matplotlib.pyplot as plt

# Extract GPS coordinates for a vehicle
gps_data = telemetry_wide[['VBOX_Lat_Min', 'VBOX_Long_Minutes']].dropna()

# Plot track layout
plt.figure(figsize=(12, 8))
plt.plot(gps_data['VBOX_Long_Minutes'], gps_data['VBOX_Lat_Min'])
plt.title('Track Layout from GPS Data')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.axis('equal')
plt.show()
```

## 🤝 Tips for Success

1. **Start Simple**: Get basic data loading and visualization working first
2. **Focus on Value**: Choose a problem that matters to drivers/teams/fans
3. **Tell a Story**: Your demo should have a clear narrative
4. **Visualize Well**: Racing data is inherently visual - make it compelling
5. **Test Early**: Make sure your code works with real data
6. **Document**: Clear README and comments for judges
7. **Demo Video**: This is crucial - make it professional and engaging

## 📅 Timeline

- **Now - Nov 10**: Data exploration and project selection
- **Nov 10-20**: Core development
- **Nov 20-23**: Polish, testing, and demo creation
- **Nov 24**: Final submission by 8:00pm EST

## 🎬 Creating Your Demo Video

Your 3-minute video should include:
1. **Problem Statement** (30s): What problem are you solving?
2. **Solution Overview** (45s): What did you build?
3. **Live Demo** (90s): Show your tool in action with real data
4. **Impact** (15s): Why does this matter?

## 📧 Contact

For questions about the competition:
- Testing: testing@devpost.com
- Toyota: trd.hackathon@toyota.com

## ⚖️ License

This is competition data. Please review the hackathon rules for usage terms.

---

**Built with ❤️ for Hack the Track 2025**

*Good luck and may the fastest code win! 🏁*

