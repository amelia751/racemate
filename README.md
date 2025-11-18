# RaceMate - AI-Powered Race Strategy Platform

> **Real-Time ML Intelligence for Professional Racing**

**Hack the Track 2025** - Toyota GR Cup Series

A production-ready, real-time race strategy platform featuring 8 specialized ML models, event-driven recommendations, and a Red Bull F1-inspired dashboard.

## 🎉 **LIVE DEMO** 🏁

**🌐 Live Application**: [racemate.site](https://racemate.site)  
**📚 Technical Documentation**: [racemate.site/documentation](https://racemate.site/documentation)  
**🔧 Backend API**: [Cloud Run Deployment](https://backend-api-533427455134.us-central1.run.app)

---

## 🚀 Quick Start (Testing the Live Site)

1. **Go to**: [racemate.site](https://racemate.site)
2. **Click**: "START STREAMING" button
3. **Watch**: Real-time telemetry + AI recommendations

**Note**: First load may take 15-30 seconds (cold start). If no recommendations appear, refresh and try again.

---

## 🏁 What is RaceMate?

RaceMate is a **real-time AI race strategist** that processes live telemetry through 8 specialized machine learning models and delivers actionable pit strategy recommendations in under 75ms.

### Core Capabilities

**1. Real-Time Telemetry Processing (100 Hz)**
- Ingests speed, RPM, throttle, brake pressure, fuel level, tire temps, G-forces
- Processes 100 data points per second through parallel ML inference
- Detects 6+ critical race events (fuel spikes, tire degradation, anomalies, FCY risk)

**2. 8-Model Ensemble Intelligence**
- **Fuel Consumption** (GradientBoosting): ±5mL/lap accuracy, R² = 0.89
- **Lap Time Transformer**: Predicts next lap time, R² = 0.94
- **Tire Degradation** (CNN-LSTM): 4-corner prediction, 94% accuracy
- **FCY Hazard** (Random Forest): Safety car probability, 89% accuracy
- **Pit Loss** (XGBoost): Circuit-specific timing, R² = 0.91
- **Anomaly Detector** (Isolation Forest): Failure prediction, 87% precision
- **Driver Embedding** (Autoencoder): Driving style clustering, 95% retention
- **Traffic GNN**: Overtaking probability, 82% accuracy

**3. Event-Driven Recommendations**
- Smart filtering (only 1+ CRITICAL or 2+ HIGH severity events)
- Human-readable, professional race strategy recommendations
- Instant formatting (no LLM delays)

**4. Professional Racing Dashboard**
- Real-time telemetry visualization (lap time, speed, RPM, G-forces)
- Fuel & tire temperature with color-coded alerts
- Brake system health monitoring
- Live ML recommendations with severity levels

---

## 📊 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FRONTEND (Next.js)                       │
│  • Real-time telemetry simulation (6 scenarios)             │
│  • Red Bull-inspired dashboard (Recharts + Framer Motion)   │
│  • Voice strategist chat interface                          │
└─────────────────────┬───────────────────────────────────────┘
                      │ HTTP POST /api/telemetry/stream
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              BACKEND API (FastAPI + Python)                 │
│  • RealtimePredictor: orchestrates all 8 models             │
│  • PredictionState: tracks fuel, tire wear, lap history     │
│  • Event detection: LOW_FUEL, TIRE_CRITICAL, ANOMALY, etc.  │
└─────────────────────┬───────────────────────────────────────┘
                      │ Parallel inference
                      ▼
┌─────────────────────────────────────────────────────────────┐
│            8 SPECIALIZED ML MODELS (PyTorch + sklearn)      │
│  1. Fuel (GradientBoosting)  5. Pit Loss (XGBoost)         │
│  2. Lap Time (Transformer)   6. Anomaly (IsolationForest)  │
│  3. Tire (CNN-LSTM)          7. Driver (Autoencoder)       │
│  4. FCY (RandomForest)       8. Traffic (GNN)              │
└─────────────────────┬───────────────────────────────────────┘
                      │ Predictions
                      ▼
┌─────────────────────────────────────────────────────────────┐
│          STRATEGY FORMATTER (Custom Python Service)         │
│  • Converts ML outputs → human-readable recommendations     │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Repository Structure

```
hack-the-track/
├── 📖 Documentation
│   ├── README.md                      # This file
│   └── DATAEXPLORE.md                 # Dataset analysis
│
├── 🤖 ML Pipeline
│   ├── models/                        # 8 PyTorch/sklearn models
│   ├── training/                      # Training scripts
│   │   ├── train_fuel_consumption.py
│   │   ├── train_lap_time.py
│   │   ├── train_tire.py
│   │   ├── train_fcy.py
│   │   ├── train_pit_loss.py
│   │   ├── train_anomaly.py
│   │   ├── train_driver_embed.py
│   │   └── train_traffic.py
│   ├── validation/                    # Model validation
│   └── config/                        # Training configs
│
├── 🌐 Backend API (FastAPI)
│   ├── main.py                        # FastAPI server
│   ├── routers/
│   │   ├── predict.py                 # ML prediction endpoints
│   │   ├── realtime.py                # Real-time processing
│   │   └── health.py                  # Health checks
│   ├── services/
│   │   ├── model_loader.py            # Load models from GCS
│   │   ├── realtime_predictor.py      # Orchestrate all 8 models
│   │   └── strategy_formatter.py      # Format recommendations
│   ├── models/
│   │   └── schemas.py                 # Pydantic schemas
│   ├── config/
│   │   └── settings.py                # Configuration
│   ├── Dockerfile                     # Cloud Run container
│   └── requirements.txt
│
├── 🎨 Frontend (Next.js 14)
│   ├── app/
│   │   ├── page.tsx                   # Main dashboard
│   │   ├── documentation/
│   │   │   └── page.tsx               # Technical docs
│   │   └── api/
│   │       └── telemetry/
│   │           └── stream/
│   │               └── route.ts       # Proxy to backend
│   ├── components/
│   │   ├── racing/
│   │   │   ├── RacingDashboard.tsx    # Main layout
│   │   │   ├── StreamingControls.tsx  # Start/stop streaming
│   │   │   ├── TelemetryCharts.tsx    # Speed/RPM charts
│   │   │   ├── EnhancedVisualizations.tsx # Fuel/tire/brake
│   │   │   ├── HeroMetrics.tsx        # Current values
│   │   │   ├── LapTimeDisplay.tsx     # Lap info
│   │   │   └── ...
│   │   ├── VoiceStrategist.tsx        # AI recommendations
│   │   ├── documentation/             # Docs components
│   │   │   ├── OverviewSection.tsx
│   │   │   ├── FuelSection.tsx
│   │   │   ├── LaptimeSection.tsx
│   │   │   ├── TireSection.tsx
│   │   │   ├── FCYSection.tsx
│   │   │   ├── PitSection.tsx
│   │   │   ├── AnomalySection.tsx
│   │   │   ├── DriverSection.tsx
│   │   │   ├── TrafficSection.tsx
│   │   │   ├── ArchitectureSection.tsx
│   │   │   └── shared/
│   │   │       ├── CodeBlock.tsx
│   │   │       ├── MetricCard.tsx
│   │   │       ├── DocumentationHeader.tsx
│   │   │       └── DocumentationTabs.tsx
│   │   └── ui/                        # shadcn/ui components
│   ├── lib/
│   │   └── store.ts                   # Zustand state
│   └── package.json
│
└── 📊 Data (2+ GB)
    ├── barber-motorsports-park/
    ├── circuit-of-the-americas/
    ├── road-america/
    ├── sebring/
    ├── sonoma/
    └── virginia-international-raceway/
```

---

## 📊 Dataset Overview

**Toyota GR Cup Racing Telemetry Dataset (2025 Season)**

- **Source**: [Hack the Track 2025 - Toyota GR Racing](https://trddev.com/hackathon-2025/)
- **6 Professional Circuits**: COTA, Barber, Road America, Sebring, Sonoma, VIR
- **12 Races** (2 per circuit)
- **200,000+ telemetry data points** at 100 Hz sampling rate
- **13 Telemetry Parameters**: Speed, RPM, throttle, brake pressure, G-forces, GPS, steering
- **120+ Drivers** with performance data
- **Weather Data**: Temperature, humidity, wind, rain indicators
- **Race Results**: Lap times, positions, fastest laps, pit stops

**See [DATAEXPLORE.md](DATAEXPLORE.md) for comprehensive dataset documentation.**

---

## 🛠️ Technology Stack

### Frontend
- **Next.js 14** (App Router, Server Components)
- **React** with TypeScript
- **Tailwind CSS v4** (Custom theme configuration)
- **shadcn/ui** (Beautiful, accessible components)
- **Zustand** (Lightweight state management)
- **Framer Motion** (Smooth animations)
- **Recharts** (Real-time data visualization)
- **Deployed on Vercel**

### Backend
- **FastAPI** (High-performance Python API)
- **Pydantic** (Data validation)
- **uvicorn** (ASGI server)
- **Google Cloud Storage** (Model storage)
- **Deployed on Google Cloud Run**

### Machine Learning
- **PyTorch** (Transformer, CNN-LSTM, GNN, Autoencoder)
- **scikit-learn** (GradientBoosting, RandomForest, IsolationForest)
- **XGBoost** (Pit loss prediction)
- **pandas + numpy** (Data processing)
- **Trained on Google Colab** (16GB GPU, 12 hours)

---

## 🚀 Local Development Setup

### Prerequisites
- Python 3.10+
- Node.js 18+
- Google Cloud Storage credentials (for model loading)

### Backend Setup

```bash
cd backend-api

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env.local
# Edit .env.local with your GCS credentials

# Start the server
python main.py
# Server runs on http://localhost:8005
```

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Set up environment variables
cp .env.example .env.local
# Edit .env.local with backend URL

# Start development server
npm run dev
# Frontend runs on http://localhost:3005
```

---

## 🎯 Key Features

### 1. Event-Driven Intelligence
- **LOW_FUEL**: Triggers when fuel < 6L remaining
- **FUEL_CONSUMPTION_SPIKE**: Detects +10% consumption increase
- **PIT_WINDOW_CLOSING**: Warns when optimal pit window is closing
- **TIRE_CRITICAL**: Alerts when tire temp > 100°C
- **ANOMALY_DETECTED**: Identifies unusual telemetry patterns
- **HIGH_SPEED**: Monitors extreme speeds (195+ km/h)

### 2. Smart Filtering
Only displays recommendations for:
- 1+ CRITICAL severity events, OR
- 2+ HIGH severity events

This prevents alert fatigue while ensuring critical issues are highlighted.

### 3. Real-Time Performance
- **Total Inference**: 75ms for all 8 models
- **Telemetry Processing**: 100 Hz (100 data points/second)
- **Frontend Updates**: 500ms for smooth visualization
- **Backend Response**: <100ms API latency

### 4. Production-Grade Deployment
- **Frontend**: Vercel (global CDN, automatic HTTPS)
- **Backend**: Google Cloud Run (auto-scaling, serverless)
- **Models**: Google Cloud Storage (versioned, cached)
- **Monitoring**: Built-in health checks and logging

---

## 🏆 Competition Details

- **Event**: Hack the Track 2025
- **Host**: Toyota Gazoo Racing North America
- **Category**: Real-Time Analytics
- **Dataset**: Toyota GR Cup Racing Telemetry (2025 Season)
- **Prize Pool**: $20,000
- **Deadline**: November 24, 2025 @ 8:00pm EST

---

## 🎬 Demo Video Highlights

1. **Live Dashboard**: Real-time telemetry streaming with 8 charts
2. **AI Recommendations**: Event-driven strategy alerts
3. **Technical Documentation**: Comprehensive ML model explanations
4. **Performance**: Sub-100ms latency with 8 models running in parallel

---

## 🤝 Acknowledgments

- **Toyota Gazoo Racing North America**: For providing professional racing telemetry data
- **SRO**: For official timing and scoring data
- **Hack the Track 2025**: For organizing this incredible competition

---

## 📧 Contact

For questions about this project:
- **GitHub**: [Your GitHub Profile]
- **Email**: [Your Email]

For competition questions:
- **Testing**: testing@devpost.com
- **Toyota**: trd.hackathon@toyota.com

---

## ⚖️ License

This project was built for Hack the Track 2025. Dataset usage subject to hackathon rules.

---

**Built with ❤️ for Hack the Track 2025**

*Real-Time Intelligence. Professional Results. 🏁*
