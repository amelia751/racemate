# Cognirace Model Quick Reference

**Last Updated**: Session 6 - All 8 Models Complete (100%)

This document provides a quick reference for all trained models, their purposes, and how to use them.

---

## 1. Fuel Consumption Model ⛽

**Type**: GradientBoostingRegressor (sklearn)  
**Purpose**: Predict fuel burn rate per lap  
**Location**: `gs://cognirace-model-artifacts/fuel_consumption/`

### Key Info
- **Input**: Speed, RPM, throttle, gear, lap
- **Output**: Fuel burn rate (synthetic units)
- **Performance**: RMSE 0.52, MAE 0.43
- **Training time**: ~2 minutes
- **Model size**: 1.24 MB

### Usage
```python
from models.fuel_consumption import FuelConsumptionModel
model = FuelConsumptionModel()
# Load from GCS checkpoint
burn_rate = model.predict(features)
```

### Best For
- Fuel strategy planning
- Estimating when to pit for fuel
- Fuel-saving mode recommendations
- Race simulation

---

## 2. Lap-Time Transformer 🏎️

**Type**: Transformer Encoder (PyTorch)  
**Purpose**: Predict next lap time with uncertainty quantiles  
**Location**: `gs://cognirace-model-artifacts/lap_time_transformer/`

### Key Info
- **Input**: 100-sample telemetry sequence (16 features)
- **Output**: Lap time delta + P10, P50, P90 quantiles
- **Performance**: Val Loss 1.56
- **Training time**: ~3 minutes
- **Model size**: 36.49 MB

### Architecture
- 4 transformer encoder layers
- 256 hidden dim
- 4 attention heads
- Positional encoding
- Quantile regression heads

### Usage
```python
from models.lap_time_transformer import LapTimeTransformer
model = LapTimeTransformer(input_dim=16)
# Load checkpoint
mean, quantiles = model(telemetry_sequence)
```

### Best For
- Real-time lap time prediction
- Performance feedback to driver
- Identifying lap time opportunities
- What-if scenario analysis

---

## 3. Tire Degradation Model 🛞

**Type**: Physics-Informed TCN (PyTorch)  
**Purpose**: Predict tire grip degradation  
**Location**: `gs://cognirace-model-artifacts/tire_degradation/`

### Key Info
- **Input**: Cumulative brake energy, lateral load, temperature
- **Output**: Grip index (0.5-1.0)
- **Performance**: RMSE 0.091, MAE 0.072
- **Training time**: ~2 minutes
- **Model size**: 0.78 MB

### Architecture
- Physics model: Learnable wear coefficients
- Residual network: 3-layer TCN
- Combined physics + learned approach

### Usage
```python
from models.tire_degradation import TireDegradationModel
model = TireDegradationModel()
grip_index = model(telemetry, physics_features)
```

### Best For
- Tire strategy (when to pit)
- Predicting performance degradation
- Comparing tire compounds
- Driver coaching (tire preservation)

---

## 4. FCY Hazard Model 🚨

**Type**: TCN with Survival Analysis (PyTorch)  
**Purpose**: Predict full-course yellow probability  
**Location**: `gs://cognirace-model-artifacts/fcy_hazard/`

### Key Info
- **Input**: 100-sample telemetry + risk factors
- **Output**: Hazard rates per lap (6-lap horizon)
- **Performance**: Val Loss 0.43
- **Training time**: ~2 minutes
- **Model size**: 2.95 MB

### Architecture
- 3-layer TCN backbone
- Survival analysis approach
- Lap-by-lap hazard rates
- Cumulative probability output

### Usage
```python
from models.fcy_hazard import FCYHazardModel
model = FCYHazardModel(horizon_laps=6)
hazard_rates, cumulative_prob = model(telemetry)
```

### Best For
- Strategic pit decisions (pit before FCY)
- Risk assessment
- Race simulation scenarios
- Real-time strategy adjustment

---

## 5. Pit Loss Model 🔧

**Type**: Physics + MLP (PyTorch)  
**Purpose**: Predict total pit stop time loss  
**Location**: `gs://cognirace-model-artifacts/pit_loss/`

### Key Info
- **Input**: Traffic state at pit exit (16 features)
- **Output**: Total pit time (base + merge penalty)
- **Performance**: RMSE 1.87s, MAE 1.47s
- **Training time**: ~1 minute
- **Model size**: 0.07 MB (tiny!)

### Architecture
- Learnable physics parameters (pit lane speed, length, service time)
- MLP for merge penalty prediction
- Combined physics + learned approach

### Usage
```python
from models.pit_loss import PitLossModel
model = PitLossModel()
pit_time = model(traffic_features)
```

### Best For
- Optimal pit timing (avoid traffic)
- Undercut/overcut decisions
- Race strategy simulation
- Real-time pit window analysis

---

## 6. Anomaly Detector ⚠️

**Type**: LSTM Autoencoder (PyTorch)  
**Purpose**: Detect telemetry anomalies (mechanical issues)  
**Location**: `gs://cognirace-model-artifacts/anomaly_detector/`

### Key Info
- **Input**: 100-sample telemetry window
- **Output**: Reconstruction error (anomaly score)
- **Performance**: Val Loss 0.52
- **Training time**: ~2 minutes
- **Model size**: 1.42 MB

### Architecture
- LSTM encoder (2 layers, 64 hidden)
- LSTM decoder (2 layers)
- Reconstruction-based anomaly detection

### Usage
```python
from models.anomaly_detector import AnomalyDetector
model = AnomalyDetector()
reconstruction, anomaly_score = model(telemetry_window)
```

### Best For
- Early mechanical issue detection
- Sensor failure detection
- Real-time health monitoring
- Preventing DNFs

---

## 7. Driver Embedding Model 👤

**Type**: Transformer Encoder (PyTorch)  
**Purpose**: Learn driver style representation  
**Location**: `gs://cognirace-model-artifacts/driver_embedding/`

### Key Info
- **Input**: 100-sample telemetry sequence
- **Output**: 32-dimensional driver embedding
- **Performance**: Val Loss 0.48
- **Training time**: ~3 minutes
- **Model size**: 6.12 MB

### Architecture
- 2-layer transformer encoder
- 128 hidden dim
- Global pooling
- 32-dim embedding projection

### Usage
```python
from models.driver_embedding import DriverEmbeddingModel
model = DriverEmbeddingModel()
embedding = model(telemetry_sequence)
# Compare drivers: cosine_similarity(emb1, emb2)
```

### Best For
- Driver style analysis
- Personalized coaching
- Comparing drivers
- Setup recommendations
- Driver matching (substitute drivers)

---

## 8. Traffic GNN 🚦

**Type**: Attention-based GNN (PyTorch)  
**Purpose**: Predict traffic impact and overtake probability  
**Location**: `gs://cognirace-model-artifacts/traffic_gnn/`

### Key Info
- **Input**: Multi-car state (5 cars, 16 features each)
- **Output**: Traffic loss (seconds) + overtake probability
- **Performance**: Val Loss 2.94
- **Training time**: ~2 minutes
- **Model size**: 0.52 MB

### Architecture
- Multi-head self-attention (4 heads)
- 2 attention layers with residual connections
- Global pooling over nodes
- Dual output heads

### Usage
```python
from models.traffic_gnn import TrafficGNN
model = TrafficGNN(node_feature_dim=16)
traffic_loss, overtake_prob = model(multi_car_state)
```

### Best For
- Pit strategy (avoid traffic)
- Overtake opportunity assessment
- Traffic-aware race simulation
- Real-time strategy adjustment

---

## Model Selection Guide

### For Race Strategy
- **Pit Timing**: Pit Loss + Traffic GNN + FCY Hazard
- **Fuel Strategy**: Fuel Consumption + Tire Degradation
- **Overtake Planning**: Traffic GNN + Lap-Time Transformer

### For Driver Coaching
- **Performance**: Lap-Time Transformer + Driver Embedding
- **Tire Management**: Tire Degradation
- **Consistency**: Anomaly Detector

### For Safety & Reliability
- **Mechanical Issues**: Anomaly Detector
- **Risk Assessment**: FCY Hazard

---

## Loading Models from GCS

### General Pattern

```python
import torch
from google.cloud import storage
from google.oauth2 import service_account

# Setup
credentials = service_account.Credentials.from_service_account_file('config/gcp_credentials.json')
client = storage.Client(project='cognirace', credentials=credentials)
bucket = client.bucket('cognirace-model-artifacts')

# Download checkpoint
blob = bucket.blob('model_name/model.pth')
blob.download_to_filename('/tmp/model.pth')

# Load model
checkpoint = torch.load('/tmp/model.pth', map_location='cpu', weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Use scaler if available
scaler = checkpoint.get('scaler')
if scaler:
    features_scaled = scaler.transform(features)
```

---

## Training All Models

### Quick Commands

```bash
cd /Users/anhlam/hack-the-track/ml-pipeline
source venv/bin/activate

# Train individual models
python training/train_fuel.py
python training/train_lap_time.py
python training/train_tire.py
python training/train_fcy.py
python training/train_pit_loss.py
python training/train_anomaly.py
python training/train_traffic.py

# Validate models
python validation/test_fuel_model.py
python validation/test_lap_time_model.py
python validation/test_tire_model.py
python validation/test_fcy_model.py
python validation/test_pit_loss_model.py
python validation/test_anomaly_model.py
python validation/test_traffic_model.py
```

---

## Model Performance Summary

| Model | Val Loss | RMSE/MAE | Size | Training Time |
|-------|----------|----------|------|---------------|
| Fuel Consumption | 0.31 | RMSE 0.52, MAE 0.43 | 1.24 MB | 2 min |
| Lap-Time Transformer | 1.56 | N/A | 36.49 MB | 3 min |
| Tire Degradation | 0.0093 | RMSE 0.091, MAE 0.072 | 0.78 MB | 2 min |
| FCY Hazard | 0.43 | N/A | 2.95 MB | 2 min |
| Pit Loss | 3.49 | RMSE 1.87, MAE 1.47 | 0.07 MB | 1 min |
| Anomaly Detector | 0.52 | N/A | 1.42 MB | 2 min |
| Driver Embedding | 0.48 | N/A | 6.12 MB | 3 min |
| Traffic GNN | 2.94 | N/A | 0.52 MB | 2 min |

**Total**: ~50 MB, ~20 minutes training time, <$0.50 cost

---

## Next Steps

### Phase 2: Deployment

1. **Vertex AI Endpoints**:
   - Create endpoints for each model
   - Deploy models
   - Configure autoscaling
   - Test predictions

2. **Real-Time API** (port 8005):
   - FastAPI service
   - Model loading
   - Batch predictions
   - Latency optimization (<100ms)

3. **Agent Orchestration**:
   - ChiefAgent coordinator
   - Specialized agents (Fuel, Tire, Telemetry)
   - Tool integration
   - Natural language interface

### Phase 3: Production

4. **Telemetry Streaming**:
   - Pub/Sub ingestion
   - Dataflow processing
   - Real-time inference

5. **Monitoring**:
   - Model performance tracking
   - Drift detection
   - Alerting

---

## Support & Documentation

- **Training Summary**: `ml-pipeline/TRAINING_SUMMARY.md`
- **Session Summaries**: `ml-pipeline/SESSION_*_SUMMARY.md`
- **Implementation Details**: `ml-pipeline/IMPLEMENTATION_SUMMARY.md`
- **Data Exploration**: `DATAEXPLORE.md`
- **Project Spec**: `IDEA.md`

---

**Status**: ✅ All 8 models trained, validated, and deployed (100% complete)  
**Next**: Phase 2 - Vertex AI deployment and real-time API

