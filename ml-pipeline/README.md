# Cognirace ML Pipeline

## Overview

Complete machine learning infrastructure for the Cognirace race engineering copilot. This pipeline processes telemetry data from 6 professional racing tracks, engineers features, trains 8 production models, and deploys them to Vertex AI.

## Project Status

### ✅ Completed
- [x] GCP Infrastructure (5 buckets created)
- [x] Service Account Authentication
- [x] Data Processing Pipeline (CSV parser, feature engineering, GCS uploader)
- [x] All 8 ML Models Implemented
- [x] Model Testing and Validation

### ⏳ In Progress
- [ ] Training scripts for all models
- [ ] Deployment automation
- [ ] Model monitoring and drift detection

## Quick Start

### 1. Setup

```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies (if not already done)
pip install -r requirements.txt
```

### 2. Process Data

```bash
# Process all tracks (30-60 minutes)
python data_processing/run_pipeline.py

# Or process single track for testing
python data_processing/csv_parser.py
```

### 3. Test Models

```bash
# Test individual models
python models/lap_time_transformer.py
python models/tire_degradation.py
python models/fuel_consumption.py
```

## Data Pipeline

### Input Data
- **Source**: 6 racing tracks × 2 races = ~23M telemetry rows
- **Format**: Long-format CSV (one row per signal per timestamp)
- **Signals**: 13 telemetry parameters (speed, RPM, brake, GPS, etc.)

### Processing Steps

1. **CSV Parsing** (`data_processing/csv_parser.py`)
   - Load raw telemetry CSVs
   - Pivot from long to wide format
   - Validate data quality

2. **Feature Engineering** (`data_processing/feature_engineering.py`)
   - Temporal features (rolling windows, EWMA slopes)
   - Energy metrics (brake energy, lateral load)
   - Throttle discipline (variance, smoothness)
   - Steering metrics (jerk, smoothness)
   - Track position (micro-sectors)
   - Tire stress proxies

3. **Upload to GCS** (`data_processing/upload_to_gcs.py`)
   - Save to Cloud Storage
   - Create train/test splits (80/20)
   - Compress with Parquet/Snappy

### Output Data
- **Features**: 40+ engineered features per row
- **Format**: Parquet (compressed)
- **Location**: GCS bucket `cognirace-processed-features`
- **Size**: ~2GB processed data

## ML Models

### 1. Lap-Time Transformer
**File**: `models/lap_time_transformer.py`
**Architecture**: 4-layer Transformer (256 hidden, 4 heads)
**Purpose**: Predict next lap time delta with quantiles
**Parameters**: 3.2M
**Inputs**: 200×16 sequence (10s at 20Hz)
**Outputs**: Mean + 3 quantiles (0.1, 0.5, 0.9)

### 2. Tire Degradation Model
**File**: `models/tire_degradation.py`
**Architecture**: Physics-informed (linear) + TCN residual (3 layers, 64 channels)
**Purpose**: Predict tire grip index and wear trajectory
**Parameters**: ~150K
**Physics**: Brake energy, lateral load, temperature effects
**Outputs**: Grip index (0.5-1.0)

### 3. Fuel Consumption Model
**File**: `models/fuel_consumption.py`
**Architecture**: XGBoost (200 estimators, depth 6)
**Purpose**: Predict fuel burn rate per lap
**Features**: RPM, throttle, gear, speed, lap
**Outputs**: Liters per lap

### 4. Traffic GNN
**File**: `models/traffic_gnn.py`
**Architecture**: GraphSAGE (2 layers, 64 hidden)
**Purpose**: Predict traffic loss and overtake probability
**Graph**: Nodes = cars, Edges = proximity
**Outputs**: Time loss (ms), overtake probability

### 5. FCY Hazard Model
**File**: `models/fcy_hazard.py`
**Architecture**: TCN (3 layers, 128 channels) + survival analysis
**Purpose**: Forecast caution/yellow flag probability
**Parameters**: 256K
**Outputs**: Hazard rates per lap (6-lap horizon)

### 6. Pit Loss Model
**File**: `models/pit_loss.py`
**Architecture**: Physics-based + MLP for merge penalty
**Purpose**: Estimate pit stop time loss
**Components**: Lane time + service time + traffic merge penalty
**Outputs**: Total pit loss (seconds)

### 7. Anomaly Detector
**File**: `models/anomaly_detector.py`
**Architecture**: LSTM Autoencoder (2 layers, 64 hidden)
**Purpose**: Detect unusual sensor patterns
**Method**: Reconstruction error
**Outputs**: Anomaly scores

### 8. Driver Embedding
**File**: `models/driver_embedding.py`
**Architecture**: Transformer (2 layers, 128 hidden) with CLS token
**Purpose**: Learn personalized driver representations
**Outputs**: 32-dim embedding + auxiliary predictions

## Configuration

All configuration is in `.env.local` (not committed):

```env
# GCP
GCP_PROJECT_ID=cognirace
GCP_REGION=us-central1

# Buckets
GCS_BUCKET_PROCESSED=cognirace-processed-features
GCS_BUCKET_MODELS=cognirace-model-artifacts

# Vertex AI
VERTEX_AI_LOCATION=us-central1
VERTEX_AI_STAGING_BUCKET=gs://cognirace-vertex-staging

# Data
LOCAL_DATA_PATH=/Users/anhlam/hack-the-track/data
PROCESSED_DATA_PATH=/Users/anhlam/hack-the-track/ml-pipeline/processed_data
```

## GCP Resources

### Created Buckets
1. `cognirace-raw-telemetry` - Raw CSV storage
2. `cognirace-processed-features` - Engineered features
3. `cognirace-model-artifacts` - Trained models
4. `cognirace-training-results` - Training logs/metrics
5. `cognirace-vertex-staging` - Vertex AI staging

### Required APIs
- Vertex AI API
- Cloud Storage API (enabled)
- Cloud Build API (optional)

### Service Account
- Email: `cognirace@cognirace.iam.gserviceaccount.com`
- Roles: Storage Admin, Vertex AI User

## Development

### Adding New Features

Edit `data_processing/feature_engineering.py`:

```python
def add_custom_feature(self, df: pd.DataFrame) -> pd.DataFrame:
    """Add your custom feature"""
    df['custom_feature'] = ...
    return df

# Add to engineer_all_features()
df = self.add_custom_feature(df)
```

### Adding New Models

1. Create model file in `models/`
2. Inherit from `nn.Module` (PyTorch) or appropriate base
3. Implement `forward()` and `__init__()`
4. Add test in `if __name__ == "__main__":`
5. Create training script in `training/`

### Testing

```bash
# Test data pipeline
python data_processing/csv_parser.py
python data_processing/feature_engineering.py
python data_processing/upload_to_gcs.py

# Test models
python models/lap_time_transformer.py

# Test GCP connectivity
python gcp_setup/create_buckets.py
python gcp_setup/setup_vertex.py
```

## Architecture Diagram

```
Data Flow:
┌─────────────┐
│  Raw CSVs   │ (6 tracks × 2 races)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ CSV Parser  │ (pivot long→wide)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Feature    │ (40+ features)
│ Engineering │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ GCS Upload  │ (train/test splits)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Vertex    │ (8 models)
│    AI       │
│  Training   │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Deployed   │ (endpoints)
│  Models     │
└─────────────┘
```

## Performance Benchmarks

### Data Processing
- Single track parsing: ~30 seconds
- Feature engineering: ~2 minutes
- GCS upload: ~1 minute
- **Total pipeline**: 30-60 minutes for all tracks

### Model Inference (estimated)
- Lap-Time Transformer: ~10ms
- Tire Degradation: ~5ms
- Fuel Consumption: ~2ms
- Traffic GNN: ~15ms
- All models combined: <50ms

## Troubleshooting

### Memory Issues
**Problem**: Out of memory during processing
**Solution**: Process tracks one at a time or increase system RAM

### GCS Upload Failures
**Problem**: Permission denied
**Solution**: Check service account has Storage Admin role

### Model Training Slow
**Problem**: Training takes too long
**Solution**: Use GPU-enabled machines, reduce batch size

## File Structure

```
ml-pipeline/
├── README.md                      # This file
├── .env.local                     # Configuration (gitignored)
├── .gitignore                     # Git ignore rules
├── requirements.txt               # Python dependencies
├── setup.sh                       # Setup script
│
├── config/                        # Configuration
│   ├── __init__.py
│   ├── settings.py                # Pydantic settings
│   └── gcp_credentials.json       # Service account (gitignored)
│
├── gcp_setup/                     # GCP provisioning
│   ├── __init__.py
│   ├── create_buckets.py          # Create GCS buckets
│   └── setup_vertex.py            # Initialize Vertex AI
│
├── data_processing/               # Data pipeline
│   ├── __init__.py
│   ├── csv_parser.py              # Parse telemetry CSVs
│   ├── feature_engineering.py     # Derive features
│   ├── upload_to_gcs.py           # Upload to Cloud Storage
│   └── run_pipeline.py            # Complete pipeline
│
├── models/                        # ML models
│   ├── __init__.py
│   ├── lap_time_transformer.py    # Core lap predictor
│   ├── tire_degradation.py        # Physics + TCN
│   ├── fuel_consumption.py        # XGBoost
│   ├── traffic_gnn.py             # Graph Neural Net
│   ├── fcy_hazard.py              # Caution predictor
│   ├── pit_loss.py                # Pit timing
│   ├── anomaly_detector.py        # LSTM autoencoder
│   └── driver_embedding.py        # Driver style
│
├── training/                      # Training scripts (WIP)
│   ├── __init__.py
│   └── train_*.py
│
├── deployment/                    # Deployment (WIP)
│   ├── __init__.py
│   └── deploy_models.py
│
└── validation/                    # Validation scripts
    ├── __init__.py
    └── validate_*.py
```

## Dependencies

See `requirements.txt` for full list. Key dependencies:
- PyTorch 2.1.2
- TensorFlow 2.15.0
- XGBoost 2.0.3
- torch-geometric 2.5.0
- google-cloud-aiplatform 1.42.1
- pandas 2.1.4
- numpy 1.26.3

## License

This is competition code for Hack the Track 2025.

## Contact

For questions about implementation:
- Check `/Users/anhlam/hack-the-track/TODO.md` for user action items
- Review GCP Console logs
- Verify service account permissions

---

**Built for**: Hack the Track 2025 - Toyota GR Cup
**Last Updated**: Phase 1 Complete (ML Pipeline Implemented)

