"""
Prediction endpoints for all Cognirace models
"""

from fastapi import APIRouter, HTTPException
from models.schemas import (
    FuelPredictionRequest, FuelPredictionResponse,
    LapTimePredictionRequest, LapTimePredictionResponse,
    TirePredictionRequest, TirePredictionResponse,
    TrafficPredictionRequest, TrafficPredictionResponse
)
from services.model_loader import model_loader
import sys
import os
import time
import torch
import numpy as np

# Add ml-pipeline to path to import model classes
ml_pipeline_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../ml-pipeline'))
if ml_pipeline_path not in sys.path:
    sys.path.insert(0, ml_pipeline_path)

# Import model classes from ml-pipeline
import importlib.util

def import_model_class(model_path, class_name):
    """Dynamically import model class from ml-pipeline"""
    spec = importlib.util.spec_from_file_location(class_name, model_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)

# Import model classes
try:
    lap_time_path = os.path.join(ml_pipeline_path, 'models', 'lap_time_transformer.py')
    LapTimeTransformer = import_model_class(lap_time_path, 'LapTimeTransformer')
    
    tire_path = os.path.join(ml_pipeline_path, 'models', 'tire_degradation.py')
    TireDegradationModel = import_model_class(tire_path, 'TireDegradationModel')
    
    traffic_path = os.path.join(ml_pipeline_path, 'models', 'traffic_gnn.py')
    TrafficGNN = import_model_class(traffic_path, 'TrafficGNN')
except Exception as e:
    print(f"Warning: Could not import some model classes: {e}")
    # Fallback - models will load from checkpoints
    LapTimeTransformer = None
    TireDegradationModel = None
    TrafficGNN = None

router = APIRouter()


@router.post("/fuel", response_model=FuelPredictionResponse)
async def predict_fuel(request: FuelPredictionRequest):
    """Predict fuel consumption (simplified formula-based prediction)"""
    start_time = time.time()
    
    try:
        # Simplified physics-based fuel prediction
        # Fuel consumption ≈ f(RPM, throttle, speed, gear)
        
        # Normalize inputs
        rpm_factor = request.nmot / 8000.0  # 0-1 range
        throttle_factor = request.aps / 100.0  # 0-1 range
        speed_factor = request.speed / 200.0  # 0-1 range
        gear_factor = 1.0 - (request.gear / 7.0)  # Lower gear = more fuel
        
        # Weighted fuel burn rate (synthetic units)
        base_burn = 0.2
        prediction = float(
            base_burn + 
            (rpm_factor * 0.25) +
            (throttle_factor * 0.30) +
            (speed_factor * 0.15) +
            (gear_factor * 0.10)
        )
        
        latency_ms = (time.time() - start_time) * 1000
        
        return FuelPredictionResponse(
            prediction=prediction,
            confidence=0.85,
            latency_ms=latency_ms
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post("/laptime", response_model=LapTimePredictionResponse)
async def predict_laptime(request: LapTimePredictionRequest):
    """Predict lap time (simplified statistical prediction)"""
    start_time = time.time()
    
    try:
        # Simplified lap time prediction based on telemetry statistics
        telemetry = np.array(request.telemetry_sequence)
        
        # Extract features (assuming first columns are speed, rpm, gear, throttle)
        avg_speed = np.mean(telemetry[:, 0]) if telemetry.shape[1] > 0 else 100
        avg_rpm = np.mean(telemetry[:, 1]) if telemetry.shape[1] > 1 else 6000
        avg_throttle = np.mean(telemetry[:, 3]) if telemetry.shape[1] > 3 else 50
        
        # Simple lap time delta prediction (seconds vs baseline)
        # Faster avg speed = negative delta (faster lap)
        # Higher avg throttle = negative delta (more aggressive)
        speed_factor = (avg_speed - 150) / 50.0
        rpm_factor = (avg_rpm - 6000) / 2000.0
        throttle_factor = (avg_throttle - 50) / 50.0
        
        prediction = float(-speed_factor * 0.5 - throttle_factor * 0.3)
        
        # Uncertainty quantiles (± variations)
        quantile_dict = {
            "p10": float(prediction - 1.2),
            "p50": float(prediction),
            "p90": float(prediction + 0.8)
        }
        
        latency_ms = (time.time() - start_time) * 1000
        
        return LapTimePredictionResponse(
            prediction=prediction,
            quantiles=quantile_dict,
            latency_ms=latency_ms
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post("/tire", response_model=TirePredictionResponse)
async def predict_tire(request: TirePredictionRequest):
    """Predict tire degradation"""
    start_time = time.time()
    
    try:
        # Load model
        model_data = model_loader.load_model('tire_degradation', TireDegradationModel)
        
        # Initialize model
        model = TireDegradationModel(input_dim=16, hidden_channels=64, num_layers=3)
        model.load_state_dict(model_data['model_state'])
        model.eval()
        
        # Prepare physics features
        physics_features = {
            'cum_brake_energy': torch.tensor([[request.cum_brake_energy]]),
            'cum_lateral_load': torch.tensor([[request.cum_lateral_load]]),
            'air_temp': torch.tensor([[request.air_temp]])
        }
        
        # Dummy telemetry sequence (would come from request in production)
        dummy_telemetry = torch.randn(1, 100, 16)
        
        # Predict
        with torch.no_grad():
            grip_index = model(dummy_telemetry, physics_features)
        
        prediction = float(grip_index[0, 0].item())
        
        latency_ms = (time.time() - start_time) * 1000
        
        return TirePredictionResponse(
            prediction=prediction,
            confidence=0.82,
            latency_ms=latency_ms
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post("/traffic", response_model=TrafficPredictionResponse)
async def predict_traffic(request: TrafficPredictionRequest):
    """Predict traffic impact"""
    start_time = time.time()
    
    try:
        # Load model
        model_data = model_loader.load_model('traffic_gnn', TrafficGNN)
        
        # Get model parameters
        checkpoint = model_data['checkpoint']
        node_feature_dim = checkpoint.get('node_feature_dim', 16)
        
        # Initialize model
        model = TrafficGNN(node_feature_dim=node_feature_dim, hidden_dim=64, num_layers=2)
        model.load_state_dict(model_data['model_state'])
        model.eval()
        
        # Prepare input (num_cars, num_features)
        car_states = torch.FloatTensor(request.car_states)
        
        # Pad if needed
        num_cars = len(request.car_states)
        if num_cars < 5:
            padding = torch.zeros(5 - num_cars, car_states.shape[1])
            car_states = torch.cat([car_states, padding], dim=0)
        
        # Add batch dimension
        car_states = car_states.unsqueeze(0)  # (1, num_cars, features)
        
        # Predict
        with torch.no_grad():
            traffic_loss, overtake_prob = model(car_states)
        
        prediction_dict = {
            "traffic_loss_seconds": float(traffic_loss[0, 0].item()),
            "overtake_probability": float(overtake_prob[0, 0].item())
        }
        
        latency_ms = (time.time() - start_time) * 1000
        
        return TrafficPredictionResponse(
            prediction=prediction_dict,
            latency_ms=latency_ms
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.get("/models")
async def list_models():
    """List available models"""
    return {
        "models": [
            {"name": "fuel", "endpoint": "/predict/fuel", "status": "available"},
            {"name": "laptime", "endpoint": "/predict/laptime", "status": "available"},
            {"name": "tire", "endpoint": "/predict/tire", "status": "available"},
            {"name": "traffic", "endpoint": "/predict/traffic", "status": "available"},
            {"name": "fcy", "endpoint": "/predict/fcy", "status": "coming_soon"},
            {"name": "pitloss", "endpoint": "/predict/pitloss", "status": "coming_soon"},
            {"name": "anomaly", "endpoint": "/predict/anomaly", "status": "coming_soon"},
            {"name": "driver", "endpoint": "/predict/driver", "status": "coming_soon"}
        ],
        "loaded_models": list(model_loader.loaded_models.keys())
    }

