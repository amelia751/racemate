"""
Pydantic models for API requests and responses
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

# Common
class PredictionResponse(BaseModel):
    """Base prediction response"""
    prediction: Any
    model_version: str = "v1"
    confidence: Optional[float] = None
    latency_ms: float
    status: str = "success"


# Fuel Consumption
class FuelPredictionRequest(BaseModel):
    """Fuel consumption prediction request"""
    speed: float = Field(..., description="Speed in km/h", ge=0, le=300)
    nmot: float = Field(..., description="Engine RPM", ge=0, le=10000)
    gear: int = Field(..., description="Current gear", ge=1, le=7)
    aps: float = Field(..., description="Throttle position %", ge=0, le=100)
    lap: int = Field(..., description="Lap number", ge=1)
    
    class Config:
        json_schema_extra = {
            "example": {
                "speed": 180.5,
                "nmot": 7200,
                "gear": 5,
                "aps": 95.2,
                "lap": 15
            }
        }


class FuelPredictionResponse(PredictionResponse):
    """Fuel consumption prediction response"""
    prediction: float = Field(..., description="Fuel burn rate (L/lap or synthetic units)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "prediction": 0.45,
                "model_version": "v1",
                "confidence": 0.92,
                "latency_ms": 25,
                "status": "success"
            }
        }


# Lap Time
class LapTimePredictionRequest(BaseModel):
    """Lap time prediction request"""
    telemetry_sequence: List[List[float]] = Field(
        ...,
        description="Telemetry sequence (seq_len x num_features)",
        min_length=10
    )
    feature_names: Optional[List[str]] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "telemetry_sequence": [[180.5, 7200, 5, 95.2] * 4] * 100,
                "feature_names": ["speed", "nmot", "gear", "aps"]
            }
        }


class LapTimePredictionResponse(PredictionResponse):
    """Lap time prediction response"""
    prediction: float = Field(..., description="Predicted lap time delta (seconds)")
    quantiles: Dict[str, float] = Field(..., description="Uncertainty quantiles")
    
    class Config:
        json_schema_extra = {
            "example": {
                "prediction": -0.5,
                "quantiles": {"p10": -1.2, "p50": -0.5, "p90": 0.1},
                "model_version": "v1",
                "latency_ms": 45,
                "status": "success"
            }
        }


# Tire Degradation
class TirePredictionRequest(BaseModel):
    """Tire degradation prediction request"""
    cum_brake_energy: float = Field(..., description="Cumulative brake energy", ge=0)
    cum_lateral_load: float = Field(..., description="Cumulative lateral load", ge=0)
    air_temp: float = Field(default=25.0, description="Air temperature (°C)")
    telemetry_sequence: List[List[float]] = Field(..., description="REQUIRED: Real telemetry sequence - NO DUMMY DATA ALLOWED")
    
    class Config:
        json_schema_extra = {
            "example": {
                "cum_brake_energy": 1500.0,
                "cum_lateral_load": 2000.0,
                "air_temp": 28.5
            }
        }


class TirePredictionResponse(PredictionResponse):
    """Tire degradation prediction response"""
    prediction: float = Field(..., description="Grip index (0.5-1.0)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "prediction": 0.85,
                "model_version": "v1",
                "confidence": 0.88,
                "latency_ms": 30,
                "status": "success"
            }
        }


# Traffic GNN
class TrafficPredictionRequest(BaseModel):
    """Traffic analysis prediction request"""
    car_states: List[List[float]] = Field(
        ...,
        description="Features for each car (num_cars x num_features)",
        min_length=1,
        max_length=10
    )
    feature_names: Optional[List[str]] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "car_states": [
                    [180.5, 7200, 5, 95.2, 0, 0] * 3,
                    [175.0, 7000, 5, 90.0, 0, 0] * 3
                ],
                "feature_names": ["speed", "nmot", "gear", "aps", "pbrake_f", "pbrake_r"]
            }
        }


class TrafficPredictionResponse(PredictionResponse):
    """Traffic analysis prediction response"""
    prediction: Dict[str, float] = Field(..., description="Traffic loss and overtake prob")
    
    class Config:
        json_schema_extra = {
            "example": {
                "prediction": {
                    "traffic_loss_seconds": 2.5,
                    "overtake_probability": 0.65
                },
                "model_version": "v1",
                "latency_ms": 35,
                "status": "success"
            }
        }


# Health Check
class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    models_loaded: int
    uptime_seconds: float


# Error Response
class ErrorResponse(BaseModel):
    """Error response"""
    error: str
    detail: Optional[str] = None
    status_code: int

