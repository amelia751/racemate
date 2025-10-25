"""
API client for calling Cognirace prediction endpoints
"""

import requests
from typing import Dict, Any, Optional, List
import time


class CogniraceAPIClient:
    """Client for Cognirace Prediction API"""
    
    def __init__(self, base_url: str = "http://localhost:8005", timeout: int = 30):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
    
    def _call_endpoint(self, method: str, endpoint: str, data: Optional[Dict] = None) -> Dict[str, Any]:
        """Make API call with error handling"""
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method == "GET":
                response = self.session.get(url, timeout=self.timeout)
            elif method == "POST":
                response = self.session.post(url, json=data, timeout=self.timeout)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            return {
                "error": str(e),
                "status": "failed"
            }
    
    def predict_fuel(self, speed: float, nmot: float, gear: int, 
                    aps: float, lap: int) -> Dict[str, Any]:
        """Predict fuel consumption"""
        data = {
            "speed": speed,
            "nmot": nmot,
            "gear": gear,
            "aps": aps,
            "lap": lap
        }
        return self._call_endpoint("POST", "/predict/fuel", data)
    
    def predict_laptime(self, telemetry_sequence: List[List[float]], 
                       feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Predict lap time"""
        data = {
            "telemetry_sequence": telemetry_sequence,
            "feature_names": feature_names or []
        }
        return self._call_endpoint("POST", "/predict/laptime", data)
    
    def predict_tire(self, cum_brake_energy: float, cum_lateral_load: float,
                    air_temp: float = 25.0) -> Dict[str, Any]:
        """Predict tire degradation"""
        data = {
            "cum_brake_energy": cum_brake_energy,
            "cum_lateral_load": cum_lateral_load,
            "air_temp": air_temp
        }
        return self._call_endpoint("POST", "/predict/tire", data)
    
    def predict_traffic(self, car_states: List[List[float]], 
                       feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Predict traffic impact"""
        data = {
            "car_states": car_states,
            "feature_names": feature_names or []
        }
        return self._call_endpoint("POST", "/predict/traffic", data)
    
    def health_check(self) -> Dict[str, Any]:
        """Check API health"""
        return self._call_endpoint("GET", "/health")
    
    def list_models(self) -> Dict[str, Any]:
        """List available models"""
        return self._call_endpoint("GET", "/predict/models")


# Tool descriptions for LLM function calling
TOOL_DEFINITIONS = [
    {
        "name": "predict_fuel_consumption",
        "description": "Predict fuel consumption rate based on current driving conditions. Returns fuel burn rate in L/lap or synthetic units.",
        "parameters": {
            "type": "object",
            "properties": {
                "speed": {"type": "number", "description": "Current speed in km/h"},
                "nmot": {"type": "number", "description": "Engine RPM"},
                "gear": {"type": "integer", "description": "Current gear (1-7)"},
                "aps": {"type": "number", "description": "Throttle position %"},
                "lap": {"type": "integer", "description": "Current lap number"}
            },
            "required": ["speed", "nmot", "gear", "aps", "lap"]
        }
    },
    {
        "name": "predict_tire_degradation",
        "description": "Predict tire grip degradation based on cumulative loads. Returns grip index (0.5-1.0).",
        "parameters": {
            "type": "object",
            "properties": {
                "cum_brake_energy": {"type": "number", "description": "Cumulative brake energy"},
                "cum_lateral_load": {"type": "number", "description": "Cumulative lateral load"},
                "air_temp": {"type": "number", "description": "Air temperature in °C"}
            },
            "required": ["cum_brake_energy", "cum_lateral_load"]
        }
    },
    {
        "name": "predict_traffic_impact",
        "description": "Predict traffic-induced time loss and overtake probability. Returns traffic loss in seconds and overtake probability.",
        "parameters": {
            "type": "object",
            "properties": {
                "car_states": {
                    "type": "array",
                    "description": "Array of car states, each with speed, rpm, gear, throttle, etc.",
                    "items": {"type": "array", "items": {"type": "number"}}
                }
            },
            "required": ["car_states"]
        }
    }
]

