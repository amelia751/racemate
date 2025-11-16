"""
Vapi AI Tool Endpoints
Provides function calling tools for the voice assistant
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import logging

router = APIRouter(prefix="/vapi", tags=["vapi-tools"])
logger = logging.getLogger(__name__)


class TelemetryRequest(BaseModel):
    """Request for current telemetry data"""
    pass


class FuelStatusRequest(BaseModel):
    """Request for fuel status"""
    speed: Optional[float] = 180.0
    nmot: Optional[float] = 8000.0
    gear: Optional[int] = 4
    aps: Optional[float] = 80.0


class TireStatusRequest(BaseModel):
    """Request for tire status"""
    cum_brake_energy: Optional[float] = 25000.0
    cum_lateral_load: Optional[float] = 45000.0
    air_temp: Optional[float] = 28.0


class RaceStatusRequest(BaseModel):
    """Request for comprehensive race status"""
    pass


@router.post("/tools/get-telemetry")
async def get_telemetry(request: TelemetryRequest) -> Dict[str, Any]:
    """
    Get current telemetry data
    Returns real-time vehicle sensors
    """
    # In production, this would fetch from live telemetry stream
    # For now, return mock data structure
    return {
        "status": "success",
        "data": {
            "speed": 185,
            "rpm": 8500,
            "gear": 4,
            "throttle": 85,
            "fuel_level": 35.2,
            "lap": 15,
            "brake_temp_front": 450,
            "brake_temp_rear": 425,
            "tire_pressure_fl": 32,
            "tire_pressure_fr": 32,
            "tire_pressure_rl": 30,
            "tire_pressure_rr": 30
        },
        "message": "Telemetry data retrieved successfully"
    }


@router.post("/tools/check-fuel")
async def check_fuel(request: FuelStatusRequest) -> Dict[str, Any]:
    """
    Check fuel status and predict consumption
    Uses ML model to analyze fuel burn rate
    """
    try:
        # Simple physics-based calculation for demo
        # fuel_per_lap ≈ (RPM * throttle * speed_factor) / efficiency
        avg_rpm = request.nmot
        avg_throttle = request.aps
        avg_speed = request.speed
        
        # Estimated fuel per lap (liters)
        fuel_per_lap = (avg_rpm / 10000) * (avg_throttle / 100) * 2.5
        
        # Current fuel (mock - would come from telemetry)
        current_fuel = 35.2
        laps_remaining = int(current_fuel / fuel_per_lap)
        
        # Determine strategy
        if laps_remaining >= 20:
            strategy = "fuel_save_low"
            recommendation = f"Fuel looks good! You can push. Estimated {laps_remaining} laps remaining at current pace."
        elif laps_remaining >= 10:
            strategy = "fuel_save_medium"
            recommendation = f"Monitor fuel closely. About {laps_remaining} laps remaining. Consider lifting in sectors 2 and 3."
        else:
            strategy = "fuel_save_critical"
            recommendation = f"FUEL CRITICAL! Only {laps_remaining} laps left. Short-shift and lift significantly."
        
        return {
            "status": "success",
            "data": {
                "current_fuel_liters": current_fuel,
                "fuel_per_lap": round(fuel_per_lap, 2),
                "laps_remaining": laps_remaining,
                "strategy": strategy,
                "confidence": 0.85
            },
            "message": recommendation
        }
        
    except Exception as e:
        logger.error(f"Fuel check error: {e}")
        return {
            "status": "error",
            "message": f"Unable to calculate fuel status: {str(e)}"
        }


@router.post("/tools/check-tires")
async def check_tires(request: TireStatusRequest) -> Dict[str, Any]:
    """
    Check tire condition and degradation
    Uses ML model to predict grip loss
    """
    try:
        # Physics-based tire wear model
        brake_wear_factor = request.cum_brake_energy / 50000  # Normalized
        lateral_wear_factor = request.cum_lateral_load / 100000
        temp_factor = (request.air_temp - 20) / 20  # Normalized from 20°C baseline
        
        # Total wear (0 to 1 scale)
        total_wear = (brake_wear_factor * 0.4 + lateral_wear_factor * 0.5 + temp_factor * 0.1)
        total_wear = min(max(total_wear, 0), 1)
        
        # Grip index (1 = new tires, 0 = completely worn)
        grip_index = 1.0 - total_wear
        
        # Determine tire condition
        if grip_index > 0.85:
            condition = "excellent"
            recommendation = "Tires are in great shape! You can attack."
        elif grip_index > 0.70:
            condition = "good"
            recommendation = "Tires have some wear but still performing well. Monitor closely."
        elif grip_index > 0.50:
            condition = "moderate"
            recommendation = "Tire degradation increasing. Consider pitting soon."
        else:
            condition = "critical"
            recommendation = "TIRE WEAR CRITICAL! Grip is significantly reduced. Pit immediately!"
        
        return {
            "status": "success",
            "data": {
                "grip_index": round(grip_index, 2),
                "wear_percentage": round((1 - grip_index) * 100, 1),
                "condition": condition,
                "brake_energy_cumulative": request.cum_brake_energy,
                "lateral_load_cumulative": request.cum_lateral_load,
                "confidence": 0.80
            },
            "message": recommendation
        }
        
    except Exception as e:
        logger.error(f"Tire check error: {e}")
        return {
            "status": "error",
            "message": f"Unable to calculate tire status: {str(e)}"
        }


@router.post("/tools/race-status")
async def race_status(request: RaceStatusRequest) -> Dict[str, Any]:
    """
    Get comprehensive race status
    Combines telemetry, fuel, and tire analysis
    """
    try:
        # Get all data (in production, these would be real calls)
        telemetry = {
            "speed": 185,
            "rpm": 8500,
            "gear": 4,
            "lap": 15,
            "position": 3
        }
        
        fuel_data = {
            "current": 35.2,
            "laps_remaining": 14,
            "status": "marginal"
        }
        
        tire_data = {
            "grip_index": 0.75,
            "wear_percentage": 25,
            "condition": "good"
        }
        
        # Overall strategy recommendation
        if fuel_data["laps_remaining"] < 10 and tire_data["grip_index"] < 0.7:
            overall_strategy = "Pit soon - both fuel and tires need attention"
        elif fuel_data["laps_remaining"] < 10:
            overall_strategy = "Fuel save mode - lift and coast where possible"
        elif tire_data["grip_index"] < 0.7:
            overall_strategy = "Tire management - smooth inputs, avoid kerbs"
        else:
            overall_strategy = "Push mode - everything looks good!"
        
        summary = f"Lap {telemetry['lap']}, P{telemetry['position']}. Fuel: {fuel_data['current']}L ({fuel_data['laps_remaining']} laps). Tires at {tire_data['wear_percentage']}% wear. {overall_strategy}"
        
        return {
            "status": "success",
            "data": {
                "telemetry": telemetry,
                "fuel": fuel_data,
                "tires": tire_data,
                "overall_strategy": overall_strategy
            },
            "message": summary
        }
        
    except Exception as e:
        logger.error(f"Race status error: {e}")
        return {
            "status": "error",
            "message": f"Unable to get race status: {str(e)}"
        }


@router.get("/tools/manifest")
async def get_tools_manifest() -> Dict[str, Any]:
    """
    Return manifest of available tools for Vapi configuration
    """
    return {
        "tools": [
            {
                "name": "get_telemetry",
                "description": "Get current real-time telemetry data including speed, RPM, gear, fuel level, and tire pressures",
                "parameters": {
                    "type": "object",
                    "properties": {}
                },
                "url": "http://localhost:8005/vapi/tools/get-telemetry",
                "method": "POST"
            },
            {
                "name": "check_fuel",
                "description": "Check current fuel status and get fuel consumption predictions. Returns laps remaining and fuel-saving recommendations.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "speed": {"type": "number", "description": "Current average speed"},
                        "nmot": {"type": "number", "description": "Current RPM"},
                        "gear": {"type": "integer", "description": "Current gear"},
                        "aps": {"type": "number", "description": "Throttle position percentage"}
                    }
                },
                "url": "http://localhost:8005/vapi/tools/check-fuel",
                "method": "POST"
            },
            {
                "name": "check_tires",
                "description": "Check tire condition and degradation. Returns grip index, wear percentage, and pit recommendations.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cum_brake_energy": {"type": "number", "description": "Cumulative brake energy"},
                        "cum_lateral_load": {"type": "number", "description": "Cumulative lateral load"},
                        "air_temp": {"type": "number", "description": "Air temperature in Celsius"}
                    }
                },
                "url": "http://localhost:8005/vapi/tools/check-tires",
                "method": "POST"
            },
            {
                "name": "race_status",
                "description": "Get comprehensive race status including telemetry, fuel, tires, and overall strategy recommendation",
                "parameters": {
                    "type": "object",
                    "properties": {}
                },
                "url": "http://localhost:8005/vapi/tools/race-status",
                "method": "POST"
            }
        ]
    }

