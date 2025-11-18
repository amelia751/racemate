"""
Real-time telemetry processing and event-driven recommendations
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import asyncio
import json
import logging
import time

# import sys
# import os
# Add parent directory to path for agents import
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

# from agents.specialized.chief_agent import ChiefAgent  # Not needed - using custom formatter
from services.realtime_predictor import RealtimePredictor
from services.strategy_formatter import StrategyFormatter

logger = logging.getLogger(__name__)

router = APIRouter()

# HTTP endpoint for testing
class TelemetryRequest(BaseModel):
    telemetry: Dict[str, Any]

@router.post("/process")
async def process_telemetry(request: TelemetryRequest):
    """
    HTTP endpoint to process telemetry and return events/recommendations
    Useful for testing and debugging
    """
    try:
        telemetry = request.telemetry
        logger.info(f"Processing telemetry via HTTP: lap={telemetry.get('lap')}, fuel={telemetry.get('fuel_level'):.1f}L, speed={telemetry.get('speed'):.0f}km/h")
        
        # Initialize predictor if needed
        predictor = RealtimePredictor()
        
        # Process telemetry - returns list of events or None
        events = predictor.process_telemetry(telemetry)
        if events is None:
            events = []
        
        logger.info(f"Detected {len(events)} events: {[e.event_type for e in events]}")
        
        # Get predictions from predictor state  
        predictions = {
            'fuel_per_lap': getattr(predictor.state, 'last_fuel_prediction', 0.06),
            'fuel_level': predictor.state.fuel_level,
            'frame_count': predictor.frame_count
        }
        
        # If we have critical or high severity events, generate recommendations
        recommendations = None
        critical_events = [e for e in events if e.severity in ['critical', 'high']]
        
        if critical_events:
            try:
                # Format ML predictions into professional recommendations (no LLM needed!)
                formatter = StrategyFormatter()
                strategy_text = formatter.format_recommendations(
                    events=[{
                        "type": e.event_type,
                        "severity": e.severity,
                        "message": e.message,
                        "data": e.data
                    } for e in critical_events],
                    predictions=predictions,
                    telemetry=telemetry
                )
                
                recommendations = {
                    "strategy": strategy_text,
                    "severity_summary": {
                        "critical": sum(1 for e in events if e.severity == 'critical'),
                        "high": sum(1 for e in events if e.severity == 'high'),
                        "medium": sum(1 for e in events if e.severity == 'medium'),
                        "info": sum(1 for e in events if e.severity == 'info')
                    },
                    "events": [
                        {
                            "type": e.event_type,
                            "severity": e.severity,
                            "message": e.message,
                            "data": e.data
                        }
                        for e in events
                    ]
                }
                
                logger.info(f"Generated professional recommendation for {len(critical_events)} events")
                
            except Exception as e:
                logger.error(f"Failed to generate recommendation: {e}")
                recommendations = {
                    "strategy": f"⚠️ Unable to generate recommendations: {str(e)}",
                    "events": [{"type": e.event_type, "severity": e.severity, "message": e.message} for e in critical_events]
                }
        
        return {
            "success": True,
            "events": [
                {
                    "type": e.event_type,
                    "severity": e.severity,
                    "message": e.message,
                    "data": e.data
                }
                for e in events
            ],
            "recommendations": recommendations,
            "predictions": predictions,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Error processing telemetry: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# WebSocket endpoint (disabled - using HTTP POST endpoint instead)
# Note: Commented out to avoid ChiefAgent dependency
# Uncomment if you need WebSocket support and have agents module available

