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

import sys
import os
# Add parent directory to path for agents import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from agents.specialized.chief_agent import ChiefAgent
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


# WebSocket endpoint (existing)
class RealtimeSession:
    """Manages a single WebSocket session for real-time telemetry processing"""
    
    def __init__(self, websocket: WebSocket):
        self.websocket = websocket
        self.predictor = RealtimePredictor()
        self.chief_agent = ChiefAgent(use_gemini=True, api_client=None)
        self.last_gemini_call = 0
        logger.info("RealtimeSession initialized with ChiefAgent (direct mode)")
    
    async def process_telemetry(self, telemetry: Dict[str, Any]):
        """Process incoming telemetry and send back recommendations if needed"""
        try:
            # Run predictions and detect events
            events = self.predictor.process_telemetry(telemetry)
            if events is None:
                events = []
            
            # Get predictions from predictor state
            predictions = {
                'fuel_per_lap': getattr(self.predictor.state, 'last_fuel_prediction', 0.06),
                'fuel_level': self.predictor.state.fuel_level,
                'frame_count': self.predictor.frame_count
            }
            
            # Filter for critical and high severity events only
            critical_events = [e for e in events if e.severity in ['critical', 'high']]
            
            if critical_events:
                logger.info(f"Detected {len(critical_events)} critical/high events: {[e.event_type for e in critical_events]}")
                
                # Generate recommendations
                recommendations = await self._generate_recommendations(critical_events, telemetry, predictions)
                
                if recommendations:
                    # Send recommendations back to frontend
                    await self.websocket.send_json({
                        "type": "recommendation",
                        "recommendations": recommendations,
                        "timestamp": time.time()
                    })
            
        except Exception as e:
            logger.error(f"Error in process_telemetry: {e}")
            await self.websocket.send_json({
                "type": "error",
                "message": str(e)
            })
    
    async def _generate_recommendations(
        self, 
        events: List[Any], 
        telemetry: Dict[str, Any],
        predictions: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Generate AI-powered recommendations based on detected events
        Rate-limited to avoid quota issues
        Only returns when Gemini is called (no quick analysis spam)
        """
        import time
        
        # Rate limiting: Only call Gemini every 10 seconds
        current_time = time.time()
        if current_time - self.last_gemini_call < 10:
            # Too soon, skip this recommendation
            return None
        
        # Build context from events
        event_summary = "\n".join([
            f"- [{e.severity.upper()}] {e.event_type}: {e.message}"
            for e in events
        ])
        
        enhanced_context = f"""
Current Race Status:
- Lap: {telemetry.get('lap', 1)}
- Speed: {telemetry.get('speed', 0):.0f} km/h
- Fuel Level: {telemetry.get('fuel_level', 0):.1f}L
- RPM: {telemetry.get('rpm', telemetry.get('nmot', 0)):.0f}
- Gear: {telemetry.get('gear', 0)}
- Throttle: {telemetry.get('throttle', telemetry.get('aps', 0)):.0f}%

Critical Events Detected:
{event_summary}

ML Model Predictions:
{json.dumps(predictions, indent=2)}
"""
        
        try:
            # Call Gemini for comprehensive analysis
            prompt = f"""Analyze these {len(events)} critical race events and provide strategic recommendations:

{enhanced_context}
"""
            gemini_response = self.chief_agent.generate_with_gemini(
                prompt=prompt,
                context=telemetry
            )
            
            self.last_gemini_call = current_time
            
            recommendations = {
                "strategy": gemini_response,
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
            
            logger.info(f"Generated Gemini recommendation for {len(events)} events")
            return recommendations
            
        except Exception as e:
            logger.error(f"Gemini recommendation failed: {e}")
            return None


@router.websocket("/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket):
    """WebSocket endpoint for real-time telemetry streaming"""
    await websocket.accept()
    logger.info("WebSocket connection established")
    
    session = RealtimeSession(websocket)
    
    try:
        while True:
            # Receive telemetry data
            data = await websocket.receive_json()
            
            if "telemetry" in data:
                telemetry = data["telemetry"]
                await session.process_telemetry(telemetry)
            
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.close()
        except:
            pass
