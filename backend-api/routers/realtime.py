"""
Real-Time WebSocket Endpoint for Cognirace
Handles streaming telemetry and triggers agent recommendations
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Dict, Any
import json
import logging
import asyncio
import sys
import os

# Add agents to path
agents_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../agents'))
if agents_path not in sys.path:
    sys.path.insert(0, agents_path)

from services.realtime_predictor import realtime_predictor
from specialized.chief_agent import ChiefAgent
from tools.api_client import CogniraceAPIClient

logger = logging.getLogger(__name__)

router = APIRouter()


class RealtimeSession:
    """Manages a real-time telemetry session"""
    
    def __init__(self):
        self.predictor = realtime_predictor
        # Don't use API client in backend - we already have the predictions!
        self.chief_agent = ChiefAgent(
            api_client=None,  # No API calls needed, we have direct access to models
            use_gemini=True
        )
        self.frame_count = 0
        self.gemini_call_count = 0
        self.last_gemini_call = 0
        logger.info("RealtimeSession initialized with ChiefAgent (direct mode)")
    
    async def process_telemetry(self, telemetry: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process telemetry through models, detect events, and generate recommendations
        Only sends recommendations for critical/high severity events
        """
        self.frame_count += 1
        
        # Run through all 8 models and detect events
        events = self.predictor.process_telemetry(telemetry)
        
        if events:
            # Filter to only critical and high severity events
            important_events = [e for e in events if e.severity in ['critical', 'high']]
            
            if important_events:
                logger.info(f"Frame {self.frame_count}: {len(important_events)} important event(s) detected")
                
                # Generate comprehensive recommendations using multi-agent system
                recommendations = await self._generate_recommendations(important_events, telemetry)
                
                return {
                    'type': 'recommendation',
                    'frame': self.frame_count,
                    'events': [e.to_dict() for e in important_events],
                    'recommendations': recommendations,
                    'timestamp': telemetry.get('timestamp')
                }
        
        # No important events, don't send anything
        return None
    
    async def _generate_recommendations(self, events, telemetry: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive recommendations using multi-agent system
        Rate-limited to avoid quota issues
        Only returns when Gemini is called (no quick analysis spam)
        """
        import time
        
        # Rate limiting: Only call Gemini every 20 seconds
        current_time = time.time()
        if current_time - self.last_gemini_call < 20:
            # Too soon, skip this recommendation
            return None
        
        # Build context from events
        event_summary = "\n".join([
            f"- [{e.severity.upper()}] {e.event_type}: {e.message}"
            for e in events
        ])
        
        # Build comprehensive analysis without API calls
        analysis_parts = []
        
        # Analyze each event type
        for event in events:
            if event.event_type.startswith('FUEL'):
                analysis_parts.append(f"🔴 FUEL ALERT: {event.message}")
            elif event.event_type.startswith('TIRE'):
                analysis_parts.append(f"🟠 TIRE WARNING: {event.message}")
            elif event.event_type.startswith('ANOMALY'):
                analysis_parts.append(f"🚨 ANOMALY: {event.message}")
            elif event.event_type.startswith('FCY'):
                analysis_parts.append(f"🟡 FCY FORECAST: {event.message}")
            elif event.event_type.startswith('PACE'):
                analysis_parts.append(f"📊 PACE UPDATE: {event.message}")
            elif event.event_type.startswith('PIT'):
                analysis_parts.append(f"⏱️ PIT STRATEGY: {event.message}")
            else:
                analysis_parts.append(f"ℹ️ {event.message}")
        
        quick_strategy = "\n".join(analysis_parts)
        
        # Use Gemini for deeper analysis (rate-limited)
        try:
            query = f"""Race Strategy Analysis:

DETECTED EVENTS:
{event_summary}

CURRENT TELEMETRY:
- Lap: {telemetry.get('lap', 1)}
- Speed: {telemetry.get('speed', 0):.0f} km/h
- Fuel: {telemetry.get('fuel_level', 35.0):.1f}L
- Throttle: {telemetry.get('aps', telemetry.get('throttle', 0)):.0f}%

Provide concise (max 3 sentences) strategic recommendation focusing on:
1. Immediate action required
2. Risk level
3. Next strategic decision point"""
            
            context = {
                'telemetry': telemetry,
                'events': event_summary,
                'race_info': {
                    'lap': telemetry.get('lap', 1),
                    'fuel_level': telemetry.get('fuel_level', 35.0),
                    'track': 'Barber Motorsports Park'
                }
            }
            
            # Call Gemini through ChiefAgent
            gemini_response = self.chief_agent.generate_with_gemini(query, context)
            
            self.last_gemini_call = current_time
            self.gemini_call_count += 1
            
            # Only show Gemini analysis (no quick strategy spam)
            strategy = f"🤖 AI RACE STRATEGIST:\n\n{gemini_response}\n\n📊 Detected Events:\n{quick_strategy}"
            
            return {
                'strategy': strategy,
                'event_count': len(events),
                'severity_summary': self._summarize_severity(events),
                'gemini_calls': self.gemini_call_count
            }
        
        except Exception as e:
            logger.error(f"Gemini call failed: {e}")
            # If Gemini fails, don't send anything
            return None
    
    def _summarize_severity(self, events) -> Dict[str, int]:
        """Count events by severity"""
        severity_count = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0, 'info': 0}
        for event in events:
            severity_count[event.severity] += 1
        return severity_count


@router.websocket("/ws/telemetry")
async def telemetry_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for real-time telemetry streaming
    
    Frontend sends: {"telemetry": {...}, "timestamp": "..."}
    Backend responds: {"type": "recommendation|status", ...}
    """
    await websocket.accept()
    session = RealtimeSession()
    
    logger.info("WebSocket connection established")
    
    try:
        while True:
            # Receive telemetry from frontend
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if 'telemetry' in message:
                telemetry = message['telemetry']
                
                # Process through models and agents
                response = await session.process_telemetry(telemetry)
                
                # Only send if there's a recommendation (filters out None)
                if response:
                    await websocket.send_text(json.dumps(response))
            
            else:
                # Invalid message format
                await websocket.send_text(json.dumps({
                    'type': 'error',
                    'message': 'Invalid message format. Expected {"telemetry": {...}}'
                }))
    
    except WebSocketDisconnect:
        logger.info("WebSocket connection closed")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        try:
            await websocket.send_text(json.dumps({
                'type': 'error',
                'message': str(e)
            }))
        except:
            pass


@router.get("/realtime/status")
async def realtime_status():
    """Check real-time system status"""
    return {
        'status': 'operational',
        'predictor_loaded': realtime_predictor.models_loaded,
        'frame_count': realtime_predictor.frame_count
    }

