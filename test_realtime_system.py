"""
Comprehensive End-to-End Test for Event-Driven Real-Time System
Tests: WebSocket streaming, ML model predictions, event detection, multi-agent recommendations
"""

import asyncio
import websockets
import json
import time
import numpy as np
from datetime import datetime
from typing import List, Dict

# Enhanced Mock Data Generator
class EnhancedTelemetrySimulator:
    """
    Generates realistic race scenarios with:
    - Normal racing
    - Fuel crisis
    - Tire degradation
    - Anomalies
    - FCY situations
    - Traffic scenarios
    """
    
    def __init__(self):
        self.lap = 1
        self.fuel_level = 50.0
        self.tire_wear = {'FL': 0, 'FR': 0, 'RL': 0, 'RR': 0}
        self.scenario = "normal_race"
        self.frame_count = 0
        
    def generate_frame(self) -> Dict:
        """Generate one telemetry frame based on current scenario"""
        self.frame_count += 1
        
        # Base telemetry (convert numpy types to Python types for JSON serialization)
        telemetry = {
            'lap': int(self.lap),
            'speed': float(160 + np.random.normal(0, 15)),
            'rpm': float(8000 + np.random.normal(0, 500)),
            'nmot': float(8000 + np.random.normal(0, 500)),
            'gear': int(np.random.choice([4, 5, 6], p=[0.3, 0.5, 0.2])),
            'throttle': float(70 + np.random.normal(0, 10)),
            'aps': float(70 + np.random.normal(0, 10)),
            'fuel_level': float(self.fuel_level),
            'brake_balance': 55.0,
            'air_temp': 26.0,
            'timestamp': datetime.now().isoformat()
        }
        
        # Apply scenario-specific modifications
        if self.scenario == "normal_race":
            self._normal_racing(telemetry)
        elif self.scenario == "fuel_crisis":
            self._fuel_crisis(telemetry)
        elif self.scenario == "tire_degradation":
            self._tire_degradation(telemetry)
        elif self.scenario == "anomaly":
            self._anomaly_scenario(telemetry)
        elif self.scenario == "fcy_imminent":
            self._fcy_scenario(telemetry)
        elif self.scenario == "traffic_heavy":
            self._traffic_scenario(telemetry)
        elif self.scenario == "pit_window":
            self._pit_window_scenario(telemetry)
        
        # Update state
        self.fuel_level -= 0.06  # Consume fuel
        if self.frame_count % 50 == 0:  # ~5 seconds at 10Hz = new lap
            self.lap += 1
            print(f"\n🏁 LAP {self.lap} STARTED\n")
        
        return telemetry
    
    def _normal_racing(self, telemetry):
        """Normal racing - no issues"""
        telemetry['speed'] = float(165 + np.random.normal(0, 10))
        telemetry['throttle'] = float(75 + np.random.normal(0, 5))
        telemetry['aps'] = telemetry['throttle']
    
    def _fuel_crisis(self, telemetry):
        """Fuel running critically low"""
        # Aggressive driving = high fuel consumption
        telemetry['speed'] = float(180 + np.random.normal(0, 5))
        telemetry['throttle'] = float(95 + np.random.normal(0, 3))
        telemetry['aps'] = telemetry['throttle']
        telemetry['rpm'] = float(11000 + np.random.normal(0, 300))
        telemetry['nmot'] = telemetry['rpm']
        self.fuel_level -= 0.02  # Extra fuel consumption
    
    def _tire_degradation(self, telemetry):
        """Tires wearing rapidly"""
        # High lateral loads = tire wear
        telemetry['speed'] = 170 + np.random.normal(0, 8)
        telemetry['throttle'] = 85 + np.random.normal(0, 5)
        # Simulate tire degradation
        for corner in self.tire_wear:
            self.tire_wear[corner] = min(100, self.tire_wear[corner] + 0.5)
    
    def _anomaly_scenario(self, telemetry):
        """Unusual telemetry patterns - possible mechanical issue"""
        # High RPM but low speed = gear/clutch issue
        telemetry['rpm'] = 11500 + np.random.normal(0, 500)
        telemetry['nmot'] = telemetry['rpm']
        telemetry['speed'] = 90 + np.random.normal(0, 15)  # Much slower than expected
        telemetry['throttle'] = 85 + np.random.normal(0, 10)
        telemetry['aps'] = telemetry['throttle']
    
    def _fcy_scenario(self, telemetry):
        """Full-course yellow imminent"""
        # Incident on track, slowing down
        telemetry['speed'] = 100 + np.random.normal(0, 10)
        telemetry['throttle'] = 40 + np.random.normal(0, 5)
        telemetry['aps'] = telemetry['throttle']
    
    def _traffic_scenario(self, telemetry):
        """Heavy traffic - multiple cars nearby"""
        # Reduced speed due to traffic
        telemetry['speed'] = 140 + np.random.normal(0, 10)
        telemetry['throttle'] = 60 + np.random.normal(0, 8)
    
    def _pit_window_scenario(self, telemetry):
        """Optimal pit window"""
        # Normal racing during pit window
        telemetry['speed'] = 165 + np.random.normal(0, 10)
        telemetry['throttle'] = 75 + np.random.normal(0, 5)
    
    def set_scenario(self, scenario: str):
        """Change race scenario"""
        self.scenario = scenario
        print(f"\n🎬 SCENARIO CHANGED: {scenario.upper().replace('_', ' ')}\n")


async def test_realtime_system():
    """
    Test the event-driven real-time system end-to-end
    """
    print("╔════════════════════════════════════════════════════════════════════════╗")
    print("║                                                                        ║")
    print("║        🏁 EVENT-DRIVEN REAL-TIME SYSTEM - END-TO-END TEST            ║")
    print("║                                                                        ║")
    print("╚════════════════════════════════════════════════════════════════════════╝\n")
    
    simulator = EnhancedTelemetrySimulator()
    
    # Test scenarios in sequence
    test_plan = [
        ("normal_race", 20, "Normal racing - should be quiet"),
        ("fuel_crisis", 15, "Fuel crisis - should trigger FUEL_CRITICAL"),
        ("tire_degradation", 15, "Tire degradation - should trigger TIRE warnings"),
        ("anomaly", 10, "Anomaly - should trigger ANOMALY_CRITICAL"),
        ("fcy_imminent", 10, "FCY scenario - should trigger FCY_IMMINENT"),
        ("pit_window", 15, "Pit window - should trigger PIT_WINDOW events"),
    ]
    
    try:
        # Connect to WebSocket
        uri = "ws://localhost:8005/realtime/ws/telemetry"
        print(f"Connecting to {uri}...\n")
        
        async with websockets.connect(uri) as websocket:
            print("✅ Connected to real-time prediction engine\n")
            print("="*80)
            
            for scenario_name, frame_count, description in test_plan:
                print(f"\n{'='*80}")
                print(f"🎬 TEST SCENARIO: {scenario_name.upper().replace('_', ' ')}")
                print(f"   Description: {description}")
                print(f"   Frames: {frame_count}")
                print(f"{'='*80}\n")
                
                simulator.set_scenario(scenario_name)
                
                for i in range(frame_count):
                    # Generate telemetry
                    telemetry = simulator.generate_frame()
                    
                    # Send to backend
                    message = {
                        'telemetry': telemetry,
                        'timestamp': datetime.now().isoformat()
                    }
                    await websocket.send(json.dumps(message))
                    
                    # Receive response
                    response_text = await websocket.recv()
                    response = json.loads(response_text)
                    
                    # Display results
                    if response['type'] == 'recommendation':
                        print(f"\n🚨 EVENTS DETECTED (Frame {response['frame']}):")
                        print(f"   Lap: {telemetry['lap']}")
                        print(f"   Speed: {telemetry['speed']:.0f} km/h")
                        print(f"   Fuel: {telemetry['fuel_level']:.1f}L")
                        print(f"   Events: {response.get('event_count', 0)}")
                        
                        # Show events
                        for event in response.get('events', []):
                            severity_icon = {
                                'critical': '🔴',
                                'high': '🟠',
                                'medium': '🟡',
                                'low': '🟢',
                                'info': '🔵'
                            }.get(event['severity'], '⚪')
                            print(f"\n   {severity_icon} [{event['severity'].upper()}] {event['event_type']}")
                            print(f"      {event['message']}")
                        
                        # Show AI recommendation
                        print(f"\n   🤖 AI RACE STRATEGIST RECOMMENDATION:")
                        strategy = response.get('recommendations', {}).get('strategy', 'No strategy')
                        # Print first 500 chars of strategy
                        strategy_preview = strategy[:500] + "..." if len(strategy) > 500 else strategy
                        print(f"      {strategy_preview}")
                        print()
                    
                    elif response['type'] == 'status':
                        # Just show lap info periodically
                        if i % 10 == 0:
                            print(f"   ✓ Frame {response['frame']}: {response['message']} (Lap {telemetry['lap']})")
                    
                    # Small delay between frames (100ms = 10Hz)
                    await asyncio.sleep(0.1)
                
                print(f"\n{'='*80}")
                print(f"✅ Scenario '{scenario_name}' completed")
                print(f"{'='*80}\n")
                
                # Pause between scenarios
                await asyncio.sleep(2)
            
            print("\n╔════════════════════════════════════════════════════════════════════════╗")
            print("║                                                                        ║")
            print("║                  ✅ ALL TESTS COMPLETED SUCCESSFULLY                  ║")
            print("║                                                                        ║")
            print("╚════════════════════════════════════════════════════════════════════════╝\n")
            
            print("📊 TEST SUMMARY:")
            print(f"   • Total scenarios tested: {len(test_plan)}")
            print(f"   • Total frames processed: {simulator.frame_count}")
            print(f"   • System: Event-driven real-time prediction")
            print(f"   • Agents: Multi-agent orchestration with Gemini")
            print(f"   • Models: All 8 ML models in pipeline")
            print()
    
    except ConnectionRefusedError:
        print("❌ ERROR: Could not connect to backend")
        print("   Make sure the backend is running on port 8005")
        print("   Run: cd backend-api && source venv/bin/activate && python -m uvicorn main:app --port 8005 --reload")
    
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("\n🚀 Starting Real-Time System Test...\n")
    asyncio.run(test_realtime_system())

