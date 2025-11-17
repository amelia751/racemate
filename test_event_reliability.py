"""
Test Event Detection Reliability
Runs multiple streaming sessions to verify events are consistently detected
"""

import asyncio
import websockets
import json
import numpy as np
from datetime import datetime

class ReliabilityTester:
    def __init__(self):
        self.total_tests = 0
        self.events_detected = 0
        self.frames_sent = 0
        
    async def test_streaming_session(self, session_num: int, frames: int = 50):
        """Test a single streaming session"""
        print(f"\n{'='*80}")
        print(f"🧪 TEST SESSION {session_num}")
        print(f"{'='*80}")
        
        uri = "ws://localhost:8005/realtime/ws/telemetry"
        session_events = 0
        
        try:
            async with websockets.connect(uri) as websocket:
                print(f"✅ Connected to backend")
                
                for frame in range(frames):
                    # Generate telemetry with varying conditions to trigger events
                    if frame < 10:
                        # Normal racing
                        telemetry = self._generate_normal_telemetry(frame)
                    elif frame < 20:
                        # High fuel consumption
                        telemetry = self._generate_high_fuel_telemetry(frame)
                    elif frame < 30:
                        # Anomaly scenario
                        telemetry = self._generate_anomaly_telemetry(frame)
                    elif frame < 40:
                        # Tire wear
                        telemetry = self._generate_tire_wear_telemetry(frame)
                    else:
                        # Mixed conditions
                        telemetry = self._generate_mixed_telemetry(frame)
                    
                    # Send telemetry
                    message = {
                        'telemetry': telemetry,
                        'timestamp': datetime.now().isoformat()
                    }
                    await websocket.send(json.dumps(message))
                    self.frames_sent += 1
                    
                    # Try to receive response (with timeout)
                    try:
                        response_text = await asyncio.wait_for(websocket.recv(), timeout=0.5)
                        response = json.loads(response_text)
                        
                        if response.get('type') == 'recommendation':
                            session_events += 1
                            self.events_detected += 1
                            event_count = response.get('event_count', 0)
                            print(f"   🚨 Frame {frame}: {event_count} event(s) detected")
                    except asyncio.TimeoutError:
                        # No response yet, continue
                        pass
                    
                    # Small delay between frames
                    await asyncio.sleep(0.05)
                
                print(f"\n📊 Session {session_num} Results:")
                print(f"   • Frames sent: {frames}")
                print(f"   • Events detected: {session_events}")
                print(f"   • Detection rate: {(session_events/frames)*100:.1f}%")
                
                return session_events > 0
                
        except Exception as e:
            print(f"❌ Session {session_num} failed: {e}")
            return False
    
    def _generate_normal_telemetry(self, frame):
        """Normal racing telemetry"""
        return {
            'lap': 1,
            'speed': float(165 + np.random.normal(0, 10)),
            'rpm': float(8000 + np.random.normal(0, 300)),
            'nmot': float(8000 + np.random.normal(0, 300)),
            'gear': int(np.random.choice([4, 5, 6])),
            'throttle': float(75 + np.random.normal(0, 5)),
            'aps': float(75 + np.random.normal(0, 5)),
            'fuel_level': float(50 - frame * 0.06),
            'timestamp': datetime.now().isoformat()
        }
    
    def _generate_high_fuel_telemetry(self, frame):
        """High fuel consumption scenario"""
        return {
            'lap': 1,
            'speed': float(180 + np.random.normal(0, 5)),
            'rpm': float(11000 + np.random.normal(0, 300)),  # High RPM
            'nmot': float(11000 + np.random.normal(0, 300)),
            'gear': int(np.random.choice([5, 6])),
            'throttle': float(95 + np.random.normal(0, 3)),  # Full throttle
            'aps': float(95 + np.random.normal(0, 3)),
            'fuel_level': float(50 - frame * 0.08),  # Higher consumption
            'timestamp': datetime.now().isoformat()
        }
    
    def _generate_anomaly_telemetry(self, frame):
        """Anomaly scenario - high RPM but low speed"""
        return {
            'lap': 1,
            'speed': float(90 + np.random.normal(0, 15)),  # Low speed
            'rpm': float(11500 + np.random.normal(0, 500)),  # Very high RPM
            'nmot': float(11500 + np.random.normal(0, 500)),
            'gear': int(np.random.choice([3, 4])),
            'throttle': float(85 + np.random.normal(0, 10)),
            'aps': float(85 + np.random.normal(0, 10)),
            'fuel_level': float(50 - frame * 0.06),
            'timestamp': datetime.now().isoformat()
        }
    
    def _generate_tire_wear_telemetry(self, frame):
        """High tire wear scenario"""
        return {
            'lap': int(15 + frame / 10),  # Higher lap number
            'speed': float(170 + np.random.normal(0, 8)),
            'rpm': float(8500 + np.random.normal(0, 400)),
            'nmot': float(8500 + np.random.normal(0, 400)),
            'gear': int(np.random.choice([4, 5])),
            'throttle': float(85 + np.random.normal(0, 5)),
            'aps': float(85 + np.random.normal(0, 5)),
            'fuel_level': float(50 - frame * 0.06),
            'timestamp': datetime.now().isoformat()
        }
    
    def _generate_mixed_telemetry(self, frame):
        """Mixed conditions"""
        return {
            'lap': int(10 + frame / 10),
            'speed': float(160 + np.random.normal(0, 20)),
            'rpm': float(9000 + np.random.normal(0, 1000)),
            'nmot': float(9000 + np.random.normal(0, 1000)),
            'gear': int(np.random.choice([3, 4, 5, 6])),
            'throttle': float(70 + np.random.normal(0, 15)),
            'aps': float(70 + np.random.normal(0, 15)),
            'fuel_level': float(50 - frame * 0.065),
            'timestamp': datetime.now().isoformat()
        }

async def run_reliability_test():
    """Run multiple streaming sessions to test reliability"""
    print("\n╔════════════════════════════════════════════════════════════════════════╗")
    print("║                                                                        ║")
    print("║           🧪 EVENT DETECTION RELIABILITY TEST                         ║")
    print("║                                                                        ║")
    print("╚════════════════════════════════════════════════════════════════════════╝")
    
    tester = ReliabilityTester()
    num_sessions = 5
    frames_per_session = 50
    successful_sessions = 0
    
    for i in range(1, num_sessions + 1):
        success = await tester.test_streaming_session(i, frames_per_session)
        if success:
            successful_sessions += 1
        
        # Wait between sessions
        if i < num_sessions:
            print(f"\n⏳ Waiting 3 seconds before next session...")
            await asyncio.sleep(3)
    
    # Final results
    print(f"\n{'='*80}")
    print(f"📊 FINAL RELIABILITY REPORT")
    print(f"{'='*80}")
    print(f"Total sessions: {num_sessions}")
    print(f"Successful sessions (events detected): {successful_sessions}")
    print(f"Success rate: {(successful_sessions/num_sessions)*100:.1f}%")
    print(f"Total frames sent: {tester.frames_sent}")
    print(f"Total events detected: {tester.events_detected}")
    print(f"Overall detection rate: {(tester.events_detected/tester.frames_sent)*100:.1f}%")
    
    if successful_sessions == num_sessions:
        print(f"\n✅ EXCELLENT: All sessions detected events!")
    elif successful_sessions >= num_sessions * 0.8:
        print(f"\n✅ GOOD: {(successful_sessions/num_sessions)*100:.0f}% of sessions detected events")
    else:
        print(f"\n⚠️  NEEDS IMPROVEMENT: Only {(successful_sessions/num_sessions)*100:.0f}% of sessions detected events")
    
    print(f"\n{'='*80}\n")

if __name__ == "__main__":
    asyncio.run(run_reliability_test())

