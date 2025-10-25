"""
TelemetryAgent - Data retrieval and formatting specialist
"""

from typing import Dict, Any, Optional, List
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from base.agent import BaseAgent


class TelemetryAgent(BaseAgent):
    """Agent specialized in telemetry data retrieval and formatting"""
    
    def __init__(self, api_client=None):
        super().__init__(
            name="TelemetryAgent",
            role="Telemetry Data Specialist"
        )
        self.api_client = api_client
        self.telemetry_buffer: List[Dict[str, Any]] = []
        self.max_buffer_size = 100
    
    def add_telemetry(self, telemetry: Dict[str, Any]):
        """Add telemetry sample to buffer"""
        self.telemetry_buffer.append(telemetry)
        
        # Keep only recent samples
        if len(self.telemetry_buffer) > self.max_buffer_size:
            self.telemetry_buffer = self.telemetry_buffer[-self.max_buffer_size:]
    
    def get_latest_telemetry(self) -> Optional[Dict[str, Any]]:
        """Get most recent telemetry sample"""
        return self.telemetry_buffer[-1] if self.telemetry_buffer else None
    
    def get_telemetry_sequence(self, length: int = 100) -> List[Dict[str, Any]]:
        """Get sequence of telemetry samples"""
        return self.telemetry_buffer[-length:] if self.telemetry_buffer else []
    
    def calculate_statistics(self, window: int = 10) -> Dict[str, Any]:
        """Calculate statistics over recent telemetry"""
        if not self.telemetry_buffer:
            return {}
        
        recent = self.telemetry_buffer[-window:]
        
        stats = {}
        
        # Calculate averages for numeric fields
        numeric_fields = ['speed', 'nmot', 'aps', 'pbrake_f', 'pbrake_r']
        
        for field in numeric_fields:
            values = [t.get(field, 0) for t in recent if field in t]
            if values:
                stats[f'{field}_avg'] = sum(values) / len(values)
                stats[f'{field}_max'] = max(values)
                stats[f'{field}_min'] = min(values)
        
        return stats
    
    def process(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Process telemetry-related queries"""
        
        self.add_message("user", query)
        
        latest = self.get_latest_telemetry()
        
        if not latest:
            response = "No telemetry data available yet."
            self.add_message("assistant", response)
            return response
        
        stats = self.calculate_statistics()
        
        response = f"""**Current Telemetry Summary**

📊 Latest Reading:
{self.format_telemetry_context(latest)}

📈 Recent Statistics (last 10 samples):
- Speed: Avg {stats.get('speed_avg', 0):.1f} km/h (Range: {stats.get('speed_min', 0):.1f}-{stats.get('speed_max', 0):.1f})
- RPM: Avg {stats.get('nmot_avg', 0):.0f} (Range: {stats.get('nmot_min', 0):.0f}-{stats.get('nmot_max', 0):.0f})
- Throttle: Avg {stats.get('aps_avg', 0):.1f}% (Range: {stats.get('aps_min', 0):.1f}-{stats.get('aps_max', 0):.1f})

💾 Buffer Status: {len(self.telemetry_buffer)}/{self.max_buffer_size} samples
"""
        
        self.add_message("assistant", response)
        return response
    
    def format_for_api(self, telemetry: Dict[str, Any]) -> Dict[str, Any]:
        """Format telemetry for API consumption"""
        return {
            "speed": telemetry.get('speed', 0),
            "nmot": telemetry.get('nmot', 0),
            "gear": telemetry.get('gear', 1),
            "aps": telemetry.get('aps', 0),
            "lap": telemetry.get('lap', 1),
            "pbrake_f": telemetry.get('pbrake_f', 0),
            "pbrake_r": telemetry.get('pbrake_r', 0),
            "cum_brake_energy": telemetry.get('cum_brake_energy', 0),
            "cum_lateral_load": telemetry.get('cum_lateral_load', 0)
        }


if __name__ == "__main__":
    # Test the agent
    agent = TelemetryAgent()
    
    # Add some mock telemetry
    for i in range(15):
        telemetry = {
            "speed": 150 + i * 2,
            "nmot": 6000 + i * 100,
            "gear": 4 + (i // 5),
            "aps": 80 + i,
            "lap": 1
        }
        agent.add_telemetry(telemetry)
    
    print("Testing TelemetryAgent...")
    response = agent.process("Show me current telemetry")
    print(response)

