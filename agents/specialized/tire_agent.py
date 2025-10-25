"""
TireAgent - Specialized agent for tire strategy
"""

from typing import Dict, Any, Optional
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from base.agent import BaseAgent


class TireAgent(BaseAgent):
    """Agent specialized in tire strategy and degradation analysis"""
    
    def __init__(self, api_client=None, use_gemini=True):
        super().__init__(
            name="TireAgent",
            role="Tire Strategy Specialist",
            api_client=api_client,
            use_gemini=use_gemini
        )
        
        # Tire thresholds
        self.critical_grip = 0.70
        self.warning_grip = 0.80
    
    def _get_expertise_description(self) -> str:
        return "Tire degradation analysis, grip monitoring, and pit timing recommendations"
    
    def process(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Process tire-related queries"""
        
        self.add_message("user", query)
        
        if not context or 'telemetry' not in context:
            response = "I need telemetry data to analyze tire condition."
            self.add_message("assistant", response)
            return response
        
        telemetry = context['telemetry']
        
        # Get tire prediction from API
        if self.api_client:
            tire_result = self.api_client.predict_tire(
                cum_brake_energy=telemetry.get('cum_brake_energy', 1500),
                cum_lateral_load=telemetry.get('cum_lateral_load', 2000),
                air_temp=context.get('weather', {}).get('air_temp', 25.0)
            )
        else:
            tire_result = {"prediction": 0.85, "status": "mock"}
        
        # Analyze tire strategy
        if "error" not in tire_result:
            grip_index = tire_result.get('prediction', 0.85)
            current_lap = telemetry.get('lap', 1)
            stint_length = current_lap - context.get('last_pit_lap', 0)
            
            response = self._generate_tire_strategy(
                grip_index, stint_length, current_lap, telemetry
            )
        else:
            response = f"Tire prediction error: {tire_result.get('error')}"
        
        self.add_message("assistant", response, metadata={"tire_result": tire_result})
        return response
    
    def _generate_tire_strategy(self, grip_index: float, stint_length: int,
                                current_lap: int, telemetry: Dict) -> str:
        """Generate tire strategy recommendation"""
        
        if grip_index < self.critical_grip:
            urgency = "CRITICAL"
            recommendation = "🔴 **TIRES CRITICAL** - Pit ASAP! Significant time loss!"
        elif grip_index < self.warning_grip:
            urgency = "WARNING"
            recommendation = f"⚠️ **TIRES DEGRADED** - Consider pitting soon"
        else:
            urgency = "OK"
            est_laps_remaining = int((grip_index - self.critical_grip) / 0.02)
            recommendation = f"✓ **TIRES OK** - Estimated {est_laps_remaining} laps remaining"
        
        # Calculate degradation rate
        deg_rate = (1.0 - grip_index) / stint_length if stint_length > 0 else 0
        
        response = f"""**Tire Strategy Analysis**

Current Status: {urgency}
- Grip Index: {grip_index:.3f} (1.0 = new, 0.5 = worn)
- Stint Length: {stint_length} laps
- Degradation Rate: {deg_rate:.4f} per lap

{recommendation}

📊 Tire Wear Factors:
- Brake Energy: {telemetry.get('cum_brake_energy', 0):.0f} units
- Lateral Load: {telemetry.get('cum_lateral_load', 0):.0f} units
- Current Speed: {telemetry.get('speed', 0):.1f} km/h

💡 Tire-saving tips:
- Smooth steering inputs
- Progressive braking
- Avoid unnecessary wheelspin
"""
        
        return response


if __name__ == "__main__":
    # Test the agent
    from tools.api_client import CogniraceAPIClient
    
    api_client = CogniraceAPIClient()
    agent = TireAgent(api_client)
    
    # Mock telemetry
    context = {
        "telemetry": {
            "speed": 165.3,
            "lap": 18,
            "cum_brake_energy": 2500,
            "cum_lateral_load": 3200
        },
        "last_pit_lap": 5,
        "weather": {
            "air_temp": 28.5
        }
    }
    
    print("Testing TireAgent...")
    response = agent.process("How are the tires looking?", context)
    print(response)

