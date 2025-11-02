"""
FuelAgent - Specialized agent for fuel strategy
"""

from typing import Dict, Any, Optional
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from base.agent import BaseAgent


class FuelAgent(BaseAgent):
    """Agent specialized in fuel strategy and consumption analysis"""
    
    def __init__(self, api_client=None, use_gemini=True):
        super().__init__(
            name="FuelAgent",
            role="Fuel Strategy Specialist",
            api_client=api_client,
            use_gemini=use_gemini
        )
        
        # Fuel strategy parameters
        self.tank_capacity = 50.0  # liters (example)
        self.safety_margin = 2.0   # liters
    
    def _get_expertise_description(self) -> str:
        return "Fuel consumption analysis, pit timing, and fuel-saving strategies"
    
    def process(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Process fuel-related queries with Gemini"""
        
        self.add_message("user", query)
        
        if not context or 'telemetry' not in context:
            response = "I need telemetry data to analyze fuel consumption."
            self.add_message("assistant", response)
            return response
        
        telemetry = context['telemetry']
        
        # Get fuel prediction from API - NO FALLBACK FOR SAFETY
        if not self.api_client:
            response = "CRITICAL ERROR: API client not available. CANNOT provide fuel analysis. SYSTEM UNSAFE FOR RACING."
            self.add_message("assistant", response)
            return response
        
        try:
            fuel_result = self.api_client.predict_fuel(
                speed=telemetry.get('speed'),
                nmot=telemetry.get('nmot'),
                gear=telemetry.get('gear'),
                aps=telemetry.get('aps'),
                lap=telemetry.get('lap', 1)
            )
            
            if fuel_result.get('error'):
                response = f"CRITICAL ERROR: Fuel prediction failed - {fuel_result['error']}. SYSTEM UNSAFE FOR RACING."
                self.add_message("assistant", response)
                return response
                
        except Exception as e:
            response = f"CRITICAL ERROR: Fuel prediction API failure - {str(e)}. SYSTEM UNSAFE FOR RACING."
            self.add_message("assistant", response)
            return response
        
        # Analyze fuel strategy
        if "error" not in fuel_result:
            burn_rate = fuel_result.get('prediction', 0.5)
            current_lap = telemetry.get('lap', 1)
            total_laps = context.get('race_info', {}).get('total_laps', 40)
            
            laps_remaining = total_laps - current_lap
            fuel_needed = laps_remaining * burn_rate
            current_fuel = telemetry.get('fuel_level', 30)
            
            # Use Gemini for natural language response
            if self.use_gemini:
                analysis_data = {
                    "burn_rate": burn_rate,
                    "current_fuel": current_fuel,
                    "laps_remaining": laps_remaining,
                    "fuel_needed": fuel_needed,
                    "margin": current_fuel - fuel_needed,
                    "safety_margin": self.safety_margin
                }
                
                enhanced_context = context.copy()
                enhanced_context['fuel_analysis'] = analysis_data
                
                response = self.generate_with_gemini(query, enhanced_context)
            else:
                response = self._generate_fuel_strategy(
                    burn_rate, current_fuel, laps_remaining, fuel_needed
                )
        else:
            response = f"Fuel prediction error: {fuel_result.get('error')}"
        
        self.add_message("assistant", response, metadata={"fuel_result": fuel_result})
        return response
    
    def _generate_fuel_strategy(self, burn_rate: float, current_fuel: float,
                                laps_remaining: int, fuel_needed: float) -> str:
        """Generate fuel strategy recommendation"""
        
        fuel_margin = current_fuel - fuel_needed
        
        if fuel_margin < self.safety_margin:
            urgency = "CRITICAL"
            recommendation = f"⚠️ **FUEL CRITICAL** - Immediate pit stop recommended!"
        elif fuel_margin < 5.0:
            urgency = "WARNING"
            recommendation = f"⚠️ **FUEL LOW** - Pit within next 3-5 laps"
        else:
            urgency = "OK"
            max_laps = int((current_fuel - self.safety_margin) / burn_rate)
            recommendation = f"✓ **FUEL OK** - Can run {max_laps} more laps"
        
        response = f"""**Fuel Strategy Analysis**

Current Status: {urgency}
- Burn Rate: {burn_rate:.3f} L/lap
- Laps Remaining: {laps_remaining}
- Fuel Needed: {fuel_needed:.2f} L
- Current Fuel: {current_fuel:.2f} L
- Margin: {fuel_margin:.2f} L

{recommendation}

💡 Fuel-saving tips:
- Lift & coast into braking zones
- Use higher gears when possible
- Smooth throttle application
"""
        
        return response


if __name__ == "__main__":
    # Test the agent
    from tools.api_client import CogniraceAPIClient
    
    api_client = CogniraceAPIClient()
    agent = FuelAgent(api_client)
    
    # Mock telemetry
    context = {
        "telemetry": {
            "speed": 180.5,
            "nmot": 7200,
            "gear": 5,
            "aps": 95.2,
            "lap": 15,
            "fuel_level": 28.0
        },
        "race_info": {
            "total_laps": 40
        }
    }
    
    print("Testing FuelAgent...")
    response = agent.process("What's our fuel situation?", context)
    print(response)

