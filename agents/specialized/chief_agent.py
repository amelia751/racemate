"""
ChiefAgent - Orchestrator that coordinates all specialized agents
"""

from typing import Dict, Any, Optional, List
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from base.agent import BaseAgent
from specialized.fuel_agent import FuelAgent
from specialized.tire_agent import TireAgent
from specialized.telemetry_agent import TelemetryAgent


class ChiefAgent(BaseAgent):
    """Chief orchestrator agent that coordinates specialized agents"""
    
    def __init__(self, api_client=None, use_gemini=True):
        super().__init__(
            name="ChiefAgent",
            role="Race Strategy Coordinator",
            api_client=api_client,
            use_gemini=use_gemini
        )
        
        # Initialize specialized agents with Gemini
        self.fuel_agent = FuelAgent(api_client, use_gemini=use_gemini)
        self.tire_agent = TireAgent(api_client, use_gemini=use_gemini)
        self.telemetry_agent = TelemetryAgent(api_client)
        
        self.agents = {
            "fuel": self.fuel_agent,
            "tire": self.tire_agent,
            "telemetry": self.telemetry_agent
        }
    
    def _get_expertise_description(self) -> str:
        return "Overall race strategy coordination, pit wall decision-making, and agent orchestration"
    
    def route_query(self, query: str) -> str:
        """Determine which agent should handle the query"""
        query_lower = query.lower()
        
        # Fuel-related keywords
        if any(word in query_lower for word in ['fuel', 'pit', 'consumption', 'laps remaining']):
            return "fuel"
        
        # Tire-related keywords
        if any(word in query_lower for word in ['tire', 'tyre', 'grip', 'degradation', 'wear']):
            return "tire"
        
        # Telemetry-related keywords
        if any(word in query_lower for word in ['telemetry', 'data', 'speed', 'rpm', 'current']):
            return "telemetry"
        
        # Default to comprehensive analysis
        return "comprehensive"
    
    def process(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Process query by routing to appropriate agent(s) with Gemini orchestration"""
        
        self.add_message("user", query)
        
        # Determine routing
        route = self.route_query(query)
        
        if route == "comprehensive":
            # Get insights from all agents
            responses = self._comprehensive_analysis(context)
            
            # Use Gemini to synthesize comprehensive response
            if self.use_gemini:
                synthesis_prompt = f"Synthesize these racing insights into a comprehensive pit wall briefing:\n\n{responses}"
                response = self.generate_with_gemini(synthesis_prompt, context)
            else:
                response = self._format_comprehensive_response(responses)
        else:
            # Route to specific agent
            agent = self.agents.get(route)
            if agent:
                response = agent.process(query, context)
            else:
                response = f"Unable to route query: {query}"
        
        self.add_message("assistant", response)
        return response
    
    def _comprehensive_analysis(self, context: Optional[Dict[str, Any]]) -> Dict[str, str]:
        """Get analysis from all specialized agents"""
        responses = {}
        
        if context:
            responses["telemetry"] = self.telemetry_agent.process(
                "Show current telemetry", context
            )
            responses["fuel"] = self.fuel_agent.process(
                "Analyze fuel situation", context
            )
            responses["tire"] = self.tire_agent.process(
                "Analyze tire condition", context
            )
        
        return responses
    
    def _format_comprehensive_response(self, responses: Dict[str, str]) -> str:
        """Format comprehensive analysis"""
        
        header = """**🏁 Cognirace Pit Wall - Comprehensive Analysis**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"""
        
        sections = []
        
        if "telemetry" in responses:
            sections.append(f"**📡 TELEMETRY STATUS**\n{responses['telemetry']}")
        
        if "fuel" in responses:
            sections.append(f"\n**⛽ FUEL ANALYSIS**\n{responses['fuel']}")
        
        if "tire" in responses:
            sections.append(f"\n**🛞 TIRE ANALYSIS**\n{responses['tire']}")
        
        footer = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💬 Ask me about fuel, tires, or telemetry for specific insights!
"""
        
        return header + "\n".join(sections) + footer
    
    def add_telemetry(self, telemetry: Dict[str, Any]):
        """Add telemetry to telemetry agent's buffer"""
        self.telemetry_agent.add_telemetry(telemetry)
    
    def get_strategy_recommendation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get strategic recommendation based on all data"""
        
        # Get predictions from all agents
        fuel_response = self.fuel_agent.process("Analyze fuel", context)
        tire_response = self.tire_agent.process("Analyze tires", context)
        
        # Simple strategy logic (would be more sophisticated with LLM)
        recommendation = {
            "should_pit": False,
            "reason": "Continue current stint",
            "urgency": "LOW",
            "fuel_analysis": fuel_response,
            "tire_analysis": tire_response
        }
        
        # Check if pit is needed
        if "CRITICAL" in fuel_response or "CRITICAL" in tire_response:
            recommendation["should_pit"] = True
            recommendation["urgency"] = "HIGH"
            recommendation["reason"] = "Critical fuel or tire condition"
        elif "WARNING" in fuel_response or "WARNING" in tire_response:
            recommendation["urgency"] = "MEDIUM"
            recommendation["reason"] = "Consider pitting soon"
        
        return recommendation


if __name__ == "__main__":
    # Test the ChiefAgent
    from tools.api_client import CogniraceAPIClient
    
    api_client = CogniraceAPIClient()
    chief = ChiefAgent(api_client)
    
    # Add some telemetry
    for i in range(10):
        telemetry = {
            "speed": 150 + i * 3,
            "nmot": 6000 + i * 100,
            "gear": 5,
            "aps": 85 + i,
            "lap": 15,
            "cum_brake_energy": 2000 + i * 100,
            "cum_lateral_load": 2500 + i * 120
        }
        chief.add_telemetry(telemetry)
    
    # Test comprehensive analysis
    context = {
        "telemetry": {
            "speed": 180.5,
            "nmot": 7200,
            "gear": 5,
            "aps": 95.2,
            "lap": 15,
            "fuel_level": 28.0,
            "cum_brake_energy": 2500,
            "cum_lateral_load": 3200
        },
        "race_info": {
            "total_laps": 40
        },
        "last_pit_lap": 5,
        "weather": {
            "air_temp": 28.5
        }
    }
    
    print("Testing ChiefAgent...")
    print("\n" + "="*70)
    response = chief.process("Give me a full analysis", context)
    print(response)
    
    print("\n" + "="*70)
    print("\nStrategy Recommendation:")
    strategy = chief.get_strategy_recommendation(context)
    print(f"Should Pit: {strategy['should_pit']}")
    print(f"Urgency: {strategy['urgency']}")
    print(f"Reason: {strategy['reason']}")

