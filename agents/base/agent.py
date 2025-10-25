"""
Base agent class for Cognirace
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json


@dataclass
class Message:
    """Agent message"""
    role: str  # 'user', 'assistant', 'system'
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }


class BaseAgent:
    """Base class for all Cognirace agents"""
    
    def __init__(self, name: str, role: str, api_client=None):
        self.name = name
        self.role = role
        self.api_client = api_client
        self.conversation_history: List[Message] = []
        self.max_history = 10
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        """Add message to conversation history"""
        msg = Message(role=role, content=content, metadata=metadata or {})
        self.conversation_history.append(msg)
        
        # Keep only last N messages
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get conversation history"""
        return [msg.to_dict() for msg in self.conversation_history]
    
    def process(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Process user query - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement process()")
    
    def call_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool/function"""
        if not self.api_client:
            return {"error": "No API client available"}
        
        # Map tool names to API client methods
        tool_map = {
            "predict_fuel_consumption": self.api_client.predict_fuel,
            "predict_tire_degradation": self.api_client.predict_tire,
            "predict_traffic_impact": self.api_client.predict_traffic,
        }
        
        if tool_name not in tool_map:
            return {"error": f"Unknown tool: {tool_name}"}
        
        try:
            return tool_map[tool_name](**parameters)
        except Exception as e:
            return {"error": str(e)}
    
    def format_telemetry_context(self, telemetry: Dict[str, Any]) -> str:
        """Format telemetry data for LLM context"""
        return f"""Current Telemetry:
- Speed: {telemetry.get('speed', 0):.1f} km/h
- RPM: {telemetry.get('nmot', 0):.0f}
- Gear: {telemetry.get('gear', 0)}
- Throttle: {telemetry.get('aps', 0):.1f}%
- Lap: {telemetry.get('lap', 0)}
- Brake Pressure (F): {telemetry.get('pbrake_f', 0):.1f} bar
- Brake Pressure (R): {telemetry.get('pbrake_r', 0):.1f} bar
"""
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', role='{self.role}')"

