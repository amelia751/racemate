"""
Base agent class for Cognirace with Gemini 1.5 integration
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json
import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), '../.env.local'))

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler(os.getenv('CONVERSATION_LOG_FILE', '/tmp/agent_conversations.log')),
        logging.StreamHandler()
    ]
)


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
    """Base class for all Cognirace agents with Gemini 1.5 integration"""
    
    def __init__(self, name: str, role: str, api_client=None, use_gemini: bool = True):
        self.name = name
        self.role = role
        self.api_client = api_client
        self.conversation_history: List[Message] = []
        self.max_history = 10
        self.use_gemini = use_gemini and GEMINI_AVAILABLE
        self.logger = logging.getLogger(f"Agent.{name}")
        
        # Initialize Gemini if available
        self.gemini_model = None
        if self.use_gemini:
            try:
                api_key = os.getenv('GOOGLE_API_KEY')
                if api_key:
                    genai.configure(api_key=api_key)
                    # Use gemini-2.5-flash - fast and cost-effective
                    self.gemini_model = genai.GenerativeModel('gemini-2.5-flash')
                    self.logger.info(f"✓ Gemini 2.5 Flash initialized for {name}")
                else:
                    self.logger.warning(f"⚠️  GOOGLE_API_KEY not found for {name}")
                    self.use_gemini = False
            except Exception as e:
                self.logger.error(f"✗ Failed to initialize Gemini: {e}")
                self.use_gemini = False
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        """Add message to conversation history and log it"""
        msg = Message(role=role, content=content, metadata=metadata or {})
        self.conversation_history.append(msg)
        
        # Log the message
        self.logger.info(f"[{self.name}] {role.upper()}: {content[:200]}...")
        
        # Keep only last N messages
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get conversation history"""
        return [msg.to_dict() for msg in self.conversation_history]
    
    def generate_with_gemini(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Generate response using Gemini 1.5"""
        if not self.use_gemini or not self.gemini_model:
            return self._fallback_response(prompt, context)
        
        try:
            # Build context-aware prompt
            full_prompt = self._build_gemini_prompt(prompt, context)
            
            # Generate response
            response = self.gemini_model.generate_content(full_prompt)
            
            if response and response.text:
                self.logger.info(f"[{self.name}] Gemini generated response ({len(response.text)} chars)")
                return response.text
            else:
                self.logger.warning(f"[{self.name}] Gemini returned empty response")
                return self._fallback_response(prompt, context)
                
        except Exception as e:
            self.logger.error(f"[{self.name}] Gemini generation failed: {e}")
            return self._fallback_response(prompt, context)
    
    def _build_gemini_prompt(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Build context-aware prompt for Gemini"""
        prompt_parts = []
        
        # Agent identity
        prompt_parts.append(f"You are {self.name}, a {self.role} for a professional racing team.")
        prompt_parts.append(f"Your expertise: {self._get_expertise_description()}")
        prompt_parts.append("")
        
        # Context
        if context:
            prompt_parts.append("**Current Race Context:**")
            if 'telemetry' in context:
                prompt_parts.append(self.format_telemetry_context(context['telemetry']))
            if 'race_info' in context:
                prompt_parts.append(f"Total Laps: {context['race_info'].get('total_laps', 'N/A')}")
            prompt_parts.append("")
        
        # Conversation history
        if self.conversation_history:
            prompt_parts.append("**Recent Conversation:**")
            for msg in self.conversation_history[-3:]:  # Last 3 messages
                prompt_parts.append(f"{msg.role.capitalize()}: {msg.content}")
            prompt_parts.append("")
        
        # Current query
        prompt_parts.append("**Current Query:**")
        prompt_parts.append(query)
        prompt_parts.append("")
        prompt_parts.append("Provide a concise, actionable response as a racing professional would.")
        
        return "\n".join(prompt_parts)
    
    def _get_expertise_description(self) -> str:
        """Get agent's expertise description - override in subclasses"""
        return "General race strategy and analysis"
    
    def _fallback_response(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Fallback response when Gemini is unavailable - BASIC ANALYSIS ONLY"""
        return (
            f"[{self.name} - BASIC MODE: AI Analysis Unavailable]\n"
            f"Query: {query}\n"
            f"WARNING: Operating without AI-powered natural language analysis. "
            f"Recommendations may be less contextual."
        )
    
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

