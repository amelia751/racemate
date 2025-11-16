"""
Update Vapi AI Assistant Configuration
Updates the Camry assistant with racing strategy prompt and tools
"""

import requests
import json
import os
from dotenv import load_dotenv

# Load environment
load_dotenv('backend-api/.env.local')

VAPI_PRIVATE_KEY = os.getenv('VAPI_PRIVATE_KEY')
VAPI_BASE_URL = "https://api.vapi.ai"

# Racing strategist system prompt
RACING_PROMPT = """You are an expert AI race strategist and engineer for the Toyota GR Cup Series. Your name is Camry (after the Toyota Camry racing in the series).

Your role is to provide real-time strategic advice to drivers during races, focusing on:

**Core Responsibilities:**
1. **Fuel Management**: Monitor fuel levels, predict consumption, recommend fuel-saving techniques
2. **Tire Strategy**: Analyze tire degradation, predict grip loss, advise on pit timing
3. **Race Strategy**: Provide comprehensive race status, position updates, and strategic decisions
4. **Performance Analysis**: Interpret telemetry data and provide actionable insights

**Communication Style:**
- Be concise and direct - drivers need quick, clear information
- Use racing terminology naturally (e.g., "lift and coast", "short-shift", "attack mode")
- Prioritize safety and car preservation while maximizing performance
- Provide specific, actionable recommendations
- Stay calm and professional even in critical situations

**Available Tools:**
You have access to real-time data through these functions:
- `get_telemetry`: Get current speed, RPM, gear, fuel level, tire pressures
- `check_fuel`: Analyze fuel consumption and predict laps remaining
- `check_tires`: Assess tire wear and grip levels
- `race_status`: Get comprehensive race overview with all systems

**Example Interactions:**

Driver: "What's my fuel looking like?"
You: "Let me check your fuel status. [calls check_fuel] You have 35 liters remaining, good for about 14 laps at current pace. You're in good shape - you can push."

Driver: "How are the tires?"
You: "Checking tire condition now. [calls check_tires] Tires are at 25% wear with a grip index of 0.75. Still good performance, but monitor closely in the next 5 laps."

Driver: "Give me a race update"
You: "Running P3 on lap 15. [calls race_status] Fuel: 35L (14 laps remaining). Tires: 25% wear, good condition. Current strategy: Push mode - everything looks good. You can attack."

**Critical Situations:**
- Fuel < 10 laps: Immediate fuel-save mode with specific techniques
- Tire grip < 0.6: Recommend pit stop soon
- Multiple issues: Prioritize safety, then strategy

Remember: You're the driver's trusted co-pilot. Be their eyes on the data while they focus on driving."""

def update_assistant():
    """Update the Camry assistant configuration"""
    
    headers = {
        "Authorization": f"Bearer {VAPI_PRIVATE_KEY}",
        "Content-Type": "application/json"
    }
    
    # First, get existing assistants to find Camry
    print("🔍 Finding Camry assistant...")
    response = requests.get(
        f"{VAPI_BASE_URL}/assistant",
        headers=headers
    )
    
    if response.status_code != 200:
        print(f"❌ Failed to get assistants: {response.status_code}")
        print(response.text)
        return False
    
    assistants = response.json()
    camry_assistant = None
    
    for assistant in assistants:
        if assistant.get('name', '').lower() == 'camry':
            camry_assistant = assistant
            break
    
    if not camry_assistant:
        print("❌ Camry assistant not found!")
        print("Available assistants:", [a.get('name') for a in assistants])
        return False
    
    assistant_id = camry_assistant['id']
    print(f"✅ Found Camry assistant: {assistant_id}")
    
    # Update assistant configuration (tools will be added later with HTTPS backend)
    print("📝 Updating assistant configuration...")
    print("⚠️  Note: Server tools require HTTPS. For hackathon demo, using prompt-only mode.")
    print("    Tools can be added later with ngrok or deployed backend.")
    
    update_payload = {
        "model": {
            "provider": "openai",
            "model": "gpt-4",
            "messages": [
                {
                    "role": "system",
                    "content": RACING_PROMPT
                }
            ]
        },
        "voice": {
            "provider": "11labs",
            "voiceId": "21m00Tcm4TlvDq8ikWAM"  # Rachel voice - professional female
        },
        "firstMessage": "Hey! I'm Camry, your AI race strategist. I'm monitoring your race. How can I help you?",
        "recordingEnabled": False,
        "serverMessages": ["end-of-call-report"]
    }
    
    response = requests.patch(
        f"{VAPI_BASE_URL}/assistant/{assistant_id}",
        headers=headers,
        json=update_payload
    )
    
    if response.status_code == 200:
        print("✅ Successfully updated Camry assistant!")
        print("📊 Configuration:")
        print(f"  - System prompt: {len(RACING_PROMPT)} characters")
        print(f"  - Model: GPT-4")
        print(f"  - Voice: ElevenLabs Rachel")
        print(f"  - First message configured")
        return True
    else:
        print(f"❌ Failed to update assistant: {response.status_code}")
        print(response.text)
        return False

if __name__ == "__main__":
    print("🏁 Cognirace - Vapi Assistant Configuration")
    print("=" * 60)
    
    if not VAPI_PRIVATE_KEY:
        print("❌ VAPI_PRIVATE_KEY not found in .env.local")
        exit(1)
    
    success = update_assistant()
    
    print("=" * 60)
    if success:
        print("✅ Assistant configuration complete!")
    else:
        print("❌ Configuration failed. Check errors above.")

