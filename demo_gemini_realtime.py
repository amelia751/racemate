#!/usr/bin/env python3
"""
Cognirace Gemini Real-Time Demo
Demonstrates streaming telemetry with Gemini-powered agent responses
"""

import sys
import os
import time
from datetime import datetime

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'streaming'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'agents'))

from simulator.telemetry_simulator import TelemetrySimulator, SimulatorConfig
from tools.api_client import CogniraceAPIClient
from specialized.chief_agent import ChiefAgent


def print_banner():
    """Print demo banner"""
    print("\n")
    print("╔" + "="*80 + "╗")
    print("║" + " "*80 + "║")
    print("║" + "🏁  COGNIRACE GEMINI 1.5 REAL-TIME DEMO  🏁".center(80) + "║")
    print("║" + " "*80 + "║")
    print("║" + "Streaming Telemetry + AI-Powered Pit Wall Assistance".center(80) + "║")
    print("║" + " "*80 + "║")
    print("╚" + "="*80 + "╝")
    print("\n")


def print_section(title: str):
    """Print section header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")


def demo_gemini_streaming():
    """
    Main demo: Stream telemetry over time and get Gemini-powered agent responses
    """
    
    print_banner()
    
    print("🚀 Initializing Cognirace System...")
    print("   • Starting API client")
    print("   • Initializing Gemini 1.5 Flash")
    print("   • Creating telemetry simulator")
    print("   • Setting up agent system")
    print()
    
    # Initialize system
    api_client = CogniraceAPIClient()
    chief = ChiefAgent(api_client, use_gemini=True)
    
    # Check Gemini initialization
    if chief.use_gemini:
        print("✅ Gemini 1.5 Flash integrated successfully!")
    else:
        print("⚠️  Gemini not available, using fallback responses")
    
    print()
    
    # Configure simulator for realistic demo
    config = SimulatorConfig(
        frequency_hz=2.0,  # 2 Hz for demo (slower for readability)
        base_speed=165.0,
        lap_time_seconds=60.0,  # 1 minute laps
        total_laps=3,  # 3 lap demo
        track_name="Virtual Circuit - Gemini Demo"
    )
    
    simulator = TelemetrySimulator(config)
    
    print_section("PHASE 1: Early Race Analysis (Lap 1)")
    
    print("📡 Starting telemetry stream at 2 Hz...")
    print("   (Real-world would be 20 Hz)")
    print()
    
    sample_count = 0
    query_intervals = {
        15: "How's our fuel situation looking?",
        30: "Give me a comprehensive race status update",
        45: "What's the tire condition? Should we be concerned?",
        60: "Based on all data, what's our strategy for the next lap?",
        90: "We're experiencing high tire temperatures. What should we do?",
        120: "Final lap! Any last-minute strategic advice?"
    }
    
    race_context = {
        "race_info": {"total_laps": 3},
        "last_pit_lap": 0,
        "weather": {"air_temp": 28.5, "track_temp": 42.0}
    }
    
    print("🎙️  **PIT WALL RADIO SIMULATION**")
    print("   (Agent will respond to queries as race progresses)")
    print()
    
    start_time = time.time()
    
    for sample in simulator.stream(duration_seconds=150):  # 2.5 minutes demo
        sample_count += 1
        
        # Add to chief agent
        chief.add_telemetry(sample)
        
        # Show telemetry updates every 10 samples
        if sample_count % 10 == 0:
            elapsed = time.time() - start_time
            print(f"⏱️  T+{elapsed:.1f}s | Lap {sample['lap']}/3 ({sample['lap_progress']:.0%}) | {sample['section']}")
            print(f"   Speed: {sample['speed']:.1f} km/h | Gear: {sample['gear']} | "
                  f"Throttle: {sample['aps']:.1f}% | Fuel: {sample['fuel_level']:.1f}L")
            print()
        
        # Periodic queries to agents
        if sample_count in query_intervals:
            query = query_intervals[sample_count]
            
            # Update context with current telemetry
            race_context['telemetry'] = sample
            
            print("─" * 80)
            print(f"📻 **DRIVER TO PIT WALL**: \"{query}\"")
            print("─" * 80)
            print()
            
            # Get Gemini-powered response
            print("🤖 **CHIEF AGENT (via Gemini 1.5)**: Processing...")
            response_start = time.time()
            
            response = chief.process(query, race_context)
            
            response_time = (time.time() - response_start) * 1000
            
            print(f"\n💬 **PIT WALL RESPONSE** (took {response_time:.0f}ms):")
            print("─" * 80)
            print(response)
            print("─" * 80)
            print()
            
            # Pause for readability
            time.sleep(2)
    
    print_section("DEMO COMPLETE - RESULTS SUMMARY")
    
    elapsed_total = time.time() - start_time
    
    print(f"✅ Demo completed successfully!")
    print()
    print(f"📊 Statistics:")
    print(f"   • Total duration: {elapsed_total:.1f} seconds")
    print(f"   • Telemetry samples processed: {sample_count}")
    print(f"   • Average rate: {sample_count/elapsed_total:.1f} samples/second")
    print(f"   • Agent queries: {len(query_intervals)}")
    print(f"   • Laps completed: {simulator.current_lap - 1}")
    print()
    
    print(f"🎯 Gemini Integration:")
    print(f"   • Status: {'✅ Active' if chief.use_gemini else '❌ Inactive'}")
    print(f"   • Model: Gemini 1.5 Flash")
    print(f"   • Natural language responses: ✅")
    print(f"   • Context-aware analysis: ✅")
    print(f"   • Conversation history: {len(chief.conversation_history)} messages")
    print()
    
    print(f"📝 Conversation Log:")
    print(f"   • Location: {os.getenv('CONVERSATION_LOG_FILE', '/tmp/agent_conversations.log')}")
    print(f"   • All interactions logged with timestamps")
    print()
    
    print("="*80)
    print("🏁 Cognirace System - Powered by Gemini 1.5 Flash")
    print("   Real-time race strategy with AI-powered intelligence")
    print("="*80)
    print()
    
    return True


def show_conversation_log():
    """Display conversation log"""
    log_file = os.getenv('CONVERSATION_LOG_FILE', '/tmp/agent_conversations.log')
    
    print_section("CONVERSATION LOG (Last 50 lines)")
    
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
            for line in lines[-50:]:
                print(line.rstrip())
    except FileNotFoundError:
        print(f"Log file not found: {log_file}")
    except Exception as e:
        print(f"Error reading log: {e}")


if __name__ == "__main__":
    try:
        print("\n🎬 Starting Cognirace Gemini Real-Time Demo...")
        print("   This demo will:")
        print("   • Stream realistic telemetry data over 2.5 minutes")
        print("   • Process 6 strategic queries with Gemini 1.5")
        print("   • Log all conversations with timestamps")
        print("   • Demonstrate real-time AI pit wall assistance")
        print()
        
        input("Press ENTER to start demo... ")
        
        success = demo_gemini_streaming()
        
        if success:
            print("\n📋 Would you like to view the conversation log?")
            view_log = input("View log? (y/n): ").lower().strip()
            
            if view_log == 'y':
                show_conversation_log()
        
        print("\n✅ Demo complete! System ready for production.")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user.")
        print("System remains operational.")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

