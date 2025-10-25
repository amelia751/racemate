#!/usr/bin/env python3
"""
Cognirace System Demo
Interactive demonstration of the complete system
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'streaming'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'agents'))

from simulator.telemetry_simulator import TelemetrySimulator, SimulatorConfig
from tools.api_client import CogniraceAPIClient
from specialized.chief_agent import ChiefAgent


def print_banner():
    """Print Cognirace banner"""
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*78 + "║")
    print("║" + "🏁  COGNIRACE - REAL-TIME RACE STRATEGY PLATFORM  🏁".center(78) + "║")
    print("║" + " "*78 + "║")
    print("║" + "Hack the Track 2025 - Toyota GR Cup Series".center(78) + "║")
    print("║" + " "*78 + "║")
    print("╚" + "="*78 + "╝")
    print("\n")


def demo_scenario_1():
    """Demo 1: Early Race - All systems green"""
    print("\n" + "="*80)
    print("SCENARIO 1: Early Race (Lap 5/40)")
    print("="*80)
    
    api_client = CogniraceAPIClient()
    chief = ChiefAgent(api_client)
    
    # Simulate early race telemetry
    context = {
        "telemetry": {
            "speed": 165.3,
            "nmot": 6800,
            "gear": 5,
            "aps": 88.5,
            "lap": 5,
            "fuel_level": 47.0,
            "cum_brake_energy": 800,
            "cum_lateral_load": 1200
        },
        "race_info": {"total_laps": 40},
        "last_pit_lap": 0,
        "weather": {"air_temp": 24.0}
    }
    
    print("\n📊 Current Situation:")
    print(f"  Lap: {context['telemetry']['lap']}/40")
    print(f"  Speed: {context['telemetry']['speed']} km/h")
    print(f"  Fuel: {context['telemetry']['fuel_level']:.1f}L")
    print(f"  Position: Early in race")
    
    print("\n🤖 Requesting ChiefAgent Analysis...")
    response = chief.process("Give me a race update", context)
    
    print("\n" + "-"*80)
    print(response)
    print("-"*80)
    
    strategy = chief.get_strategy_recommendation(context)
    print(f"\n🎯 Strategic Decision: {'PIT NOW' if strategy['should_pit'] else 'STAY OUT'}")
    print(f"   Urgency Level: {strategy['urgency']}")


def demo_scenario_2():
    """Demo 2: Mid-race - Tire warning"""
    print("\n" + "="*80)
    print("SCENARIO 2: Mid-Race Tire Warning (Lap 18/40)")
    print("="*80)
    
    api_client = CogniraceAPIClient()
    chief = ChiefAgent(api_client)
    
    # Simulate mid-race with tire degradation
    context = {
        "telemetry": {
            "speed": 158.2,
            "nmot": 6500,
            "gear": 5,
            "aps": 85.0,
            "lap": 18,
            "fuel_level": 32.0,
            "cum_brake_energy": 3200,
            "cum_lateral_load": 4500
        },
        "race_info": {"total_laps": 40},
        "last_pit_lap": 0,
        "weather": {"air_temp": 28.5}
    }
    
    print("\n📊 Current Situation:")
    print(f"  Lap: {context['telemetry']['lap']}/40")
    print(f"  Stint Length: 18 laps (no pit yet)")
    print(f"  Cumulative Brake Energy: {context['telemetry']['cum_brake_energy']}")
    print(f"  Cumulative Lateral Load: {context['telemetry']['cum_lateral_load']}")
    print(f"  ⚠️  High tire stress detected")
    
    print("\n🤖 Requesting Tire Analysis...")
    tire_response = chief.tire_agent.process("How are the tires?", context)
    
    print("\n" + "-"*80)
    print(tire_response)
    print("-"*80)


def demo_scenario_3():
    """Demo 3: Critical pit window"""
    print("\n" + "="*80)
    print("SCENARIO 3: Critical Pit Window (Lap 25/40)")
    print("="*80)
    
    api_client = CogniraceAPIClient()
    chief = ChiefAgent(api_client)
    
    # Simulate critical situation
    context = {
        "telemetry": {
            "speed": 152.8,
            "nmot": 6200,
            "gear": 4,
            "aps": 78.0,
            "lap": 25,
            "fuel_level": 18.5,  # Low fuel!
            "cum_brake_energy": 4800,
            "cum_lateral_load": 6200
        },
        "race_info": {"total_laps": 40},
        "last_pit_lap": 0,
        "weather": {"air_temp": 30.0}
    }
    
    print("\n📊 Current Situation:")
    print(f"  Lap: {context['telemetry']['lap']}/40")
    print(f"  Fuel: {context['telemetry']['fuel_level']:.1f}L 🔴 CRITICAL")
    print(f"  Stint: 25 laps without pit")
    print(f"  Tire Stress: VERY HIGH")
    
    print("\n🤖 Emergency Strategic Analysis...")
    strategy = chief.get_strategy_recommendation(context)
    
    print(f"\n{'='*80}")
    print(f"  🚨 URGENT RECOMMENDATION 🚨")
    print(f"{'='*80}")
    print(f"  Should Pit: {'YES!' if strategy['should_pit'] else 'NO'}")
    print(f"  Urgency: {strategy['urgency']}")
    print(f"  Reason: {strategy['reason']}")
    print(f"{'='*80}")
    
    if strategy['should_pit']:
        print("\n  📻 Radio Message:")
        print("  'Box, box, box! Come in this lap for fuel and tires.'")


def demo_streaming():
    """Demo 4: Live streaming telemetry"""
    print("\n" + "="*80)
    print("SCENARIO 4: Live Telemetry Stream (10 seconds)")
    print("="*80)
    
    api_client = CogniraceAPIClient()
    chief = ChiefAgent(api_client)
    
    config = SimulatorConfig(
        frequency_hz=10.0,  # 10 Hz for demo
        base_speed=160.0,
        lap_time_seconds=30.0,
        total_laps=2
    )
    
    simulator = TelemetrySimulator(config)
    
    print("\n📡 Starting live telemetry stream...")
    print("   Frequency: 10 Hz")
    print("   Duration: 10 seconds")
    print()
    
    sample_count = 0
    
    for sample in simulator.stream(duration_seconds=10):
        chief.add_telemetry(sample)
        sample_count += 1
        
        if sample_count % 10 == 0:
            stats = chief.telemetry_agent.calculate_statistics(window=10)
            print(f"\n  Sample {sample_count}:")
            print(f"    Lap {sample['lap']} ({sample['lap_progress']:.0%}) - {sample['section']}")
            print(f"    Avg Speed: {stats.get('speed_avg', 0):.1f} km/h")
            print(f"    Avg RPM: {stats.get('nmot_avg', 0):.0f}")
            print(f"    Avg Throttle: {stats.get('aps_avg', 0):.1f}%")
    
    print(f"\n✓ Stream complete: {sample_count} samples processed")
    print(f"  Buffer size: {len(chief.telemetry_agent.telemetry_buffer)} samples")


def main():
    """Run complete demo"""
    print_banner()
    
    print("🎬 Welcome to the Cognirace Live Demo!")
    print("\nThis demonstration showcases:")
    print("  • Real-time ML predictions")
    print("  • Multi-agent coordination")
    print("  • Strategic decision-making")
    print("  • Live telemetry streaming")
    
    input("\n Press ENTER to start Scenario 1... ")
    demo_scenario_1()
    
    input("\n Press ENTER for Scenario 2... ")
    demo_scenario_2()
    
    input("\n Press ENTER for Scenario 3 (Critical!)... ")
    demo_scenario_3()
    
    input("\n Press ENTER for Scenario 4 (Live Stream)... ")
    demo_streaming()
    
    print("\n" + "="*80)
    print("✅ DEMO COMPLETE")
    print("="*80)
    print("\n🏁 Cognirace System is fully operational and ready for race day!")
    print("\n📊 System Capabilities Demonstrated:")
    print("   ✓ Real-time telemetry processing")
    print("   ✓ ML-powered predictions (Fuel, Tires, Lap Time)")
    print("   ✓ Intelligent agent coordination")
    print("   ✓ Strategic pit decision-making")
    print("   ✓ Live streaming at 10-20 Hz")
    print("\n🚀 Ready for deployment and production use!")
    print("\nThank you for watching! 🏁")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted. System remains operational.")
        sys.exit(0)

