#!/usr/bin/env python3
"""
Comprehensive End-to-End Test
Tests the complete Cognirace system:
1. Telemetry Simulator generates data
2. Agents process the data
3. API provides predictions
4. ChiefAgent coordinates everything
"""

import sys
import os
import time

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../streaming'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../agents'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../backend-api'))

from simulator.telemetry_simulator import TelemetrySimulator, SimulatorConfig
from tools.api_client import CogniraceAPIClient
from specialized.chief_agent import ChiefAgent


def print_header(title: str):
    """Print formatted header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")


def test_api_connectivity():
    """Test 1: Verify API is running"""
    print_header("TEST 1: API CONNECTIVITY")
    
    api_client = CogniraceAPIClient()
    
    print("Checking API health...")
    health = api_client.health_check()
    
    if health.get('status') == 'healthy':
        print("✓ API is healthy")
        print(f"  Uptime: {health.get('uptime_seconds', 0):.1f}s")
        print(f"  Models loaded: {health.get('models_loaded', 0)}")
        return True
    else:
        print("✗ API is not responding")
        print("  Make sure the API is running on port 8005")
        print("  Run: cd backend-api && ./start_api.sh")
        return False


def test_telemetry_simulator():
    """Test 2: Verify telemetry simulator"""
    print_header("TEST 2: TELEMETRY SIMULATOR")
    
    config = SimulatorConfig(
        frequency_hz=20.0,
        base_speed=160.0,
        total_laps=1
    )
    
    simulator = TelemetrySimulator(config)
    
    print("Generating 10 telemetry samples...")
    samples = []
    
    for i, sample in enumerate(simulator.stream(max_samples=10)):
        samples.append(sample)
        if i < 3:  # Show first 3 samples
            print(f"\n  Sample {i+1}:")
            print(f"    Speed: {sample['speed']:.1f} km/h | Gear: {sample['gear']}")
            print(f"    RPM: {sample['nmot']:.0f} | Throttle: {sample['aps']:.1f}%")
            print(f"    Section: {sample['section']}")
    
    print(f"\n✓ Generated {len(samples)} samples successfully")
    return samples


def test_individual_agents(api_client):
    """Test 3: Test individual agents"""
    print_header("TEST 3: INDIVIDUAL AGENTS")
    
    from specialized.fuel_agent import FuelAgent
    from specialized.tire_agent import TireAgent
    from specialized.telemetry_agent import TelemetryAgent
    
    # Test context
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
        "race_info": {"total_laps": 40},
        "last_pit_lap": 5,
        "weather": {"air_temp": 28.5}
    }
    
    print("Testing FuelAgent...")
    fuel_agent = FuelAgent(api_client)
    fuel_response = fuel_agent.process("Analyze fuel", context)
    print("✓ FuelAgent responded")
    print(f"  Response length: {len(fuel_response)} chars")
    
    print("\nTesting TireAgent...")
    tire_agent = TireAgent(api_client)
    tire_response = tire_agent.process("Analyze tires", context)
    print("✓ TireAgent responded")
    print(f"  Response length: {len(tire_response)} chars")
    
    print("\nTesting TelemetryAgent...")
    telem_agent = TelemetryAgent()
    telem_agent.add_telemetry(context["telemetry"])
    telem_response = telem_agent.process("Show telemetry")
    print("✓ TelemetryAgent responded")
    print(f"  Response length: {len(telem_response)} chars")
    
    return True


def test_chief_agent(api_client):
    """Test 4: Test ChiefAgent orchestration"""
    print_header("TEST 4: CHIEF AGENT ORCHESTRATION")
    
    chief = ChiefAgent(api_client)
    
    # Add some telemetry samples
    print("Adding telemetry samples to ChiefAgent...")
    for i in range(20):
        telemetry = {
            "speed": 150 + i * 2,
            "nmot": 6000 + i * 100,
            "gear": 5,
            "aps": 85 + i * 0.5,
            "lap": 15,
            "cum_brake_energy": 2000 + i * 100,
            "cum_lateral_load": 2500 + i * 120,
            "fuel_level": 30 - i * 0.1
        }
        chief.add_telemetry(telemetry)
    
    print(f"✓ Added 20 telemetry samples")
    
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
        "race_info": {"total_laps": 40},
        "last_pit_lap": 5,
        "weather": {"air_temp": 28.5}
    }
    
    print("\nRequesting comprehensive analysis...")
    response = chief.process("Give me a full race analysis", context)
    
    print("\n" + "-"*80)
    print(response)
    print("-"*80)
    
    print("\n✓ ChiefAgent provided comprehensive analysis")
    
    # Test strategy recommendation
    print("\nRequesting strategy recommendation...")
    strategy = chief.get_strategy_recommendation(context)
    
    print(f"\n📊 Strategy Recommendation:")
    print(f"   Should Pit: {strategy['should_pit']}")
    print(f"   Urgency: {strategy['urgency']}")
    print(f"   Reason: {strategy['reason']}")
    
    return True


def test_streaming_pipeline(api_client):
    """Test 5: Full streaming pipeline"""
    print_header("TEST 5: STREAMING PIPELINE (30 SECONDS)")
    
    chief = ChiefAgent(api_client)
    
    config = SimulatorConfig(
        frequency_hz=5.0,  # 5 Hz for testing
        base_speed=160.0,
        lap_time_seconds=30.0,  # Short lap for testing
        total_laps=3
    )
    
    simulator = TelemetrySimulator(config)
    
    print("Starting 30-second streaming test...")
    print("  Frequency: 5 Hz")
    print("  Duration: 30 seconds")
    print()
    
    sample_count = 0
    analysis_interval = 25  # Analyze every 25 samples (5 seconds)
    
    for sample in simulator.stream(duration_seconds=30):
        # Add to chief agent
        chief.add_telemetry(sample)
        sample_count += 1
        
        # Show progress
        if sample_count % 10 == 0:
            print(f"  📡 Processed {sample_count} samples... "
                  f"(Lap {sample['lap']}, {sample['lap_progress']:.0%})")
        
        # Periodic analysis
        if sample_count % analysis_interval == 0:
            context = {
                "telemetry": sample,
                "race_info": {"total_laps": config.total_laps},
                "last_pit_lap": 0,
                "weather": {"air_temp": 25.0}
            }
            
            # Get quick strategy check
            strategy = chief.get_strategy_recommendation(context)
            print(f"\n  🎯 Strategy Check (Sample {sample_count}):")
            print(f"     Urgency: {strategy['urgency']}")
            print(f"     Should Pit: {strategy['should_pit']}")
            print()
    
    print(f"\n✓ Streaming test complete!")
    print(f"  Total samples processed: {sample_count}")
    print(f"  Duration: 30 seconds")
    print(f"  Average rate: {sample_count/30:.1f} samples/second")
    
    return True


def run_all_tests():
    """Run complete test suite"""
    
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*78 + "║")
    print("║" + "  🏁 COGNIRACE END-TO-END COMPREHENSIVE TEST".center(78) + "║")
    print("║" + " "*78 + "║")
    print("╚" + "="*78 + "╝")
    
    results = {}
    
    # Test 1: API Connectivity
    results["api"] = test_api_connectivity()
    if not results["api"]:
        print("\n❌ Cannot proceed without API. Please start the API first.")
        return False
    
    # Create API client for remaining tests
    api_client = CogniraceAPIClient()
    
    # Test 2: Telemetry Simulator
    results["simulator"] = test_telemetry_simulator() is not None
    
    # Test 3: Individual Agents
    results["agents"] = test_individual_agents(api_client)
    
    # Test 4: Chief Agent
    results["chief"] = test_chief_agent(api_client)
    
    # Test 5: Streaming Pipeline
    results["streaming"] = test_streaming_pipeline(api_client)
    
    # Summary
    print_header("TEST SUMMARY")
    
    total = len(results)
    passed = sum(1 for r in results.values() if r)
    
    print("Test Results:")
    print(f"  1. API Connectivity:        {'✓ PASSED' if results['api'] else '✗ FAILED'}")
    print(f"  2. Telemetry Simulator:     {'✓ PASSED' if results['simulator'] else '✗ FAILED'}")
    print(f"  3. Individual Agents:       {'✓ PASSED' if results['agents'] else '✗ FAILED'}")
    print(f"  4. Chief Agent:             {'✓ PASSED' if results['chief'] else '✗ FAILED'}")
    print(f"  5. Streaming Pipeline:      {'✓ PASSED' if results['streaming'] else '✗ FAILED'}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    
    if passed == total:
        print("\n✅ ALL TESTS PASSED! System is fully operational!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review errors above.")
    
    print("\n" + "="*80)
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

