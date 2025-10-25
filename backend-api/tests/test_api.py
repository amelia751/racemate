#!/usr/bin/env python3
"""
Comprehensive API endpoint testing
Tests all prediction endpoints with real requests
"""

import requests
import json
import time
from typing import Dict, Any

API_BASE_URL = "http://localhost:8005"

def print_header(title: str):
    """Print formatted header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def test_endpoint(name: str, method: str, url: str, data: Dict[str, Any] = None) -> bool:
    """Test an API endpoint"""
    print(f"\n[TEST] {name}")
    print(f"  URL: {method} {url}")
    
    try:
        start_time = time.time()
        
        if method == "GET":
            response = requests.get(url, timeout=10)
        elif method == "POST":
            print(f"  Request: {json.dumps(data, indent=2)[:200]}...")
            response = requests.post(url, json=data, timeout=10)
        
        latency_ms = (time.time() - start_time) * 1000
        
        print(f"  Status: {response.status_code}")
        print(f"  Latency: {latency_ms:.2f}ms")
        
        if response.status_code == 200:
            result = response.json()
            print(f"  Response: {json.dumps(result, indent=2)[:300]}...")
            print(f"  ✓ PASSED")
            return True
        else:
            print(f"  Error: {response.text}")
            print(f"  ✗ FAILED")
            return False
            
    except Exception as e:
        print(f"  Exception: {e}")
        print(f"  ✗ FAILED")
        return False


def run_all_tests():
    """Run all API tests"""
    
    print_header("COGNIRACE API ENDPOINT TESTS")
    
    results = []
    
    # Test 1: Root endpoint
    results.append(test_endpoint(
        "Root Endpoint",
        "GET",
        f"{API_BASE_URL}/"
    ))
    
    # Test 2: Health check
    results.append(test_endpoint(
        "Health Check",
        "GET",
        f"{API_BASE_URL}/health"
    ))
    
    # Test 3: Readiness check
    results.append(test_endpoint(
        "Readiness Check",
        "GET",
        f"{API_BASE_URL}/ready"
    ))
    
    # Test 4: List models
    results.append(test_endpoint(
        "List Models",
        "GET",
        f"{API_BASE_URL}/predict/models"
    ))
    
    # Test 5: Fuel prediction
    results.append(test_endpoint(
        "Fuel Consumption Prediction",
        "POST",
        f"{API_BASE_URL}/predict/fuel",
        data={
            "speed": 180.5,
            "nmot": 7200,
            "gear": 5,
            "aps": 95.2,
            "lap": 15
        }
    ))
    
    # Test 6: Lap time prediction
    results.append(test_endpoint(
        "Lap Time Prediction",
        "POST",
        f"{API_BASE_URL}/predict/laptime",
        data={
            "telemetry_sequence": [[180.5, 7200, 5, 95.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] * 100,
            "feature_names": ["speed", "nmot", "gear", "aps"] + ["f"+str(i) for i in range(12)]
        }
    ))
    
    # Test 7: Tire degradation prediction
    results.append(test_endpoint(
        "Tire Degradation Prediction",
        "POST",
        f"{API_BASE_URL}/predict/tire",
        data={
            "cum_brake_energy": 1500.0,
            "cum_lateral_load": 2000.0,
            "air_temp": 28.5
        }
    ))
    
    # Test 8: Traffic analysis prediction
    results.append(test_endpoint(
        "Traffic Analysis Prediction",
        "POST",
        f"{API_BASE_URL}/predict/traffic",
        data={
            "car_states": [
                [180.5, 7200, 5, 95.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [175.0, 7000, 5, 90.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            ],
            "feature_names": ["speed", "nmot", "gear", "aps"] + ["f"+str(i) for i in range(12)]
        }
    ))
    
    # Summary
    print_header("TEST SUMMARY")
    
    passed = sum(results)
    total = len(results)
    
    print(f"\nTests Passed: {passed}/{total} ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n✅ ALL TESTS PASSED!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
    
    print("\n" + "="*70)
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)

