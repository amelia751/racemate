#!/usr/bin/env python3
"""
Comprehensive System Test for Cognirace
Tests the entire flow: Backend API → Telemetry Simulation → Agent Responses
"""

import requests
import json
import time
import sys
from datetime import datetime
from typing import Dict, Any

# Configuration
BACKEND_URL = "http://localhost:8005"
FRONTEND_URL = "http://localhost:3005"

# Colors for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

class SystemTester:
    def __init__(self):
        self.results = []
        self.errors = []
        
    def log(self, message: str, level: str = "INFO"):
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        
        if level == "SUCCESS":
            print(f"{GREEN}✓ [{timestamp}]{RESET} {message}")
        elif level == "ERROR":
            print(f"{RED}✗ [{timestamp}]{RESET} {message}")
            self.errors.append(message)
        elif level == "WARNING":
            print(f"{YELLOW}⚠ [{timestamp}]{RESET} {message}")
        else:
            print(f"{BLUE}ℹ [{timestamp}]{RESET} {message}")
    
    def test_backend_health(self) -> bool:
        """Test if backend API is responding"""
        self.log("Testing backend health...", "INFO")
        
        try:
            response = requests.get(f"{BACKEND_URL}/health", timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                self.log(f"Backend is healthy: {json.dumps(data, indent=2)}", "SUCCESS")
                return True
            else:
                self.log(f"Backend returned status {response.status_code}", "ERROR")
                return False
                
        except requests.exceptions.ConnectionError:
            self.log(f"Cannot connect to backend at {BACKEND_URL}", "ERROR")
            self.log("Make sure backend is running: cd backend-api && python -m uvicorn main:app --port 8005", "WARNING")
            return False
        except Exception as e:
            self.log(f"Backend health check failed: {e}", "ERROR")
            return False
    
    def test_fuel_prediction(self) -> bool:
        """Test fuel prediction endpoint"""
        self.log("Testing fuel prediction API...", "INFO")
        
        payload = {
            "speed": 245,
            "nmot": 12500,
            "gear": 6,
            "aps": 92,
            "lap": 15
        }
        
        try:
            response = requests.post(
                f"{BACKEND_URL}/predict/fuel",
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                self.log(f"Fuel prediction: {json.dumps(data, indent=2)}", "SUCCESS")
                
                # Check for actual prediction vs hardcoded
                if 'prediction' in data:
                    if isinstance(data['prediction'], (int, float)):
                        self.log("✓ Fuel prediction returns numeric value", "SUCCESS")
                        return True
                    else:
                        self.log("⚠ Fuel prediction format unexpected", "WARNING")
                        return True
                else:
                    self.log("⚠ No prediction field in response", "WARNING")
                    return False
            else:
                self.log(f"Fuel prediction failed with status {response.status_code}: {response.text}", "ERROR")
                return False
                
        except Exception as e:
            self.log(f"Fuel prediction error: {e}", "ERROR")
            return False
    
    def test_tire_prediction(self) -> bool:
        """Test tire prediction endpoint"""
        self.log("Testing tire prediction API...", "INFO")
        
        # Generate proper 16-feature telemetry sequences
        # Features: [speed, nmot, gear, aps, pbrake_f, pbrake_r, accx_can, accy_can, 
        #            steering_angle, brake_energy, lateral_load, throttle_var, 
        #            cum_brake_energy, cum_lateral_load, air_temp, extra]
        telemetry_sequence_16 = []
        for i in range(3):
            telemetry_sequence_16.append([
                245 - i*5,  # speed
                9500 - i*200,  # nmot (under 13000 limit)
                6,  # gear
                92 - i*2,  # aps
                50.0,  # pbrake_f
                45.0,  # pbrake_r
                0.8,  # accx_can
                1.2,  # accy_can
                15.0,  # steering_angle
                100.0,  # brake_energy
                200.0,  # lateral_load
                5.0,  # throttle_variance
                18750.0,  # cum_brake_energy
                51000.0,  # cum_lateral_load
                28.0,  # air_temp
                0.0  # extra feature
            ])
        
        payload = {
            "cum_brake_energy": 18750,
            "cum_lateral_load": 51000,
            "air_temp": 28,
            "telemetry_sequence": telemetry_sequence_16
        }
        
        try:
            response = requests.post(
                f"{BACKEND_URL}/predict/tire",
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                self.log(f"Tire prediction: {json.dumps(data, indent=2)}", "SUCCESS")
                
                if 'prediction' in data or 'grip_index' in data:
                    self.log("✓ Tire prediction returns grip data", "SUCCESS")
                    return True
                else:
                    self.log("⚠ Unexpected tire prediction format", "WARNING")
                    return False
            else:
                self.log(f"Tire prediction failed with status {response.status_code}: {response.text}", "ERROR")
                return False
                
        except Exception as e:
            self.log(f"Tire prediction error: {e}", "ERROR")
            return False
    
    def test_laptime_prediction(self) -> bool:
        """Test laptime prediction endpoint"""
        self.log("Testing laptime prediction API...", "INFO")
        
        # Create a realistic telemetry sequence (200 timesteps x 16 features)
        telemetry_sequence = []
        for i in range(200):
            telemetry_sequence.append([
                200 + (i % 50),  # speed
                8000 + (i % 2000),  # nmot (under 13000 limit)
                5 + (i % 2),  # gear
                80 + (i % 20),  # aps
                40.0 + (i % 20),  # pbrake_f
                35.0 + (i % 20),  # pbrake_r
                0.5 + (i % 10) * 0.1,  # accx_can
                0.8 + (i % 10) * 0.1,  # accy_can
                10.0 + (i % 30),  # steering_angle
                80.0 + (i % 40),  # brake_energy
                150.0 + (i % 100),  # lateral_load
                3.0 + (i % 5),  # throttle_variance
                5000.0 + i * 25,  # cum_brake_energy
                15000.0 + i * 75,  # cum_lateral_load
                26.0 + (i % 5),  # air_temp
                0.0  # extra feature
            ])
        
        payload = {
            "telemetry_sequence": telemetry_sequence
        }
        
        try:
            response = requests.post(
                f"{BACKEND_URL}/predict/laptime",
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                self.log(f"Laptime prediction: {json.dumps(data, indent=2)}", "SUCCESS")
                return True
            else:
                self.log(f"Laptime prediction failed with status {response.status_code}: {response.text}", "ERROR")
                return False
                
        except Exception as e:
            self.log(f"Laptime prediction error: {e}", "ERROR")
            return False
    
    def simulate_telemetry_stream(self, duration: int = 30) -> bool:
        """Simulate a telemetry stream for N seconds"""
        self.log(f"Simulating telemetry stream for {duration} seconds...", "INFO")
        
        lap = 1
        success_count = 0
        error_count = 0
        
        start_time = time.time()
        
        while time.time() - start_time < duration:
            # Generate realistic telemetry with proper RPM range
            telemetry = {
                "speed": 180 + (time.time() % 80),
                "nmot": 6000 + (time.time() % 3000),  # Stay under 13000 limit
                "gear": 4 + int(time.time() % 3),
                "aps": 70 + (time.time() % 30),
                "lap": lap
            }
            
            # Test fuel prediction with this telemetry
            try:
                response = requests.post(
                    f"{BACKEND_URL}/predict/fuel",
                    json=telemetry,
                    timeout=5
                )
                
                if response.status_code == 200:
                    success_count += 1
                    if success_count % 5 == 0:
                        self.log(f"Lap {lap}: Fuel prediction successful ({success_count} total)", "SUCCESS")
                else:
                    error_count += 1
                    self.log(f"Lap {lap}: Fuel prediction failed", "ERROR")
                    
            except Exception as e:
                error_count += 1
                self.log(f"Lap {lap}: Request error - {e}", "ERROR")
            
            # Increment lap every 10 seconds
            if int(time.time() - start_time) % 10 == 0 and int(time.time() - start_time) > 0:
                lap += 1
            
            time.sleep(1)  # 1 Hz update rate
        
        self.log(f"Stream complete: {success_count} successful, {error_count} errors", "INFO")
        return error_count == 0
    
    def test_frontend_accessibility(self) -> bool:
        """Test if frontend is accessible"""
        self.log("Testing frontend accessibility...", "INFO")
        
        try:
            response = requests.get(FRONTEND_URL, timeout=5)
            
            if response.status_code == 200:
                self.log(f"Frontend is accessible at {FRONTEND_URL}", "SUCCESS")
                return True
            else:
                self.log(f"Frontend returned status {response.status_code}", "WARNING")
                return False
                
        except requests.exceptions.ConnectionError:
            self.log(f"Cannot connect to frontend at {FRONTEND_URL}", "ERROR")
            self.log("Make sure frontend is running: cd frontend && npm run dev", "WARNING")
            return False
        except Exception as e:
            self.log(f"Frontend accessibility check failed: {e}", "ERROR")
            return False
    
    def run_full_test(self):
        """Run comprehensive system test"""
        print("\n" + "="*80)
        print("🏎️  COGNIRACE COMPREHENSIVE SYSTEM TEST")
        print("="*80 + "\n")
        
        tests = [
            ("Backend Health", self.test_backend_health),
            ("Fuel Prediction API", self.test_fuel_prediction),
            ("Tire Prediction API", self.test_tire_prediction),
            ("Laptime Prediction API", self.test_laptime_prediction),
            ("Frontend Accessibility", self.test_frontend_accessibility),
            ("Telemetry Stream Simulation", lambda: self.simulate_telemetry_stream(30)),
        ]
        
        passed = 0
        failed = 0
        
        for test_name, test_func in tests:
            print(f"\n{'='*80}")
            print(f"TEST: {test_name}")
            print('='*80)
            
            try:
                result = test_func()
                if result:
                    passed += 1
                    self.results.append((test_name, "PASS"))
                else:
                    failed += 1
                    self.results.append((test_name, "FAIL"))
            except Exception as e:
                failed += 1
                self.results.append((test_name, "ERROR"))
                self.log(f"Test crashed: {e}", "ERROR")
        
        # Final summary
        print("\n" + "="*80)
        print("📊 TEST SUMMARY")
        print("="*80 + "\n")
        
        for test_name, status in self.results:
            if status == "PASS":
                print(f"{GREEN}✓{RESET} {test_name}: {status}")
            else:
                print(f"{RED}✗{RESET} {test_name}: {status}")
        
        print(f"\n{BLUE}Total:{RESET} {len(tests)} tests")
        print(f"{GREEN}Passed:{RESET} {passed}")
        print(f"{RED}Failed:{RESET} {failed}")
        
        if self.errors:
            print(f"\n{RED}Errors encountered:{RESET}")
            for error in self.errors:
                print(f"  • {error}")
        
        print("\n" + "="*80)
        
        if failed == 0:
            print(f"{GREEN}🎉 ALL TESTS PASSED!{RESET}")
            print("System is ready for manual testing at: http://localhost:3005")
        else:
            print(f"{RED}⚠️  SOME TESTS FAILED{RESET}")
            print("Please review errors above and fix before manual testing")
        
        print("="*80 + "\n")
        
        return failed == 0


if __name__ == "__main__":
    tester = SystemTester()
    success = tester.run_full_test()
    sys.exit(0 if success else 1)

