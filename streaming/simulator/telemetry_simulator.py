"""
Telemetry Simulator - Generates realistic mock racing telemetry
"""

import random
import time
import math
from typing import Dict, Any, List, Generator
from datetime import datetime
from dataclasses import dataclass


@dataclass
class SimulatorConfig:
    """Configuration for telemetry simulator"""
    frequency_hz: float = 20.0  # Samples per second
    base_speed: float = 150.0    # km/h
    base_rpm: float = 6000.0
    lap_time_seconds: float = 90.0  # Average lap time
    total_laps: int = 40
    track_name: str = "Virtual Circuit"


class TelemetrySimulator:
    """Simulates realistic racing telemetry stream"""
    
    def __init__(self, config: SimulatorConfig = None):
        self.config = config or SimulatorConfig()
        self.current_lap = 1
        self.lap_progress = 0.0  # 0.0 to 1.0
        self.fuel_level = 50.0
        self.cum_brake_energy = 0.0
        self.cum_lateral_load = 0.0
        self.vehicle_id = "CAR_01"
        
        # Track sections (simulate different characteristics)
        self.sections = self._define_track_sections()
    
    def _define_track_sections(self) -> List[Dict[str, Any]]:
        """Define track sections with different characteristics"""
        return [
            {"name": "Straight", "length": 0.3, "speed_mult": 1.2, "throttle": 0.95, "brake": 0.0},
            {"name": "Hard Braking", "length": 0.05, "speed_mult": 0.5, "throttle": 0.0, "brake": 0.9},
            {"name": "Slow Corner", "length": 0.15, "speed_mult": 0.6, "throttle": 0.6, "brake": 0.2},
            {"name": "Acceleration", "length": 0.2, "speed_mult": 0.85, "throttle": 0.9, "brake": 0.0},
            {"name": "Fast Sweeper", "length": 0.25, "speed_mult": 1.0, "throttle": 0.75, "brake": 0.0},
            {"name": "Heavy Braking", "length": 0.05, "speed_mult": 0.4, "throttle": 0.0, "brake": 1.0}
        ]
    
    def _get_current_section(self) -> Dict[str, Any]:
        """Get current track section based on lap progress"""
        cumulative = 0.0
        for section in self.sections:
            cumulative += section["length"]
            if self.lap_progress <= cumulative:
                return section
        return self.sections[-1]
    
    def _calculate_gear(self, speed: float, rpm: float) -> int:
        """Calculate appropriate gear"""
        if speed < 60:
            return 2
        elif speed < 100:
            return 3
        elif speed < 140:
            return 4
        elif speed < 180:
            return 5
        else:
            return 6
    
    def generate_sample(self) -> Dict[str, Any]:
        """Generate a single telemetry sample"""
        
        section = self._get_current_section()
        
        # Add some noise/variation
        noise = lambda base, var: base * (1.0 + random.uniform(-var, var))
        
        # Calculate speed based on section
        target_speed = self.config.base_speed * section["speed_mult"]
        speed = noise(target_speed, 0.05)
        
        # RPM correlates with speed and throttle
        target_rpm = self.config.base_rpm * (speed / self.config.base_speed)
        nmot = noise(target_rpm, 0.08)
        
        # Throttle position
        aps = section["throttle"] * 100 * noise(1.0, 0.1)
        aps = max(0, min(100, aps))
        
        # Braking
        pbrake_f = section["brake"] * noise(280, 0.15)  # Front brake pressure (bar)
        pbrake_r = section["brake"] * noise(220, 0.15)  # Rear brake pressure
        
        # Gear
        gear = self._calculate_gear(speed, nmot)
        
        # Accelerations
        accel_long = (aps / 100.0) - (section["brake"] * 2.0) + random.uniform(-0.3, 0.3)
        accel_lat = math.sin(self.lap_progress * 4 * math.pi) * random.uniform(0.5, 1.5)
        
        # Steering (correlates with lateral acceleration)
        steering = accel_lat * 45.0  # degrees
        
        # Update cumulative metrics
        self.cum_brake_energy += pbrake_f * speed * 0.01  # Simplified
        self.cum_lateral_load += abs(accel_lat) * speed * 0.01
        
        # Fuel consumption (simplified)
        fuel_burn = (aps / 100.0) * (nmot / 8000.0) * 0.0001
        self.fuel_level = max(0, self.fuel_level - fuel_burn)
        
        # Create telemetry sample
        telemetry = {
            "timestamp": datetime.now().isoformat(),
            "vehicle_id": self.vehicle_id,
            "lap": self.current_lap,
            "lap_progress": self.lap_progress,
            "speed": round(speed, 2),
            "nmot": round(nmot, 1),
            "gear": gear,
            "aps": round(aps, 2),
            "ath": round(aps, 2),  # Throttle (same as APS)
            "pbrake_f": round(pbrake_f, 2),
            "pbrake_r": round(pbrake_r, 2),
            "accx_can": round(accel_long, 3),
            "accy_can": round(accel_lat, 3),
            "Steering_Angle": round(steering, 2),
            "cum_brake_energy": round(self.cum_brake_energy, 2),
            "cum_lateral_load": round(self.cum_lateral_load, 2),
            "fuel_level": round(self.fuel_level, 2),
            "track": self.config.track_name,
            "section": section["name"]
        }
        
        return telemetry
    
    def advance_time(self, dt: float):
        """Advance simulation time"""
        # Progress through lap
        lap_duration = self.config.lap_time_seconds
        self.lap_progress += dt / lap_duration
        
        # Check lap completion
        if self.lap_progress >= 1.0:
            self.lap_progress = 0.0
            self.current_lap += 1
            
            # Reset cumulative metrics per lap
            self.cum_brake_energy = 0.0
            self.cum_lateral_load = 0.0
            
            print(f"📍 Lap {self.current_lap}/{self.config.total_laps} complete")
    
    def stream(self, duration_seconds: float = None, 
               max_samples: int = None) -> Generator[Dict[str, Any], None, None]:
        """Generate continuous telemetry stream"""
        
        dt = 1.0 / self.config.frequency_hz
        samples_generated = 0
        start_time = time.time()
        
        print(f"🏁 Starting telemetry stream...")
        print(f"   Frequency: {self.config.frequency_hz} Hz")
        print(f"   Track: {self.config.track_name}")
        print(f"   Total Laps: {self.config.total_laps}")
        print()
        
        while True:
            # Check stopping conditions
            if duration_seconds and (time.time() - start_time) >= duration_seconds:
                break
            if max_samples and samples_generated >= max_samples:
                break
            if self.current_lap > self.config.total_laps:
                break
            
            # Generate sample
            sample = self.generate_sample()
            yield sample
            
            # Advance time
            self.advance_time(dt)
            samples_generated += 1
            
            # Sleep to maintain real-time frequency
            time.sleep(dt)
        
        print(f"\n✓ Stream completed: {samples_generated} samples generated")


if __name__ == "__main__":
    # Test the simulator
    config = SimulatorConfig(
        frequency_hz=10.0,  # 10 Hz for testing
        total_laps=2
    )
    
    simulator = TelemetrySimulator(config)
    
    print("Testing Telemetry Simulator...")
    print("="*70)
    
    # Generate 100 samples
    for i, sample in enumerate(simulator.stream(max_samples=100)):
        if i % 10 == 0:  # Print every 10th sample
            print(f"\nSample {i+1}:")
            print(f"  Lap: {sample['lap']} ({sample['lap_progress']:.1%})")
            print(f"  Speed: {sample['speed']:.1f} km/h | Gear: {sample['gear']}")
            print(f"  RPM: {sample['nmot']:.0f} | Throttle: {sample['aps']:.1f}%")
            print(f"  Brake F/R: {sample['pbrake_f']:.1f}/{sample['pbrake_r']:.1f} bar")
            print(f"  Section: {sample['section']}")

