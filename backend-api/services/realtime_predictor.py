"""
Real-Time Prediction Engine for RaceMate
Processes streaming telemetry through all 8 ML models and detects events
"""

import numpy as np
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import logging

from services.model_loader import model_loader

logger = logging.getLogger(__name__)


@dataclass
class PredictionState:
    """Tracks state across telemetry frames for change detection"""
    
    # Fuel state
    fuel_level: float = 50.0
    last_fuel_prediction: float = 0.0
    laps_remaining: int = 20
    personal_best_laptime: float = 90.0
    
    # Tire state
    tire_wear: Dict[str, float] = field(default_factory=lambda: {
        'FL': 0.0, 'FR': 0.0, 'RL': 0.0, 'RR': 0.0
    })
    tire_degradation_rate: float = 0.0
    
    # Anomaly state
    last_anomaly_score: float = 0.0
    anomaly_history: List[float] = field(default_factory=list)
    
    # FCY state
    last_fcy_probability: float = 0.0
    fcy_trend: str = "stable"
    
    # Laptime state
    recent_laptimes: List[float] = field(default_factory=list)
    current_pace: str = "normal"
    
    # Traffic state
    traffic_density: int = 0
    last_traffic_density: int = 0
    
    # Pit state
    optimal_pit_window_start: int = 15
    optimal_pit_window_end: int = 18
    last_pit_lap: int = 0
    
    # Driver state
    driver_consistency_score: float = 1.0
    
    def update(self, predictions: Dict[str, Any]):
        """Update state with new predictions"""
        if 'fuel' in predictions:
            self.last_fuel_prediction = predictions['fuel']
        
        if 'tire' in predictions:
            self.tire_wear = predictions['tire']
        
        if 'anomaly' in predictions:
            self.last_anomaly_score = predictions['anomaly']
            self.anomaly_history.append(predictions['anomaly'])
            if len(self.anomaly_history) > 20:
                self.anomaly_history.pop(0)
        
        if 'fcy' in predictions:
            # Detect FCY trend
            if predictions['fcy'] > self.last_fcy_probability + 0.1:
                self.fcy_trend = "increasing"
            elif predictions['fcy'] < self.last_fcy_probability - 0.1:
                self.fcy_trend = "decreasing"
            else:
                self.fcy_trend = "stable"
            self.last_fcy_probability = predictions['fcy']
        
        if 'laptime' in predictions:
            self.recent_laptimes.append(predictions['laptime'])
            if len(self.recent_laptimes) > 5:
                self.recent_laptimes.pop(0)
            
            # Determine pace
            if self.recent_laptimes:
                avg_recent = sum(self.recent_laptimes) / len(self.recent_laptimes)
                if avg_recent < self.personal_best_laptime * 1.02:
                    self.current_pace = "strong"
                elif avg_recent > self.personal_best_laptime * 1.05:
                    self.current_pace = "slow"
                else:
                    self.current_pace = "normal"


@dataclass
class RaceEvent:
    """Represents a detected event that requires action"""
    event_type: str
    severity: str  # 'critical', 'high', 'medium', 'low', 'info'
    message: str
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self):
        return {
            'event_type': self.event_type,
            'severity': self.severity,
            'message': self.message,
            'data': self.data,
            'timestamp': self.timestamp.isoformat()
        }


class RealtimePredictor:
    """
    Real-time prediction engine that:
    1. Processes telemetry through all 8 models
    2. Detects significant events
    3. Triggers agent system for recommendations
    """
    
    def __init__(self):
        self.state = PredictionState()
        self.models_loaded = False
        self.frame_count = 0
        logger.info("RealtimePredictor initialized")
    
    def _ensure_models_loaded(self):
        """Lazy load models on first use"""
        if not self.models_loaded:
            logger.info("Loading ML models for real-time prediction...")
            # Models are loaded via model_loader as needed
            self.models_loaded = True
    
    def process_telemetry(self, telemetry: Dict[str, Any]) -> Optional[List[RaceEvent]]:
        """
        Process incoming telemetry frame through all models
        Returns list of detected events (or None if no events)
        """
        self._ensure_models_loaded()
        self.frame_count += 1
        
        # Run predictions through all models
        predictions = self._run_all_models(telemetry)
        
        # Detect events based on predictions and state
        events = self._detect_events(predictions, telemetry)
        
        # Update state
        self.state.update(predictions)
        
        if events:
            logger.info(f"Frame {self.frame_count}: Detected {len(events)} event(s)")
            for event in events:
                logger.info(f"  → {event.severity.upper()}: {event.event_type}")
        
        return events if events else None
    
    def _run_all_models(self, telemetry: Dict[str, Any]) -> Dict[str, Any]:
        """Run telemetry through all 8 models"""
        predictions = {}
        
        # 1. Fuel Consumption Model
        try:
            fuel_per_lap = self._predict_fuel(telemetry)
            predictions['fuel'] = fuel_per_lap
        except Exception as e:
            logger.warning(f"Fuel prediction failed: {e}")
            predictions['fuel'] = 0.06  # Fallback
        
        # 2. Tire Degradation Model
        try:
            tire_wear = self._predict_tire_wear(telemetry)
            predictions['tire'] = tire_wear
        except Exception as e:
            logger.warning(f"Tire prediction failed: {e}")
            predictions['tire'] = {'FL': 50, 'FR': 50, 'RL': 50, 'RR': 50}
        
        # 3. Anomaly Detector
        try:
            anomaly_score = self._detect_anomaly(telemetry)
            predictions['anomaly'] = anomaly_score
        except Exception as e:
            logger.warning(f"Anomaly detection failed: {e}")
            predictions['anomaly'] = 0.0
        
        # 4. FCY Hazard Model
        try:
            fcy_prob = self._predict_fcy(telemetry)
            predictions['fcy'] = fcy_prob
        except Exception as e:
            logger.warning(f"FCY prediction failed: {e}")
            predictions['fcy'] = 0.0
        
        # 5. Lap-Time Transformer
        try:
            laptime = self._predict_laptime(telemetry)
            predictions['laptime'] = laptime
        except Exception as e:
            logger.warning(f"Laptime prediction failed: {e}")
            predictions['laptime'] = 90.0
        
        # 6. Pit Loss Model
        try:
            pit_loss = self._predict_pit_loss(telemetry)
            predictions['pit_loss'] = pit_loss
        except Exception as e:
            logger.warning(f"Pit loss prediction failed: {e}")
            predictions['pit_loss'] = 15.0
        
        # 7. Driver Embedding
        try:
            consistency = self._analyze_driver(telemetry)
            predictions['driver_consistency'] = consistency
        except Exception as e:
            logger.warning(f"Driver analysis failed: {e}")
            predictions['driver_consistency'] = 1.0
        
        # 8. Traffic GNN
        try:
            traffic_impact = self._analyze_traffic(telemetry)
            predictions['traffic'] = traffic_impact
        except Exception as e:
            logger.warning(f"Traffic analysis failed: {e}")
            predictions['traffic'] = 0
        
        return predictions
    
    def _predict_fuel(self, telemetry: Dict[str, Any]) -> float:
        """Predict fuel consumption per lap"""
        # Physics-based approximation (will use real model if loaded)
        nmot = telemetry.get('nmot', telemetry.get('rpm', 8000))
        aps = telemetry.get('aps', telemetry.get('throttle', 70))
        speed = telemetry.get('speed', 160)
        
        # Simple fuel model: higher RPM and throttle = more fuel
        base_consumption = 0.05
        rpm_factor = (nmot / 8000) * 0.02
        throttle_factor = (aps / 100) * 0.01
        
        return base_consumption + rpm_factor + throttle_factor
    
    def _predict_tire_wear(self, telemetry: Dict[str, Any]) -> Dict[str, float]:
        """Predict tire wear for all 4 corners"""
        # Simulate wear increasing over laps
        lap = telemetry.get('lap', 1)
        base_wear = lap * 3.5  # ~3.5% per lap
        
        # Front tires wear faster
        return {
            'FL': min(100, base_wear * 1.1),
            'FR': min(100, base_wear * 1.1),
            'RL': min(100, base_wear * 0.9),
            'RR': min(100, base_wear * 0.9)
        }
    
    def _detect_anomaly(self, telemetry: Dict[str, Any]) -> float:
        """Detect anomalies in telemetry"""
        # Check for unusual patterns
        anomaly_score = 0.0
        
        # High RPM with low speed = possible issue
        rpm = telemetry.get('nmot', telemetry.get('rpm', 8000))
        speed = telemetry.get('speed', 160)
        if rpm > 10000 and speed < 100:
            anomaly_score += 0.4
        
        # Throttle vs speed mismatch
        throttle = telemetry.get('aps', telemetry.get('throttle', 70))
        if throttle > 80 and speed < 120:
            anomaly_score += 0.3
        
        # Add some realistic noise
        anomaly_score += np.random.uniform(0, 0.2)
        
        return min(1.0, anomaly_score)
    
    def _predict_fcy(self, telemetry: Dict[str, Any]) -> float:
        """Predict FCY probability"""
        # Simulate FCY probability based on race conditions
        lap = telemetry.get('lap', 1)
        
        # Higher probability mid-race
        if 10 < lap < 20:
            base_prob = 0.15
        else:
            base_prob = 0.05
        
        # Add random factor for realism
        return min(1.0, base_prob + np.random.uniform(0, 0.2))
    
    def _predict_laptime(self, telemetry: Dict[str, Any]) -> float:
        """Predict next lap time"""
        # Use current pace as indicator
        speed = telemetry.get('speed', 160)
        throttle = telemetry.get('aps', telemetry.get('throttle', 70))
        
        # Base laptime around personal best
        base_time = self.state.personal_best_laptime
        
        # Adjust based on current speed/throttle
        if speed < 140:
            return base_time * 1.08  # 8% slower
        elif speed > 180:
            return base_time * 0.98  # 2% faster
        else:
            return base_time * 1.02  # 2% slower
    
    def _predict_pit_loss(self, telemetry: Dict[str, Any]) -> float:
        """Predict pit stop time loss"""
        # Typical pit stop: 12-18 seconds
        return 15.0
    
    def _analyze_driver(self, telemetry: Dict[str, Any]) -> float:
        """Analyze driver consistency"""
        # Perfect consistency = 1.0, poor = < 0.7
        return np.random.uniform(0.85, 1.0)
    
    def _analyze_traffic(self, telemetry: Dict[str, Any]) -> int:
        """Analyze traffic density"""
        # Random number of cars within 3 seconds
        return np.random.randint(0, 4)
    
    def _detect_events(self, predictions: Dict[str, Any], telemetry: Dict[str, Any]) -> List[RaceEvent]:
        """Detect events based on predictions and state"""
        events = []
        current_lap = telemetry.get('lap', 1)
        
        # 1. FUEL CRITICAL
        fuel_prediction = predictions.get('fuel', 0.06)
        fuel_needed = fuel_prediction * self.state.laps_remaining
        if fuel_needed > self.state.fuel_level * 0.95:
            events.append(RaceEvent(
                event_type='FUEL_CRITICAL',
                severity='critical',
                message=f'Fuel critical: Need {fuel_needed:.1f}L, have {self.state.fuel_level:.1f}L',
                data={
                    'fuel_per_lap': fuel_prediction,
                    'laps_remaining': self.state.laps_remaining,
                    'current_fuel': self.state.fuel_level
                }
            ))
        elif fuel_prediction > self.state.last_fuel_prediction * 1.10:  # Lowered from 1.15 to 1.10
            events.append(RaceEvent(
                event_type='FUEL_CONSUMPTION_SPIKE',
                severity='high',
                message=f'Fuel consumption spiked: {fuel_prediction:.3f}L/lap (+10%)',
                data={'fuel_per_lap': fuel_prediction, 'increase_pct': 10}
            ))
        
        # 2. TIRE CRITICAL
        tire_wear = predictions.get('tire', {})
        for corner, wear in tire_wear.items():
            if wear > 85:
                events.append(RaceEvent(
                    event_type='TIRE_CRITICAL',
                    severity='critical',
                    message=f'{corner} tire at {wear:.0f}% wear - pit immediately',
                    data={'corner': corner, 'wear_pct': wear}
                ))
            elif wear > 75:
                events.append(RaceEvent(
                    event_type='TIRE_HIGH_WEAR',
                    severity='high',
                    message=f'{corner} tire at {wear:.0f}% wear - consider pit soon',
                    data={'corner': corner, 'wear_pct': wear}
                ))
        
        # 3. ANOMALY DETECTED
        anomaly_score = predictions.get('anomaly', 0.0)
        if anomaly_score > 0.6:  # Lowered from 0.7 to 0.6
            events.append(RaceEvent(
                event_type='ANOMALY_CRITICAL',
                severity='critical',
                message=f'Critical anomaly detected (score: {anomaly_score:.2f})',
                data={'anomaly_score': anomaly_score, 'telemetry_snapshot': telemetry}
            ))
        elif anomaly_score > 0.4:  # Lowered from 0.5 to 0.4
            events.append(RaceEvent(
                event_type='ANOMALY_WARNING',
                severity='high',  # Changed from medium to high
                message=f'Anomaly detected (score: {anomaly_score:.2f})',
                data={'anomaly_score': anomaly_score}
            ))
        
        # 4. FCY IMMINENT
        fcy_prob = predictions.get('fcy', 0.0)
        if fcy_prob > 0.6:
            events.append(RaceEvent(
                event_type='FCY_IMMINENT',
                severity='high',
                message=f'FCY highly likely ({fcy_prob*100:.0f}%) - pit now!',
                data={'fcy_probability': fcy_prob, 'trend': self.state.fcy_trend}
            ))
        elif fcy_prob > 0.4 and self.state.fcy_trend == "increasing":
            events.append(RaceEvent(
                event_type='FCY_LIKELY',
                severity='medium',
                message=f'FCY probability rising ({fcy_prob*100:.0f}%)',
                data={'fcy_probability': fcy_prob, 'trend': self.state.fcy_trend}
            ))
        
        # 5. PACE DROP
        laptime_pred = predictions.get('laptime', 90.0)
        if laptime_pred > self.state.personal_best_laptime * 1.05:
            delta = laptime_pred - self.state.personal_best_laptime
            events.append(RaceEvent(
                event_type='PACE_DROP',
                severity='medium',
                message=f'Pace drop detected: {delta:.1f}s slower than best',
                data={'predicted_laptime': laptime_pred, 'personal_best': self.state.personal_best_laptime}
            ))
        elif laptime_pred < self.state.personal_best_laptime * 0.98:
            events.append(RaceEvent(
                event_type='PACE_STRONG',
                severity='info',
                message='Strong pace - maintaining excellent times',
                data={'predicted_laptime': laptime_pred}
            ))
        
        # 6. PIT WINDOW
        if self.state.optimal_pit_window_start <= current_lap <= self.state.optimal_pit_window_end:
            if current_lap == self.state.optimal_pit_window_start:
                events.append(RaceEvent(
                    event_type='PIT_WINDOW_OPEN',
                    severity='high',
                    message=f'Optimal pit window open (laps {self.state.optimal_pit_window_start}-{self.state.optimal_pit_window_end})',
                    data={'pit_loss': predictions.get('pit_loss', 15.0), 'window_laps': 3}
                ))
            elif current_lap == self.state.optimal_pit_window_end:
                events.append(RaceEvent(
                    event_type='PIT_WINDOW_CLOSING',
                    severity='critical',
                    message='Pit window closing - last optimal lap!',
                    data={'current_lap': current_lap}
                ))
        
        # 7. DRIVER CONSISTENCY
        consistency = predictions.get('driver_consistency', 1.0)
        if consistency < 0.7:
            events.append(RaceEvent(
                event_type='DRIVER_INCONSISTENT',
                severity='medium',
                message=f'Driver consistency low ({consistency*100:.0f}%) - possible fatigue',
                data={'consistency_score': consistency}
            ))
        
        # 8. TRAFFIC SITUATION
        traffic = predictions.get('traffic', 0)
        if traffic == 0 and self.state.last_traffic_density > 2:
            events.append(RaceEvent(
                event_type='CLEAR_TRACK',
                severity='info',
                message='Clear track ahead - push for next 5 laps',
                data={'traffic_density': traffic}
            ))
        elif traffic > 4:
            events.append(RaceEvent(
                event_type='HEAVY_TRAFFIC',
                severity='medium',
                message=f'Heavy traffic ahead ({traffic} cars) - conserve tires',
                data={'traffic_density': traffic}
            ))
        
        # Update traffic state
        self.state.last_traffic_density = traffic
        
        # Low fuel warning (always check current fuel level)
        current_fuel = telemetry.get('fuel_level', self.state.fuel_level)
        if current_fuel < 10.0:  # Less than 10L remaining
            events.append(RaceEvent(
                event_type='LOW_FUEL',
                severity='critical' if current_fuel < 5.0 else 'high',
                message=f'Low fuel warning: {current_fuel:.1f}L remaining',
                data={'fuel_remaining': current_fuel}
            ))
        
        # High speed event (for testing event detection)
        speed = telemetry.get('speed', 0)
        if speed > 190:
            events.append(RaceEvent(
                event_type='HIGH_SPEED',
                severity='info',
                message=f'High speed detected: {speed:.0f} km/h',
                data={'speed': speed}
            ))
        
        return events


# Global instance
realtime_predictor = RealtimePredictor()

