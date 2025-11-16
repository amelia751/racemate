# Event-Driven Recommendation System Architecture

## Current State (Naive):
```
Frontend → Timer (15s) → Call Gemini → Recommendation
```
**Problem**: Calls API regardless of whether anything important happened

---

## Proposed Architecture (Intelligent):

```
Frontend Telemetry Stream
    ↓
Backend Real-Time Processor
    ↓
8 ML Models (Continuous Inference)
    ↓
Change Detection + Thresholds
    ↓
Event Trigger System
    ↓
Gemini (Only when event detected)
    ↓
Actionable Recommendation
```

---

## Model-Driven Event Triggers

### 1. Fuel Consumption Model
**Current Prediction**: Fuel per lap
**State to Track**: 
- `last_fuel_prediction`
- `laps_remaining`
- `current_fuel_level`

**Trigger Conditions**:
```python
if predicted_fuel_consumption * laps_remaining > current_fuel_level * 0.9:
    trigger_event("FUEL_CRITICAL", severity="high")
elif predicted_fuel_consumption > last_prediction * 1.15:
    trigger_event("FUEL_SPIKE", severity="medium")
```

**Recommendation Examples**:
- "⛽ FUEL CRITICAL: Pit within 2 laps or risk DNF"
- "⚠️ Fuel consumption 15% higher than expected, reduce aggressive braking"

---

### 2. Tire Degradation Model
**Current Prediction**: Tire wear percentage per corner
**State to Track**:
- `tire_wear_FL/FR/RL/RR`
- `degradation_rate`

**Trigger Conditions**:
```python
if any(tire_wear) > 85:
    trigger_event("TIRE_CRITICAL", severity="high")
elif degradation_rate > historical_avg * 1.3:
    trigger_event("TIRE_RAPID_WEAR", severity="medium")
```

**Recommendation Examples**:
- "🛞 TIRES CRITICAL: Front-left at 88% wear, pit immediately"
- "⚠️ Tire degradation accelerating, consider earlier pit stop"

---

### 3. Anomaly Detector (LSTM Autoencoder)
**Current Prediction**: Anomaly score (0-1)
**State to Track**:
- `anomaly_score`
- `anomaly_pattern` (which sensors triggered)

**Trigger Conditions**:
```python
if anomaly_score > 0.7:
    trigger_event("ANOMALY_CRITICAL", severity="high")
elif anomaly_score > 0.5:
    trigger_event("ANOMALY_WARNING", severity="medium")
```

**Recommendation Examples**:
- "🚨 ANOMALY: Unusual brake pattern detected, check brake balance"
- "⚠️ Throttle application inconsistent, driver fatigue possible"

---

### 4. FCY Hazard Model
**Current Prediction**: Probability of full-course yellow
**State to Track**:
- `fcy_probability`
- `trend` (increasing/stable/decreasing)

**Trigger Conditions**:
```python
if fcy_probability > 0.6:
    trigger_event("FCY_IMMINENT", severity="high")
elif fcy_probability > 0.3 and trend == "increasing":
    trigger_event("FCY_LIKELY", severity="medium")
```

**Recommendation Examples**:
- "🚦 FCY IMMINENT (85%): Pit NOW before caution"
- "⚠️ FCY probability rising, prepare for early pit"

---

### 5. Lap-Time Transformer
**Current Prediction**: Predicted next lap time
**State to Track**:
- `predicted_lap_time`
- `personal_best`
- `recent_average`

**Trigger Conditions**:
```python
if predicted_lap_time > personal_best * 1.05:
    trigger_event("PACE_DROP", severity="medium")
elif predicted_lap_time < personal_best * 0.98:
    trigger_event("PACE_IMPROVEMENT", severity="info")
```

**Recommendation Examples**:
- "📉 PACE DROP: 3% slower than best, check tire pressures"
- "🚀 Strong pace! Maintain current strategy"

---

### 6. Pit Loss Model
**Current Prediction**: Time loss for pit stop at current lap
**State to Track**:
- `predicted_pit_loss`
- `optimal_window_start/end`

**Trigger Conditions**:
```python
if current_lap in optimal_pit_window:
    trigger_event("PIT_WINDOW_OPEN", severity="medium")
elif current_lap == optimal_pit_window.end - 1:
    trigger_event("PIT_WINDOW_CLOSING", severity="high")
```

**Recommendation Examples**:
- "⏱️ PIT WINDOW OPEN: Minimal time loss (12.3s) for next 3 laps"
- "🚨 PIT WINDOW CLOSING: Last optimal lap for pit stop"

---

### 7. Driver Embedding
**Current Prediction**: Driver behavior vector
**State to Track**:
- `current_embedding`
- `baseline_embedding` (learned during training)
- `cosine_similarity`

**Trigger Conditions**:
```python
similarity = cosine_similarity(current, baseline)
if similarity < 0.7:
    trigger_event("DRIVER_INCONSISTENT", severity="medium")
```

**Recommendation Examples**:
- "👤 Driving style deviating from baseline, consider hydration break"
- "⚠️ Input consistency dropping, driver fatigue possible"

---

### 8. Traffic GNN
**Current Prediction**: Optimal racing line given traffic
**State to Track**:
- `traffic_density` (cars within 3 seconds)
- `clear_track_opportunity`

**Trigger Conditions**:
```python
if traffic_density == 0 and last_traffic_density > 2:
    trigger_event("CLEAR_TRACK", severity="info")
elif upcoming_traffic > 5:
    trigger_event("TRAFFIC_JAM", severity="medium")
```

**Recommendation Examples**:
- "🏁 CLEAR TRACK: Push for next 5 laps, no traffic ahead"
- "🚗 Heavy traffic ahead, conserve tires for overtake"

---

## Backend Implementation

### Real-Time Prediction Engine

```python
# backend-api/services/realtime_predictor.py

class RealtimePredictor:
    def __init__(self):
        self.models = self.load_all_models()
        self.state = PredictionState()
        self.thresholds = load_thresholds()
        
    def process_telemetry(self, telemetry: TelemetryFrame):
        """Process incoming telemetry through all models"""
        
        # Run all 8 models
        predictions = {
            'fuel': self.models.fuel.predict(telemetry),
            'tire': self.models.tire.predict(telemetry),
            'anomaly': self.models.anomaly.predict(telemetry),
            'fcy': self.models.fcy.predict(telemetry),
            'laptime': self.models.laptime.predict(telemetry),
            'pit_loss': self.models.pit_loss.predict(telemetry),
            'driver': self.models.driver.predict(telemetry),
            'traffic': self.models.traffic.predict(telemetry)
        }
        
        # Detect changes and trigger events
        events = self.detect_events(predictions, self.state)
        
        # Update state
        self.state.update(predictions)
        
        # Generate recommendations for events
        if events:
            recommendations = self.generate_recommendations(events, telemetry)
            return recommendations
        
        return None  # No events, no recommendation needed
    
    def detect_events(self, predictions, state):
        """Compare current predictions vs state to detect events"""
        events = []
        
        # Fuel check
        if predictions['fuel'] * state.laps_remaining > state.fuel_level * 0.9:
            events.append(Event('FUEL_CRITICAL', severity='high', data=predictions['fuel']))
        
        # Tire check
        for corner, wear in predictions['tire'].items():
            if wear > 85:
                events.append(Event('TIRE_CRITICAL', severity='high', corner=corner, wear=wear))
        
        # Anomaly check
        if predictions['anomaly'] > 0.7:
            events.append(Event('ANOMALY_CRITICAL', severity='high', score=predictions['anomaly']))
        
        # FCY check
        if predictions['fcy'] > 0.6:
            events.append(Event('FCY_IMMINENT', severity='high', probability=predictions['fcy']))
        
        # Laptime check
        if predictions['laptime'] > state.personal_best * 1.05:
            events.append(Event('PACE_DROP', severity='medium', delta=predictions['laptime'] - state.personal_best))
        
        # ... more checks for other models
        
        return events
```

### WebSocket Endpoint

```python
# backend-api/routers/realtime.py

@router.websocket("/ws/telemetry")
async def telemetry_websocket(websocket: WebSocket):
    await websocket.accept()
    predictor = RealtimePredictor()
    
    try:
        while True:
            # Receive telemetry from frontend
            data = await websocket.receive_json()
            telemetry = TelemetryFrame(**data)
            
            # Process through models and detect events
            recommendations = predictor.process_telemetry(telemetry)
            
            # Send recommendations only if events detected
            if recommendations:
                await websocket.send_json({
                    'type': 'recommendation',
                    'data': recommendations
                })
    except WebSocketDisconnect:
        pass
```

---

## Frontend Changes

```typescript
// frontend/lib/realtime-processor.ts

export class RealtimeProcessor {
  private ws: WebSocket;
  
  connect() {
    this.ws = new WebSocket('ws://localhost:8005/ws/telemetry');
    
    this.ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      if (data.type === 'recommendation') {
        // Only shows recommendations when backend detects events
        addRecommendation(data.data);
      }
    };
  }
  
  sendTelemetry(telemetry: TelemetryData) {
    // Send every frame, backend decides if recommendation needed
    this.ws.send(JSON.stringify(telemetry));
  }
}
```

---

## Benefits of Event-Driven System

✅ **Efficient**: Only calls Gemini when models detect significant changes
✅ **Responsive**: Immediate recommendations when issues arise (not waiting for timer)
✅ **Intelligent**: Leverages your 8 trained ML models properly
✅ **Quota-Friendly**: ~90% fewer API calls (only on events, not every 15s)
✅ **Relevant**: Recommendations are always actionable and timely

---

## Priority Events (Most Critical First)

1. **FUEL_CRITICAL** - Risk of DNF
2. **ANOMALY_CRITICAL** - Mechanical issue possible
3. **FCY_IMMINENT** - Strategic opportunity
4. **TIRE_CRITICAL** - Performance/safety issue
5. **PIT_WINDOW_CLOSING** - Strategic timing
6. **PACE_DROP** - Performance concern
7. **TRAFFIC_CLEAR** - Opportunity to push
8. **DRIVER_INCONSISTENT** - Human factors

---

## Implementation Steps

1. ✅ **Models Trained & Deployed** (Done!)
2. 🔲 Create `RealtimePredictor` class
3. 🔲 Implement event detection logic
4. 🔲 Add WebSocket endpoint
5. 🔲 Frontend: Replace timer with WebSocket
6. 🔲 Test event triggering
7. 🔲 Tune thresholds based on testing

---

## Estimated Impact

**Current System**:
- Gemini calls: 4 per minute (every 15s)
- Relevance: ~30% (many calls with no important changes)
- Latency: Up to 15s to detect issue

**Event-Driven System**:
- Gemini calls: ~0.5 per minute (only on events)
- Relevance: ~95% (only calls when models detect something)
- Latency: <1s from issue detection to recommendation

**Quota Savings**: ~87% fewer API calls! 🎉

