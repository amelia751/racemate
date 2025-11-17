# Frontend Debugging Guide

## ✅ React Error Fixed

All `setState` calls are now properly deferred to avoid "Cannot update component while rendering" error.

## 🔍 How to Debug the Chat Interface

### Step 1: Open Browser Console
- **Chrome/Edge**: Press `F12` or `Cmd+Option+I` (Mac) / `Ctrl+Shift+I` (Windows)
- **Safari**: Enable Developer Menu in Preferences, then `Cmd+Option+C`

### Step 2: Start Streaming
1. Open `http://localhost:3005`
2. Click **"START STREAMING"** button
3. Watch the console for these logs:

### Expected Console Output

```javascript
// 1. Streaming starts
[StreamingControls] START STREAMING clicked

// 2. After 1 second (first frame sent)
[StreamingControls] Backend response: { success: true, events: [...], recommendations: {...} }
[StreamingControls] Storing recommendation in window: { strategy: "...", severity_summary: {...} }
[StreamingControls] Window object updated: { strategy: "...", scenario: "FUEL_CRISIS", timestamp: 1234567890 }

// 3. VoiceStrategist polls every 500ms
[VoiceStrategist] Starting to poll for recommendations
[VoiceStrategist] Found recommendation in window: { strategy: "...", scenario: "FUEL_CRISIS" }
[VoiceStrategist] Adding recommendation to chat: { type: "critical", message: "..." }

// 4. Next scenario (after 5 seconds)
[StreamingControls] Backend response: { success: true, events: [...], recommendations: {...} }
...
```

---

## 🐛 Troubleshooting

### Problem: No console logs at all
**Solution**: Check if frontend is running
```bash
lsof -ti:3005
```
If nothing, restart frontend:
```bash
cd frontend && npm run dev
```

### Problem: Logs show "Backend response" but no recommendations
**Check**: `data.recommendations` might be `null` or `undefined`
**Reason**: No critical/high severity events detected
**Solution**: Wait for FUEL_CRISIS scenario (0-5s) or EXTREME scenario (20-25s)

### Problem: Logs show recommendation stored but VoiceStrategist doesn't find it
**Check**: `isStreaming` state in VoiceStrategist
**Debug**: Add this to console:
```javascript
// In browser console
window.__latestRecommendation
```
Should show the recommendation object if it's stored.

### Problem: VoiceStrategist finds recommendation but chat doesn't update
**Check**: React state update
**Debug**: Look for `setRecommendations` errors in console
**Solution**: Check if `addRecommendation` function is working

---

## 🎯 What Should Happen

### Scenario Cycle (every 5 seconds)
1. **FUEL_CRISIS** (0-5s) → 🚨 CRITICAL + ⚠️ HIGH events → AI recommendation
2. **HIGH_SPEED_LOW_FUEL** (5-10s) → ⚠️ HIGH + ℹ️ INFO events → AI recommendation
3. **ANOMALY_DETECTED** (10-15s) → ⚠️ HIGH + ⚡ MEDIUM events → AI recommendation
4. **OPTIMAL** (15-20s) → Minimal events → Might not generate recommendation
5. **EXTREME** (20-25s) → 🚨 Multiple CRITICAL events → AI recommendation
6. **TIRE_STRESS** (25-30s) → ⚠️ HIGH events → AI recommendation

### Chat Interface Behavior
- Each recommendation appears as a card
- Shows scenario name (FUEL_CRISIS, etc.)
- Displays AI-generated strategy text
- Lists detected events above the recommendation
- Scrolls automatically to newest recommendation

---

## 🧪 Manual Test Commands

### Test Backend Directly
```bash
curl -X POST http://localhost:8005/realtime/process \
  -H "Content-Type: application/json" \
  -d '{"telemetry":{"speed":195,"fuel_level":5,"lap":18,"rpm":8500,"gear":5,"throttle":80,"aps":80,"nmot":8500,"cum_brake_energy":30000,"cum_lateral_load":50000,"air_temp":28}}'
```

Should return:
```json
{
  "success": true,
  "events": [
    {"type": "FUEL_CONSUMPTION_SPIKE", "severity": "high", ...},
    {"type": "LOW_FUEL", "severity": "critical", ...},
    ...
  ],
  "recommendations": {
    "strategy": "Based on the critical fuel situation...",
    "severity_summary": {"critical": 1, "high": 2, ...}
  }
}
```

### Test Next.js API Route
```bash
curl -X POST http://localhost:3005/api/telemetry/stream \
  -H "Content-Type: application/json" \
  -d '{"speed":195,"fuel_level":5,"lap":18,"rpm":8500,"gear":5,"throttle":80,"aps":80,"nmot":8500,"cum_brake_energy":30000,"cum_lateral_load":50000,"air_temp":28}'
```

---

## 📊 Debug Panel (🐛 Button)

Click the 🐛 button (bottom-right) to see:

### Logs Tab
- Shows all streaming events
- Auto-clears on START STREAMING
- "Copy All" button copies entire history

### Real-Time Tab
- Backend API status (online/offline)
- Current telemetry values
- ML models status
- AI rate limiting info

---

## ✅ Success Indicators

You'll know it's working when you see:

1. ✅ Console shows `[StreamingControls] Backend response` every second
2. ✅ Console shows `[VoiceStrategist] Found recommendation` 
3. ✅ Chat interface shows cards with AI recommendations
4. ✅ Scenario badge updates every 5 seconds (FUEL_CRISIS → HIGH_SPEED → etc.)
5. ✅ Debug panel logs show events detected

---

## 🆘 Still Not Working?

Copy the entire console output and share it. Look for:
- Red error messages
- Failed fetch requests
- React component errors
- State update warnings

