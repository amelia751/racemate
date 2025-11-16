# 🏎️ Cognirace - Quick Start Guide

## ✅ System Status

**Backend API**: ✅ Running on `http://localhost:8005`
**Frontend**: ✅ Running on `http://localhost:3005`

---

## 🚀 Access the Application

**Open your browser to:**
```
http://localhost:3005
```

---

## 🎮 How to Use

### 1️⃣ **Connect to Voice Agent** (Top Right)

1. Room name: `race-session-001` (or your choice)
2. Driver name: `Driver` (or your name)
3. Click **"🚀 Connect to Strategy Agent"**
4. Wait for green **"Agent Ready"** indicator

### 2️⃣ **Start Telemetry Streaming** (Bottom Left)

1. Find the **"🏁 Telemetry Simulator"** panel
2. Click the big green **"🚀 START STREAMING"** button
3. Watch the magic happen:
   - ✅ Telemetry updates every second
   - ✅ Laps increment every 10 seconds
   - ✅ API predictions every 5 seconds
   - ✅ Agent gets lap completion messages
   - ✅ All logged to debug panel

### 3️⃣ **Monitor Debug Panel** (Bottom)

- **📋 All events logged here!**
- See every API call
- See every agent message
- See every error (if any)
- **Click "📋 Copy All"** to share logs with developers

### 4️⃣ **Ask the Agent Questions**

**Quick Actions:**
- ⛽ **Fuel Check** - Current fuel status
- 🛞 **Tire Check** - Tire degradation analysis
- 📊 **Race Status** - Comprehensive briefing
- 🏁 **Pit Strategy** - Optimal pit stop plan

**Or type custom questions:**
- "How many laps can I go on current fuel?"
- "When should I pit?"
- "What's my tire grip level?"
- "Give me comprehensive race status"

**Or use your voice:**
- Just speak your question!
- Agent responds with voice and text

---

## 🐛 Debug Panel - Your Best Friend

### What It Shows:
- ✅ **Connection events** - LiveKit, agent status
- ✅ **Telemetry data** - Real-time car data
- ✅ **API calls** - Fuel, tire predictions
- ✅ **Agent messages** - All communications
- ✅ **Errors** - Any issues (highlighted in red)

### How to Use:
1. **Watch logs appear in real-time**
2. **Filter by level** - All/Errors/Warnings/Success/Info
3. **Search** - Type keywords to find specific logs
4. **Copy logs**:
   - Click 📋 on individual log
   - Or click **"📋 Copy All"** for everything
5. **Share with developer** for debugging

### Log Colors:
- 🔴 **Red** - Errors (something failed)
- 🟡 **Yellow** - Warnings (attention needed)
- 🟢 **Green** - Success (operation completed)
- 🔵 **Blue** - Info (general information)

---

## 🧪 Test Scenario

### Full Integration Test:

1. **Open http://localhost:3005**

2. **Connect to agent:**
   - Room: `race-session-001`
   - Click "Connect"
   - Wait for "Agent Ready"

3. **Start streaming:**
   - Click "🚀 START STREAMING"
   - Watch telemetry panel update

4. **Check debug panel:**
   - Should see "SIMULATOR: Starting telemetry stream"
   - Should see "TELEMETRY: Lap X | Speed: XXX km/h"
   - Should see "API: Requesting fuel prediction..."
   - Should see "API: Fuel prediction received"

5. **Ask agent:**
   - Click "⛽ Fuel Check" button
   - Or type "What's my fuel status?"
   - Check message panel for response
   - Check debug panel for agent logs

6. **If everything works:**
   - ✅ Telemetry streaming
   - ✅ API calls succeeding
   - ✅ Agent responding
   - ✅ All logged in debug panel
   - 🎉 **SUCCESS!**

7. **If something fails:**
   - 🔍 Check debug panel for red errors
   - 📋 Click "Copy All"
   - 📧 Send logs to developer

---

## 🔧 Troubleshooting

### Backend Not Responding

```bash
# Check if backend is running
curl http://localhost:8005/health

# Should return: {"status":"healthy",...}
```

**If not working:**
```bash
# Restart backend
cd /Users/anhlam/hack-the-track/backend-api
source venv/bin/activate
python -m uvicorn main:app --host 0.0.0.0 --port 8005 --reload
```

### Frontend Not Loading

```bash
# Check if frontend is running
curl http://localhost:3005

# Should return HTML
```

**If not working:**
```bash
# Restart frontend
cd /Users/anhlam/hack-the-track/frontend
npm run dev
```

### Agent Not Connecting

1. Check browser console (F12)
2. Look for WebRTC errors
3. Check microphone permissions
4. Verify LiveKit credentials in `.env.local`

### API Calls Failing

1. **Check debug panel** - Look for red API errors
2. **Test backend directly:**
   ```bash
   curl -X POST http://localhost:8005/predict/fuel \
     -H "Content-Type: application/json" \
     -d '{"speed":200,"nmot":8000,"gear":5,"aps":90,"lap":10}'
   ```
3. **Check backend logs** in terminal

---

## 📊 What's Happening Under the Hood

### Telemetry Simulator:
- Generates realistic racing data
- Updates at 1 Hz (once per second)
- Cycles through laps (10 seconds each)
- Makes API predictions every 5 seconds
- Sends lap messages to agent

### API Integration:
- **Fuel Prediction**: XGBoost model on Vertex AI
- **Tire Prediction**: Physics-informed TCN
- **Lap Time**: Transformer model
- All models trained on real Toyota GR Cup data

### Voice Agent:
- **LiveKit** for real-time audio
- **Gemini 2.5 Flash** for natural language
- Multi-agent system (Fuel, Tire, Chief)
- Real-time strategy recommendations

---

## 🎯 Expected Behavior

### ✅ Good Signs:
- Telemetry updates smoothly every second
- Lap counter increments every 10 seconds
- Debug panel shows green success messages
- API calls return in <500ms
- Agent responds to questions
- No red errors in debug panel

### 🚨 Bad Signs:
- Red errors in debug panel
- API calls timing out
- Agent not responding
- Telemetry not updating
- Browser console errors

**If you see bad signs:**
1. Stop streaming
2. Copy all debug logs
3. Check browser console
4. Check backend terminal
5. Share logs with developer

---

## 🎬 Demo Flow

### Perfect Run:

```
1. Open http://localhost:3005
   → Frontend loads ✅

2. Click "Connect to Strategy Agent"
   → "Agent Ready" appears ✅
   → Debug: "Agent joined the room" ✅

3. Click "🚀 START STREAMING"
   → Telemetry updates appear ✅
   → Debug: "SIMULATOR: Starting telemetry stream" ✅
   → Debug: "TELEMETRY: Lap 1 | Speed: 245 km/h..." ✅

4. Wait 5 seconds
   → Debug: "API: Requesting fuel prediction..." ✅
   → Debug: "API: Fuel prediction received" ✅

5. Click "⛽ Fuel Check"
   → Debug: "AGENT: Sent message to agent" ✅
   → Message appears in chat panel ✅
   → Agent response appears ✅

6. Check all panels
   → Telemetry Display: Shows live data ✅
   → Race Context: Shows track and lap ✅
   → Message Display: Shows conversation ✅
   → Debug Panel: Shows all logs ✅

🎉 SUCCESS - SYSTEM WORKING!
```

---

## 📝 Quick Reference

| Component | Port | URL |
|-----------|------|-----|
| Frontend | 3005 | http://localhost:3005 |
| Backend API | 8005 | http://localhost:8005 |
| API Health | 8005 | http://localhost:8005/health |
| API Docs | 8005 | http://localhost:8005/docs |

| Action | Location | Button |
|--------|----------|--------|
| Connect Agent | Top Right | "🚀 Connect to Strategy Agent" |
| Start Streaming | Bottom Left | "🚀 START STREAMING" |
| Stop Streaming | Bottom Left | "⏹️ STOP STREAMING" |
| Copy Debug Logs | Debug Panel | "📋 Copy All" |
| Fuel Check | Voice Agent Panel | "⛽ Fuel Check" |
| Tire Check | Voice Agent Panel | "🛞 Tire Check" |

---

## 🚀 Ready to Race!

Everything is set up and running. Just open **http://localhost:3005** and start testing!

**Any issues?** Check the debug panel first - it's your best friend for troubleshooting! 🐛

---

**Built with**: ❤️ for Toyota GR Cup Racing
**Powered by**: Gemini 2.5 Flash • LiveKit • 8 ML Models • Vertex AI

