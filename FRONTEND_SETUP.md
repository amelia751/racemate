# Cognirace Frontend Setup Complete! 🏎️

## ✅ What's Been Created

### New Features Added:

1. **📡 Telemetry Simulator**
   - Start/stop streaming button
   - Simulates real-time telemetry at 1 Hz
   - Auto-increments laps every 10 seconds
   - Makes API calls every 5 seconds
   - Sends lap completion messages to agent
   - All actions logged to debug panel

2. **🐛 Debug Panel**
   - Comprehensive logging system
   - Copy individual logs or all logs
   - Filter by level (error, warning, success, info)
   - Search functionality
   - Auto-scroll toggle
   - Color-coded by severity
   - Expandable data inspection
   - Statistics summary

3. **🎙️ Voice Agent Integration**
   - LiveKit-powered voice communication
   - Real-time transcription
   - Text message fallback
   - Quick action buttons
   - Connection status indicators

4. **📊 Dashboard Components**
   - Real-time telemetry display
   - Race context management
   - Message history
   - Track selection
   - Session configuration

## 🚀 Running the System

### Ports Configuration:
- **Backend API**: `http://localhost:8005`
- **Frontend**: `http://localhost:3005`

### Start Both Servers:

```bash
# Terminal 1 - Backend API
cd /Users/anhlam/hack-the-track/backend-api
source venv/bin/activate
python -m uvicorn main:app --host 0.0.0.0 --port 8005 --reload

# Terminal 2 - Frontend
cd /Users/anhlam/hack-the-track/frontend
npm run dev
```

### Access the Application:

Open your browser to: **http://localhost:3005**

## 🧪 Testing the System

### Step 1: Connect to Voice Agent
1. Enter a room name (e.g., `race-session-001`)
2. Enter your driver name
3. Click "Connect to Strategy Agent"
4. Wait for green "Agent Ready" indicator

### Step 2: Configure Race Context
1. Select track (default: Suzuka Circuit)
2. Set session type (practice/qualifying/race)
3. Set total laps (default: 53)
4. Context is auto-sent to agent

### Step 3: Start Telemetry Streaming
1. Click the big **"🚀 START STREAMING"** button
2. Watch telemetry update in real-time
3. Observe API calls in debug panel
4. See agent responses in message panel
5. Track logs in debug panel

### Step 4: Interact with Agent

**Via Voice:**
- Just speak your questions
- Agent responds with voice and text

**Via Text:**
- Type in message box
- Or use quick action buttons:
  - ⛽ Fuel Check
  - 🛞 Tire Check
  - 📊 Race Status
  - 🏁 Pit Strategy

### Step 5: Debug Panel Usage

**View Logs:**
- All events logged with timestamps
- Color-coded by severity
- Click to expand data

**Copy for Debugging:**
- Click 📋 on individual logs
- Or click "📋 Copy All" for everything
- Paste to share with developers

**Filter & Search:**
- Filter by log level
- Search by keyword
- Auto-scroll toggle

## 🎯 What to Test

### 1. Connection Flow
- [ ] Frontend loads without errors
- [ ] LiveKit connection establishes
- [ ] Agent joins the room
- [ ] Agent status shows "Ready"

### 2. Telemetry Streaming
- [ ] Click "Start Streaming" button
- [ ] Telemetry updates every second
- [ ] Laps increment every 10 seconds
- [ ] API calls every 5 seconds
- [ ] All events logged to debug panel

### 3. API Integration
- [ ] Fuel prediction API responds
- [ ] Tire prediction API responds
- [ ] Health check passes
- [ ] Errors logged properly

### 4. Agent Communication
- [ ] Voice input works
- [ ] Text messages send
- [ ] Agent responds
- [ ] Messages display in chat
- [ ] Transcriptions appear

### 5. Debug Panel
- [ ] Logs appear in real-time
- [ ] Copy functions work
- [ ] Filters work correctly
- [ ] Search works
- [ ] Statistics update

## 🐛 Troubleshooting

### Backend Not Starting (Port 8005)

**Check if port is in use:**
```bash
lsof -ti:8005 | xargs kill -9  # Kill process on port 8005
```

**Check virtual environment:**
```bash
cd /Users/anhlam/hack-the-track/backend-api
source venv/bin/activate
python --version  # Should be 3.10+
```

**Check dependencies:**
```bash
pip install -r requirements.txt
```

**Start with verbose logging:**
```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8005 --reload --log-level debug
```

### Frontend Not Starting (Port 3005)

**Check if port is in use:**
```bash
lsof -ti:3005 | xargs kill -9  # Kill process on port 3005
```

**Reinstall dependencies:**
```bash
cd /Users/anhlam/hack-the-track/frontend
rm -rf node_modules package-lock.json
npm install
```

**Check Node version:**
```bash
node --version  # Should be 18+
```

### LiveKit Connection Issues

**Check environment variables:**
```bash
cat /Users/anhlam/hack-the-track/frontend/.env.local
```

Should have:
- `NEXT_PUBLIC_LIVEKIT_URL`
- `NEXT_PUBLIC_LIVEKIT_API_KEY`
- `LIVEKIT_API_SECRET`

**Check browser console:**
- Open DevTools (F12)
- Look for WebRTC errors
- Check microphone permissions

### API Calls Failing

**Test backend directly:**
```bash
# Health check
curl http://localhost:8005/health

# Fuel prediction
curl -X POST http://localhost:8005/predict/fuel \
  -H "Content-Type: application/json" \
  -d '{"speed":200,"nmot":8000,"gear":5,"aps":90,"lap":10}'
```

**Check CORS:**
- Backend should allow `http://localhost:3005`
- Check `backend-api/main.py` CORS settings

### Debug Panel Not Showing Logs

**Check browser console:**
```javascript
// In browser console:
console.log('Test debug log');
```

**Force a log entry:**
- Click any button in the simulator
- Should trigger log entries

## 📋 Debug Panel Log Categories

| Category | Description |
|----------|-------------|
| `SIMULATOR` | Telemetry streaming events |
| `TELEMETRY` | Real-time telemetry data |
| `API` | Backend API calls and responses |
| `API TEST` | Manual API test results |
| `AGENT` | Voice agent communication |
| `CONNECTION` | LiveKit connection events |
| `DEBUG` | Debug panel internal events |

## 🎨 Features Demo

### Telemetry Simulator:
- **Real-time streaming**: Updates UI at 1 Hz
- **Realistic data**: Randomized within racing ranges
- **Auto lap progression**: Lap changes every 10 seconds
- **API integration**: Automatic predictions every 5 seconds
- **Agent notifications**: Lap completion messages

### Debug Panel:
- **Live logging**: All events captured in real-time
- **Rich filtering**: By level, category, or keyword
- **Copy functionality**: Individual or bulk copy
- **Data inspection**: Expandable JSON data
- **Statistics**: Summary of log counts by level

## 🚦 Current Status

✅ **Frontend Structure**: Complete
✅ **Telemetry Simulator**: Complete  
✅ **Debug Panel**: Complete
✅ **Voice Agent UI**: Complete
✅ **Port Configuration**: 3005 (frontend), 8005 (backend)
✅ **LiveKit Integration**: Complete
✅ **API Client**: Complete
✅ **State Management**: Complete

## 📝 Next Steps

1. **Test the complete flow:**
   - Start both servers
   - Open http://localhost:3005
   - Connect to voice agent
   - Start telemetry streaming
   - Monitor debug panel

2. **If issues occur:**
   - Check debug panel for errors
   - Copy all logs (📋 Copy All button)
   - Share with developer for debugging

3. **Expected behavior:**
   - Telemetry updates smoothly
   - API calls succeed
   - Agent responds to queries
   - All events logged properly

## 🎯 Success Criteria

✅ Both servers running on correct ports
✅ Frontend loads without errors
✅ Telemetry simulator starts and streams
✅ Debug panel shows all logs
✅ API calls succeed (check debug panel)
✅ Voice agent connects
✅ Messages sent and received
✅ Copy functionality works

## 📞 Support

If you encounter any issues:

1. **Check Debug Panel** - All errors logged there
2. **Copy Logs** - Use "📋 Copy All" button
3. **Check Browser Console** - Press F12
4. **Check Backend Logs** - Terminal running uvicorn
5. **Verify Ports** - Run `lsof -ti:8005` and `lsof -ti:3005`

---

**Built with**: Next.js 14, TypeScript, Tailwind CSS, LiveKit, Zustand
**Ready for**: Real-time race strategy testing! 🏁

