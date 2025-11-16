# Cognirace Frontend

Real-time race strategy platform with AI-powered voice agent for the Toyota GR Cup Series.

## 🏎️ Features

- **Voice Agent Integration**: Real-time voice communication with AI race strategist powered by LiveKit
- **Live Telemetry Display**: Real-time visualization of car telemetry data
- **Race Context Management**: Configure track, laps, and session type
- **Message History**: Full conversation log between driver and strategy agent
- **Quick Actions**: One-click queries for fuel, tires, and strategy recommendations
- **ML-Powered Predictions**: Integration with 8 trained models on Vertex AI:
  - Fuel Consumption (XGBoost)
  - Lap Time (Transformer)
  - Tire Degradation (Physics-informed TCN)
  - FCY Hazard (Survival Analysis)
  - Pit Loss Prediction
  - Anomaly Detection
  - Driver Embedding
  - Traffic GNN

## 🚀 Quick Start

### Prerequisites

- Node.js 18+ and npm
- LiveKit account and credentials
- Cognirace backend API running (port 8005)
- Python backend for token generation (port 5001)

### Installation

```bash
cd frontend
npm install
```

### Environment Configuration

Create a `.env.local` file:

```env
# LiveKit Configuration
NEXT_PUBLIC_LIVEKIT_URL=wss://your-livekit-url.livekit.cloud
NEXT_PUBLIC_LIVEKIT_API_KEY=your-api-key
LIVEKIT_API_SECRET=your-api-secret

# Cognirace API
NEXT_PUBLIC_API_URL=http://localhost:8005

# Backend Server
NEXT_PUBLIC_BACKEND_URL=http://localhost:5001
```

### Development

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

### Production Build

```bash
npm run build
npm start
```

## 📁 Project Structure

```
frontend/
├── app/
│   ├── api/
│   │   └── livekit/
│   │       └── token/
│   │           └── route.ts          # LiveKit token generation API
│   ├── layout.tsx                    # Root layout with LiveKit provider
│   ├── page.tsx                      # Main dashboard
│   └── globals.css                   # Global styles
├── components/
│   ├── VoiceAgentPanel.tsx          # Voice agent controls
│   ├── TelemetryDisplay.tsx         # Real-time telemetry
│   ├── MessageDisplay.tsx           # Chat conversation
│   └── RaceContextPanel.tsx         # Race configuration
├── lib/
│   ├── livekit/
│   │   ├── LiveKitContext.tsx       # LiveKit React context
│   │   └── useLiveKitRacing.ts      # Custom LiveKit hook
│   ├── api-client.ts                # API client for backend
│   └── store.ts                     # Zustand state management
└── .env.local                        # Environment variables
```

## 🎮 Usage

### 1. Connect to Voice Agent

1. Enter a room name (e.g., `race-session-001`)
2. Enter your driver name
3. Click "Connect to Strategy Agent"
4. Wait for agent to be ready (green indicator)

### 2. Configure Race Context

1. Select your track from the dropdown
2. Set session type (practice, qualifying, or race)
3. Configure total laps and current lap
4. Context is automatically sent to the agent

### 3. Interact with Agent

**Voice Interaction:**
- Speak directly to ask questions
- Agent responds with voice and text

**Text Interaction:**
- Type questions in the message input
- Use quick action buttons for common queries

**Quick Actions:**
- ⛽ **Fuel Check**: Get current fuel status and predictions
- 🛞 **Tire Check**: Analyze tire degradation
- 📊 **Race Status**: Comprehensive race briefing
- 🏁 **Pit Strategy**: Optimal pit stop recommendations

### 4. View Telemetry

Real-time display of:
- Speed (km/h)
- RPM
- Gear
- Throttle position
- Current lap
- Fuel level
- Brake energy
- Air temperature

## 🔧 Technology Stack

### Frontend
- **Next.js 14**: React framework with App Router
- **TypeScript**: Type-safe development
- **Tailwind CSS**: Utility-first styling
- **LiveKit Client SDK**: Real-time audio/video
- **Zustand**: Lightweight state management
- **Axios**: HTTP client

### Voice Agent
- **LiveKit**: Real-time communication infrastructure
- **Google Gemini 2.5 Flash**: Natural language processing
- **WebRTC**: Browser-based audio streaming

### Backend Integration
- **FastAPI**: ML prediction API (port 8005)
- **Flask**: Token generation and LiveKit management (port 5001)
- **Vertex AI**: Model serving
- **Google Cloud Storage**: Model artifacts

## 🎯 API Endpoints

### Frontend API Routes

**POST `/api/livekit/token`**
- Generate LiveKit access token
- Body: `{ roomName, participantName, metadata }`
- Returns: `{ token }`

### Backend API (FastAPI - Port 8005)

**POST `/predict/fuel`**
- Predict fuel consumption
- Body: `{ speed, nmot, gear, aps, lap }`

**POST `/predict/laptime`**
- Predict lap time
- Body: `{ telemetry_sequence }`

**POST `/predict/tire`**
- Predict tire degradation
- Body: `{ cum_brake_energy, cum_lateral_load, air_temp, telemetry_sequence }`

**GET `/health`**
- Health check

## 🤖 Agent Communication

### RPC Methods

The frontend communicates with the voice agent using LiveKit RPC:

**`set_instructions`**: Initialize agent with race context
**`user_message`**: Send text message to agent
**`ping`**: Health check
**`get_current_context`**: Retrieve current agent context

### Data Topics

**`chat_message`**: Agent text responses
**`agent_status`**: Agent state updates
**`user_transcription`**: Speech-to-text results

## 🎨 Customization

### Styling

Modify `app/globals.css` for global styles and `tailwind.config.ts` for theme configuration.

### Race Tracks

Add new tracks in `components/RaceContextPanel.tsx`:

```typescript
const tracks = [
  'Your New Track',
  // ... existing tracks
];
```

### Telemetry Metrics

Customize displayed metrics in `components/TelemetryDisplay.tsx`.

## 🐛 Troubleshooting

### Agent Not Connecting

1. Check LiveKit credentials in `.env.local`
2. Verify backend server is running on port 5001
3. Check browser console for WebRTC errors
4. Ensure microphone permissions are granted

### API Connection Issues

1. Verify FastAPI server is running on port 8005
2. Check CORS configuration
3. Verify `.env.local` API URLs are correct

### Voice Not Working

1. Grant microphone permissions in browser
2. Check browser compatibility (Chrome/Edge recommended)
3. Verify LiveKit room connection status
4. Check agent status indicator (should be green)

## 📊 Performance

- **Connection Time**: < 3 seconds
- **Voice Latency**: < 200ms
- **Prediction API**: < 500ms
- **State Updates**: Real-time

## 🔐 Security

- LiveKit tokens are server-side generated
- API secrets stored in environment variables only
- No sensitive data in client-side code
- CORS configured for specific origins

## 🚢 Deployment

### Vercel (Recommended)

```bash
npm run build
vercel deploy
```

### Docker

```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build
EXPOSE 3000
CMD ["npm", "start"]
```

## 📝 License

Part of the Cognirace project - Real-time race strategy platform for Toyota GR Cup Series.

## 🤝 Contributing

This is part of a larger Cognirace ecosystem. See main project README for contribution guidelines.

## 📞 Support

For issues related to:
- Voice agent: Check LiveKit documentation
- ML predictions: See backend-api documentation
- Agent behavior: See agents documentation

---

Built with ❤️ for the Toyota GR Cup Series
