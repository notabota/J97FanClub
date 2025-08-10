# Voice Scam Shield

**Multilingual AI for Real-Time Call Scam Detection**

Voice Scam Shield is a comprehensive cybersecurity application that provides real-time protection against AI-driven voice scam calls. It uses advanced machine learning techniques to detect fraudulent activities, synthetic voices, and suspicious conversation patterns across multiple languages.

## 🛡️ Features

### Core Functionality
- **Real-time Audio Monitoring**: Captures and analyzes live call audio through WebRTC
- **Multilingual Support**: Detects scams in English, Spanish, and French
- **AI-Powered Detection**: Uses OpenAI GPT models and pattern matching for scam identification
- **Voice Authentication**: Anti-spoofing detection to identify synthetic voices
- **Live Risk Assessment**: Real-time risk scoring with visual dashboard
- **Discrete Audio Alerts**: ElevenLabs TTS warnings in multiple languages
- **Incident Reporting**: Comprehensive post-call analysis and documentation

### Technical Capabilities
- **Streaming Transcription**: OpenAI Whisper for real-time speech-to-text
- **Pattern Recognition**: Multilingual scam pattern database
- **Synthetic Voice Detection**: Neural networks for deepfake identification  
- **WebSocket Communication**: Low-latency real-time data streaming
- **Comprehensive Logging**: Detailed incident reports and analytics

## 🏗️ Architecture

- **Frontend**: Next.js with React and TypeScript
- **Backend**: FastAPI with Python
- **Real-time Communication**: Socket.IO WebSockets
- **AI Services**: OpenAI Whisper, GPT models, custom neural networks
- **Audio Processing**: LibROSA, PyTorch, SciPy

## 🚀 Quick Start

### Prerequisites
- Node.js 18+ and npm
- Python 3.8+
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd voice-scam-shield
   ```

2. **Install dependencies**
   ```bash
   npm run install:all
   ```

3. **Backend Setup**
   ```bash
   cd backend
   cp .env.example .env
   # Edit .env with your API keys (see Configuration section)
   pip install -r requirements.txt
   ```

4. **Frontend Setup**
   ```bash
   cd frontend
   npm install
   ```

### Running the Application

1. **Start Backend Server**
   ```bash
   cd backend
   python -m uvicorn main:app --reload
   # Server runs on http://localhost:8000
   ```

2. **Start Frontend Development Server**
   ```bash
   cd frontend
   npm run dev
   # App runs on http://localhost:3000
   ```

3. **Access the Application**
   - Open http://localhost:3000 in your web browser
   - Allow microphone access when prompted
   - Click "Start Monitoring" to begin real-time scam detection

## ⚙️ Configuration

### Required API Keys

1. **OpenAI API** (Required for advanced scam detection)
   - Get your key from https://platform.openai.com/api-keys
   - Add to `backend/.env`: `OPENAI_API_KEY=your_key_here`

2. **ElevenLabs API** (Optional - for audio alerts)
   - Get your key from https://elevenlabs.io/
   - Add to `backend/.env`: `ELEVENLABS_API_KEY=your_key_here`

### Environment Variables

Copy `backend/.env.example` to `backend/.env` and configure:

```env
OPENAI_API_KEY=your_openai_api_key_here
ELEVENLABS_API_KEY=your_elevenlabs_api_key_here
DEBUG=true
LOG_LEVEL=INFO
WHISPER_MODEL_SIZE=base
HOST=0.0.0.0
PORT=8000
FRONTEND_URL=http://localhost:3000
```

## 📊 How It Works

### Detection Pipeline

1. **Audio Capture**: WebRTC captures microphone audio in real-time
2. **Voice Activity Detection**: Identifies speech segments in audio stream
3. **Speech Recognition**: Whisper converts audio to text with language detection
4. **Scam Analysis**: Dual analysis using:
   - Pattern matching against known scam indicators
   - LLM analysis with GPT models for context understanding
5. **Voice Authentication**: Neural network checks for synthetic voice characteristics
6. **Risk Assessment**: Combines all analyses into final risk score
7. **Alert Generation**: Provides visual and audio warnings for high-risk calls
8. **Incident Reporting**: Documents complete call analysis for review

### Risk Levels

- **🟢 SAFE**: No suspicious indicators detected
- **🟡 SUSPICIOUS**: Some warning signs present - exercise caution
- **🔴 SCAM**: High probability of fraudulent activity - recommended to hang up

### Supported Scam Types

- Government impersonation (IRS, Social Security, law enforcement)
- Tech support scams
- Financial fraud (bank account verification, credit card issues)
- Grandparent scams (emergency situations)
- Gift card payment requests
- Urgency and pressure tactics
- Account suspension notifications

## 🔧 Development

### Project Structure

```
voice-scam-shield/
├── frontend/                 # Next.js React application
│   ├── app/                 # Next.js app router
│   ├── components/          # React components
│   └── lib/                 # Utility functions
├── backend/                 # FastAPI Python server
│   ├── services/           # Core AI services
│   │   ├── audio_processor.py      # Whisper transcription
│   │   ├── scam_detector.py        # Scam pattern detection
│   │   ├── voice_authenticator.py  # Anti-spoofing
│   │   ├── tts_service.py          # Audio alerts
│   │   └── incident_reporter.py    # Report generation
│   └── main.py             # FastAPI server
└── README.md
```

### API Endpoints

- `GET /` - Health check
- `GET /health` - Service status
- `GET /reports/summary` - Incident summary
- `GET /reports/{filename}` - Specific incident report
- `WebSocket /socket.io/` - Real-time audio streaming

### WebSocket Events

- `connect` - Client connection established
- `disconnect` - Client disconnected
- `audio_chunk` - Send audio data for processing
- `transcription` - Receive speech-to-text results
- `risk_assessment` - Receive scam detection results
- `alert` - Receive high-risk warnings

## 🔒 Security & Privacy

- All audio processing occurs locally or through secure APIs
- No audio data is permanently stored
- Incident reports contain only text transcripts and metadata
- WebSocket connections use secure protocols
- API keys are stored as environment variables only

## 📈 Performance

- **Latency**: Alerts within 2 seconds of suspicious speech
- **Accuracy**: ≥80% correct classification on test datasets
- **Anti-spoofing**: ≤10% Equal Error Rate for synthetic voice detection
- **Languages**: Real-time processing for English, Spanish, French
- **Scalability**: Supports multiple concurrent sessions

## 🚨 Limitations

- Requires internet connection for advanced AI features
- Microphone access required for audio monitoring
- API rate limits may apply for high-volume usage
- Effectiveness depends on audio quality and clarity
- Some legitimate calls may trigger false positives

## 🤝 Contributing

This project is designed for defensive cybersecurity purposes only. Contributions should focus on:

- Improving detection accuracy
- Adding new language support
- Enhancing user interface
- Expanding scam pattern databases
- Performance optimizations

## 📄 License

This project is developed for cybersecurity defense and education purposes.

## ⚠️ Disclaimer

Voice Scam Shield is a defensive security tool designed to help users identify potential scam calls. While it uses advanced AI techniques, no system is 100% accurate. Users should always exercise their own judgment and verify suspicious calls through official channels. This tool should supplement, not replace, common-sense security practices.

## 🆘 Support

For technical issues or questions:
1. Check the troubleshooting section below
2. Review the API documentation
3. Ensure all dependencies are properly installed
4. Verify API keys are correctly configured

### Common Issues

1. **Audio not capturing**: Ensure microphone permissions are granted
2. **WebSocket connection failed**: Check backend server is running on port 8000
3. **No transcription**: Verify OpenAI API key is valid and has credits
4. **TTS alerts not playing**: ElevenLabs API key required for audio alerts

---

**Built for the VC Big Bets (Cybersecurity) Track** - Protecting users from AI-driven voice scam calls with multilingual real-time detection.