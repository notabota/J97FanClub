from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import socketio
import uvicorn
import asyncio
import json
import numpy as np
from typing import Dict, Any
import logging

from services.audio_processor import AudioProcessor
from services.scam_detector import ScamDetector
from services.voice_authenticator import VoiceAuthenticator
from services.tts_service import TTSService, FallbackTTS
from services.incident_reporter import IncidentReporter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="Voice Scam Shield API", version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create Socket.IO server
sio = socketio.AsyncServer(
    async_mode='asgi',
    cors_allowed_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    logger=False,
    engineio_logger=False
)

# Initialize services
audio_processor = AudioProcessor()
scam_detector = ScamDetector()
voice_authenticator = VoiceAuthenticator()
tts_service = TTSService() if TTSService().is_ready() else FallbackTTS()
incident_reporter = IncidentReporter()

# Store active sessions
active_sessions: Dict[str, Dict[str, Any]] = {}

@app.get("/")
async def root():
    return {"message": "Voice Scam Shield API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "services": {
        "audio_processor": audio_processor.is_ready(),
        "scam_detector": scam_detector.is_ready(),
        "voice_authenticator": voice_authenticator.is_ready(),
        "tts_service": tts_service.is_ready()
    }}

@sio.event
async def connect(sid, environ):
    logger.info(f"Client {sid} connected")
    active_sessions[sid] = {
        "audio_buffer": [],
        "transcript": "",
        "risk_level": "safe",
        "language": "en"
    }
    # Start incident tracking
    incident_reporter.start_session(sid, {"client_info": environ.get('HTTP_USER_AGENT', 'Unknown')})
    await sio.emit('status', {'message': 'Connected to Voice Scam Shield'}, room=sid)

@sio.event
async def disconnect(sid):
    logger.info(f"Client {sid} disconnected")
    if sid in active_sessions:
        # Generate incident report before cleanup
        report_path = incident_reporter.end_session(sid)
        if report_path:
            logger.info(f"Incident report saved: {report_path}")
        del active_sessions[sid]

@sio.event
async def audio_chunk(sid, data):
    """Process incoming audio chunks from clients"""
    try:
        if sid not in active_sessions:
            await sio.emit('error', {'message': 'Session not found'}, room=sid)
            return

        # Convert audio data to numpy array
        try:
            audio_buffer = data['data']
            logger.info(f"Received audio chunk - type: {type(audio_buffer)}, size: {len(audio_buffer) if hasattr(audio_buffer, '__len__') else 'unknown'}")
            
            if isinstance(audio_buffer, (list, tuple)):
                # Handle audio data from Web Audio API
                audio_data = np.array(audio_buffer, dtype=np.float32)
                
                # Check if data needs normalization (if values are in 0-255 range)
                if audio_data.max() > 1.0:
                    logger.info(f"Normalizing audio data from 0-255 range to -1 to 1")
                    audio_data = (audio_data / 255.0) * 2.0 - 1.0  # Convert 0-255 to -1 to 1
                
                logger.info(f"Final audio data - shape: {audio_data.shape}, min: {audio_data.min():.3f}, max: {audio_data.max():.3f}")
            elif isinstance(audio_buffer, bytes):
                # Handle bytes (fallback)
                audio_data = np.frombuffer(audio_buffer, dtype=np.float32)
            else:
                logger.error(f"Unexpected audio buffer type: {type(audio_buffer)}")
                return
            
        except Exception as e:
            logger.error(f"Error converting audio buffer: {str(e)}")
            return

        # Add to session buffer
        active_sessions[sid]['audio_buffer'].extend(audio_data)
        current_buffer_size = len(active_sessions[sid]['audio_buffer'])
        logger.info(f"Current buffer size: {current_buffer_size}, target: 48000")

        # Process audio if buffer is large enough (2 seconds of audio at 16kHz)
        if len(active_sessions[sid]['audio_buffer']) >= 32000:  # 2 seconds at 16kHz
            logger.info(f"Processing audio segment for session {sid}")
            await process_audio_segment(sid)

    except Exception as e:
        logger.error(f"Error processing audio chunk for {sid}: {str(e)}")
        await sio.emit('error', {'message': 'Error processing audio'}, room=sid)

async def process_audio_segment(sid: str):
    """Process accumulated audio segment for transcription and analysis"""
    try:
        session = active_sessions[sid]
        audio_buffer = np.array(session['audio_buffer'])

        # Clear buffer for next segment
        session['audio_buffer'] = []

        # Transcribe audio
        logger.info(f"Starting transcription for audio buffer shape: {audio_buffer.shape}")
        transcript_result = await audio_processor.transcribe(audio_buffer)
        logger.info(f"Transcription result: {transcript_result}")
        if not transcript_result:
            logger.warning("No transcription result received")
            return

        transcript_text = transcript_result['text']
        detected_language = transcript_result.get('language', 'en')

        # Update session
        session['transcript'] += transcript_text + " "
        session['language'] = detected_language

        # Send transcription to client
        await sio.emit('transcription', {
            'text': transcript_text,
            'language': detected_language
        }, room=sid)

        # Analyze for scam patterns
        scam_analysis = await scam_detector.analyze_text(
            transcript_text,
            detected_language
        )

        # Check for voice spoofing
        voice_analysis = await voice_authenticator.analyze_audio(audio_buffer)

        # Combine analyses for risk assessment
        risk_assessment = combine_risk_assessments(scam_analysis, voice_analysis)

        # Update session risk level
        session['risk_level'] = risk_assessment['risk']

        # Add segment to incident report
        incident_reporter.add_segment(sid, {
            'transcript': transcript_text,
            'risk_level': risk_assessment['risk'],
            'confidence': risk_assessment.get('scam_confidence', 0),
            'indicators': scam_analysis.get('indicators', []),
            'language': detected_language,
            'voice_analysis': voice_analysis
        })

        # Send risk assessment to client
        await sio.emit('risk_assessment', risk_assessment, room=sid)

        # Generate alert if high risk
        if risk_assessment['risk'] in ['suspicious', 'scam']:
            await generate_alert(sid, risk_assessment)

    except Exception as e:
        logger.error(f"Error processing audio segment for {sid}: {str(e)}")
        await sio.emit('error', {'message': 'Error analyzing audio'}, room=sid)

def combine_risk_assessments(scam_analysis: Dict, voice_analysis: Dict) -> Dict:
    """Combine scam detection and voice authentication results"""
    risk_scores = {
        'safe': 0,
        'suspicious': 1,
        'scam': 2
    }

    scam_risk = risk_scores.get(scam_analysis.get('risk', 'safe'), 0)
    voice_risk = risk_scores.get(voice_analysis.get('risk', 'safe'), 0)

    # Take the maximum risk level
    final_risk_score = max(scam_risk, voice_risk)
    risk_levels = ['safe', 'suspicious', 'scam']
    final_risk = risk_levels[final_risk_score]

    reasons = []
    if scam_analysis.get('reason'):
        reasons.append(scam_analysis['reason'])
    if voice_analysis.get('reason'):
        reasons.append(voice_analysis['reason'])

    return {
        'risk': final_risk,
        'reason': '; '.join(reasons) if reasons else 'No issues detected',
        'scam_confidence': scam_analysis.get('confidence', 0),
        'voice_confidence': voice_analysis.get('confidence', 0)
    }

async def generate_alert(sid: str, risk_assessment: Dict):
    """Generate and send alert for suspicious/scam calls"""
    session = active_sessions[sid]
    language = session.get('language', 'en')

    alert_messages = {
        'en': {
            'suspicious': "Caution: This call may contain suspicious elements. Avoid sharing personal information.",
            'scam': "WARNING: This appears to be a fraudulent call. Do not share personal information, codes, or make payments. Hang up immediately."
        },
        'es': {
            'suspicious': "Precaución: Esta llamada puede contener elementos sospechosos. Evite compartir información personal.",
            'scam': "ADVERTENCIA: Esta parece ser una llamada fraudulenta. No comparta información personal, códigos o haga pagos. Cuelgue inmediatamente."
        },
        'fr': {
            'suspicious': "Attention: Cet appel peut contenir des éléments suspects. Évitez de partager des informations personnelles.",
            'scam': "AVERTISSEMENT: Cet appel semble frauduleux. Ne partagez pas d'informations personnelles, de codes ou n'effectuez pas de paiements. Raccrochez immédiatement."
        }
    }

    risk_level = risk_assessment['risk']
    message = alert_messages.get(language, alert_messages['en']).get(risk_level, '')

    # Generate TTS audio alert
    audio_alert = None
    try:
        audio_alert = await tts_service.generate_discrete_alert(risk_level, language)
    except Exception as e:
        logger.error(f"Error generating TTS alert: {str(e)}")

    alert_data = {
        'level': risk_level,
        'message': message,
        'language': language
    }

    # Add audio data if available
    if audio_alert:
        import base64
        alert_data['audio'] = base64.b64encode(audio_alert).decode('utf-8')
        alert_data['audio_format'] = 'mp3'

    await sio.emit('alert', alert_data, room=sid)

@app.get("/reports/summary")
async def get_summary_report(period: str = "24h"):
    """Get summary of recent incidents"""
    try:
        summary = await incident_reporter.generate_summary_report(period)
        return summary
    except Exception as e:
        logger.error(f"Error generating summary report: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate summary report")

@app.get("/reports/{report_filename}")
async def get_incident_report(report_filename: str):
    """Get specific incident report"""
    try:
        reports_dir = incident_reporter.reports_dir
        report_path = reports_dir / report_filename

        if not report_path.exists():
            raise HTTPException(status_code=404, detail="Report not found")

        report = incident_reporter.get_report(str(report_path))
        if not report:
            raise HTTPException(status_code=500, detail="Failed to load report")

        return report
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving report {report_filename}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve report")

# Create Socket.IO ASGI app with FastAPI as the fallback
sio_app = socketio.ASGIApp(sio, other_asgi_app=app)

if __name__ == "__main__":
    uvicorn.run(sio_app, host="0.0.0.0", port=8000, log_level="info")
