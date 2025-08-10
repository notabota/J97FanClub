import requests
import asyncio
import logging
from typing import Optional, Dict, Any
import io
import base64
from pathlib import Path
import tempfile
import os
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class TTSService:
    def __init__(self):
        self.api_key = os.getenv('ELEVENLABS_API_KEY')
        self.base_url = "https://api.elevenlabs.io/v1"
        self.voice_ids = {
            'en': 'pNInz6obpgDQGcFmaJgB',  # Adam - English
            'es': 'VR6AewLTigWG4xSOukaG',  # Pablo - Spanish  
            'fr': 'cgSgspJ2msm6clMCkdW9'   # Remi - French
        }
        self.available = self.api_key is not None
        
    def is_ready(self) -> bool:
        """Check if TTS service is ready"""
        return self.available
    
    async def generate_alert_audio(self, text: str, language: str = 'en') -> Optional[bytes]:
        """Generate audio alert using ElevenLabs TTS"""
        if not self.available:
            logger.warning("ElevenLabs API key not available")
            return None
            
        try:
            voice_id = self.voice_ids.get(language, self.voice_ids['en'])
            
            url = f"{self.base_url}/text-to-speech/{voice_id}"
            
            headers = {
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
                "xi-api-key": self.api_key
            }
            
            data = {
                "text": text,
                "model_id": "eleven_multilingual_v2",
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.7,
                    "style": 0.2,
                    "use_speaker_boost": True
                }
            }
            
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: requests.post(url, json=data, headers=headers)
            )
            
            if response.status_code == 200:
                return response.content
            else:
                logger.error(f"ElevenLabs API error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error generating TTS audio: {str(e)}")
            return None
    
    async def create_alert_file(self, text: str, language: str = 'en') -> Optional[str]:
        """Create temporary audio file for alert"""
        try:
            audio_data = await self.generate_alert_audio(text, language)
            if not audio_data:
                return None
                
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
                temp_file.write(audio_data)
                temp_file_path = temp_file.name
                
            return temp_file_path
            
        except Exception as e:
            logger.error(f"Error creating alert file: {str(e)}")
            return None
    
    def get_alert_messages(self) -> Dict[str, Dict[str, str]]:
        """Get predefined alert messages in different languages"""
        return {
            'en': {
                'suspicious': "Alert: This call shows suspicious patterns. Please be cautious with any personal information.",
                'scam': "Warning: This appears to be a fraudulent call. Do not share personal information or make any payments. Consider hanging up."
            },
            'es': {
                'suspicious': "Alerta: Esta llamada muestra patrones sospechosos. Por favor, tenga cuidado con cualquier información personal.",
                'scam': "Advertencia: Esta parece ser una llamada fraudulenta. No comparta información personal ni realice pagos. Considere colgar."
            },
            'fr': {
                'suspicious': "Alerte: Cet appel présente des motifs suspects. Veuillez faire attention à toute information personnelle.",
                'scam': "Avertissement: Cet appel semble frauduleux. Ne partagez pas d'informations personnelles et n'effectuez aucun paiement. Envisagez de raccrocher."
            }
        }
    
    async def generate_discrete_alert(self, risk_level: str, language: str = 'en') -> Optional[bytes]:
        """Generate a discrete audio alert based on risk level"""
        messages = self.get_alert_messages()
        
        message = messages.get(language, messages['en']).get(risk_level, '')
        if not message:
            return None
            
        return await self.generate_alert_audio(message, language)
    
    def cleanup_temp_files(self, file_paths: list):
        """Clean up temporary audio files"""
        for file_path in file_paths:
            try:
                if os.path.exists(file_path):
                    os.unlink(file_path)
            except Exception as e:
                logger.error(f"Error cleaning up temp file {file_path}: {str(e)}")

# Fallback TTS using system speech synthesis (for demo without API key)
class FallbackTTS:
    def __init__(self):
        self.available = True
        
    def is_ready(self) -> bool:
        return True
        
    async def generate_alert_audio(self, text: str, language: str = 'en') -> Optional[bytes]:
        """Generate simple beep or return None for fallback"""
        logger.info(f"Fallback TTS: {text} (Language: {language})")
        return None  # Would generate simple alert tone in production
        
    async def generate_discrete_alert(self, risk_level: str, language: str = 'en') -> Optional[bytes]:
        logger.info(f"Fallback alert: {risk_level} risk detected (Language: {language})")
        return None