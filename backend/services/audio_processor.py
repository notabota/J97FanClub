import whisper
import numpy as np
import torch
import logging
from typing import Optional, Dict, Any
import asyncio
from concurrent.futures import ThreadPoolExecutor
import io
import soundfile as sf

logger = logging.getLogger(__name__)

class AudioProcessor:
    def __init__(self, model_size: str = "base"):
        self.model_size = model_size
        self.model = None
        self.executor = ThreadPoolExecutor(max_workers=2)
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize Whisper model for transcription"""
        try:
            logger.info(f"Loading Whisper model: {self.model_size}")
            self.model = whisper.load_model(self.model_size)
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {str(e)}")
            self.model = None
    
    def is_ready(self) -> bool:
        """Check if the audio processor is ready"""
        return self.model is not None
    
    async def transcribe(self, audio_data: np.ndarray, sample_rate: int = 16000) -> Optional[Dict[str, Any]]:
        """Transcribe audio data to text with language detection"""
        if not self.is_ready():
            logger.warning("Audio processor not ready")
            return None
        
        try:
            # Ensure audio is in the right format for Whisper
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # Normalize audio
            if np.max(np.abs(audio_data)) > 0:
                audio_data = audio_data / np.max(np.abs(audio_data))
            
            # Whisper expects audio at 16kHz
            if sample_rate != 16000:
                # Simple resampling (for production, use proper resampling)
                audio_data = self._resample_audio(audio_data, sample_rate, 16000)
            
            # Run transcription in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self._transcribe_sync,
                audio_data
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in transcription: {str(e)}")
            return None
    
    def _transcribe_sync(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """Synchronous transcription method"""
        try:
            result = self.model.transcribe(
                audio_data,
                language=None,  # Auto-detect language
                task="transcribe",
                verbose=False
            )
            
            return {
                'text': result['text'].strip(),
                'language': result['language'],
                'segments': result.get('segments', []),
                'confidence': self._calculate_confidence(result)
            }
        except Exception as e:
            logger.error(f"Synchronous transcription error: {str(e)}")
            raise
    
    def _calculate_confidence(self, result: Dict) -> float:
        """Calculate average confidence from segments"""
        segments = result.get('segments', [])
        if not segments:
            return 0.0
        
        total_confidence = 0.0
        total_duration = 0.0
        
        for segment in segments:
            duration = segment.get('end', 0) - segment.get('start', 0)
            confidence = segment.get('avg_logprob', -1.0)
            # Convert log probability to confidence score
            confidence = max(0, min(1, (confidence + 1) / 1))
            
            total_confidence += confidence * duration
            total_duration += duration
        
        return total_confidence / total_duration if total_duration > 0 else 0.0
    
    def _resample_audio(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Simple resampling (for production use librosa.resample)"""
        if orig_sr == target_sr:
            return audio
        
        # Simple linear interpolation resampling
        ratio = target_sr / orig_sr
        new_length = int(len(audio) * ratio)
        
        indices = np.linspace(0, len(audio) - 1, new_length)
        resampled = np.interp(indices, np.arange(len(audio)), audio)
        
        return resampled.astype(np.float32)
    
    async def detect_voice_activity(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """Detect voice activity in audio segment"""
        try:
            # Simple energy-based VAD
            frame_length = 400  # 25ms at 16kHz
            hop_length = 160    # 10ms at 16kHz
            
            # Calculate short-time energy
            energy = []
            for i in range(0, len(audio_data) - frame_length, hop_length):
                frame = audio_data[i:i + frame_length]
                energy.append(np.sum(frame ** 2))
            
            energy = np.array(energy)
            
            # Threshold-based VAD
            threshold = np.mean(energy) * 0.1
            voice_frames = energy > threshold
            
            voice_ratio = np.sum(voice_frames) / len(voice_frames) if len(voice_frames) > 0 else 0
            
            return {
                'has_voice': voice_ratio > 0.3,
                'voice_ratio': voice_ratio,
                'energy_profile': energy.tolist()
            }
            
        except Exception as e:
            logger.error(f"Error in voice activity detection: {str(e)}")
            return {'has_voice': False, 'voice_ratio': 0.0}
    
    async def separate_speakers(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """Basic speaker diarization (placeholder for more sophisticated methods)"""
        try:
            # This is a simplified implementation
            # In production, use pyannote.audio or similar libraries
            
            # For now, assume single speaker
            return {
                'num_speakers': 1,
                'speaker_segments': [{
                    'speaker_id': 0,
                    'start': 0.0,
                    'end': len(audio_data) / 16000.0,
                    'confidence': 0.8
                }]
            }
            
        except Exception as e:
            logger.error(f"Error in speaker separation: {str(e)}")
            return {'num_speakers': 1, 'speaker_segments': []}