import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Dict, Any, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor
import librosa
import scipy.signal

logger = logging.getLogger(__name__)

class SimpleAntiSpoofModel(nn.Module):
    """Simplified anti-spoofing model for demonstration"""
    def __init__(self, input_dim=80):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(64, 2)  # Real vs Fake
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.global_pool(x).squeeze(-1)
        x = self.dropout(x)
        return torch.softmax(self.classifier(x), dim=1)

class VoiceAuthenticator:
    def __init__(self):
        # Feature extraction parameters - define these first
        self.sample_rate = 16000
        self.n_mels = 80
        self.n_fft = 512
        self.hop_length = 160
        
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.executor = ThreadPoolExecutor(max_workers=1)
        self._initialize_model()
        
    def _initialize_model(self):
        """Initialize anti-spoofing model"""
        try:
            # In production, load pre-trained weights
            self.model = SimpleAntiSpoofModel(input_dim=self.n_mels)
            self.model.to(self.device)
            self.model.eval()
            
            # Initialize with random weights for demo
            # In production, load trained weights:
            # self.model.load_state_dict(torch.load('anti_spoof_model.pth'))
            
            logger.info("Voice authentication model initialized")
        except Exception as e:
            logger.error(f"Failed to initialize voice authentication model: {str(e)}")
            self.model = None
    
    def is_ready(self) -> bool:
        """Check if voice authenticator is ready"""
        return self.model is not None
    
    async def analyze_audio(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """Analyze audio for voice spoofing detection"""
        try:
            if not self.is_ready():
                return {
                    'risk': 'safe',
                    'confidence': 0.0,
                    'reason': 'Voice authenticator not available'
                }
            
            # Run analysis in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self._analyze_audio_sync,
                audio_data
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in voice authentication: {str(e)}")
            return {
                'risk': 'safe',
                'confidence': 0.0,
                'reason': 'Analysis failed'
            }
    
    def _analyze_audio_sync(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """Synchronous audio analysis"""
        try:
            # Extract features
            features = self._extract_features(audio_data)
            
            if features is None:
                return {
                    'risk': 'safe',
                    'confidence': 0.0,
                    'reason': 'Feature extraction failed'
                }
            
            # Run model inference
            with torch.no_grad():
                features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
                predictions = self.model(features_tensor)
                
                # Get probabilities [real_prob, fake_prob]
                real_prob = predictions[0][0].item()
                fake_prob = predictions[0][1].item()
            
            # Determine risk based on fake probability
            if fake_prob > 0.8:
                risk = 'scam'
                reason = f'High probability of synthetic voice ({fake_prob:.2f})'
            elif fake_prob > 0.6:
                risk = 'suspicious'
                reason = f'Possible synthetic voice detected ({fake_prob:.2f})'
            else:
                risk = 'safe'
                reason = f'Natural voice detected ({real_prob:.2f})'
            
            # Additional analysis
            spectral_analysis = self._analyze_spectral_features(audio_data)
            
            return {
                'risk': risk,
                'confidence': max(real_prob, fake_prob),
                'reason': reason,
                'real_probability': real_prob,
                'fake_probability': fake_prob,
                'spectral_analysis': spectral_analysis
            }
            
        except Exception as e:
            logger.error(f"Synchronous voice analysis error: {str(e)}")
            return {
                'risk': 'safe',
                'confidence': 0.0,
                'reason': 'Analysis failed'
            }
    
    def _extract_features(self, audio_data: np.ndarray) -> Optional[np.ndarray]:
        """Extract mel-spectrogram features from audio"""
        try:
            # Ensure audio is 1D
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # Normalize
            if np.max(np.abs(audio_data)) > 0:
                audio_data = audio_data / np.max(np.abs(audio_data))
            
            # Extract mel-spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio_data,
                sr=self.sample_rate,
                n_mels=self.n_mels,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                power=2.0
            )
            
            # Convert to log scale
            log_mel = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Normalize features
            log_mel = (log_mel - np.mean(log_mel)) / (np.std(log_mel) + 1e-8)
            
            return log_mel
            
        except Exception as e:
            logger.error(f"Feature extraction error: {str(e)}")
            return None
    
    def _analyze_spectral_features(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """Analyze spectral characteristics that might indicate synthesis"""
        try:
            # Calculate spectral features that differ between real and synthetic speech
            
            # Spectral centroid
            spectral_centroids = librosa.feature.spectral_centroid(
                y=audio_data, sr=self.sample_rate
            )[0]
            
            # Spectral rolloff
            spectral_rolloff = librosa.feature.spectral_rolloff(
                y=audio_data, sr=self.sample_rate
            )[0]
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
            
            # MFCC features
            mfccs = librosa.feature.mfcc(
                y=audio_data, sr=self.sample_rate, n_mfcc=13
            )
            
            # Calculate statistics
            features = {
                'spectral_centroid_mean': np.mean(spectral_centroids),
                'spectral_centroid_std': np.std(spectral_centroids),
                'spectral_rolloff_mean': np.mean(spectral_rolloff),
                'spectral_rolloff_std': np.std(spectral_rolloff),
                'zcr_mean': np.mean(zcr),
                'zcr_std': np.std(zcr),
                'mfcc_means': np.mean(mfccs, axis=1).tolist(),
                'mfcc_stds': np.std(mfccs, axis=1).tolist()
            }
            
            # Simple heuristics for synthetic voice detection
            # These would be replaced with learned thresholds in production
            anomaly_score = 0.0
            
            # Check for unnaturally consistent spectral features
            if np.std(spectral_centroids) < 200:  # Too consistent
                anomaly_score += 0.3
            
            if np.mean(zcr) < 0.02:  # Unusually low zero crossing rate
                anomaly_score += 0.2
            
            # Check MFCC characteristics
            if np.std(mfccs[1]) < 10:  # Too consistent first MFCC coefficient
                anomaly_score += 0.2
            
            features['anomaly_score'] = anomaly_score
            features['is_anomalous'] = anomaly_score > 0.4
            
            return features
            
        except Exception as e:
            logger.error(f"Spectral analysis error: {str(e)}")
            return {'error': str(e)}
    
    def enroll_voice(self, audio_data: np.ndarray, speaker_id: str) -> Dict[str, Any]:
        """Enroll a speaker's voice for future verification (placeholder)"""
        try:
            # Extract speaker embedding/features for enrollment
            features = self._extract_features(audio_data)
            
            if features is None:
                return {'success': False, 'reason': 'Feature extraction failed'}
            
            # In production, save features to database with speaker_id
            # For now, just return success
            
            return {
                'success': True,
                'speaker_id': speaker_id,
                'features_shape': features.shape,
                'message': 'Voice enrolled successfully (demo mode)'
            }
            
        except Exception as e:
            logger.error(f"Voice enrollment error: {str(e)}")
            return {'success': False, 'reason': str(e)}
    
    def verify_speaker(self, audio_data: np.ndarray, claimed_speaker_id: str) -> Dict[str, Any]:
        """Verify if audio matches enrolled speaker (placeholder)"""
        try:
            # Extract features from current audio
            features = self._extract_features(audio_data)
            
            if features is None:
                return {
                    'verified': False,
                    'confidence': 0.0,
                    'reason': 'Feature extraction failed'
                }
            
            # In production, compare with enrolled features
            # For demo, return random verification
            verification_score = np.random.random()
            
            return {
                'verified': verification_score > 0.5,
                'confidence': verification_score,
                'speaker_id': claimed_speaker_id,
                'reason': 'Demo verification (random result)'
            }
            
        except Exception as e:
            logger.error(f"Speaker verification error: {str(e)}")
            return {
                'verified': False,
                'confidence': 0.0,
                'reason': str(e)
            }