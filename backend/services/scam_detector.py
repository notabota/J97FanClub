import openai
import re
import logging
from typing import Dict, List, Any, Optional
import asyncio
import json
from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

@dataclass
class ScamIndicator:
    pattern: str
    weight: float
    description: str
    languages: List[str]

class ScamDetector:
    def __init__(self):
        self.client = None
        self._initialize_openai()
        self.scam_patterns = self._load_scam_patterns()
        self.language_models = {
            'en': 'english',
            'es': 'spanish',
            'fr': 'french'
        }
    
    def _initialize_openai(self):
        """Initialize OpenAI client"""
        try:
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key:
                # Initialize OpenAI client with just the API key
                self.client = openai.OpenAI(api_key=api_key)
                logger.info("OpenAI client initialized successfully")
            else:
                logger.warning("OpenAI API key not found. Using pattern-based detection only.")
                self.client = None
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {str(e)}")
            self.client = None
    
    def is_ready(self) -> bool:
        """Check if the scam detector is ready"""
        return len(self.scam_patterns) > 0
    
    def _load_scam_patterns(self) -> List[ScamIndicator]:
        """Load multilingual scam detection patterns"""
        patterns = [
            # English patterns
            ScamIndicator(
                pattern=r'\b(?:urgent|immediate|act now|limited time|expires today)\b',
                weight=0.7,
                description='Urgency tactics',
                languages=['en']
            ),
            ScamIndicator(
                pattern=r'\b(?:verify your account|suspended|locked|frozen)\b',
                weight=0.8,
                description='Account verification scam',
                languages=['en']
            ),
            ScamIndicator(
                pattern=r'\b(?:social security|ssn|tax refund|irs|government|arrest)\b',
                weight=0.9,
                description='Government impersonation',
                languages=['en']
            ),
            ScamIndicator(
                pattern=r'\b(?:gift card|itunes|amazon card|walmart|target)\b',
                weight=0.9,
                description='Gift card payment scam',
                languages=['en']
            ),
            ScamIndicator(
                pattern=r'\b(?:grandma|grandson|granddaughter|emergency|hospital|jail|bail)\b',
                weight=0.8,
                description='Grandparent scam',
                languages=['en']
            ),
            
            # Spanish patterns
            ScamIndicator(
                pattern=r'\b(?:urgente|inmediato|actuar ahora|tiempo limitado|expira hoy)\b',
                weight=0.7,
                description='Táticas de urgencia',
                languages=['es']
            ),
            ScamIndicator(
                pattern=r'\b(?:verifique su cuenta|suspendido|bloqueado|congelado)\b',
                weight=0.8,
                description='Estafa de verificación de cuenta',
                languages=['es']
            ),
            ScamIndicator(
                pattern=r'\b(?:seguro social|reembolso de impuestos|gobierno|arresto)\b',
                weight=0.9,
                description='Suplantación del gobierno',
                languages=['es']
            ),
            
            # French patterns
            ScamIndicator(
                pattern=r'\b(?:urgent|immédiat|agir maintenant|temps limité|expire aujourd\'hui)\b',
                weight=0.7,
                description='Tactiques d\'urgence',
                languages=['fr']
            ),
            ScamIndicator(
                pattern=r'\b(?:vérifiez votre compte|suspendu|bloqué|gelé)\b',
                weight=0.8,
                description='Arnaque de vérification de compte',
                languages=['fr']
            ),
            ScamIndicator(
                pattern=r'\b(?:sécurité sociale|remboursement d\'impôt|gouvernement|arrestation)\b',
                weight=0.9,
                description='Usurpation du gouvernement',
                languages=['fr']
            ),
        ]
        
        return patterns
    
    async def analyze_text(self, text: str, language: str = 'en') -> Dict[str, Any]:
        """Analyze text for scam indicators"""
        try:
            # Pattern-based analysis
            pattern_analysis = self._analyze_patterns(text, language)
            
            # LLM-based analysis if available
            llm_analysis = None
            if self.client:
                llm_analysis = await self._analyze_with_llm(text, language)
            
            # Combine analyses
            final_analysis = self._combine_analyses(pattern_analysis, llm_analysis)
            
            return final_analysis
            
        except Exception as e:
            logger.error(f"Error in scam analysis: {str(e)}")
            return {
                'risk': 'safe',
                'confidence': 0.0,
                'reason': 'Analysis failed',
                'indicators': []
            }
    
    def _analyze_patterns(self, text: str, language: str) -> Dict[str, Any]:
        """Analyze text using predefined patterns"""
        text_lower = text.lower()
        detected_indicators = []
        total_weight = 0.0
        
        for pattern in self.scam_patterns:
            if language in pattern.languages:
                matches = re.findall(pattern.pattern, text_lower, re.IGNORECASE)
                if matches:
                    detected_indicators.append({
                        'type': pattern.description,
                        'matches': matches,
                        'weight': pattern.weight
                    })
                    total_weight += pattern.weight
        
        # Determine risk level based on total weight
        if total_weight >= 1.5:
            risk = 'scam'
        elif total_weight >= 0.5:
            risk = 'suspicious'
        else:
            risk = 'safe'
        
        confidence = min(1.0, total_weight / 2.0)
        
        return {
            'risk': risk,
            'confidence': confidence,
            'indicators': detected_indicators,
            'total_weight': total_weight,
            'method': 'pattern_analysis'
        }
    
    async def _analyze_with_llm(self, text: str, language: str) -> Optional[Dict[str, Any]]:
        """Analyze text using OpenAI LLM"""
        try:
            language_name = self.language_models.get(language, 'english')
            
            system_prompt = f"""You are a multilingual scam detection expert. Analyze the following {language_name} text for potential scam indicators.

Look for these common scam patterns:
- Urgency tactics (immediate action required)
- Request for personal information (SSN, passwords, account details)
- Government/authority impersonation
- Financial requests (gift cards, wire transfers, cryptocurrency)
- Emotional manipulation (emergency situations, threats)
- Too-good-to-be-true offers
- Verification/security alerts
- Technical support scams

Respond with a JSON object containing:
- risk: "safe", "suspicious", or "scam"
- confidence: float between 0.0 and 1.0
- reason: brief explanation in English
- specific_indicators: list of detected scam indicators"""

            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ],
                temperature=0.1,
                max_tokens=300
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Try to parse JSON response
            try:
                result = json.loads(result_text)
                result['method'] = 'llm_analysis'
                return result
            except json.JSONDecodeError:
                # Fallback parsing if JSON is malformed
                logger.warning("LLM response not valid JSON, attempting fallback parsing")
                return self._parse_llm_fallback(result_text)
                
        except Exception as e:
            logger.error(f"Error in LLM analysis: {str(e)}")
            return None
    
    def _parse_llm_fallback(self, text: str) -> Dict[str, Any]:
        """Fallback parsing for non-JSON LLM responses"""
        risk = 'safe'
        confidence = 0.0
        reason = 'LLM analysis completed'
        
        text_lower = text.lower()
        
        if 'scam' in text_lower or 'fraud' in text_lower:
            risk = 'scam'
            confidence = 0.8
        elif 'suspicious' in text_lower or 'caution' in text_lower:
            risk = 'suspicious' 
            confidence = 0.6
        
        return {
            'risk': risk,
            'confidence': confidence,
            'reason': reason,
            'method': 'llm_fallback'
        }
    
    def _combine_analyses(self, pattern_analysis: Dict, llm_analysis: Optional[Dict]) -> Dict[str, Any]:
        """Combine pattern-based and LLM analyses"""
        if not llm_analysis:
            return pattern_analysis
        
        # Weight the analyses
        pattern_weight = 0.4
        llm_weight = 0.6
        
        risk_scores = {'safe': 0, 'suspicious': 1, 'scam': 2}
        
        pattern_score = risk_scores[pattern_analysis['risk']]
        llm_score = risk_scores[llm_analysis['risk']]
        
        # Weighted average
        combined_score = (pattern_score * pattern_weight + llm_score * llm_weight)
        
        # Convert back to risk level
        if combined_score >= 1.5:
            final_risk = 'scam'
        elif combined_score >= 0.7:
            final_risk = 'suspicious'
        else:
            final_risk = 'safe'
        
        # Combine confidences
        combined_confidence = (
            pattern_analysis['confidence'] * pattern_weight + 
            llm_analysis['confidence'] * llm_weight
        )
        
        # Combine reasons
        reasons = []
        if pattern_analysis.get('indicators'):
            reasons.append(f"Pattern detection: {len(pattern_analysis['indicators'])} indicators")
        if llm_analysis.get('reason'):
            reasons.append(f"LLM analysis: {llm_analysis['reason']}")
        
        return {
            'risk': final_risk,
            'confidence': min(1.0, combined_confidence),
            'reason': '; '.join(reasons) if reasons else 'No significant indicators detected',
            'pattern_analysis': pattern_analysis,
            'llm_analysis': llm_analysis,
            'method': 'combined_analysis'
        }