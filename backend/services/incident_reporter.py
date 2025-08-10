import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import asyncio
import os
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class CallSegment:
    start_time: float
    end_time: float
    transcript: str
    risk_level: str
    confidence: float
    indicators: List[str]
    language: str

@dataclass
class IncidentReport:
    session_id: str
    start_time: datetime
    end_time: Optional[datetime]
    duration: Optional[float]
    language: str
    overall_risk: str
    max_risk_level: str
    segments: List[CallSegment]
    summary: str
    recommendations: List[str]
    scam_indicators: List[str]
    voice_analysis: Dict[str, Any]
    metadata: Dict[str, Any]

class IncidentReporter:
    def __init__(self, reports_dir: str = "reports"):
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(exist_ok=True)
        self.active_sessions: Dict[str, Dict] = {}
        
    def start_session(self, session_id: str, metadata: Dict[str, Any] = None):
        """Start tracking a new call session"""
        self.active_sessions[session_id] = {
            'start_time': datetime.now(timezone.utc),
            'end_time': None,
            'segments': [],
            'metadata': metadata or {},
            'max_risk': 'safe',
            'languages': set(),
            'scam_indicators': [],
            'voice_analyses': []
        }
        logger.info(f"Started incident tracking for session {session_id}")
    
    def add_segment(self, session_id: str, segment_data: Dict[str, Any]):
        """Add a call segment to the active session"""
        if session_id not in self.active_sessions:
            logger.warning(f"Session {session_id} not found")
            return
            
        session = self.active_sessions[session_id]
        
        # Create segment
        segment = CallSegment(
            start_time=segment_data.get('start_time', 0),
            end_time=segment_data.get('end_time', 0),
            transcript=segment_data.get('transcript', ''),
            risk_level=segment_data.get('risk_level', 'safe'),
            confidence=segment_data.get('confidence', 0.0),
            indicators=segment_data.get('indicators', []),
            language=segment_data.get('language', 'en')
        )
        
        session['segments'].append(segment)
        session['languages'].add(segment.language)
        
        # Update max risk level
        risk_priority = {'safe': 0, 'suspicious': 1, 'scam': 2}
        current_max = risk_priority.get(session['max_risk'], 0)
        new_risk = risk_priority.get(segment.risk_level, 0)
        if new_risk > current_max:
            session['max_risk'] = segment.risk_level
        
        # Collect indicators
        session['scam_indicators'].extend(segment.indicators)
        
        # Add voice analysis if present
        if 'voice_analysis' in segment_data:
            session['voice_analyses'].append(segment_data['voice_analysis'])
    
    def end_session(self, session_id: str) -> Optional[str]:
        """End a session and generate incident report"""
        if session_id not in self.active_sessions:
            logger.warning(f"Session {session_id} not found")
            return None
            
        session = self.active_sessions[session_id]
        session['end_time'] = datetime.now(timezone.utc)
        
        # Generate report
        report = self._generate_report(session_id, session)
        
        # Save report
        report_path = self._save_report(report)
        
        # Clean up session
        del self.active_sessions[session_id]
        
        logger.info(f"Generated incident report for session {session_id}: {report_path}")
        return report_path
    
    def _generate_report(self, session_id: str, session_data: Dict) -> IncidentReport:
        """Generate comprehensive incident report"""
        start_time = session_data['start_time']
        end_time = session_data['end_time'] or datetime.now(timezone.utc)
        duration = (end_time - start_time).total_seconds()
        
        # Determine primary language
        languages = session_data['languages']
        primary_language = list(languages)[0] if languages else 'en'
        
        # Generate summary
        summary = self._generate_summary(session_data)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(session_data)
        
        # Consolidate voice analysis
        voice_analysis = self._consolidate_voice_analysis(session_data['voice_analyses'])
        
        # Create report
        report = IncidentReport(
            session_id=session_id,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            language=primary_language,
            overall_risk=session_data['max_risk'],
            max_risk_level=session_data['max_risk'],
            segments=session_data['segments'],
            summary=summary,
            recommendations=recommendations,
            scam_indicators=list(set(session_data['scam_indicators'])),
            voice_analysis=voice_analysis,
            metadata=session_data['metadata']
        )
        
        return report
    
    def _generate_summary(self, session_data: Dict) -> str:
        """Generate human-readable summary of the call"""
        max_risk = session_data['max_risk']
        segments_count = len(session_data['segments'])
        duration = (session_data['end_time'] - session_data['start_time']).total_seconds()
        indicators_count = len(set(session_data['scam_indicators']))
        
        if max_risk == 'scam':
            summary = f"HIGH RISK CALL DETECTED: This {duration:.1f}-second call showed strong indicators of fraudulent activity across {segments_count} analyzed segments. {indicators_count} unique scam indicators were identified."
        elif max_risk == 'suspicious':
            summary = f"SUSPICIOUS CALL: This {duration:.1f}-second call contained potentially suspicious elements across {segments_count} analyzed segments. {indicators_count} warning indicators were detected."
        else:
            summary = f"SAFE CALL: This {duration:.1f}-second call appeared legitimate based on analysis of {segments_count} segments. No significant risk indicators were detected."
        
        return summary
    
    def _generate_recommendations(self, session_data: Dict) -> List[str]:
        """Generate actionable recommendations based on call analysis"""
        max_risk = session_data['max_risk']
        indicators = set(session_data['scam_indicators'])
        
        recommendations = []
        
        if max_risk == 'scam':
            recommendations.extend([
                "IMMEDIATE ACTION: Do not provide any personal information, passwords, or verification codes",
                "Do not make any payments or transfer money as requested in this call",
                "Report this incident to your bank/financial institution if financial information was discussed",
                "Consider reporting this scam to local authorities or the FTC",
                "Block the caller's number to prevent future contact"
            ])
        elif max_risk == 'suspicious':
            recommendations.extend([
                "Exercise caution: Verify the caller's identity through official channels",
                "Do not provide personal information until identity is confirmed",
                "Call back using official phone numbers from legitimate sources",
                "Monitor your accounts for any unauthorized activity"
            ])
        else:
            recommendations.extend([
                "No immediate action required - call appeared legitimate",
                "Continue to monitor for any unusual follow-up communications",
                "Stay vigilant for future calls from unknown numbers"
            ])
        
        # Add specific recommendations based on indicators
        if 'urgency tactics' in indicators:
            recommendations.append("Be wary of any pressure to act immediately - legitimate organizations typically allow time for verification")
        
        if 'government impersonation' in indicators:
            recommendations.append("Government agencies typically do not make unsolicited calls requesting personal information")
        
        if 'gift card payment' in indicators:
            recommendations.append("Legitimate organizations never request payment via gift cards")
        
        return recommendations
    
    def _consolidate_voice_analysis(self, voice_analyses: List[Dict]) -> Dict[str, Any]:
        """Consolidate voice analysis results from all segments"""
        if not voice_analyses:
            return {}
        
        synthetic_detections = [va for va in voice_analyses if va.get('fake_probability', 0) > 0.6]
        avg_real_prob = sum(va.get('real_probability', 0) for va in voice_analyses) / len(voice_analyses)
        avg_fake_prob = sum(va.get('fake_probability', 0) for va in voice_analyses) / len(voice_analyses)
        
        return {
            'total_segments_analyzed': len(voice_analyses),
            'synthetic_voice_detections': len(synthetic_detections),
            'average_real_probability': avg_real_prob,
            'average_fake_probability': avg_fake_prob,
            'likely_synthetic': avg_fake_prob > 0.6,
            'confidence': max(avg_real_prob, avg_fake_prob)
        }
    
    def _save_report(self, report: IncidentReport) -> str:
        """Save incident report to file"""
        try:
            # Create filename with timestamp
            timestamp = report.start_time.strftime('%Y%m%d_%H%M%S')
            filename = f"incident_report_{report.session_id}_{timestamp}.json"
            file_path = self.reports_dir / filename
            
            # Convert to JSON-serializable format
            report_dict = asdict(report)
            
            # Handle datetime serialization
            report_dict['start_time'] = report.start_time.isoformat()
            if report.end_time:
                report_dict['end_time'] = report.end_time.isoformat()
            
            # Save to file
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(report_dict, f, indent=2, ensure_ascii=False)
            
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Error saving incident report: {str(e)}")
            return ""
    
    async def generate_summary_report(self, time_period: str = "24h") -> Dict[str, Any]:
        """Generate summary report of recent incidents"""
        try:
            # This would query recent reports in production
            # For now, return a sample summary
            
            return {
                'period': time_period,
                'total_calls_analyzed': 0,
                'scam_calls_detected': 0,
                'suspicious_calls': 0,
                'safe_calls': 0,
                'most_common_scam_types': [],
                'languages_processed': [],
                'average_call_duration': 0,
                'synthetic_voice_detections': 0
            }
            
        except Exception as e:
            logger.error(f"Error generating summary report: {str(e)}")
            return {}
    
    def get_report(self, report_path: str) -> Optional[Dict[str, Any]]:
        """Load and return a specific incident report"""
        try:
            if not os.path.exists(report_path):
                return None
                
            with open(report_path, 'r', encoding='utf-8') as f:
                return json.load(f)
                
        except Exception as e:
            logger.error(f"Error loading report {report_path}: {str(e)}")
            return None