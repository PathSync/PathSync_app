import re
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import threading


class SpeechRecognizer(ABC):
    
    def __init__(self, name: str):
        self.name = name
        self.is_listening = False
        self.recognition_history: List[Dict[str, Any]] = []
        self.confidence_threshold = 0.7
    
    @abstractmethod
    def recognize_speech(self, audio_input: str) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def start_listening(self) -> bool:
        pass
    
    @abstractmethod
    def stop_listening(self) -> bool:
        pass
    
    def get_recognition_stats(self) -> Dict[str, Any]:
        if not self.recognition_history:
            return {"message": "No recognition attempts recorded"}
        
        total_attempts = len(self.recognition_history)
        successful_attempts = len([r for r in self.recognition_history if r.get('success', False)])
        
        return {
            'total_attempts': total_attempts,
            'successful_attempts': successful_attempts,
            'success_rate': successful_attempts / total_attempts if total_attempts > 0 else 0,
            'average_confidence': sum(r.get('confidence', 0) for r in self.recognition_history) / total_attempts if total_attempts > 0 else 0
        }


class HealthcareSpeechRecognizer(SpeechRecognizer):
    
    def __init__(self):
        super().__init__("HealthcareSpeechRecognizer")
        self.medical_vocabulary = self._initialize_medical_vocabulary()
        self.symptom_patterns = self._initialize_symptom_patterns()
        self.vital_signs_patterns = self._initialize_vital_signs_patterns()
    
    def _initialize_medical_vocabulary(self) -> Dict[str, List[str]]:
        return {
            'symptoms': [
                'chest pain', 'shortness of breath', 'headache', 'dizziness', 'nausea',
                'vomiting', 'fever', 'chills', 'fatigue', 'weakness', 'pain', 'ache',
                'cough', 'sore throat', 'runny nose', 'congestion', 'rash', 'swelling'
            ],
            'body_parts': [
                'head', 'neck', 'chest', 'back', 'arm', 'leg', 'hand', 'foot',
                'stomach', 'abdomen', 'heart', 'lungs', 'throat', 'eyes', 'ears'
            ],
            'vital_signs': [
                'heart rate', 'blood pressure', 'temperature', 'pulse', 'breathing',
                'respiratory rate', 'oxygen saturation', 'pain level', 'pain score'
            ],
            'medical_terms': [
                'systolic', 'diastolic', 'hypertension', 'hypotension', 'tachycardia',
                'bradycardia', 'arrhythmia', 'medication', 'prescription', 'allergy'
            ]
        }
    
    def _initialize_symptom_patterns(self) -> List[Dict[str, Any]]:
        return [
            {
                'pattern': r'i have (.*?) pain',
                'type': 'pain_location',
                'extract': lambda m: {'symptom': 'pain', 'location': m.group(1).strip()}
            },
            {
                'pattern': r'my (.*?) hurts?',
                'type': 'pain_location',
                'extract': lambda m: {'symptom': 'pain', 'location': m.group(1).strip()}
            },
            {
                'pattern': r'i feel (.*)',
                'type': 'general_symptom',
                'extract': lambda m: {'symptom': m.group(1).strip()}
            },
            {
                'pattern': r'i am experiencing (.*)',
                'type': 'general_symptom',
                'extract': lambda m: {'symptom': m.group(1).strip()}
            },
            {
                'pattern': r'pain level (\d+)',
                'type': 'pain_score',
                'extract': lambda m: {'pain_score': int(m.group(1))}
            }
        ]
    
    def _initialize_vital_signs_patterns(self) -> List[Dict[str, Any]]:
        return [
            {
                'pattern': r'heart rate (\d+)',
                'type': 'heart_rate',
                'extract': lambda m: {'heart_rate': int(m.group(1))}
            },
            {
                'pattern': r'pulse (\d+)',
                'type': 'heart_rate', 
                'extract': lambda m: {'heart_rate': int(m.group(1))}
            },
            {
                'pattern': r'blood pressure (\d+) over (\d+)',
                'type': 'blood_pressure',
                'extract': lambda m: {'systolic_bp': int(m.group(1)), 'diastolic_bp': int(m.group(2))}
            },
            {
                'pattern': r'temperature (\d+\.?\d*)',
                'type': 'temperature',
                'extract': lambda m: {'temperature': float(m.group(1))}
            },
            {
                'pattern': r'oxygen saturation (\d+)',
                'type': 'oxygen_saturation',
                'extract': lambda m: {'oxygen_saturation': int(m.group(1))}
            },
            {
                'pattern': r'breathing rate (\d+)',
                'type': 'respiratory_rate',
                'extract': lambda m: {'respiratory_rate': int(m.group(1))}
            }
        ]
    
    def recognize_speech(self, audio_input: str) -> Dict[str, Any]:
        start_time = time.time()
        time.sleep(0.1)
        normalized_text = self._normalize_text(audio_input)
        extracted_info = self._extract_medical_information(normalized_text)
        confidence = self._calculate_confidence(normalized_text, extracted_info)
        processing_time = time.time() - start_time
        result = {
            'success': True,
            'recognized_text': normalized_text,
            'original_input': audio_input,
            'extracted_info': extracted_info,
            'confidence': confidence,
            'processing_time': processing_time,
            'timestamp': datetime.now(),
            'medical_terms_found': len(extracted_info.get('symptoms', [])) + len(extracted_info.get('vital_signs', []))
        }
        self.recognition_history.append(result)
        return result
    
    def _normalize_text(self, text: str) -> str:
        normalized = text.lower().strip()
        replacements = {
            'hart': 'heart',
            'beet': 'beat',
            'blud': 'blood',
            'presha': 'pressure',
            'temprature': 'temperature',
            'oxigen': 'oxygen',
            'brething': 'breathing'
        }
        for wrong, correct in replacements.items():
            normalized = normalized.replace(wrong, correct)
        return normalized
    
    def _extract_medical_information(self, text: str) -> Dict[str, Any]:
        extracted = {
            'symptoms': [],
            'vital_signs': [],
            'body_parts': [],
            'medical_terms': []
        }
        for pattern_info in self.symptom_patterns:
            matches = re.finditer(pattern_info['pattern'], text, re.IGNORECASE)
            for match in matches:
                symptom_info = pattern_info['extract'](match)
                symptom_info['pattern_type'] = pattern_info['type']
                extracted['symptoms'].append(symptom_info)
        for pattern_info in self.vital_signs_patterns:
            matches = re.finditer(pattern_info['pattern'], text, re.IGNORECASE)
            for match in matches:
                vital_info = pattern_info['extract'](match)
                vital_info['pattern_type'] = pattern_info['type']
                extracted['vital_signs'].append(vital_info)
        for category, terms in self.medical_vocabulary.items():
            for term in terms:
                if term.lower() in text.lower():
                    extracted[category].append({
                        'term': term,
                        'category': category,
                        'found_in_context': self._get_context(text, term)
                    })
        return extracted
    
    def _get_context(self, text: str, term: str, window: int = 20) -> str:
        text_lower = text.lower()
        term_lower = term.lower()
        start_pos = text_lower.find(term_lower)
        if start_pos == -1:
            return ""
        context_start = max(0, start_pos - window)
        context_end = min(len(text), start_pos + len(term) + window)
        return text[context_start:context_end].strip()
    
    def _calculate_confidence(self, text: str, extracted_info: Dict[str, Any]) -> float:
        base_confidence = 0.8
        medical_terms_bonus = 0.0
        for category, items in extracted_info.items():
            if items:
                medical_terms_bonus += len(items) * 0.05
        length_penalty = 0.0
        if len(text.split()) < 3:
            length_penalty = 0.2
        final_confidence = min(1.0, base_confidence + medical_terms_bonus - length_penalty)
        return round(final_confidence, 2)
    
    def start_listening(self) -> bool:
        if self.is_listening:
            return False
        self.is_listening = True
        print(f"{self.name}: Started listening for healthcare-related speech...")
        return True
    
    def stop_listening(self) -> bool:
        if not self.is_listening:
            return False
        self.is_listening = False
        print(f"{self.name}: Stopped listening.")
        return True
    
    def process_continuous_speech(self, speech_inputs: List[str]) -> List[Dict[str, Any]]:
        results = []
        for i, speech_input in enumerate(speech_inputs):
            print(f"Processing speech input {i+1}/{len(speech_inputs)}: '{speech_input[:50]}...'")
            result = self.recognize_speech(speech_input)
            results.append(result)
            time.sleep(0.05)
        return results


class SpeechRecognitionEngine:
    
    def __init__(self):
        self.recognizers: Dict[str, SpeechRecognizer] = {}
        self.active_recognizer: Optional[str] = None
        self.session_history: List[Dict[str, Any]] = []
        self.add_recognizer('healthcare', HealthcareSpeechRecognizer())
        self.set_active_recognizer('healthcare')
    
    def add_recognizer(self, name: str, recognizer: SpeechRecognizer) -> bool:
        if name in self.recognizers:
            return False
        self.recognizers[name] = recognizer
        return True
    
    def set_active_recognizer(self, name: str) -> bool:
        if name not in self.recognizers:
            return False
        self.active_recognizer = name
        return True
    
    def recognize(self, audio_input: str) -> Dict[str, Any]:
        if not self.active_recognizer or self.active_recognizer not in self.recognizers:
            return {
                'success': False,
                'error': 'No active recognizer available',
                'timestamp': datetime.now()
            }
        recognizer = self.recognizers[self.active_recognizer]
        result = recognizer.recognize_speech(audio_input)
        session_record = {
            'recognizer_used': self.active_recognizer,
            'result': result,
            'session_time': datetime.now()
        }
        self.session_history.append(session_record)
        return result
    
    def start_session(self) -> Dict[str, Any]:
        if not self.active_recognizer:
            return {'success': False, 'error': 'No active recognizer'}
        recognizer = self.recognizers[self.active_recognizer]
        success = recognizer.start_listening()
        return {
            'success': success,
            'recognizer': self.active_recognizer,
            'session_started': datetime.now(),
            'message': f"Speech recognition session started with {recognizer.name}"
        }
    
    def end_session(self) -> Dict[str, Any]:
        if not self.active_recognizer:
            return {'success': False, 'error': 'No active session'}
        recognizer = self.recognizers[self.active_recognizer]
        success = recognizer.stop_listening()
        return {
            'success': success,
            'session_ended': datetime.now(),
            'recognitions_processed': len([r for r in self.session_history if r['recognizer_used'] == self.active_recognizer])
        }
    
    def get_engine_stats(self) -> Dict[str, Any]:
        stats = {
            'total_recognizers': len(self.recognizers),
            'active_recognizer': self.active_recognizer,
            'total_sessions': len(self.session_history),
            'recognizer_stats': {}
        }
        for name, recognizer in self.recognizers.items():
            stats['recognizer_stats'][name] = recognizer.get_recognition_stats()
        return stats
    
    def demonstrate_recognition(self) -> Dict[str, Any]:
        sample_inputs = [
            "I have chest pain and my heart rate is 110",
            "My temperature is 38.5 degrees and I feel dizzy", 
            "Blood pressure 140 over 90, experiencing shortness of breath",
            "Pain level 7 in my lower back",
            "I am 45 years old from Gauteng, oxygen saturation 96"
        ]
        print("=" * 60)
        print("SPEECH RECOGNITION DEMONSTRATION")
        print("=" * 60)
        results = []
        for i, sample_input in enumerate(sample_inputs, 1):
            print(f"\n{i}. Processing: '{sample_input}'")
            result = self.recognize(sample_input)
            if result.get('success'):
                print(f"   ✓ Recognized: '{result['recognized_text']}'")
                print(f"   ✓ Confidence: {result['confidence']:.2f}")
                extracted = result.get('extracted_info', {})
                if extracted.get('symptoms'):
                    print(f"   ✓ Symptoms: {len(extracted['symptoms'])} found")
                if extracted.get('vital_signs'):
                    print(f"   ✓ Vital Signs: {len(extracted['vital_signs'])} found")
            else:
                print(f"   ✗ Recognition failed: {result.get('error', 'Unknown error')}")
            results.append(result)
        print("\n" + "=" * 60)
        print("DEMONSTRATION COMPLETED")
        print("=" * 60)
        return {
            'demonstration_completed': True,
            'samples_processed': len(sample_inputs),
            'successful_recognitions': len([r for r in results if r.get('success')]),
            'results': results,
            'engine_stats': self.get_engine_stats()
        }
