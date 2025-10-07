import time
import re
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from datetime import datetime


class SpeechSynthesizer(ABC):
    
    def __init__(self, name: str):
        self.name = name
        self.synthesis_history: List[Dict[str, Any]] = []
        self.voice_settings = {
            'speed': 1.0,
            'pitch': 1.0,
            'volume': 0.8,
            'voice_type': 'neutral'
        }
    
    @abstractmethod
    def synthesize_speech(self, text: str) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def set_voice_parameters(self, **kwargs) -> bool:
        pass
    
    def get_synthesis_stats(self) -> Dict[str, Any]:
        if not self.synthesis_history:
            return {"message": "No synthesis attempts recorded"}
        
        total_syntheses = len(self.synthesis_history)
        successful_syntheses = len([s for s in self.synthesis_history if s.get('success', False)])
        
        total_text_length = sum(s.get('text_length', 0) for s in self.synthesis_history)
        total_synthesis_time = sum(s.get('synthesis_time', 0) for s in self.synthesis_history)
        
        return {
            'total_syntheses': total_syntheses,
            'successful_syntheses': successful_syntheses,
            'success_rate': successful_syntheses / total_syntheses if total_syntheses > 0 else 0,
            'average_text_length': total_text_length / total_syntheses if total_syntheses > 0 else 0,
            'average_synthesis_time': total_synthesis_time / total_syntheses if total_syntheses > 0 else 0,
            'words_per_second': (total_text_length / 5) / total_synthesis_time if total_synthesis_time > 0 else 0
        }


class HealthcareSpeechSynthesizer(SpeechSynthesizer):
    
    def __init__(self):
        super().__init__("HealthcareSpeechSynthesizer")
        self.medical_pronunciations = self._initialize_medical_pronunciations()
        self.response_templates = self._initialize_response_templates()
        self.voice_settings.update({
            'speed': 0.9,
            'pitch': 0.95,
            'voice_type': 'professional',
            'emphasis_medical_terms': True
        })
    
    def _initialize_medical_pronunciations(self) -> Dict[str, str]:
        return {
            'tachycardia': 'tak-ih-KAR-dee-ah',
            'bradycardia': 'brad-ih-KAR-dee-ah', 
            'hypertension': 'hahy-per-TEN-shuhn',
            'hypotension': 'hahy-poh-TEN-shuhn',
            'arrhythmia': 'uh-RITH-mee-ah',
            'systolic': 'sis-TOL-ik',
            'diastolic': 'dahy-uh-STOL-ik',
            'saturation': 'sach-uh-REY-shuhn',
            'respiratory': 'RES-per-ah-tor-ee',
            'cardiovascular': 'kar-dee-oh-VAS-kyuh-ler',
            'pneumonia': 'noo-MOHN-yah',
            'bronchitis': 'brong-KAHY-tis',
            'dyspnea': 'DISP-nee-ah',
            'tachypnea': 'tak-ip-NEE-ah'
        }
    
    def _initialize_response_templates(self) -> Dict[str, List[str]]:
        return {
            'assessment_complete': [
                "Thank you for providing your information. Based on your symptoms, here is my assessment:",
                "I have analyzed your health information. Here are my findings:",
                "Your health assessment is complete. Here is what I found:"
            ],
            'vital_signs_analysis': [
                "Your vital signs indicate the following:",
                "Based on your vital signs, I observe:",
                "Your current vital signs show:"
            ],
            'recommendations': [
                "Based on this assessment, I recommend:",
                "My recommendations for you are:",
                "Here are my suggested next steps:"
            ],
            'emergency_alert': [
                "ATTENTION: Your symptoms may require immediate medical attention.",
                "WARNING: These vital signs are concerning and need urgent evaluation.",
                "URGENT: Please seek immediate medical care."
            ],
            'reassurance': [
                "Your vital signs appear to be within normal ranges.",
                "Based on your information, there are no immediate concerns.",
                "Your health indicators look stable."
            ]
        }
    
    def synthesize_speech(self, text: str) -> Dict[str, Any]:
        start_time = time.time()
        processed_text = self._preprocess_medical_text(text)
        synthesis_time = self._calculate_synthesis_time(processed_text)
        time.sleep(min(synthesis_time, 0.1))
        speech_metadata = self._generate_speech_metadata(processed_text)
        processing_time = time.time() - start_time
        result = {
            'success': True,
            'original_text': text,
            'processed_text': processed_text,
            'speech_metadata': speech_metadata,
            'synthesis_time': processing_time,
            'text_length': len(text),
            'word_count': len(text.split()),
            'medical_terms_processed': len([term for term in self.medical_pronunciations.keys() if term.lower() in text.lower()]),
            'voice_settings': self.voice_settings.copy(),
            'timestamp': datetime.now(),
            'audio_duration_estimate': synthesis_time
        }
        self.synthesis_history.append(result)
        return result
    
    def _preprocess_medical_text(self, text: str) -> str:
        processed = text
        for term, pronunciation in self.medical_pronunciations.items():
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            if pattern.search(processed):
                processed = pattern.sub(f"{term} ({pronunciation})", processed)
        processed = self._add_natural_pauses(processed)
        return processed
    
    def _add_natural_pauses(self, text: str) -> str:
        text = text.replace(',', ', [pause:short]')
        text = text.replace('.', '. [pause:medium]')
        text = text.replace('!', '! [pause:medium]')
        text = text.replace('?', '? [pause:medium]')
        medical_indicators = ['your heart rate', 'blood pressure', 'temperature', 'oxygen saturation']
        for indicator in medical_indicators:
            text = text.replace(indicator, f'[pause:short] {indicator}')
        return text
    
    def _calculate_synthesis_time(self, text: str) -> float:
        base_time = len(text) * 0.01
        word_count = len(text.split())
        medical_term_bonus = 0
        for term in self.medical_pronunciations.keys():
            if term.lower() in text.lower():
                medical_term_bonus += 0.2
        speed_factor = 1.0 / self.voice_settings['speed']
        total_time = (base_time + medical_term_bonus) * speed_factor
        return total_time
    
    def _generate_speech_metadata(self, text: str) -> Dict[str, Any]:
        return {
            'estimated_duration_seconds': self._calculate_synthesis_time(text),
            'word_count': len(text.split()),
            'sentence_count': len([s for s in text.split('.') if s.strip()]),
            'medical_terms_count': len([term for term in self.medical_pronunciations.keys() if term.lower() in text.lower()]),
            'pause_markers': text.count('[pause:'),
            'voice_profile': self.voice_settings['voice_type'],
            'synthesis_quality': 'high' if len(text) > 50 else 'standard'
        }
    
    def set_voice_parameters(self, **kwargs) -> bool:
        valid_params = ['speed', 'pitch', 'volume', 'voice_type', 'emphasis_medical_terms']
        updated = False
        for param, value in kwargs.items():
            if param in valid_params:
                if param in ['speed', 'pitch', 'volume']:
                    if isinstance(value, (int, float)) and 0.1 <= value <= 2.0:
                        self.voice_settings[param] = value
                        updated = True
                elif param == 'voice_type':
                    valid_voices = ['neutral', 'professional', 'warm', 'authoritative']
                    if value in valid_voices:
                        self.voice_settings[param] = value
                        updated = True
                elif param == 'emphasis_medical_terms':
                    if isinstance(value, bool):
                        self.voice_settings[param] = value
                        updated = True
        return updated
    
    def generate_healthcare_response(self, response_type: str, content: str) -> Dict[str, Any]:
        if response_type not in self.response_templates:
            response_type = 'assessment_complete'
        templates = self.response_templates[response_type]
        import random
        selected_template = random.choice(templates)
        full_response = f"{selected_template} {content}"
        original_settings = self.voice_settings.copy()
        if response_type == 'emergency_alert':
            self.set_voice_parameters(speed=0.8, pitch=1.1, volume=1.0)
        elif response_type == 'reassurance':
            self.set_voice_parameters(speed=0.9, pitch=0.9, volume=0.8)
        result = self.synthesize_speech(full_response)
        result['response_type'] = response_type
        result['template_used'] = selected_template
        self.voice_settings = original_settings
        return result


class TextToSpeechEngine:
    
    def __init__(self):
        self.synthesizers: Dict[str, SpeechSynthesizer] = {}
        self.active_synthesizer: Optional[str] = None
        self.session_history: List[Dict[str, Any]] = []
        self.add_synthesizer('healthcare', HealthcareSpeechSynthesizer())
        self.set_active_synthesizer('healthcare')
    
    def add_synthesizer(self, name: str, synthesizer: SpeechSynthesizer) -> bool:
        if name in self.synthesizers:
            return False
        self.synthesizers[name] = synthesizer
        return True
    
    def set_active_synthesizer(self, name: str) -> bool:
        if name not in self.synthesizers:
            return False
        self.active_synthesizer = name
        return True
    
    def speak(self, text: str) -> Dict[str, Any]:
        if not self.active_synthesizer or self.active_synthesizer not in self.synthesizers:
            return {
                'success': False,
                'error': 'No active synthesizer available',
                'timestamp': datetime.now()
            }
        synthesizer = self.synthesizers[self.active_synthesizer]
        result = synthesizer.synthesize_speech(text)
        session_record = {
            'synthesizer_used': self.active_synthesizer,
            'result': result,
            'session_time': datetime.now()
        }
        self.session_history.append(session_record)
        return result
    
    def speak_healthcare_response(self, response_type: str, content: str) -> Dict[str, Any]:
        if self.active_synthesizer != 'healthcare':
            return {
                'success': False,
                'error': 'Healthcare synthesizer not active',
                'timestamp': datetime.now()
            }
        healthcare_synth = self.synthesizers['healthcare']
        if isinstance(healthcare_synth, HealthcareSpeechSynthesizer):
            result = healthcare_synth.generate_healthcare_response(response_type, content)
            session_record = {
                'synthesizer_used': self.active_synthesizer,
                'result': result,
                'session_time': datetime.now(),
                'response_type': response_type
            }
            self.session_history.append(session_record)
            return result
        return self.speak(content)
    
    def get_engine_stats(self) -> Dict[str, Any]:
        stats = {
            'total_synthesizers': len(self.synthesizers),
            'active_synthesizer': self.active_synthesizer,
            'total_sessions': len(self.session_history),
            'synthesizer_stats': {}
        }
        for name, synthesizer in self.synthesizers.items():
            stats['synthesizer_stats'][name] = synthesizer.get_synthesis_stats()
        return stats
    
    def demonstrate_synthesis(self) -> Dict[str, Any]:
        sample_responses = [
            {
                'type': 'assessment_complete',
                'content': 'Your heart rate of 110 BPM and temperature of 38.2°C indicate possible tachycardia and fever.'
            },
            {
                'type': 'vital_signs_analysis', 
                'content': 'Blood pressure 140 over 90 suggests hypertension. Oxygen saturation of 96% is slightly below normal.'
            },
            {
                'type': 'recommendations',
                'content': 'I recommend monitoring your temperature and seeking medical evaluation if symptoms persist.'
            },
            {
                'type': 'emergency_alert',
                'content': 'Your oxygen saturation of 88% requires immediate medical attention.'
            },
            {
                'type': 'reassurance',
                'content': 'Your vital signs are within normal ranges. Continue routine monitoring.'
            }
        ]
        print("=" * 60)
        print("SPEECH SYNTHESIS DEMONSTRATION")
        print("=" * 60)
        results = []
        for i, sample in enumerate(sample_responses, 1):
            response_type = sample['type']
            content = sample['content']
            print(f"\n{i}. Synthesizing {response_type}:")
            print(f"   Content: '{content[:60]}...'")
            if self.active_synthesizer == 'healthcare':
                result = self.speak_healthcare_response(response_type, content)
            else:
                result = self.speak(content)
            if result.get('success'):
                print(f"   ✓ Synthesis successful")
                print(f"   ✓ Estimated duration: {result.get('audio_duration_estimate', 0):.2f}s")
                print(f"   ✓ Medical terms processed: {result.get('medical_terms_processed', 0)}")
            else:
                print(f"   ✗ Synthesis failed: {result.get('error', 'Unknown error')}")
            results.append(result)
        print("\n" + "=" * 60)
        print("DEMONSTRATION COMPLETED")
        print("=" * 60)
        return {
            'demonstration_completed': True,
            'samples_processed': len(sample_responses),
            'successful_syntheses': len([r for r in results if r.get('success')]),
            'results': results,
            'engine_stats': self.get_engine_stats()
        }
