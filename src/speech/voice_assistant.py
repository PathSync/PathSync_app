import time
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime

from .speech_recognition import SpeechRecognitionEngine, HealthcareSpeechRecognizer
from .speech_synthesis import TextToSpeechEngine, HealthcareSpeechSynthesizer


class VoiceAssistant:
    
    def __init__(self, name: str):
        self.name = name
        self.speech_engine = SpeechRecognitionEngine()
        self.tts_engine = TextToSpeechEngine()
        self.conversation_history: List[Dict[str, Any]] = []
        self.is_active = False
        
    def start_conversation(self) -> Dict[str, Any]:
        if self.is_active:
            return {'success': False, 'message': 'Conversation already active'}
        
        self.is_active = True
        speech_session = self.speech_engine.start_session()
        
        welcome_message = "Hello! I'm your voice assistant. How can I help you today?"
        welcome_response = self.tts_engine.speak(welcome_message)
        
        session_info = {
            'success': True,
            'session_started': datetime.now(),
            'speech_recognition': speech_session.get('success', False),
            'welcome_spoken': welcome_response.get('success', False),
            'message': f'{self.name} is now ready for voice interaction'
        }
        
        self.conversation_history.append({
            'type': 'session_start',
            'timestamp': datetime.now(),
            'details': session_info
        })
        
        return session_info
    
    def process_voice_input(self, voice_input: str) -> Dict[str, Any]:
        if not self.is_active:
            return {'success': False, 'error': 'Voice assistant not active'}
        
        recognition_result = self.speech_engine.recognize(voice_input)
        
        if not recognition_result.get('success'):
            error_response = "I'm sorry, I didn't understand that. Could you please repeat?"
            tts_result = self.tts_engine.speak(error_response)
            return {
                'success': False,
                'recognition_result': recognition_result,
                'response_text': error_response,
                'tts_result': tts_result
            }
        
        response_text = self._generate_response(recognition_result)
        
        tts_result = self.tts_engine.speak(response_text)
        
        conversation_entry = {
            'type': 'voice_interaction',
            'timestamp': datetime.now(),
            'user_input': voice_input,
            'recognized_text': recognition_result.get('recognized_text', ''),
            'response_text': response_text,
            'recognition_confidence': recognition_result.get('confidence', 0),
            'tts_success': tts_result.get('success', False)
        }
        self.conversation_history.append(conversation_entry)
        
        return {
            'success': True,
            'recognition_result': recognition_result,
            'response_text': response_text,
            'tts_result': tts_result,
            'conversation_entry': conversation_entry
        }
    
    def _generate_response(self, recognition_result: Dict[str, Any]) -> str:
        recognized_text = recognition_result.get('recognized_text', '')
        return f"I heard you say: {recognized_text}. Thank you for your input."
    
    def end_conversation(self) -> Dict[str, Any]:
        if not self.is_active:
            return {'success': False, 'message': 'No active conversation'}
        
        self.is_active = False
        speech_session_end = self.speech_engine.end_session()
        
        goodbye_message = "Thank you for using the voice assistant. Goodbye!"
        goodbye_response = self.tts_engine.speak(goodbye_message)
        
        session_summary = {
            'success': True,
            'session_ended': datetime.now(),
            'total_interactions': len([entry for entry in self.conversation_history if entry['type'] == 'voice_interaction']),
            'speech_recognition_ended': speech_session_end.get('success', False),
            'goodbye_spoken': goodbye_response.get('success', False)
        }
        
        self.conversation_history.append({
            'type': 'session_end',
            'timestamp': datetime.now(),
            'details': session_summary
        })
        
        return session_summary
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        interactions = [entry for entry in self.conversation_history if entry['type'] == 'voice_interaction']
        
        if not interactions:
            return {'message': 'No voice interactions recorded'}
        
        total_interactions = len(interactions)
        successful_recognitions = len([i for i in interactions if i.get('recognition_confidence', 0) > 0.5])
        successful_responses = len([i for i in interactions if i.get('tts_success', False)])
        
        return {
            'total_interactions': total_interactions,
            'successful_recognitions': successful_recognitions,
            'successful_responses': successful_responses,
            'recognition_success_rate': successful_recognitions / total_interactions if total_interactions > 0 else 0,
            'response_success_rate': successful_responses / total_interactions if total_interactions > 0 else 0,
            'average_confidence': sum(i.get('recognition_confidence', 0) for i in interactions) / total_interactions if total_interactions > 0 else 0,
            'conversation_duration': (self.conversation_history[-1]['timestamp'] - self.conversation_history[0]['timestamp']).total_seconds() if len(self.conversation_history) >= 2 else 0
        }


class HealthcareVoiceAssistant(VoiceAssistant):
    
    def __init__(self):
        super().__init__("Healthcare Voice Assistant")
        
        self.speech_engine = SpeechRecognitionEngine()
        self.tts_engine = TextToSpeechEngine()
        
        self.medical_knowledge = self._initialize_medical_knowledge()
        self.current_assessment = {}
        self.patient_callback: Optional[Callable] = None
    
    def _initialize_medical_knowledge(self) -> Dict[str, Any]:
        return {
            'symptom_responses': {
                'chest pain': 'Chest pain can be serious. I recommend seeking immediate medical attention if the pain is severe or accompanied by shortness of breath.',
                'headache': 'Headaches can have various causes. Monitor the severity and frequency. Seek medical care if severe or persistent.',
                'fever': 'A fever indicates your body is fighting an infection. Monitor your temperature and seek medical care if it exceeds 39°C.',
                'shortness of breath': 'Difficulty breathing requires immediate attention. Please seek emergency medical care.',
                'dizziness': 'Dizziness can indicate various conditions. Sit down safely and seek medical evaluation if it persists.'
            },
            'vital_signs_guidance': {
                'heart_rate': {
                    'normal': (60, 100),
                    'response_high': 'Your heart rate appears elevated. This could indicate stress, exercise, or a medical condition requiring evaluation.',
                    'response_low': 'Your heart rate appears low. This may be normal for athletes or could indicate a medical condition.',
                    'response_normal': 'Your heart rate is within the normal range.'
                },
                'blood_pressure': {
                    'normal_systolic': (90, 140),
                    'normal_diastolic': (60, 90),
                    'response_high': 'Your blood pressure appears elevated. This may indicate hypertension requiring medical evaluation.',
                    'response_low': 'Your blood pressure appears low. Monitor for symptoms and seek medical advice if concerned.',
                    'response_normal': 'Your blood pressure readings are within normal ranges.'
                },
                'temperature': {
                    'normal': (36.1, 37.2),
                    'response_high': 'Your temperature indicates a fever. Monitor closely and seek medical care if it continues to rise.',
                    'response_low': 'Your temperature is below normal. This could indicate hypothermia or other conditions.',
                    'response_normal': 'Your temperature is normal.'
                }
            },
            'emergency_keywords': [
                'chest pain', 'shortness of breath', 'severe pain', 'unconscious', 
                'bleeding', 'emergency', 'help', 'urgent', 'critical'
            ]
        }
    
    def set_patient_callback(self, callback: Callable) -> None:
        self.patient_callback = callback
    
    def _generate_response(self, recognition_result: Dict[str, Any]) -> str:
        recognized_text = recognition_result.get('recognized_text', '').lower()
        extracted_info = recognition_result.get('extracted_info', {})
        
        if self._contains_emergency_keywords(recognized_text):
            return self._generate_emergency_response(recognized_text)
        
        if extracted_info.get('symptoms'):
            return self._generate_symptom_response(extracted_info['symptoms'])
        
        if extracted_info.get('vital_signs'):
            return self._generate_vital_signs_response(extracted_info['vital_signs'])
        
        if any(keyword in recognized_text for keyword in ['health', 'medical', 'doctor', 'symptom']):
            return self._generate_general_health_response(recognized_text)
        
        return "I understand you're seeking healthcare assistance. Could you please describe your symptoms or provide your vital signs?"
    
    def _contains_emergency_keywords(self, text: str) -> bool:
        return any(keyword in text for keyword in self.medical_knowledge['emergency_keywords'])
    
    def _generate_emergency_response(self, text: str) -> str:
        return "This sounds like a medical emergency. Please call emergency services immediately or go to the nearest emergency room. Do not delay seeking immediate medical attention."
    
    def _generate_symptom_response(self, symptoms: List[Dict[str, Any]]) -> str:
        responses = []
        
        for symptom_info in symptoms:
            symptom = symptom_info.get('symptom', '').lower()
            
            for known_symptom, response in self.medical_knowledge['symptom_responses'].items():
                if known_symptom in symptom:
                    responses.append(response)
                    break
            else:
                responses.append(f"I understand you're experiencing {symptom}. I recommend consulting with a healthcare provider for proper evaluation.")
        
        if len(responses) == 1:
            return responses[0]
        elif len(responses) > 1:
            return "Based on your symptoms: " + " Additionally, ".join(responses)
        else:
            return "Thank you for describing your symptoms. I recommend consulting with a healthcare provider for proper evaluation and treatment."
    
    def _generate_vital_signs_response(self, vital_signs: List[Dict[str, Any]]) -> str:
        responses = []
        
        for vital_info in vital_signs:
            vital_type = vital_info.get('pattern_type', '')
            
            if vital_type == 'heart_rate':
                hr = vital_info.get('heart_rate', 0)
                normal_range = self.medical_knowledge['vital_signs_guidance']['heart_rate']['normal']
                
                if hr < normal_range[0]:
                    responses.append(self.medical_knowledge['vital_signs_guidance']['heart_rate']['response_low'])
                elif hr > normal_range[1]:
                    responses.append(self.medical_knowledge['vital_signs_guidance']['heart_rate']['response_high'])
                else:
                    responses.append(self.medical_knowledge['vital_signs_guidance']['heart_rate']['response_normal'])
            
            elif vital_type == 'blood_pressure':
                systolic = vital_info.get('systolic_bp', 0)
                diastolic = vital_info.get('diastolic_bp', 0)
                
                bp_guidance = self.medical_knowledge['vital_signs_guidance']['blood_pressure']
                
                if (systolic > bp_guidance['normal_systolic'][1] or 
                    diastolic > bp_guidance['normal_diastolic'][1]):
                    responses.append(bp_guidance['response_high'])
                elif (systolic < bp_guidance['normal_systolic'][0] or 
                      diastolic < bp_guidance['normal_diastolic'][0]):
                    responses.append(bp_guidance['response_low'])
                else:
                    responses.append(bp_guidance['response_normal'])
            
            elif vital_type == 'temperature':
                temp = vital_info.get('temperature', 0)
                normal_range = self.medical_knowledge['vital_signs_guidance']['temperature']['normal']
                
                if temp > normal_range[1]:
                    responses.append(self.medical_knowledge['vital_signs_guidance']['temperature']['response_high'])
                elif temp < normal_range[0]:
                    responses.append(self.medical_knowledge['vital_signs_guidance']['temperature']['response_low'])
                else:
                    responses.append(self.medical_knowledge['vital_signs_guidance']['temperature']['response_normal'])
        
        if responses:
            return " ".join(responses)
        else:
            return "Thank you for providing your vital signs. I recommend keeping track of these values and sharing them with your healthcare provider."
    
    def _generate_general_health_response(self, text: str) -> str:
        if 'appointment' in text:
            return "I can help you understand your health concerns, but I cannot schedule appointments. Please contact your healthcare provider directly."
        elif 'medication' in text:
            return "For medication questions, please consult with your doctor or pharmacist. They can provide proper guidance about your prescriptions."
        elif 'diagnosis' in text:
            return "I cannot provide medical diagnoses. Please consult with a qualified healthcare professional for proper medical evaluation."
        else:
            return "I'm here to help with basic health information and symptom assessment. For specific medical advice, please consult with a healthcare professional."
    
    def conduct_voice_assessment(self, voice_inputs: List[str]) -> Dict[str, Any]:
        if not self.is_active:
            start_result = self.start_conversation()
            if not start_result.get('success'):
                return {'success': False, 'error': 'Could not start voice session'}
        
        assessment_results = []
        collected_info = {
            'symptoms': [],
            'vital_signs': [],
            'concerns': []
        }
        
        print("=" * 60)
        print("VOICE-BASED HEALTH ASSESSMENT")
        print("=" * 60)
        
        for i, voice_input in enumerate(voice_inputs, 1):
            print(f"\n{i}. Processing: '{voice_input}'")
            
            result = self.process_voice_input(voice_input)
            assessment_results.append(result)
            
            if result.get('success'):
                recognition = result['recognition_result']
                extracted = recognition.get('extracted_info', {})
                
                if extracted.get('symptoms'):
                    collected_info['symptoms'].extend(extracted['symptoms'])
                
                if extracted.get('vital_signs'):
                    collected_info['vital_signs'].extend(extracted['vital_signs'])
                
                print(f"   ✓ Recognized: '{recognition.get('recognized_text', '')}'")
                print(f"   ✓ Response: '{result['response_text'][:100]}...'")
            else:
                print(f"   ✗ Processing failed")
        
        final_assessment = self._generate_final_assessment(collected_info)
        final_response = self.tts_engine.speak_healthcare_response('assessment_complete', final_assessment)
        
        print(f"\n📋 Final Assessment:")
        print(f"   {final_assessment}")
        print("\n" + "=" * 60)
        print("VOICE ASSESSMENT COMPLETED")
        print("=" * 60)
        
        return {
            'success': True,
            'assessment_completed': True,
            'inputs_processed': len(voice_inputs),
            'successful_interactions': len([r for r in assessment_results if r.get('success')]),
            'collected_information': collected_info,
            'final_assessment': final_assessment,
            'final_response': final_response,
            'conversation_summary': self.get_conversation_summary()
        }
    
    def _generate_final_assessment(self, collected_info: Dict[str, Any]) -> str:
        assessment_parts = []
        
        if collected_info['symptoms']:
            symptom_count = len(collected_info['symptoms'])
            assessment_parts.append(f"You reported {symptom_count} symptom(s).")
        
        if collected_info['vital_signs']:
            vital_count = len(collected_info['vital_signs'])
            assessment_parts.append(f"I recorded {vital_count} vital sign measurement(s).")
        
        if collected_info['symptoms'] or collected_info['vital_signs']:
            assessment_parts.append("Based on the information provided, I recommend consulting with a healthcare provider for proper evaluation and treatment.")
        else:
            assessment_parts.append("Thank you for using the voice assessment system. For any health concerns, please consult with a healthcare professional.")
        
        return " ".join(assessment_parts)


class VoiceInteractionManager:
    
    def __init__(self):
        self.voice_assistants: Dict[str, VoiceAssistant] = {}
        self.active_assistant: Optional[str] = None
        self.interaction_logs: List[Dict[str, Any]] = []
        
        self.add_assistant('healthcare', HealthcareVoiceAssistant())
        self.set_active_assistant('healthcare')
    
    def add_assistant(self, name: str, assistant: VoiceAssistant) -> bool:
        if name in self.voice_assistants:
            return False
        
        self.voice_assistants[name] = assistant
        return True
    
    def set_active_assistant(self, name: str) -> bool:
        if name not in self.voice_assistants:
            return False
        
        self.active_assistant = name
        return True
    
    def start_voice_interaction(self) -> Dict[str, Any]:
        if not self.active_assistant:
            return {'success': False, 'error': 'No active voice assistant'}
        
        assistant = self.voice_assistants[self.active_assistant]
        result = assistant.start_conversation()
        
        self.interaction_logs.append({
            'type': 'interaction_start',
            'assistant': self.active_assistant,
            'timestamp': datetime.now(),
            'result': result
        })
        
        return result
    
    def process_voice_command(self, voice_input: str) -> Dict[str, Any]:
        if not self.active_assistant:
            return {'success': False, 'error': 'No active voice assistant'}
        
        assistant = self.voice_assistants[self.active_assistant]
        result = assistant.process_voice_input(voice_input)
        
        self.interaction_logs.append({
            'type': 'voice_command',
            'assistant': self.active_assistant,
            'input': voice_input,
            'timestamp': datetime.now(),
            'result': result
        })
        
        return result
    
    def end_voice_interaction(self) -> Dict[str, Any]:
        if not self.active_assistant:
            return {'success': False, 'error': 'No active voice assistant'}
        
        assistant = self.voice_assistants[self.active_assistant]
        result = assistant.end_conversation()
        
        self.interaction_logs.append({
            'type': 'interaction_end',
            'assistant': self.active_assistant,
            'timestamp': datetime.now(),
            'result': result
        })
        
        return result
    
    def get_interaction_summary(self) -> Dict[str, Any]:
        total_interactions = len(self.interaction_logs)
        
        if total_interactions == 0:
            return {'message': 'No voice interactions recorded'}
        
        interaction_types = {}
        assistants_used = set()
        
        for log in self.interaction_logs:
            interaction_type = log.get('type', 'unknown')
            interaction_types[interaction_type] = interaction_types.get(interaction_type, 0) + 1
            assistants_used.add(log.get('assistant', 'unknown'))
        
        return {
            'total_interactions': total_interactions,
            'interaction_types': interaction_types,
            'assistants_used': list(assistants_used),
            'active_assistant': self.active_assistant,
            'latest_interaction': self.interaction_logs[-1]['timestamp'] if self.interaction_logs else None
        }
