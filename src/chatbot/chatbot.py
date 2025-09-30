import re
import random
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from predicts import HealthcarePredictor

try:
    from speech import create_healthcare_voice_assistant, create_tts_engine

    SPEECH_AVAILABLE = True
except ImportError:
    SPEECH_AVAILABLE = False
    print("Speech capabilities not available - running in text-only mode")


class HealthcareChatbot:
    def __init__(self, enable_speech=False):
        self.predictor = HealthcarePredictor()
        self.enable_speech = enable_speech and SPEECH_AVAILABLE

        if self.enable_speech:
            self.voice_assistant = create_healthcare_voice_assistant()
            self.tts_engine = create_tts_engine()
            print("✓ Speech capabilities enabled")
        else:
            self.voice_assistant = None
            self.tts_engine = None

        self.responses = {
            'greeting': [
                "Hello! I'm your healthcare assistant. How can I help you today?",
                "Hi there! I'm here to assist with your healthcare needs.",
                "Welcome! How can I help you with healthcare services today?"
            ],
            'triage': "Based on your symptoms, I recommend {} priority care. {}",
            'nationality': "Based on our verification, you appear to be a {} (confidence: {:.2f}).",
            'unknown': "I'm not sure how to help with that. I can assist with triage assessment, identity verification, or general healthcare questions.",
            'goodbye': [
                "Thank you for using our healthcare services. Stay healthy!",
                "Goodbye! Take care of your health.",
                "Have a great day! Remember to prioritize your health."
            ]
        }

        self.triage_explanations = {
            'Red': "Please seek immediate medical attention (ER).",
            'Yellow': "You should see a doctor within 24 hours.",
            'Green': "You can schedule a routine appointment."
        }

    def respond(self, message, use_voice=False):
        message = message.lower()

        if use_voice and self.enable_speech:
            return self.process_voice_input(message)

        if any(word in message for word in ['hello', 'hi', 'hey', 'greetings']):
            response = random.choice(self.responses['greeting'])
            return self._format_response(response, 'greeting')

        elif any(word in message for word in ['bye', 'goodbye', 'exit', 'quit']):
            response = random.choice(self.responses['goodbye'])
            return self._format_response(response, 'goodbye')

        elif any(word in message for word in ['voice', 'speech', 'speak']):
            return self.handle_speech_request(message)

        elif any(word in message for word in ['symptom', 'pain', 'hurt', 'not feeling', 'unwell', 'triage']):
            return self.assess_triage(message)

        elif any(word in message for word in ['identity', 'verify', 'nationality', 'citizen', 'id']):
            return self.verify_identity(message)

        elif 'help' in message:
            help_text = "I can help with: 1) Triage assessment 2) Identity verification 3) General healthcare questions"
            if self.enable_speech:
                help_text += " 4) Voice-based interactions (say 'enable voice' or 'speak')"
            help_text += ". What do you need help with?"
            return help_text

        else:
            return self.responses['unknown']

    def assess_triage(self, message):
        age = self.extract_value(message, r'age[:]?[\s]*(\d+)')
        hr_bpm = self.extract_value(message, r'heart[:]?[\s]*(\d+)') or self.extract_value(message, r'hr[:]?[\s]*(\d+)')
        temp = self.extract_value(message, r'temp[:]?[\s]*(\d+\.?\d*)') or self.extract_value(message,
                                                                                              r'temperature[:]?[\s]*(\d+\.?\d*)')
        resp_rate = self.extract_value(message, r'respiratory[:]?[\s]*(\d+)') or self.extract_value(message,
                                                                                                    r'breath[:]?[\s]*(\d+)')
        systolic = self.extract_value(message, r'systolic[:]?[\s]*(\d+)')
        diastolic = self.extract_value(message, r'diastolic[:]?[\s]*(\d+)')
        o2_sat = self.extract_value(message, r'oxygen[:]?[\s]*(\d+)') or self.extract_value(message,
                                                                                            r'o2[:]?[\s]*(\d+)')
        pain = self.extract_value(message, r'pain[:]?[\s]*(\d+)')

        default_values = {
            'age': age if age else 35,
            'gender': 'Male',
            'hr_bpm': hr_bpm if hr_bpm else 75,
            'temp_c': temp if temp else 36.6,
            'resp_rate': resp_rate if resp_rate else 16,
            'systolic_bp': systolic if systolic else 120,
            'diastolic_bp': diastolic if diastolic else 80,
            'o2_sat': o2_sat if o2_sat else 98,
            'pain_score': pain if pain else 3
        }

        if 'female' in message:
            default_values['gender'] = 'Female'
        elif 'other' in message:
            default_values['gender'] = 'Other'

        try:
            priority, confidence = self.predictor.predict_triage(
                int(default_values['age']),
                default_values['gender'],
                int(default_values['hr_bpm']),
                float(default_values['temp_c']),
                int(default_values['resp_rate']),
                int(default_values['systolic_bp']),
                int(default_values['diastolic_bp']),
                int(default_values['o2_sat']),
                int(default_values['pain_score'])
            )

            return self.responses['triage'].format(priority, self.triage_explanations[priority])
        except Exception as e:
            return f"I encountered an error processing your symptoms: {str(e)}. Please provide clearer information."

    def verify_identity(self, message):
        age = self.extract_value(message, r'age[:]?[\s]*(\d+)')
        biometric_score = self.extract_value(message, r'biometric[:]?[\s]*(\d+\.?\d*)') or 0.85

        default_values = {
            'age': age if age else 35,
            'gender': 'Male',
            'province': 'Gauteng',
            'biometric_score': float(biometric_score)
        }

        if 'female' in message:
            default_values['gender'] = 'Female'
        elif 'other' in message:
            default_values['gender'] = 'Other'

        for province in ['Gauteng', 'Western Cape', 'Eastern Cape', 'KwaZulu-Natal',
                         'Free State', 'Limpopo', 'Mpumalanga', 'North West', 'Northern Cape']:
            if province.lower() in message:
                default_values['province'] = province
                break

        try:
            citizenship, confidence = self.predictor.predict_biometric(
                int(default_values['age']),
                default_values['gender'],
                default_values['province'],
                default_values['biometric_score']
            )

            citizenship_display = {
                'SA': 'South African Citizen',
                'Non-SA': 'Non-South African Citizen',
                'Review': 'Requires Manual Review'
            }.get(citizenship, citizenship)

            return self.responses['nationality'].format(citizenship_display, confidence)
        except Exception as e:
            return f"I encountered an error verifying your identity: {str(e)}"

    def extract_value(self, text, pattern):
        match = re.search(pattern, text, re.IGNORECASE)
        return match.group(1) if match else None

    def _format_response(self, response_text, response_type='general'):
        if self.enable_speech and self.tts_engine:
            try:
                tts_result = self.tts_engine.speak_healthcare_response(response_type, response_text)
                if tts_result.get('success'):
                    return f"{response_text}\n[🔊 Response also available as speech - duration: {tts_result.get('audio_duration_estimate', 0):.1f}s]"
            except Exception as e:
                print(f"Speech synthesis error: {e}")

        return response_text

    def handle_speech_request(self, message):
        if not self.enable_speech:
            return "Speech features are not currently available. Please ensure speech components are properly installed."

        if any(word in message for word in ['enable', 'start', 'activate']):
            return self._format_response(
                "Voice mode activated! You can now speak your symptoms or questions, and I'll respond with both text and speech.",
                'assessment_complete'
            )
        elif any(word in message for word in ['disable', 'stop', 'deactivate']):
            return "Voice mode deactivated. Continuing with text-only responses."
        else:
            return self._format_response(
                "I can process voice input and provide speech responses. Say 'enable voice' to activate voice mode.",
                'general'
            )

    def process_voice_input(self, voice_input):
        if not self.enable_speech or not self.voice_assistant:
            return "Voice processing is not available."

        try:
            if not self.voice_assistant.is_active:
                self.voice_assistant.start_conversation()

            result = self.voice_assistant.process_voice_input(voice_input)

            if result.get('success'):
                response_text = result.get('response_text', 'I processed your voice input.')
                recognition_confidence = result.get('recognition_result', {}).get('confidence', 0)

                voice_info = f"\n[🎙️ Voice recognized with {recognition_confidence:.1%} confidence]"
                return response_text + voice_info
            else:
                return "I had trouble processing your voice input. Please try speaking more clearly or use text input."

        except Exception as e:
            return f"Voice processing error: {e}. Falling back to text mode."

    def demonstrate_speech_capabilities(self):
        if not self.enable_speech:
            return "Speech capabilities are not enabled for this chatbot instance."

        print("=" * 60)
        print("CHATBOT SPEECH CAPABILITIES DEMONSTRATION")
        print("=" * 60)

        sample_interactions = [
            "Hello, I need help with my health",
            "I have chest pain and my heart rate is 110",
            "My temperature is 38.5 and I feel dizzy",
            "Thank you for your help"
        ]

        results = []

        for i, interaction in enumerate(sample_interactions, 1):
            print(f"\n{i}. Processing: '{interaction}'")

            response = self.process_voice_input(interaction)
            results.append({
                'input': interaction,
                'response': response,
                'has_speech_synthesis': '[🔊' in response
            })

            print(f"   Response: {response[:100]}...")

        print(f"\n📊 Speech Integration Summary:")
        print(f"   • Interactions processed: {len(sample_interactions)}")
        print(f"   • Responses with speech: {sum(1 for r in results if r['has_speech_synthesis'])}")
        print(f"   • Voice assistant active: {self.voice_assistant.is_active if self.voice_assistant else False}")

        return results

    def get_speech_status(self):
        return {
            'speech_available': SPEECH_AVAILABLE,
            'speech_enabled': self.enable_speech,
            'voice_assistant_active': self.voice_assistant.is_active if self.voice_assistant else False,
            'tts_engine_ready': self.tts_engine is not None,
            'capabilities': [
                'Voice input processing',
                'Healthcare speech recognition',
                'Medical text-to-speech synthesis',
                'Voice-guided health assessment'
            ] if self.enable_speech else ['Text-only mode']
        }