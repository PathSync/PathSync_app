import re
import random
from predict import HealthcarePredictor


class HealthcareChatbot:
    def __init__(self):
        self.predictor = HealthcarePredictor()
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

    def respond(self, message):
        """Generate a response based on user input"""
        message = message.lower()

        # Greeting
        if any(word in message for word in ['hello', 'hi', 'hey', 'greetings']):
            return random.choice(self.responses['greeting'])

        # Goodbye
        elif any(word in message for word in ['bye', 'goodbye', 'exit', 'quit']):
            return random.choice(self.responses['goodbye'])

        # Triage assessment
        elif any(word in message for word in ['symptom', 'pain', 'hurt', 'not feeling', 'unwell', 'triage']):
            return self.assess_triage(message)

        # Identity verification
        elif any(word in message for word in ['identity', 'verify', 'nationality', 'citizen', 'id']):
            return self.verify_identity(message)

        # Help
        elif 'help' in message:
            return "I can help with: 1) Triage assessment 2) Identity verification 3) General healthcare questions. What do you need help with?"

        # Default response
        else:
            return self.responses['unknown']

    def assess_triage(self, message):
        """Assess triage priority based on symptoms described"""
        # Extract values from message using regex
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

        # Default values if not provided
        default_values = {
            'age': age if age else 35,
            'gender': 'Male',  # Assume male if not specified
            'hr_bpm': hr_bpm if hr_bpm else 75,
            'temp_c': temp if temp else 36.6,
            'resp_rate': resp_rate if resp_rate else 16,
            'systolic_bp': systolic if systolic else 120,
            'diastolic_bp': diastolic if diastolic else 80,
            'o2_sat': o2_sat if o2_sat else 98,
            'pain_score': pain if pain else 3
        }

        # Try to detect gender from message
        if 'female' in message:
            default_values['gender'] = 'Female'
        elif 'other' in message:
            default_values['gender'] = 'Other'

        # Predict triage priority
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
        """Verify identity and nationality"""
        # Extract values from message
        age = self.extract_value(message, r'age[:]?[\s]*(\d+)')
        biometric_score = self.extract_value(message, r'biometric[:]?[\s]*(\d+\.?\d*)') or 0.85

        # Default values
        default_values = {
            'age': age if age else 35,
            'gender': 'Male',
            'province': 'Gauteng',
            'biometric_score': float(biometric_score)
        }

        # Try to detect gender and province from message
        if 'female' in message:
            default_values['gender'] = 'Female'
        elif 'other' in message:
            default_values['gender'] = 'Other'

        for province in ['Gauteng', 'Western Cape', 'Eastern Cape', 'KwaZulu-Natal',
                         'Free State', 'Limpopo', 'Mpumalanga', 'North West', 'Northern Cape']:
            if province.lower() in message:
                default_values['province'] = province
                break

        # Predict citizenship
        try:
            citizenship, confidence = self.predictor.predict_biometric(
                int(default_values['age']),
                default_values['gender'],
                default_values['province'],
                default_values['biometric_score']
            )

            # Format citizenship for display
            citizenship_display = {
                'SA': 'South African Citizen',
                'Non-SA': 'Non-South African Citizen',
                'Review': 'Requires Manual Review'
            }.get(citizenship, citizenship)

            return self.responses['nationality'].format(citizenship_display, confidence)
        except Exception as e:
            return f"I encountered an error verifying your identity: {str(e)}"

    def extract_value(self, text, pattern):
        """Extract a value using regex pattern"""
        match = re.search(pattern, text, re.IGNORECASE)
        return match.group(1) if match else None