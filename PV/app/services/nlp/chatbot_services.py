
class HealthcareChatbot:
    """AI chatbot for healthcare information"""

    def __init__(self):
        # In a real implementation, we would load a pre-trained model here
        self.conversation_history = []

    def get_response(self, user_input):

        user_input = user_input.lower()
        response = ""

        # Define responses based on keywords
        if any(word in user_input for word in ['hello', 'hi', 'greetings', ]):
            response = "Hello! I'm here to help with your healthcare questions. How can I assist you today?"

        elif any(word in user_input for word in ['triage', 'priority', 'emergency']):
            response = ("Triage priority is determined based on the severity of your condition. "
                        "Red indicates emergency, Yellow is urgent, and Green is non-urgent.")

        elif any(word in user_input for word in ['medical aid', 'insurance', 'payment']):
            response = ("South Africa has various medical aid schemes like Discovery, Bonitas, "
                        "and Momentum. Public hospitals provide care regardless of medical aid status.")

        elif any(word in user_input for word in ['clinic', 'hospital', 'facility']):
            response = ("Healthcare facilities in South Africa include public clinics, community health centers,"
                        " and private hospitals. Your location and medical aid will determine which facilities you can access.")

        elif any(word in user_input for word in ['id', 'document', 'verify']):
            response = ("To verify your identity, we need your South African ID number and a photo for facial recognition. "
                        "This helps us access your medical records and ensure you receive proper care.")

        elif any(word in user_input for word in ['thanks', 'thank you', 'appreciate']):
            response = "You're welcome! Is there anything else I can help you with?"

        elif any(word in user_input for word in ['symptoms', 'pain', 'hurt']):
            response = ("I'm not a doctor, but if you're experiencing severe symptoms like chest pain, difficulty breathing, "
                        "or heavy bleeding, please seek emergency care immediately.")

        else:
            response = ("I'm here to help with healthcare information. "
                        " Could you please provide more details or ask about medical aid, triage, facilities, or ID verification?")

        return response