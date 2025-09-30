import sys
import os
import warnings
from pathlib import Path

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "src"))

from src.chatbot.chatbot import HealthcareChatbot
from predicts import HealthcarePredictor



def main():
    print("Initializing MedBAot AI...")

    try:
      chatbot = HealthcareChatbot(enable_speech=True)
      speech_status = chatbot.get_speech_status()
      print(" MedBAot initialized successfully!")

      if speech_status['speech_enabled']:
        print("🎙️ Speech Recognition & Synthesis: ENABLED")
      else:
        print("📝 Text-only mode: Speech components not available")

    except Exception as e:
        print(f"Warning: {e}")
        chatbot = HealthcareChatbot(enable_speech=False)
        print("MedBAot AI System initialized in text-only mode")

    print("\n" + "=" * 50)
    print("🏥MedBAot CHATBOT WITH VOICE SUPPORT")
    print("=" * 50)
    print("Type 'exit' to quit the application")
    print("You can ask about:")
    print("- Symptoms (e.g., 'I have chest pain, age 45, heart rate 110')")
    print("- Identity verification (e.g., 'Verify my identity, I'm from Gauteng')")
    print("- General questions")

    if chatbot.enable_speech:
        print("- Voice interactions (say 'enable voice' or 'speak')")
        print("- Speech-enabled responses")

    print("=" * 50)

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("Chatbot: Thank you for using our healthcare services. Goodbye!")
            break

        if user_input:
            response = chatbot.respond(user_input)
            print(f"Chatbot: {response}")


if __name__ == "__main__":
    main()