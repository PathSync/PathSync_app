from chatbot import HealthcareChatbot

def main():
    print("Initializing Healthcare AI System...")

    # Initialize chatbot
    chatbot = HealthcareChatbot()

    print("Healthcare AI System initialized successfully!")
    print("\n" + "=" * 50)
    print("HEALTHCARE CHATBOT")
    print("=" * 50)
    print("Type 'exit' to quit the application")
    print("You can ask about:")
    print("- Symptoms (e.g., 'I have chest pain, age 45, heart rate 110')")
    print("- Identity verification (e.g., 'Verify my identity, I'm from Gauteng')")
    print("- General questions")
    print("=" * 50)

    # Chat interface
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