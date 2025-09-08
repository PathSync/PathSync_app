from flask import Flask, render_template, request, jsonify
from .bot_engine import HealthChatbot
import json

app = Flask(__name__)
chatbot = HealthChatbot()


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/get_response', methods=['POST'])
def get_bot_response():
    try:
        user_message = request.json['message']
        # In a real implementation, you'd extract user_data from the request
        user_data = None  # This would come from a form or user session

        response = chatbot.get_response(user_message, user_data)
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'response': f"Sorry, I encountered an error: {str(e)}"})


if __name__ == '__main__':
    app.run(debug=True)