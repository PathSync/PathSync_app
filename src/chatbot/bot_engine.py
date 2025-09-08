import json
import random
import nltk
import numpy as np
from nltk.stem import PorterStemmer
from src.predict import predict_biometric, predict_triage

# Ensure necessary NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

stemmer = PorterStemmer()


class HealthChatbot:
    def __init__(self, intents_path="src/chatbot/intents.json"):
        self.intents = self.load_intents(intents_path)
        self.words = []
        self.classes = []
        self.ignore_words = ['?', '!', '.', ',']
        self.stemmed_words = []

        # Prepare data for processing
        self.prepare_data()

    def load_intents(self, path):
        with open(path, 'r') as file:
            return json.load(file)

    def prepare_data(self):
        for intent in self.intents['intents']:
            for pattern in intent['patterns']:
                # Tokenize each word
                w = nltk.word_tokenize(pattern)
                self.words.extend(w)
                # Add to classes list
                if intent['tag'] not in self.classes:
                    self.classes.append(intent['tag'])

        # Stem and lower each word
        self.words = [stemmer.stem(w.lower()) for w in self.words if w not in self.ignore_words]
        self.words = sorted(list(set(self.words)))
        self.classes = sorted(list(set(self.classes)))

    def preprocess_sentence(self, sentence):
        # Tokenize and stem
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
        return sentence_words

    def bag_of_words(self, sentence):
        sentence_words = self.preprocess_sentence(sentence)
        bag = [0] * len(self.words)
        for s in sentence_words:
            for i, w in enumerate(self.words):
                if w == s:
                    bag[i] = 1
        return np.array(bag)

    def predict_class(self, sentence):
        bow = self.bag_of_words(sentence)
        # In a real implementation, you'd use a trained model here
        # For simplicity, we'll use pattern matching

        # Check for specific keywords
        sentence_lower = sentence.lower()

        if any(word in sentence_lower for word in ['hello', 'hi', 'hey', 'greetings']):
            return 'greeting'
        elif any(word in sentence_lower for word in ['bye', 'goodbye', 'see you']):
            return 'goodbye'
        elif any(word in sentence_lower for word in ['predict', 'analysis', 'health', 'biometric']):
            return 'health_prediction'
        elif any(word in sentence_lower for word in ['triage', 'urgency', 'emergency']):
            return 'triage_request'
        elif any(word in sentence_lower for word in ['help', 'support', 'assist']):
            return 'help'
        else:
            return 'unknown'

    def get_response(self, message, user_data=None):
        intent = self.predict_class(message)

        for i in self.intents['intents']:
            if i['tag'] == intent:
                response = random.choice(i['responses'])

                # Handle specific intents with custom logic
                if intent == 'health_prediction' and user_data:
                    # Here you would integrate with your actual model
                    prediction = predict_biometric(user_data)
                    response = response.format(prediction=prediction)

                elif intent == 'triage_request' and user_data:
                    prediction = predict_triage(user_data)
                    response = response.format(prediction=prediction)

                return response

        return "I'm not sure how to help with that. Could you try rephrasing?"