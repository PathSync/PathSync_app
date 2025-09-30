import numpy as np
from typing import Tuple, List
from .base_models import NeuralNetwork
import pandas as pd


class BiometricNN(NeuralNetwork):
    
    def __init__(self):
        super().__init__("BiometricNN")
        self.label_map = {'SA': 0, 'Non-SA': 1, 'Review': 2}
        self.reverse_map = {v: k for k, v in self.label_map.items()}
    
    def prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        features = []
        labels = []
        for _, row in data.iterrows():
            feature_vector = [
                row['age'] / 100.0,
                1.0 if row['gender'] == 'Male' else 0.0,
                row['biometric_score'],
                len(row['province']) / 20.0
            ]
            features.append(feature_vector)
            labels.append(self.label_map[row['citizenship_status']])
        return np.array(features), np.array(labels)
    
    def train(self, X: np.ndarray, y: np.ndarray) -> float:
        print(f"Training {self.name}...")
        input_size = X.shape[1]
        hidden_size = 8
        output_size = 3
        np.random.seed(42)
        self.weights = {
            'W1': np.random.randn(input_size, hidden_size) * 0.1,
            'b1': np.zeros((1, hidden_size)),
            'W2': np.random.randn(hidden_size, output_size) * 0.1,
            'b2': np.zeros((1, output_size))
        }
        learning_rate = 0.01
        epochs = 100
        for epoch in range(epochs):
            z1 = np.dot(X, self.weights['W1']) + self.weights['b1']
            a1 = self._sigmoid(z1)
            z2 = np.dot(a1, self.weights['W2']) + self.weights['b2']
            a2 = self._softmax(z2)
            y_onehot = np.zeros((len(y), output_size))
            y_onehot[np.arange(len(y)), y] = 1
            loss = -np.mean(np.sum(y_onehot * np.log(a2 + 1e-8), axis=1))
            dz2 = a2 - y_onehot
            dW2 = np.dot(a1.T, dz2) / len(X)
            db2 = np.mean(dz2, axis=0, keepdims=True)
            da1 = np.dot(dz2, self.weights['W2'].T)
            dz1 = da1 * a1 * (1 - a1)
            dW1 = np.dot(X.T, dz1) / len(X)
            db1 = np.mean(dz1, axis=0, keepdims=True)
            self.weights['W1'] -= learning_rate * dW1
            self.weights['b1'] -= learning_rate * db1
            self.weights['W2'] -= learning_rate * dW2
            self.weights['b2'] -= learning_rate * db2
            if (epoch + 1) % 20 == 0:
                predictions = np.argmax(a2, axis=1)
                accuracy = np.mean(predictions == y)
                print(f"Epoch {epoch + 1}: Loss={loss:.4f}, Accuracy={accuracy:.4f}")
        predictions = self.predict(X)
        self.accuracy = np.mean(predictions == y)
        self.is_trained = True
        print(f"Training completed! Accuracy: {self.accuracy:.4f}")
        return self.accuracy
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        z1 = np.dot(X, self.weights['W1']) + self.weights['b1']
        a1 = self._sigmoid(z1)
        z2 = np.dot(a1, self.weights['W2']) + self.weights['b2']
        a2 = self._softmax(z2)
        predicted_classes = np.argmax(a2, axis=1)
        return np.array([self.reverse_map[cls] for cls in predicted_classes])
    
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def _softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)


class TriageNN(NeuralNetwork):
    
    def __init__(self):
        super().__init__("TriageNN")
        self.label_map = {'Red': 0, 'Yellow': 1, 'Green': 2}
        self.reverse_map = {v: k for k, v in self.label_map.items()}
    
    def prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        features = []
        labels = []
        for _, row in data.iterrows():
            feature_vector = [
                row['age'] / 100.0,
                1.0 if row['gender'] == 'Male' else 0.0,
                row['hr_bpm'] / 200.0,
                row['temp_c'] / 45.0,
                row['resp_rate'] / 50.0,
                row['systolic_bp'] / 250.0,
                row['diastolic_bp'] / 150.0,
                row['o2_sat'] / 100.0,
                row['pain_score'] / 10.0
            ]
            features.append(feature_vector)
            labels.append(self.label_map[row['triage_priority']])
        return np.array(features), np.array(labels)
    
    def train(self, X: np.ndarray, y: np.ndarray) -> float:
        print(f"Training {self.name}...")
        input_size = X.shape[1]
        hidden_size = 12
        output_size = 3
        np.random.seed(42)
        self.weights = {
            'W1': np.random.randn(input_size, hidden_size) * 0.1,
            'b1': np.zeros((1, hidden_size)),
            'W2': np.random.randn(hidden_size, output_size) * 0.1,
            'b2': np.zeros((1, output_size))
        }
        learning_rate = 0.01
        epochs = 150
        for epoch in range(epochs):
            z1 = np.dot(X, self.weights['W1']) + self.weights['b1']
            a1 = self._relu(z1)
            z2 = np.dot(a1, self.weights['W2']) + self.weights['b2']
            a2 = self._softmax(z2)
            y_onehot = np.zeros((len(y), output_size))
            y_onehot[np.arange(len(y)), y] = 1
            loss = -np.mean(np.sum(y_onehot * np.log(a2 + 1e-8), axis=1))
            dz2 = a2 - y_onehot
            dW2 = np.dot(a1.T, dz2) / len(X)
            db2 = np.mean(dz2, axis=0, keepdims=True)
            da1 = np.dot(dz2, self.weights['W2'].T)
            dz1 = da1 * (z1 > 0)
            dW1 = np.dot(X.T, dz1) / len(X)
            db1 = np.mean(dz1, axis=0, keepdims=True)
            self.weights['W1'] -= learning_rate * dW1
            self.weights['b1'] -= learning_rate * db1
            self.weights['W2'] -= learning_rate * dW2
            self.weights['b2'] -= learning_rate * db2
            if (epoch + 1) % 30 == 0:
                predictions = np.argmax(a2, axis=1)
                accuracy = np.mean(predictions == y)
                print(f"Epoch {epoch + 1}: Loss={loss:.4f}, Accuracy={accuracy:.4f}")
        predictions = self.predict(X)
        self.accuracy = np.mean(predictions == y)
        self.is_trained = True
        print(f"Training completed! Accuracy: {self.accuracy:.4f}")
        return self.accuracy
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        z1 = np.dot(X, self.weights['W1']) + self.weights['b1']
        a1 = self._relu(z1)
        z2 = np.dot(a1, self.weights['W2']) + self.weights['b2']
        a2 = self._softmax(z2)
        predicted_classes = np.argmax(a2, axis=1)
        return np.array([self.reverse_map[cls] for cls in predicted_classes])
    
    def _relu(self, x):
        return np.maximum(0, x)
    
    def _softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)


class NeuralNetworkFactory:
    
    @staticmethod
    def create_biometric_nn() -> BiometricNN:
        return BiometricNN()
    
    @staticmethod
    def create_triage_nn() -> TriageNN:
        return TriageNN()
    
    @staticmethod
    def get_available_models() -> List[str]:
        return ['biometric', 'triage']
