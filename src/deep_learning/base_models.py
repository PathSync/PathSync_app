from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd

class ModelMetrics:
    
    def __init__(self, accuracy: float, loss: float = 0.0):
        self.accuracy = accuracy
        self.loss = loss
        self.training_time = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'accuracy': self.accuracy,
            'loss': self.loss,
            'training_time': self.training_time
        }

class TrainingHistory:
    
    def __init__(self):
        self.losses = []
        self.accuracies = []
    
    def add_epoch(self, loss: float, accuracy: float):
        self.losses.append(loss)
        self.accuracies.append(accuracy)
    
    def get_final_accuracy(self) -> float:
        return self.accuracies[-1] if self.accuracies else 0.0

class NeuralNetwork(ABC):
    
    def __init__(self, name: str):
        self.name = name
        self.is_trained = False
        self.weights = None
        self.accuracy = 0.0
    
    @abstractmethod
    def prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        pass
    
    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray) -> float:
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass
    
    def get_metrics(self) -> ModelMetrics:
        return ModelMetrics(accuracy=self.accuracy)

class SimpleDeepLearningPredictor:
    
    def __init__(self):
        self.models: Dict[str, NeuralNetwork] = {}
    
    def add_model(self, name: str, model: NeuralNetwork):
        self.models[name] = model
    
    def train_model(self, name: str, data: pd.DataFrame) -> float:
        if name not in self.models:
            raise ValueError(f"Model {name} not found")
        
        model = self.models[name]
        X, y = model.prepare_data(data)
        accuracy = model.train(X, y)
        
        return accuracy
    
    def predict(self, name: str, data: pd.DataFrame) -> np.ndarray:
        if name not in self.models:
            raise ValueError(f"Model {name} not found")
        
        model = self.models[name]
        X, _ = model.prepare_data(data)
        
        return model.predict(X)
    
    def get_model_info(self) -> Dict[str, Any]:
        info = {}
        for name, model in self.models.items():
            info[name] = {
                'name': model.name,
                'is_trained': model.is_trained,
                'accuracy': model.accuracy
            }
        return info
