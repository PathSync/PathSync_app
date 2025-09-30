from typing import Dict, Any
from .base_models import TrainingHistory

class TrainingConfig:
    def __init__(self):
        self.epochs = 50
        self.learning_rate = 0.01
        self.verbose = True

    def with_epochs(self, epochs: int) -> 'TrainingConfig':
        self.epochs = epochs
        return self

    def with_learning_rate(self, lr: float) -> 'TrainingConfig':
        self.learning_rate = lr
        return self

    def with_verbose(self, verbose: bool) -> 'TrainingConfig':
        self.verbose = verbose
        return self

class TrainingManager:
    def __init__(self, config: TrainingConfig = None):
        self.config = config or TrainingConfig()
        self.training_sessions = []

    def train_model(self, model, X, y) -> TrainingHistory:
        print(f"Starting training with {self.config.epochs} epochs...")
        accuracy = model.train(X, y)
        history = TrainingHistory()
        history.add_epoch(loss=0.1, accuracy=accuracy)
        session = {
            'model_name': model.name,
            'final_accuracy': accuracy,
            'epochs': self.config.epochs
        }
        self.training_sessions.append(session)
        return history

    def get_training_summary(self) -> Dict[str, Any]:
        if not self.training_sessions:
            return {'message': 'No training sessions recorded'}
        return {
            'total_sessions': len(self.training_sessions),
            'sessions': self.training_sessions
        }