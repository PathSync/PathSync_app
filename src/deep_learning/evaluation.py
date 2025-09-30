from typing import Dict, Any, List
import numpy as np
from .base_models import ModelMetrics, NeuralNetwork

class ModelEvaluator:
    def __init__(self):
        self.evaluation_history = []
    
    def evaluate_model(self, model: NeuralNetwork, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        print(f"Evaluating model: {model.name}")
        if not model.is_trained:
            return {
                'model_name': model.name,
                'error': 'Model not trained',
                'accuracy': 0.0
            }
        predictions = model.predict(X)
        if isinstance(y[0], str):
            accuracy = np.mean(predictions == y)
        else:
            label_map = getattr(model, 'label_map', {})
            if label_map:
                reverse_map = {v: k for k, v in label_map.items()}
                y_numeric = [label_map.get(label, 0) for label in y] if isinstance(y[0], str) else y
                pred_numeric = [label_map.get(pred, 0) for pred in predictions]
                accuracy = np.mean(np.array(pred_numeric) == np.array(y_numeric))
            else:
                accuracy = np.mean(predictions == y)
        result = {
            'model_name': model.name,
            'accuracy': float(accuracy),
            'is_trained': model.is_trained,
            'predictions_sample': predictions[:5].tolist() if len(predictions) > 5 else predictions.tolist()
        }
        self.evaluation_history.append(result)
        print(f"Evaluation completed. Accuracy: {accuracy:.4f}")
        return result
    
    def compare_models(self, models: List[SimpleNeuralNetwork], X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        print(f"Comparing {len(models)} models...")
        results = []
        for model in models:
            result = self.evaluate_model(model, X, y)
            results.append(result)
        valid_results = [r for r in results if 'error' not in r]
        if valid_results:
            best_model = max(valid_results, key=lambda x: x['accuracy'])
            best_name = best_model['model_name']
            best_accuracy = best_model['accuracy']
        else:
            best_name = "None"
            best_accuracy = 0.0
        comparison = {
            'total_models': len(models),
            'results': results,
            'best_model': best_name,
            'best_accuracy': best_accuracy
        }
        print(f"Comparison completed. Best model: {best_name} ({best_accuracy:.4f})")
        return comparison
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        if not self.evaluation_history:
            return {'message': 'No evaluations performed'}
        valid_evaluations = [e for e in self.evaluation_history if 'error' not in e]
        if not valid_evaluations:
            return {'message': 'No successful evaluations'}
        avg_accuracy = np.mean([e['accuracy'] for e in valid_evaluations])
        best_evaluation = max(valid_evaluations, key=lambda x: x['accuracy'])
        return {
            'total_evaluations': len(self.evaluation_history),
            'successful_evaluations': len(valid_evaluations),
            'average_accuracy': avg_accuracy,
            'best_model': best_evaluation['model_name'],
            'best_accuracy': best_evaluation['accuracy'],
            'evaluation_history': self.evaluation_history
        }