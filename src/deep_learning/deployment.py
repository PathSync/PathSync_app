from typing import Dict, Any, List
import pickle
import os
from datetime import datetime
from .base_models import SimpleNeuralNetwork


class ModelRepository:
    def __init__(self, base_path: str = "models/simple_deep_learning"):
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)
        self.models = {}
    
    def save_model(self, model: SimpleNeuralNetwork, model_id: str) -> str:
        if not model.is_trained:
            raise ValueError("Cannot save untrained model")
        
        filepath = os.path.join(self.base_path, f"{model_id}.pkl")
        
        model_data = {
            'name': model.name,
            'weights': model.weights,
            'is_trained': model.is_trained,
            'accuracy': model.accuracy,
            'saved_at': datetime.now().isoformat()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        self.models[model_id] = {
            'filepath': filepath,
            'model_name': model.name,
            'saved_at': model_data['saved_at'],
            'accuracy': model.accuracy
        }
        
        print(f"Model saved: {model_id}")
        return model_id
    
    def load_model(self, model_id: str, model_class) -> SimpleNeuralNetwork:
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        filepath = self.models[model_id]['filepath']
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        model = model_class()
        model.name = model_data['name']
        model.weights = model_data['weights']
        model.is_trained = model_data['is_trained']
        model.accuracy = model_data['accuracy']
        
        print(f"Model loaded: {model_id}")
        return model
    
    def list_models(self) -> List[Dict[str, Any]]:
        return [
            {
                'model_id': model_id,
                'model_name': info['model_name'],
                'saved_at': info['saved_at'],
                'accuracy': info['accuracy']
            }
            for model_id, info in self.models.items()
        ]


class SimpleInferenceEngine:
    def __init__(self, model: SimpleNeuralNetwork):
        self.model = model
        self.prediction_count = 0
        
        if not model.is_trained:
            raise ValueError("Cannot create inference engine with untrained model")
    
    def predict_single(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        start_time = datetime.now()
        
        if 'biometric' in self.model.name.lower():
            import pandas as pd
            data = pd.DataFrame([{
                'age': input_data.get('age', 35),
                'gender': input_data.get('gender', 'Male'),
                'province': input_data.get('province', 'Gauteng'),
                'biometric_score': input_data.get('biometric_score', 0.8),
                'citizenship_status': 'SA'
            }])
        else:
            import pandas as pd
            data = pd.DataFrame([{
                'age': input_data.get('age', 35),
                'gender': input_data.get('gender', 'Male'),
                'hr_bpm': input_data.get('hr_bpm', 75),
                'temp_c': input_data.get('temp_c', 36.6),
                'resp_rate': input_data.get('resp_rate', 16),
                'systolic_bp': input_data.get('systolic_bp', 120),
                'diastolic_bp': input_data.get('diastolic_bp', 80),
                'o2_sat': input_data.get('o2_sat', 98),
                'pain_score': input_data.get('pain_score', 3),
                'triage_priority': 'Green'
            }])
        
        X, _ = self.model.prepare_data(data)
        prediction = self.model.predict(X)[0]
        
        end_time = datetime.now()
        inference_time = (end_time - start_time).total_seconds()
        
        self.prediction_count += 1
        
        return {
            'prediction': prediction,
            'inference_time': inference_time,
            'timestamp': end_time.isoformat()
        }
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            'model_name': self.model.name,
            'total_predictions': self.prediction_count,
            'model_accuracy': self.model.accuracy
        }


class SimpleModelDeployment:
    def __init__(self, repository: SimpleModelRepository):
        self.repository = repository
        self.deployed_models: Dict[str, SimpleInferenceEngine] = {}
        self.deployments = []
    
    def deploy_model(self, model: SimpleNeuralNetwork, deployment_name: str) -> SimpleInferenceEngine:
        print(f"Deploying model as {deployment_name}...")
        engine = SimpleInferenceEngine(model)
        self.deployed_models[deployment_name] = engine
        deployment_record = {
            'deployment_name': deployment_name,
            'model_name': model.name,
            'deployed_at': datetime.now().isoformat(),
            'status': 'active'
        }
        self.deployments.append(deployment_record)
        print(f"Model deployed successfully: {deployment_name}")
        return engine
    
    def get_inference_engine(self, deployment_name: str) -> SimpleInferenceEngine:
        if deployment_name not in self.deployed_models:
            raise ValueError(f"Deployment {deployment_name} not found")
        return self.deployed_models[deployment_name]
    
    def list_deployments(self) -> List[Dict[str, Any]]:
        deployments = []
        for deployment in self.deployments:
            deployment_copy = deployment.copy()
            deployment_name = deployment['deployment_name']
            if deployment_name in self.deployed_models:
                engine = self.deployed_models[deployment_name]
                deployment_copy['stats'] = engine.get_stats()
            deployments.append(deployment_copy)
        return deployments
    
    def get_deployment_summary(self) -> Dict[str, Any]:
        active_deployments = len(self.deployed_models)
        total_predictions = sum(
            engine.prediction_count for engine in self.deployed_models.values()
        )
        return {
            'active_deployments': active_deployments,
            'total_deployments': len(self.deployments),
            'total_predictions': total_predictions,
            'deployments': self.deployments
        }