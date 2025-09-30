import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent))

from predicts import HealthcarePredictor
try:
    from deep_learning import DeepLearningPredictor, NeuralNetworkFactory
    from deep_learning.training import TrainingConfig, TrainingManager
    DEEP_LEARNING_AVAILABLE = True
except ImportError:
    DEEP_LEARNING_AVAILABLE = False
    print("Deep learning components not available - using traditional ML only")
class EnhancedHealthcarePredictor(HealthcarePredictor):
    def __init__(self, use_deep_learning: bool = True):
        super().__init__()
        self.use_deep_learning = use_deep_learning and DEEP_LEARNING_AVAILABLE
        if self.use_deep_learning:
            self._initialize_deep_learning_models()
    def _initialize_deep_learning_models(self):
        try:
            print("Initializing deep learning models...")
            self.deep_learning_predictor = DeepLearningPredictor()
            biometric_model = NeuralNetworkFactory.create_biometric_nn()
            triage_model = NeuralNetworkFactory.create_triage_nn()
            self.deep_learning_predictor.add_model('biometric', biometric_model)
            self.deep_learning_predictor.add_model('triage', triage_model)
            print("Deep learning models initialized successfully!")
        except Exception as e:
            print(f"Warning: Failed to initialize deep learning models: {e}")
            self.use_deep_learning = False
    def predict_biometric_enhanced(self, age: int, gender: str, province: str, biometric_score: float):
        traditional_result, traditional_confidence = super().predict_biometric(age, gender, province, biometric_score)
        if not self.use_deep_learning or not hasattr(self, 'deep_learning_predictor'):
            return traditional_result, traditional_confidence
        try:
            data = pd.DataFrame([{
                'age': age,
                'gender': gender,
                'province': province,
                'biometric_score': biometric_score,
                'citizenship_status': traditional_result
            }])
            model_info = self.deep_learning_predictor.get_model_info()
            if 'biometric' in model_info and model_info['biometric']['is_trained']:
                dl_predictions = self.deep_learning_predictor.predict('biometric', data)
                dl_prediction = dl_predictions[0]
                if traditional_result == dl_prediction:
                    final_result = traditional_result
                    final_confidence = min(traditional_confidence * 1.1, 1.0)
                else:
                    final_result = traditional_result
                    final_confidence = traditional_confidence * 0.8
                return final_result, final_confidence
            else:
                print("Deep learning biometric model not trained, using traditional prediction")
        except Exception as e:
            print(f"Deep learning prediction failed, using traditional: {e}")
        return traditional_result, traditional_confidence
    def predict_triage_enhanced(self, age: int, gender: str, hr_bpm: int, temp_c: float,
                              resp_rate: int, systolic_bp: int, diastolic_bp: int,
                              o2_sat: int, pain_score: int):
        traditional_result, traditional_confidence = super().predict_triage(
            age, gender, hr_bpm, temp_c, resp_rate, systolic_bp, diastolic_bp, o2_sat, pain_score
        )
        if not self.use_deep_learning or not hasattr(self, 'deep_learning_predictor'):
            return traditional_result, traditional_confidence
        try:
            data = pd.DataFrame([{
                'age': age,
                'gender': gender,
                'hr_bpm': hr_bpm,
                'temp_c': temp_c,
                'resp_rate': resp_rate,
                'systolic_bp': systolic_bp,
                'diastolic_bp': diastolic_bp,
                'o2_sat': o2_sat,
                'pain_score': pain_score,
                'triage_priority': traditional_result
            }])
            model_info = self.deep_learning_predictor.get_model_info()
            if 'triage' in model_info and model_info['triage']['is_trained']:
                dl_predictions = self.deep_learning_predictor.predict('triage', data)
                dl_prediction = dl_predictions[0]
                priority_weights = {'Red': 3, 'Yellow': 2, 'Green': 1}
                if traditional_result == 'Red' or dl_prediction == 'Red':
                    final_result = 'Red'
                    final_confidence = max(traditional_confidence, 0.9)
                elif traditional_result == dl_prediction:
                    final_result = traditional_result
                    final_confidence = min(traditional_confidence * 1.1, 1.0)
                else:
                    trad_weight = priority_weights.get(traditional_result, 1) * traditional_confidence
                    dl_weight = priority_weights.get(dl_prediction, 1) * 0.8
                    if trad_weight >= dl_weight:
                        final_result = traditional_result
                        final_confidence = traditional_confidence * 0.9
                    else:
                        final_result = dl_prediction
                        final_confidence = 0.8
                return final_result, final_confidence
            else:
                print("Deep learning triage model not trained, using traditional prediction")
        except Exception as e:
            print(f"Deep learning prediction failed, using traditional: {e}")
        return traditional_result, traditional_confidence
    def train_deep_learning_models(self, biometric_data: pd.DataFrame = None, triage_data: pd.DataFrame = None):
        if not self.use_deep_learning or not hasattr(self, 'deep_learning_predictor'):
            print("Deep learning not enabled")
            return
        print("Training deep learning models...")
        if biometric_data is not None:
            try:
                print("Training biometric deep learning model...")
                accuracy = self.deep_learning_predictor.train_model('biometric', biometric_data)
                print(f"Biometric deep learning model training completed! Accuracy: {accuracy:.4f}")
            except Exception as e:
                print(f"Failed to train biometric deep learning model: {e}")
        if triage_data is not None:
            try:
                print("Training triage deep learning model...")
                accuracy = self.deep_learning_predictor.train_model('triage', triage_data)
                print(f"Triage deep learning model training completed! Accuracy: {accuracy:.4f}")
            except Exception as e:
                print(f"Failed to train triage deep learning model: {e}")
    def get_model_info(self):
        info = {
            'traditional_ml': {
                'biometric_model': 'Random Forest (trained)',
                'triage_model': 'Random Forest (trained)'
            },
            'deep_learning_enabled': self.use_deep_learning,
            'deep_learning_models': {}
        }
        if self.use_deep_learning and hasattr(self, 'deep_learning_predictor'):
            try:
                dl_info = self.deep_learning_predictor.get_model_info()
                info['deep_learning_models'] = dl_info
            except Exception as e:
                print(f"Error getting deep learning model info: {e}")
                info['deep_learning_models'] = {'error': str(e)}
        return info

def demonstrate_enhanced_predictor():
    print("=" * 60)
    print("ENHANCED HEALTHCARE PREDICTOR DEMONSTRATION")
    print("=" * 60)
    print()
    print("1. Initializing Enhanced Predictor...")
    enhanced_predictor = EnhancedHealthcarePredictor(use_deep_learning=True)
    model_info = enhanced_predictor.get_model_info()
    print("Available models:")
    print(f"- Traditional ML: {list(model_info['traditional_ml'].keys())}")
    print(f"- Deep Learning Enabled: {model_info['deep_learning_enabled']}")
    if model_info['deep_learning_enabled'] and model_info['deep_learning_models']:
        if 'error' not in model_info['deep_learning_models']:
            print(f"- Deep Learning Models: {list(model_info['deep_learning_models'].keys())}")
        else:
            print(f"- Deep Learning Models: Error - {model_info['deep_learning_models']['error']}")
    print()
    print("2. Testing Biometric Prediction...")
    bio_result, bio_confidence = enhanced_predictor.predict_biometric_enhanced(
        age=35, gender='Male', province='Gauteng', biometric_score=0.85
    )
    print(f"Biometric Result: {bio_result} (confidence: {bio_confidence:.4f})")
    print()
    print("3. Testing Triage Prediction...")
    triage_result, triage_confidence = enhanced_predictor.predict_triage_enhanced(
        age=45, gender='Female', hr_bpm=110, temp_c=38.2, resp_rate=22,
        systolic_bp=140, diastolic_bp=90, o2_sat=96, pain_score=7
    )
    print(f"Triage Result: {triage_result} (confidence: {triage_confidence:.4f})")
    print()
    print("4. Comparing with Traditional Predictor...")
    traditional_predictor = HealthcarePredictor()
    trad_bio_result, trad_bio_conf = traditional_predictor.predict_biometric(
        age=35, gender='Male', province='Gauteng', biometric_score=0.85
    )
    trad_triage_result, trad_triage_conf = traditional_predictor.predict_triage(
        age=45, gender='Female', hr_bpm=110, temp_c=38.2, resp_rate=22,
        systolic_bp=140, diastolic_bp=90, o2_sat=96, pain_score=7
    )
    print("Comparison Results:")
    print(f"Biometric - Traditional: {trad_bio_result} ({trad_bio_conf:.4f}) vs Enhanced: {bio_result} ({bio_confidence:.4f})")
    print(f"Triage - Traditional: {trad_triage_result} ({trad_triage_conf:.4f}) vs Enhanced: {triage_result} ({triage_confidence:.4f})")
    print()
    print("Enhanced predictor demonstration completed!")
    print("Note: Deep learning models need training data to show full capabilities.")

if __name__ == "__main__":
    demonstrate_enhanced_predictor()
