
import joblib
import numpy as np


class HealthcarePredictor:
    """Unified predictor for both biometric and triage models"""

    def __init__(self, biometric_model_path, triage_model_path):
        self.biometric_model = joblib.load(biometric_model_path)
        self.triage_model = joblib.load(triage_model_path)

    def predict_identity_verification(self, patient_data):
        """Predict identity verification status"""

        features = self._prepare_biometric_features(patient_data)

        # Make prediction
        prediction = self.biometric_model['model'].predict([features])[0]
        probability = self.biometric_model['model'].predict_proba([features])[0]

        return bool(prediction), probability

    def predict_triage_priority(self, clinical_data):
        """Predict triage priority"""
        # Preprocess input data
        features = self._prepare_triage_features(clinical_data)

        # Make prediction
        prediction_encoded = self.triage_model['model'].predict([features])[0]
        priority = self.triage_model['encoders']['triage_encoder'].inverse_transform([prediction_encoded])[0]
        probability = self.triage_model['model'].predict_proba([features])[0]

        return priority, probability

    def _prepare_biometric_features(self, data):
        """Prepare biometric features for prediction"""
        # Convert categorical data using stored encoders
        gender_encoded = self.biometric_model['preprocessor']['gender_encoder'].transform([data['gender']])[0]
        citizenship_encoded = \
        self.biometric_model['preprocessor']['citizenship_encoder'].transform([data['citizenship_status']])[0]

        return [
            data['age'],
            gender_encoded,
            data['facial_match_score'],
            data['fingerprint_match_score'],
            data['has_medical_aid'],
            citizenship_encoded
        ]

    def _prepare_triage_features(self, data):
        """Prepare triage features for prediction"""
        # Encode categorical data
        arrival_encoded = self.triage_model['encoders']['arrival_encoder'].transform([data['arrival_mode']])[0]

        # Scale numerical features
        numerical_features = [
            data['heart_rate'],
            data['temp_c'],
            data['resp_rate'],
            data['o2_sat'],
            data['systolic_bp'],
            data['pain_score']
        ]

        scaled_features = self.triage_model['scaler'].transform([numerical_features])[0]

        return list(scaled_features) + [arrival_encoded]