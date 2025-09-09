# predict.py
import joblib
import numpy as np


class HealthcarePredictor:
    def __init__(self):
        # Load trained models
        self.biometric_model = joblib.load('models/biometric_model.pkl')
        self.triage_model = joblib.load('models/triage_model.pkl')

        # Load preprocessing info
        self.biometric_preprocessing = joblib.load('models/biometric_preprocessing.pkl')
        self.triage_preprocessing = joblib.load('models/triage_preprocessing.pkl')

    def predict_biometric(self, age, gender, province, biometric_score):
        # Preprocess input
        gender_encoded = 0 if gender.lower() == 'male' else 1 if gender.lower() == 'female' else 2
        province_encoded = self.biometric_preprocessing['province_map'].get(province, 0)

        # Make prediction
        features = [age, gender_encoded, province_encoded, biometric_score]
        prediction = self.biometric_model.predict([features])[0]
        confidence = np.max(self.biometric_model.predict_proba([features]))

        # Map back to original labels
        citizenship_map = {v: k for k, v in self.biometric_preprocessing['citizenship_map'].items()}
        return citizenship_map.get(prediction, 'Review'), confidence

    def predict_triage(self, age, gender, hr_bpm, temp_c, resp_rate, systolic_bp, diastolic_bp, o2_sat, pain_score):
        # Preprocess input
        gender_encoded = 0 if gender.lower() == 'male' else 1 if gender.lower() == 'female' else 2

        # Make prediction
        features = [age, gender_encoded, hr_bpm, temp_c, resp_rate, systolic_bp, diastolic_bp, o2_sat, pain_score]
        prediction = self.triage_model.predict([features])[0]
        confidence = np.max(self.triage_model.predict_proba([features]))

        # Map back to original labels
        priority_map = {v: k for k, v in self.triage_preprocessing.items()}
        return priority_map.get(prediction, 'Green'), confidence


# For testing
if __name__ == "__main__":
    predictor = HealthcarePredictor()
    result = predictor.predict_biometric(35, 'Male', 'Gauteng', 0.85)
    print(f"Biometric prediction: {result}")