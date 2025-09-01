# app/ml/training/train_triage.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os

from app.services.data.data_loader import DataLoader
from app.core.config import settings


class TriagePredictor:
    """Predict triage priority based on patient data"""

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.target_encoder = LabelEncoder()
        self.data_loader = DataLoader()
        self.is_trained = False

    def load_and_preprocess_data(self):
        """Load and preprocess the triage data"""
        # Load the dataset
        df = self.data_loader.load_triage_data()

        # Preprocess the data
        categorical_cols = ['gender', 'province', 'facility_type', 'visit_type',
                            'medical_scheme', 'icd10_code']

        # Encode categorical variables
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le

        # Encode the target variable
        if 'triage_priority' in df.columns:
            self.target_encoder.fit(df['triage_priority'])
            df['triage_priority_encoded'] = self.target_encoder.transform(df['triage_priority'])

        return df

    def train_model(self):
        """Train the triage prediction model"""
        df = self.load_and_preprocess_data()

        # Prepare features and target
        feature_cols = [col for col in df.columns if
                        col not in ['triage_priority', 'triage_priority_encoded', 'patient_id', 'admission_time']]
        X = df[feature_cols]
        y = df['triage_priority_encoded']

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train the model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        self.model.fit(X_train_scaled, y_train)

        # Evaluate the model
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)

        print(f"Model training complete. Train accuracy: {train_score:.3f}, Test accuracy: {test_score:.3f}")
        self.is_trained = True

        # Save the model
        self.save_model()

        return train_score, test_score

    def save_model(self):
        """Save the trained model and preprocessing objects"""
        if not os.path.exists(settings.ML_MODELS_DIR):
            os.makedirs(settings.ML_MODELS_DIR)

        # Save model
        joblib.dump(self.model, settings.TRIAGE_MODEL_PATH)

        # Save preprocessing objects
        joblib.dump(self.scaler, os.path.join(settings.ML_MODELS_DIR, 'scaler.pkl'))
        joblib.dump(self.label_encoders, os.path.join(settings.ML_MODELS_DIR, 'label_encoders.pkl'))
        joblib.dump(self.target_encoder, os.path.join(settings.ML_MODELS_DIR, 'target_encoder.pkl'))

        print(f"Model saved to {settings.TRIAGE_MODEL_PATH}")

    def load_model(self):
        """Load a pre-trained model and preprocessing objects"""
        try:
            self.model = joblib.load(settings.TRIAGE_MODEL_PATH)
            self.scaler = joblib.load(os.path.join(settings.ML_MODELS_DIR, 'scaler.pkl'))
            self.label_encoders = joblib.load(os.path.join(settings.ML_MODELS_DIR, 'label_encoders.pkl'))
            self.target_encoder = joblib.load(os.path.join(settings.ML_MODELS_DIR, 'target_encoder.pkl'))
            self.is_trained = True
            print("Model loaded successfully")
            return True
        except FileNotFoundError:
            print("No pre-trained model found. Please train the model first.")
            return False

    def predict_triage(self, patient_data):
        """
        Predict triage priority based on patient data
        """
        if not self.is_trained:
            # Try to load a pre-trained model
            if not self.load_model():
                # If no model exists, use simple heuristics
                return self._predict_with_heuristics(patient_data)

        try:
            # Convert patient data to DataFrame
            patient_df = pd.DataFrame([patient_data])

            # Preprocess the data (same as training)
            categorical_cols = ['gender', 'province', 'facility_type', 'visit_type',
                                'medical_scheme', 'icd10_code']

            for col in categorical_cols:
                if col in patient_df.columns and col in self.label_encoders:
                    try:
                        patient_df[col] = self.label_encoders[col].transform(patient_df[col].astype(str))
                    except ValueError:
                        # Handle unseen labels by using the most common class
                        patient_df[col] = 0

            # Ensure all expected columns are present
            expected_cols = list(self.label_encoders.keys()) + [
                'age', 'has_medical_aid', 'arrival_via_ambulance',
                'hr_bpm', 'temp_c', 'resp_rate', 'systolic_bp', 'diastolic_bp',
                'o2_sat', 'pain_score'
            ]

            for col in expected_cols:
                if col not in patient_df.columns:
                    patient_df[col] = 0  # Default value for missing columns

            # Scale the features
            patient_scaled = self.scaler.transform(patient_df[expected_cols])

            # Make prediction
            prediction_encoded = self.model.predict(patient_scaled)
            prediction_proba = self.model.predict_proba(patient_scaled)

            # Decode the prediction
            prediction = self.target_encoder.inverse_transform(prediction_encoded)

            return prediction[0], prediction_proba[0]

        except Exception as e:
            print(f"Error in model prediction: {e}")
            # Fall back to heuristic prediction
            return self._predict_with_heuristics(patient_data)

    def _predict_with_heuristics(self, patient_data):
        """
        Fallback prediction using simple heuristics when ML model is not available
        """
        conditions = ['hr_bpm', 'temp_c', 'pain_score', 'o2_sat']
        scores = []

        # Simple heuristic for demo purposes
        if patient_data.get('pain_score', 0) > 7:
            scores.append(2)  # Red
        elif patient_data.get('pain_score', 0) > 4:
            scores.append(1)  # Yellow
        else:
            scores.append(0)  # Green

        if patient_data.get('hr_bpm', 70) > 100 or patient_data.get('hr_bpm', 70) < 50:
            scores.append(2)
        elif patient_data.get('hr_bpm', 70) > 90 or patient_data.get('hr_bpm', 70) < 60:
            scores.append(1)
        else:
            scores.append(0)

        if patient_data.get('o2_sat', 98) < 90:
            scores.append(2)
        elif patient_data.get('o2_sat', 98) < 95:
            scores.append(1)
        else:
            scores.append(0)

        # Determine final priority based on worst score
        final_score = max(scores) if scores else 0
        priority = ['Green', 'Yellow', 'Red'][final_score]

        return priority, [0.0, 0.0, 0.0]  # Placeholder for probabilities

    def get_demo_data(self):
        """Get demo data for visualization purposes"""
        return self.data_loader.load_triage_data()