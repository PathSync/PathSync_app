# src/triage_train.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib


class TriageModelTrainer:
    """Train triage priority prediction model"""

    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        self.scaler = StandardScaler()
        self.encoders = {}

    def preprocess_data(self, df):
        """Preprocess triage data"""
        df_processed = df.copy()

        # Encode categorical variables
        le_arrival = LabelEncoder()
        le_triage = LabelEncoder()

        df_processed['arrival_encoded'] = le_arrival.fit_transform(df['arrival_mode'])
        df_processed['triage_encoded'] = le_triage.fit_transform(df['triage_priority'])

        # Store encoders
        self.encoders['arrival_encoder'] = le_arrival
        self.encoders['triage_encoder'] = le_triage

        # Scale numerical features
        numerical_features = ['heart_rate', 'temp_c', 'resp_rate', 'o2_sat', 'systolic_bp', 'pain_score']
        df_processed[numerical_features] = self.scaler.fit_transform(df_processed[numerical_features])

        return df_processed

    def train(self, df):
        """Train the triage prediction model"""
        # Preprocess data
        df_processed = self.preprocess_data(df)

        # Features and target
        features = ['heart_rate', 'temp_c', 'resp_rate', 'o2_sat',
                    'systolic_bp', 'pain_score', 'arrival_encoded']
        X = df_processed[features]
        y = df_processed['triage_encoded']

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Train model
        self.model.fit(X_train, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test)
        print("Triage Model Performance:")
        print(classification_report(y_test, y_pred,
                                    target_names=self.encoders['triage_encoder'].classes_))

        return self.model

    def save_model(self, filepath):
        """Save model and preprocessors"""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'encoders': self.encoders
        }, filepath)